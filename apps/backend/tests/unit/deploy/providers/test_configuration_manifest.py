"""Tests for deploying published configuration manifests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import llm_d_lens.deploy.providers.configuration_manifest as configuration_manifest
from llm_d_lens.deploy.providers.configuration_manifest import ConfigurationManifestAdapter
from llm_d_lens.deploy.providers.guide_adapter import GuideDefinition, GuideDeploymentArtifact
from llm_d_lens.deploy.providers.target_node import apply_placement


class _Runner:
    def __init__(self) -> None:
        self.copy_model_secret = AsyncMock()
        self.commands: list[list[str]] = []

    async def __call__(self, command: list[str]) -> tuple[int, str, str]:
        self.commands.append(command)
        return 0, "ok", ""


class _RoutedDelegate:
    published_manifest_lifecycle = True

    def __init__(self) -> None:
        self.router_installs: list[str] = []

    async def install_published_router(self, artifact, context) -> None:
        self.router_installs.append(context["namespace"])
        context.update(
            {
                "router_effective_values": "router: {}\n",
                "router_effective_values_checksum": "sha256:values",
                "router_rendered_manifest": "kind: Deployment\n",
                "router_rendered_manifest_checksum": "sha256:manifest",
            }
        )


def _asset(name: str, content: str) -> dict[str, str]:
    return {
        "name": name,
        "content": content,
        "checksum": f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize(
    "policy,available,expected",
    [
        ({"mode": "auto"}, [], None),
        ({"mode": "auto"}, ["nvidia"], "nvidia"),
        ({"mode": "node-default"}, [], None),
        ({"mode": "named", "name": "custom"}, ["custom"], "custom"),
    ],
)
async def test_runtime_policy_derives_deployment_manifest_without_mutating_configuration(
    monkeypatch, tmp_path, directory, policy, available, expected
):
    manifest = yaml.safe_dump(
        {
            "kind": "Pod",
            "spec": {
                **({"runtimeClassName": "nvidia"} if policy["mode"] == "node-default" else {}),
                "containers": [{"name": "modelserver", "resources": {"limits": {"nvidia.com/gpu": "1"}}}],
            },
        }
    )
    source = tmp_path / "source.yaml"
    source.write_text(manifest)
    if directory:
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        (overlay / "manifest.yaml").write_text(manifest)
    delegate = SimpleNamespace(
        _accelerator="nvidia",
        render=AsyncMock(
            return_value=GuideDeploymentArtifact(
                guide_id="baseline-vllm",
                artifact_hash="source",
                manifest_ref=str(overlay if directory else source),
                manifest_checksum="sha256:source",
            )
        ),
    )
    discovery = AsyncMock(
        return_value={
            "items": [{"metadata": {"name": name}} for name in available],
            "nodes": [{"metadata": {"name": "gpu-node"}}],
        }
    )
    monkeypatch.setattr(configuration_manifest, "discover_runtime_classes", discovery)
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llmd-", 5, tmp_path / "output")
    artifact = await adapter.render(
        GuideDefinition("baseline-vllm", "source", "hash", "stable"),
        {
            "_runtime_class_policy": policy,
            "_cluster_id": "cluster-a",
        },
    )
    path = Path(artifact.manifest_ref)
    if directory:
        path = path / "manifest.yaml"
    assert yaml.safe_load(path.read_text())["spec"].get("runtimeClassName") == expected
    assert source.read_text() == manifest
    assert artifact.deployment_contract["runtimeClass"]["policy"] == policy
    assert artifact.artifact_hash != "source"
    if policy["mode"] == "node-default":
        discovery.assert_not_awaited()
    else:
        discovery.assert_awaited_once_with("cluster-a")


@pytest.mark.asyncio
async def test_directory_runtime_policies_do_not_overwrite_another_deployment(monkeypatch, tmp_path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    manifest = yaml.safe_dump({"kind": "Pod", "spec": {"containers": [{"name": "modelserver"}]}})
    (overlay / "manifest.yaml").write_text(manifest)
    (overlay / "kustomization.yaml").write_text("resources:\n- manifest.yaml\n")
    delegate = SimpleNamespace(
        render=AsyncMock(
            return_value=GuideDeploymentArtifact(
                guide_id="optimized-baseline",
                artifact_hash="source",
                manifest_ref=str(overlay),
                manifest_checksum="sha256:source",
            )
        )
    )
    monkeypatch.setattr(
        configuration_manifest,
        "discover_runtime_classes",
        AsyncMock(
            return_value={
                "items": [{"metadata": {"name": name}} for name in ("first", "second")],
                "nodes": [{"metadata": {"name": "gpu-node"}}],
            }
        ),
    )
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llmd-", 5, tmp_path / "output")
    definition = GuideDefinition("optimized-baseline", "source", "hash", "stable")
    first = await adapter.render(
        definition,
        {
            "_runtime_class_policy": {"mode": "named", "name": "first"},
            "_cluster_id": "cluster-a",
        },
    )
    first_manifest = (Path(first.manifest_ref) / "manifest.yaml").read_text()
    second = await adapter.render(
        definition,
        {
            "_runtime_class_policy": {"mode": "named", "name": "second"},
            "_cluster_id": "cluster-a",
        },
    )
    assert first.manifest_ref != second.manifest_ref
    assert (Path(first.manifest_ref) / "manifest.yaml").read_text() == first_manifest
    assert yaml.safe_load(first_manifest)["spec"]["runtimeClassName"] == "first"
    assert (
        yaml.safe_load((Path(second.manifest_ref) / "manifest.yaml").read_text())["spec"]["runtimeClassName"]
        == "second"
    )
    assert (overlay / "manifest.yaml").read_text() == manifest
    assert (Path(first.manifest_ref) / "kustomization.yaml").read_text() == "resources:\n- manifest.yaml\n"


@pytest.mark.parametrize(
    "provider",
    ["baseline-vllm", "pd-disaggregation", "tiered-prefix-cache", "precise-prefix-cache-routing", "optimized-baseline"],
)
@pytest.mark.parametrize("invalid", [False, True])
def test_provider_serving_controls_validate_explicit_values_and_fall_back_to_model(tmp_path, provider, invalid):
    from llm_d_lens.deploy.providers.baseline_vllm import BaselineVllmAdapter
    from llm_d_lens.deploy.providers.helm_kustomize import HelmKustomizeGuideAdapter
    from llm_d_lens.deploy.providers.optimized_baseline import OptimizedBaselineGuideAdapter
    from llm_d_lens.deploy.providers.pd_disaggregation import PdDisaggregationAdapter
    from llm_d_lens.deploy.providers.precise_prefix_cache_routing import PrecisePrefixCacheRoutingAdapter

    overrides = {
        "model": {"name": "org/model", "maxModelLen": 4096},
        "runtime": {"image": "example/vllm:1", "imageMode": "use-upstream-image"},
        "prefill": {"replicaCount": 1, "tensorParallelSize": 1},
        "decode": {"replicaCount": 1, "tensorParallelSize": 1},
    }
    if invalid:
        overrides["decode"]["maxModelLen"] = 0

    def parameters():
        if provider == "optimized-baseline":
            adapter = OptimizedBaselineGuideAdapter.__new__(OptimizedBaselineGuideAdapter)
            return adapter._deployment_parameters(overrides)[2]["serving_arguments"]
        if provider == "baseline-vllm":
            return BaselineVllmAdapter._parameters(overrides)["serving_arguments"]
        if provider == "pd-disaggregation":
            adapter = PdDisaggregationAdapter.__new__(PdDisaggregationAdapter)
            adapter._accelerator = "xpu"
            adapter._sources = {".": tmp_path}
            parsed = adapter._parameters(overrides)
            assert parsed["prefill"]["serving_arguments"] == {"max-model-len": 4096}
            return parsed["decode"]["serving_arguments"]
        if provider == "tiered-prefix-cache":
            adapter = HelmKustomizeGuideAdapter.__new__(HelmKustomizeGuideAdapter)
            adapter._accelerator = "xpu"
            adapter._descriptor = SimpleNamespace(variants={"native/cpu": tmp_path}, guide_id=provider)
        else:
            adapter = PrecisePrefixCacheRoutingAdapter.__new__(PrecisePrefixCacheRoutingAdapter)
        return adapter._parameters(overrides)["serving_arguments"]

    if invalid:
        with pytest.raises(ValueError, match="maxModelLen"):
            parameters()
    else:
        assert parameters() == {"max-model-len": 4096}


def test_same_node_placement_adds_required_pod_affinity():
    manifest = yaml.safe_dump(
        {
            "kind": "Deployment",
            "metadata": {"name": "model"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "model"}},
                    "spec": {"containers": [{"name": "modelserver"}]},
                }
            },
        }
    )
    rendered, count = apply_placement(manifest, {"replicas": "same-node"})

    pod = yaml.safe_load(rendered)["spec"]["template"]
    assert count == 1
    term = pod["spec"]["affinity"]["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][0]
    assert term["labelSelector"]["matchLabels"] == {
        "lens.llm-d.ai/placement-group": pod["metadata"]["labels"]["lens.llm-d.ai/placement-group"]
    }
    assert term["topologyKey"] == "kubernetes.io/hostname"


@pytest.mark.parametrize("replicas,affinity_type", [("spread", "podAntiAffinity"), ("none", "podAffinity")])
def test_placement_spread_and_default_prefer_same_node(replicas, affinity_type):
    manifest = yaml.safe_dump(
        {
            "kind": "Deployment",
            "metadata": {"name": "model"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "model"}},
                    "spec": {"containers": [{"name": "modelserver"}]},
                }
            },
        }
    )
    rendered, count = apply_placement(manifest, {"replicas": replicas})
    affinity = yaml.safe_load(rendered)["spec"]["template"]["spec"]["affinity"]
    assert count == 1
    if replicas == "spread":
        assert (
            affinity[affinity_type]["requiredDuringSchedulingIgnoredDuringExecution"][0]["topologyKey"]
            == "kubernetes.io/hostname"
        )
    else:
        assert affinity["podAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0]["weight"] == 100


def test_balanced_placement_spreads_replicas_across_eligible_nodes():
    manifest = yaml.safe_dump(
        {
            "kind": "Deployment",
            "metadata": {"name": "model"},
            "spec": {
                "template": {
                    "metadata": {"labels": {"app": "model"}},
                    "spec": {"containers": [{"name": "modelserver"}]},
                }
            },
        }
    )
    rendered, count = apply_placement(manifest, {"replicas": "balanced"})
    template = yaml.safe_load(rendered)["spec"]["template"]
    assert count == 1
    assert template["spec"]["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "DoNotSchedule",
            "labelSelector": {
                "matchLabels": {
                    "lens.llm-d.ai/placement-group": template["metadata"]["labels"]["lens.llm-d.ai/placement-group"]
                }
            },
        }
    ]
    assert "affinity" not in template["spec"]


@pytest.mark.parametrize("replicas", ["prefer-same-node", "same-node", "balanced", "spread"])
def test_placement_groups_modelserver_deployments_in_one_manifest(replicas):
    documents = [
        {
            "kind": "Deployment",
            "metadata": {"name": name},
            "spec": {"template": {"spec": {"containers": [{"name": container}]}}},
        }
        for name, container in [("pd-prefill", "modelserver"), ("pd-router", "router"), ("pd-decode", "modelserver")]
    ]
    rendered, count = apply_placement(yaml.safe_dump_all(documents), {"replicas": replicas})
    prefill, router, decode = list(yaml.safe_load_all(rendered))
    assert count == 2
    assert "lens.llm-d.ai/placement-group" not in router["spec"]["template"].get("metadata", {}).get("labels", {})
    prefill_template = prefill["spec"]["template"]
    decode_template = decode["spec"]["template"]
    group = prefill_template["metadata"]["labels"]["lens.llm-d.ai/placement-group"]
    assert decode_template["metadata"]["labels"]["lens.llm-d.ai/placement-group"] == group
    for template in [prefill_template, decode_template]:
        pod_spec = template["spec"]
        if replicas == "balanced":
            selector = pod_spec["topologySpreadConstraints"][0]["labelSelector"]
        else:
            affinity_type = "podAntiAffinity" if replicas == "spread" else "podAffinity"
            strength = (
                "requiredDuringSchedulingIgnoredDuringExecution"
                if replicas in {"spread", "same-node"}
                else "preferredDuringSchedulingIgnoredDuringExecution"
            )
            term = pod_spec["affinity"][affinity_type][strength][0]
            selector = (term.get("podAffinityTerm") or term)["labelSelector"]
        assert selector == {"matchLabels": {"lens.llm-d.ai/placement-group": group}}


def test_placement_node_label_only_with_same_node():
    manifest = yaml.safe_dump(
        {
            "kind": "Deployment",
            "metadata": {"name": "model"},
            "spec": {"template": {"spec": {"containers": [{"name": "modelserver"}]}}},
        }
    )
    rendered, _ = apply_placement(manifest, {"replicas": "same-node", "nodeLabel": {"key": "pool", "value": "gpu"}})
    assert yaml.safe_load(rendered)["spec"]["template"]["spec"]["nodeSelector"] == {"pool": "gpu"}
    with pytest.raises(ValueError, match="node label requires same-node"):
        apply_placement(manifest, {"replicas": "spread", "nodeLabel": {"key": "pool", "value": "gpu"}})


def test_leader_worker_templates_prefer_same_node_without_forcing_it():
    template = {"metadata": {"labels": {"app": "model"}}, "spec": {"containers": [{"name": "worker"}]}}
    manifest = yaml.safe_dump(
        {
            "kind": "LeaderWorkerSet",
            "metadata": {"name": "model"},
            "spec": {"leaderWorkerTemplate": {"leaderTemplate": template, "workerTemplate": template.copy()}},
        }
    )
    rendered, count = apply_placement(manifest, {"replicas": "prefer-same-node"})
    templates = yaml.safe_load(rendered)["spec"]["leaderWorkerTemplate"]
    assert count == 2
    for key in ("leaderTemplate", "workerTemplate"):
        assert (
            templates[key]["spec"]["affinity"]["podAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0][
                "weight"
            ]
            == 100
        )
    for replicas in ("spread", "balanced"):
        with pytest.raises(ValueError, match="not supported for LeaderWorkerSet"):
            apply_placement(manifest, {"replicas": replicas})


@pytest.mark.asyncio
async def test_render_preserves_validated_deployment_bundle_and_router_profile(monkeypatch, tmp_path: Path):
    manifest = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: model\n"
    bundle = {
        "schemaVersion": "guide-deployment-bundle.v1",
        "guide": "optimized-baseline",
        "sourceCommit": "a" * 40,
        "helm": {
            "chart": "oci://router",
            "version": "v1",
            "releaseName": "optimized-baseline",
            "values": [_asset("router-effective.yaml", "router: {}\n")],
        },
        "resources": [],
    }
    deployment = {
        "readinessDeployments": ["model"],
        "endpoint": {"protocol": "http", "serviceName": "model", "port": 8000},
    }
    monkeypatch.setattr(
        configuration_manifest,
        "validate_configuration_manifest_content",
        lambda *_: (
            manifest,
            "sha256:source",
            deployment,
            {
                "guide": "optimized-baseline",
                "commit": "a" * 40,
            },
        ),
    )
    adapter = ConfigurationManifestAdapter(AsyncMock(), _Runner(), "llm-d-bench-", 30, tmp_path)
    definition = GuideDefinition("optimized-baseline", "source", "hash", "supported")

    artifact = await adapter.render(
        definition,
        {
            "_deployment_namespace": "llm-d-bench-run",
            "routerProfile": "load-only",
            "officialGuide": {"deploymentBundle": bundle},
        },
    )

    assert artifact.deployment_contract["deploymentBundle"] == bundle
    assert artifact.deployment_contract["routerProfile"] == "load-only"


@pytest.mark.asyncio
async def test_target_node_does_not_read_kustomize_directory_as_manifest(monkeypatch, tmp_path: Path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.yaml").write_text("kind: Service\n", encoding="utf-8")
    delegate = AsyncMock()
    delegate.render.return_value = GuideDeploymentArtifact(
        guide_id="optimized-baseline", artifact_hash="hash", manifest_ref=str(overlay)
    )
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llm-d-bench-", 30, tmp_path)
    definition = GuideDefinition("optimized-baseline", "source", "hash", "supported")
    monkeypatch.setattr(configuration_manifest, "configured_target_node", lambda: "smc-19")

    artifact = await adapter.render(definition, {})

    assert Path(artifact.manifest_ref).is_dir()
    assert Path(artifact.manifest_ref) != overlay
    assert (Path(artifact.manifest_ref) / "manifest.yaml").read_text() == "kind: Service\n"


@pytest.mark.asyncio
async def test_placement_accepts_overlay_when_delegate_applied_it(monkeypatch, tmp_path: Path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.yaml").write_text("kind: Service\n", encoding="utf-8")
    delegate = AsyncMock()
    delegate.render.return_value = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(overlay),
        deployment_contract={"placementApplied": True},
    )
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llm-d-bench-", 30, tmp_path)
    monkeypatch.setattr(configuration_manifest, "configured_target_node", lambda: None)

    artifact = await adapter.render(
        GuideDefinition("optimized-baseline", "source", "hash", "supported"),
        {"_placement": {"replicas": "spread"}},
    )

    assert Path(artifact.manifest_ref).is_dir()
    assert Path(artifact.manifest_ref) != overlay
    assert artifact.deployment_contract["placementApplied"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("replicas", ["spread", "balanced"])
async def test_render_applies_advanced_placement_to_modelserver_manifest(monkeypatch, tmp_path: Path, replicas: str):
    manifest = yaml.safe_dump(
        {
            "kind": "Deployment",
            "metadata": {"name": "model"},
            "spec": {"template": {"spec": {"containers": [{"name": "modelserver"}]}}},
        }
    )
    source = tmp_path / "source.yaml"
    source.write_text(manifest, encoding="utf-8")
    delegate = AsyncMock()
    delegate.render.return_value = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="source",
        manifest_ref=str(source),
        manifest_checksum="sha256:source",
    )
    monkeypatch.setattr(configuration_manifest, "instrument_manifest", lambda artifact, _: artifact)
    monkeypatch.setattr(configuration_manifest, "configured_target_node", lambda: None)
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llm-d-bench-", 30, tmp_path)
    artifact = await adapter.render(
        GuideDefinition("optimized-baseline", "source", "hash", "supported"),
        {"_placement": {"replicas": replicas}},
    )
    pod_spec = yaml.safe_load(Path(artifact.manifest_ref).read_text(encoding="utf-8"))["spec"]["template"]["spec"]
    if replicas == "spread":
        assert pod_spec["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    else:
        assert pod_spec["topologySpreadConstraints"][0]["whenUnsatisfiable"] == "DoNotSchedule"
    assert artifact.manifest_checksum != "sha256:source"


@pytest.mark.asyncio
async def test_deploy_delegates_kustomize_directory_artifact(tmp_path: Path):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    artifact = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(overlay),
        deployment_contract={"routerProfile": "optimized-baseline"},
    )
    runner = _Runner()
    delegate = AsyncMock()
    delegate.deploy.return_value = {"namespace": "llm-d-bench-run"}
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)
    context = {"namespace": "llm-d-bench-run"}

    assert await adapter.deploy(artifact, context) == {"namespace": "llm-d-bench-run"}
    delegate.deploy.assert_awaited_once_with(artifact, context)
    assert runner.commands == []


@pytest.mark.asyncio
async def test_direct_data_plane_contract_delegates_lifecycle(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("kind: Deployment\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="baseline-vllm",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={"data_plane_kind": "direct"},
    )
    runner = _Runner()
    delegate = AsyncMock()
    delegate.deploy.return_value = {"namespace": "llm-d-bench-run"}
    delegate.diagnostics.return_value = {}
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)
    context = {"namespace": "llm-d-bench-run", "_artifact": artifact}

    await adapter.deploy(artifact, context)
    await adapter.readiness(context)
    await adapter.diagnostics(context)
    await adapter.stop(context, artifact)
    await adapter.cleanup(context, artifact, force=True)

    delegate.deploy.assert_awaited_once_with(artifact, context)
    delegate.readiness.assert_awaited_once_with(context)
    delegate.diagnostics.assert_awaited_once_with(context)
    delegate.stop.assert_awaited_once_with(context, artifact)
    delegate.cleanup.assert_awaited_once_with(context, artifact, force=True)
    assert runner.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["pd-disaggregation", "baseline-vllm", "manifest", "legacy"])
async def test_startup_diagnostics_cover_delegate_and_manifest_lifecycles(tmp_path, lifecycle):
    from tests.unit.deploy.test_readiness_transient import startup_snapshot

    snapshot = startup_snapshot()
    pod = snapshot["pod_statuses"][0]
    raw_pod = {
        "metadata": {"name": pod["name"]},
        "spec": {
            "containers": [
                {
                    **pod["containers"][0],
                    "env": [{"name": "PRIVATE_VALUE", "value": "not-for-diagnostics"}],
                }
            ]
        },
        "status": pod["status"],
    }

    class Runner(_Runner):
        async def __call__(self, command):
            self.commands.append(command)
            if command[-2:] == ["-o", "json"]:
                return 0, json.dumps({"items": [raw_pod]}), ""
            if command[1:3] == ["get", "pods"]:
                return 0, snapshot["pods"], ""
            return 0, "ok", ""

    delegate = AsyncMock()
    delegate.diagnostics.return_value = {"pods": snapshot["pods"]}
    runner = Runner()
    contract = {
        "readinessDeployments": ["model"],
        "endpoint": {"protocol": "http", "serviceName": "model", "port": 8000},
    }
    artifact = GuideDeploymentArtifact(
        lifecycle, "hash", deployment_contract={} if lifecycle == "baseline-vllm" else contract
    )
    context = {"namespace": "llm-d-bench-run"}
    if lifecycle != "legacy":
        context["_artifact"] = artifact
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)

    diagnostics = await adapter.diagnostics(context)

    assert diagnostics["pod_statuses"] == [pod]
    assert "not-for-diagnostics" not in json.dumps(diagnostics)
    assert "PRIVATE_VALUE" not in json.dumps(diagnostics)
    assert runner.commands[-1] == ["kubectl", "get", "pods", "--namespace", "llm-d-bench-run", "-o", "json"]


@pytest.mark.asyncio
async def test_failed_startup_query_is_reported_in_diagnostics(tmp_path, caplog):
    delegate = AsyncMock()
    delegate.diagnostics.return_value = {"pods": "model-0 0/1 Running"}
    runner = AsyncMock(return_value=(1, "", "Unable to connect to the server"))
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)

    diagnostics = await adapter.diagnostics({"namespace": "llm-d-bench-run"})

    assert diagnostics["pod_statuses_error"] == "Unable to connect to the server"
    assert "pod_statuses" not in diagnostics
    assert "Pod startup diagnostics failed" in caplog.text


@pytest.mark.asyncio
async def test_deploy_copies_frontend_selected_existing_secret(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("env:\n  - secretKeyRef:\n      name: llm-d-hf-token\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={
            "readinessDeployments": ["decode"],
            "endpoint": {"protocol": "http", "serviceName": "model", "port": 8000},
        },
    )
    runner = _Runner()
    adapter = ConfigurationManifestAdapter(AsyncMock(), runner, "llm-d-bench-", 30, tmp_path)

    await adapter.deploy(
        artifact,
        {
            "namespace": "llm-d-bench-run",
            "deployment_policy": {
                "model_secret": {
                    "mode": "existing-secret",
                    "sourceNamespace": "model-secrets",
                    "sourceName": "huggingface",
                }
            },
        },
    )

    runner.copy_model_secret.assert_awaited_once_with("llm-d-bench-run", "model-secrets", "huggingface")
    assert runner.commands[-1][:3] == ["kubectl", "apply", "--namespace"]


@pytest.mark.asyncio
async def test_pd_manifest_uses_delegate_for_router_lifecycle(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("kind: Deployment\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="pd-disaggregation",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={
            "readinessDeployments": ["decode", "prefill"],
            "endpoint": {"protocol": "http", "serviceName": "model", "port": 8200},
        },
    )
    runner = _Runner()
    delegate = AsyncMock()
    delegate.deploy.return_value = {"namespace": "llm-d-bench-run"}
    delegate.readiness.return_value = object()
    delegate.diagnostics.return_value = {"router": "ready"}
    delegate.stop.return_value = {"stopped": True}
    delegate.cleanup.return_value = {"cleaned_up": True}
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)
    context = {"namespace": "llm-d-bench-run", "_artifact": artifact}

    assert await adapter.deploy(artifact, context) == {"namespace": "llm-d-bench-run"}
    await adapter.readiness(context)
    await adapter.diagnostics(context)
    await adapter.stop(context, artifact)
    await adapter.cleanup(context, artifact, force=True)

    delegate.deploy.assert_awaited_once_with(artifact, context)
    delegate.readiness.assert_awaited_once_with(context)
    delegate.diagnostics.assert_awaited_once_with(context)
    delegate.stop.assert_awaited_once_with(context, artifact)
    delegate.cleanup.assert_awaited_once_with(context, artifact, force=True)
    assert all(command[:2] != ["kubectl", "apply"] for command in runner.commands)


@pytest.mark.asyncio
async def test_optimized_manifest_installs_router_then_applies_exact_model_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("kind: Deployment\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={
            "readinessDeployments": ["optimized-baseline-xpu-vllm-decode"],
            "endpoint": {"protocol": "http", "serviceName": "optimized-baseline-modelserver", "port": 8000},
        },
    )
    runner = _Runner()
    delegate = _RoutedDelegate()
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)

    result = await adapter.deploy(artifact, {"namespace": "llm-d-bench-run"})

    assert delegate.router_installs == ["llm-d-bench-run"]
    assert runner.commands[-1] == ["kubectl", "apply", "--namespace", "llm-d-bench-run", "--filename", str(manifest)]
    assert result["namespace"] == "llm-d-bench-run"
    assert result["router_effective_values"] == "router: {}\n"
    assert result["router_rendered_manifest_checksum"] == "sha256:manifest"


@pytest.mark.asyncio
async def test_readiness_retries_until_newly_applied_deployment_is_visible(monkeypatch, tmp_path: Path):
    """An apply/read visibility gap must not become a terminal rollout failure."""
    artifact = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(tmp_path / "manifest.yaml"),
        deployment_contract={
            "readinessDeployments": ["optimized-baseline-xpu-vllm-decode"],
            "endpoint": {"protocol": "http", "serviceName": "model", "port": 8000},
        },
    )
    responses = iter(
        [
            (1, "", 'Error from server (NotFound): deployments.apps "optimized-baseline-xpu-vllm-decode" not found'),
            (1, "", 'Error from server (NotFound): deployments.apps "optimized-baseline-xpu-vllm-decode" not found'),
            (0, "deployment successfully rolled out", ""),
        ]
    )

    async def runner(_command):
        return next(responses)

    sleep = AsyncMock()
    monkeypatch.setattr(configuration_manifest.asyncio, "sleep", sleep)
    adapter = ConfigurationManifestAdapter(AsyncMock(), runner, "llm-d-bench-", 30, tmp_path)

    result = await adapter.readiness({"namespace": "llm-d-bench-run", "_artifact": artifact})

    assert result.accepted
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_exact_manifest_ensures_storage_volume_pvc_in_deployment_namespace(monkeypatch, tmp_path: Path):
    """Exact manifests come from Configuration with a baked PVC claimName, so
    the manifest adapter must prepare that claim inside the per-run deployment
    namespace before applying the manifest.
    """
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("kind: Deployment\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="optimized-baseline",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={
            "readinessDeployments": ["optimized-baseline-xpu-vllm-decode"],
            "endpoint": {"protocol": "http", "serviceName": "optimized-baseline-modelserver", "port": 8000},
        },
    )
    runner = _Runner()
    delegate = _RoutedDelegate()
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)
    volume = object()

    async def fake_get_ready_volume(volume_id, *, cluster_id=None):
        assert volume_id == "storage-6bb85a3e"
        assert cluster_id == "d459a19d"
        return volume

    ensure_calls = []

    async def fake_ensure_mount_in_namespace(resolved_volume, *, namespace, cluster_id):
        ensure_calls.append((resolved_volume, namespace, cluster_id))
        return "prism-storage-storage-6bb85a3e"

    monkeypatch.setattr(configuration_manifest, "get_ready_volume", fake_get_ready_volume)
    monkeypatch.setattr(configuration_manifest, "ensure_mount_in_namespace", fake_ensure_mount_in_namespace)

    await adapter.deploy(
        artifact,
        {
            "namespace": "llm-d-bench-run",
            "provenance": {"cluster_server_id": "d459a19d"},
            "runtime": {"storageVolumeId": "storage-6bb85a3e"},
        },
    )

    assert ensure_calls == [(volume, "llm-d-bench-run", "d459a19d")]
    assert runner.commands[-1] == ["kubectl", "apply", "--namespace", "llm-d-bench-run", "--filename", str(manifest)]


@pytest.mark.asyncio
async def test_pd_manifest_ensures_storage_volume_pvc_before_delegate_lifecycle(monkeypatch, tmp_path: Path):
    """Regression test: pd-disaggregation (and any other
    ``published_manifest_lifecycle`` guide) must still get its Configuration-
    baked PVC claim prepared in the deployment namespace even though its own
    ``deploy()`` is fully delegated for Router/EPP installation. Before this
    fix, the storage-mount step ran after the delegate-lifecycle early
    return, so Pods referencing the storage volume's PVC failed to schedule
    with "persistentvolumeclaim ... not found".
    """
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("kind: Deployment\n", encoding="utf-8")
    artifact = GuideDeploymentArtifact(
        guide_id="pd-disaggregation",
        artifact_hash="hash",
        manifest_ref=str(manifest),
        deployment_contract={
            "readinessDeployments": ["decode", "prefill"],
            "endpoint": {"protocol": "http", "serviceName": "model", "port": 8200},
        },
    )
    runner = _Runner()
    delegate = AsyncMock()
    delegate.deploy.return_value = {"namespace": "llm-d-bench-run"}
    adapter = ConfigurationManifestAdapter(delegate, runner, "llm-d-bench-", 30, tmp_path)
    volume = object()

    async def fake_get_ready_volume(volume_id, *, cluster_id=None):
        assert volume_id == "storage-fcb73102"
        return volume

    ensure_calls = []

    async def fake_ensure_mount_in_namespace(resolved_volume, *, namespace, cluster_id):
        ensure_calls.append((resolved_volume, namespace, cluster_id))
        return "prism-storage-storage-fcb73102"

    monkeypatch.setattr(configuration_manifest, "get_ready_volume", fake_get_ready_volume)
    monkeypatch.setattr(configuration_manifest, "ensure_mount_in_namespace", fake_ensure_mount_in_namespace)

    context = {
        "namespace": "llm-d-bench-run",
        "provenance": {"cluster_server_id": "d459a19d"},
        "runtime": {"storageVolumeId": "storage-fcb73102"},
    }
    result = await adapter.deploy(artifact, context)

    assert ensure_calls == [(volume, "llm-d-bench-run", "d459a19d")]
    delegate.deploy.assert_awaited_once_with(artifact, context)
    assert result == {"namespace": "llm-d-bench-run"}


def _vllm_deployment(name: str, *, args=None, shell_style: bool = False, grace=None, prestop=None):
    container = {
        "name": "modelserver",
        "image": "vllm/vllm-openai:v1",
        "command": ["/bin/sh", "-c"] if shell_style else ["vllm", "serve"],
        "args": ["vllm serve model --max-model-len=4096"] if shell_style else ["model", "--max-model-len=4096"],
    }
    if args is not None:
        container["args"] = args
    if prestop is not None:
        container["lifecycle"] = {"preStop": prestop}
    pod = {"containers": [container]}
    if grace is not None:
        pod["terminationGracePeriodSeconds"] = grace
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {"template": {"spec": pod}},
    }


def test_graceful_shutdown_manifest_applies_vllm_settings():
    manifest = yaml.safe_dump_all(
        [_vllm_deployment("decode"), _vllm_deployment("prefill", shell_style=True)],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    documents = list(yaml.safe_load_all(rendered))
    for document in documents:
        pod = document["spec"]["template"]["spec"]
        assert pod["terminationGracePeriodSeconds"] == 120
        container = pod["containers"][0]
        assert container["lifecycle"]["preStop"]["exec"]["command"] == ["/bin/sh", "-c", "sleep 5"]

    decode = documents[0]["spec"]["template"]["spec"]["containers"][0]
    assert decode["args"] == ["model", "--max-model-len=4096", "--shutdown-timeout=90"]
    prefill = documents[1]["spec"]["template"]["spec"]["containers"][0]
    assert prefill["args"][0].endswith("--shutdown-timeout=90")
    assert [entry["runtime"] for entry in report["containers"]] == ["vllm", "vllm"]
    assert report["warnings"] == []


def test_graceful_shutdown_manifest_patches_shell_command_script():
    from llm_d_lens.configuration.manifest_facts import _invocation_tokens

    document = _vllm_deployment("command-shell")
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["/bin/sh", "-c", "vllm serve model --max-model-len=4096"]
    container.pop("args")

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(
        yaml.safe_dump_all([document], sort_keys=False)
    )

    rendered_container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    assert rendered_container["command"][2].endswith("--shutdown-timeout=90")
    assert not rendered_container.get("args")
    assert report["containers"][0]["shutdownTimeoutSeconds"] == 90
    assert "--shutdown-timeout=90" in _invocation_tokens(rendered_container, "command-shell")


def test_graceful_shutdown_manifest_patches_python_module_invocation():
    from llm_d_lens.configuration.manifest_facts import _invocation_tokens

    document = _vllm_deployment("python-module")
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    container["args"] = ["--model", "model"]

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(
        yaml.safe_dump_all([document], sort_keys=False)
    )

    rendered_container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    assert "--shutdown-timeout=90" in rendered_container["args"]
    assert report["containers"][0]["shutdownTimeoutSeconds"] == 90
    assert "--shutdown-timeout=90" in _invocation_tokens(rendered_container, "python-module")


@pytest.mark.parametrize(
    ("command", "args"),
    [
        (["/bin/sh", "-c"], ["exec vllm serve model \\\n  --tensor-parallel-size=4\n"]),
        (["/bin/sh", "-c"], ['vllm serve model --host "$HOST_IP" --shutdown-timeout 30']),
        (["vllm"], ["serve", "model"]),
    ],
)
def test_graceful_shutdown_manifest_patches_recognizable_invocations(command, args):
    from llm_d_lens.configuration.manifest_facts import vllm_argument_value

    document = _vllm_deployment("recognizable")
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["command"] = command
    container["args"] = args

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(
        yaml.safe_dump_all([document], sort_keys=False)
    )

    rendered_container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    assert vllm_argument_value(rendered_container, {"--shutdown-timeout"}, "recognizable") == "90"
    assert report["containers"][0]["shutdownTimeoutSeconds"] == 90
    assert report["warnings"] == []


def test_graceful_shutdown_manifest_keeps_non_literal_shutdown_timeout():
    document = _vllm_deployment("variable-timeout")
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["/bin/sh", "-c", "vllm serve model --shutdown-timeout=$TIMEOUT"]
    container.pop("args")

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(
        yaml.safe_dump_all([document], sort_keys=False)
    )

    rendered_container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    assert rendered_container["command"][2] == "vllm serve model --shutdown-timeout=$TIMEOUT"
    assert report["containers"][0]["shutdownTimeoutSeconds"] is None


def test_graceful_shutdown_manifest_skips_unsafe_shell_command_with_warning():
    document = _vllm_deployment("unsafe-shell")
    container = document["spec"]["template"]["spec"]["containers"][0]
    container["command"] = ["/bin/sh", "-c", "vllm serve model && echo done"]
    container.pop("args")

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(
        yaml.safe_dump_all([document], sort_keys=False)
    )

    rendered_container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    assert rendered_container["command"] == ["/bin/sh", "-c", "vllm serve model && echo done"]
    assert report["containers"][0]["shutdownTimeoutSeconds"] is None
    assert any("could not safely apply" in warning for warning in report["warnings"])


def test_graceful_shutdown_manifest_preserves_existing_values_and_handles_sglang():
    manifest = yaml.safe_dump_all(
        [
            _vllm_deployment(
                "decode",
                args=["model", "--shutdown-timeout=180"],
                grace=300,
                prestop={"exec": {"command": ["custom-drain"]}},
            ),
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "sglang"},
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "modelserver",
                                    "image": "lmsysorg/sglang:latest",
                                    "command": ["python", "-m", "sglang.launch_server"],
                                    "args": ["--model", "test"],
                                }
                            ]
                        }
                    }
                },
            },
        ],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    documents = list(yaml.safe_load_all(rendered))
    decode_pod = documents[0]["spec"]["template"]["spec"]
    decode = decode_pod["containers"][0]
    assert decode_pod["terminationGracePeriodSeconds"] == 300
    assert decode["args"] == ["model", "--shutdown-timeout=180"]
    assert decode["lifecycle"]["preStop"]["exec"]["command"] == ["custom-drain"]
    sglang = documents[1]["spec"]["template"]["spec"]["containers"][0]
    assert not any("--shutdown-timeout" in str(item) for item in sglang["args"])
    assert sglang["lifecycle"]["preStop"]["exec"]["command"] == ["/bin/sh", "-c", "sleep 5"]
    assert report["containers"][0]["shutdownTimeoutSeconds"] == 180
    assert report["containers"][1]["runtime"] == "sglang"
    assert report["warnings"]


def test_graceful_shutdown_manifest_expands_grace_for_longer_timeout():
    manifest = yaml.safe_dump_all(
        [_vllm_deployment("decode", args=["model", "--shutdown-timeout=180"])],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    pod = yaml.safe_load(rendered)["spec"]["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] == 195
    assert report["containers"][0]["terminationGracePeriodSeconds"] == 195


def test_graceful_shutdown_manifest_is_idempotent():
    manifest = yaml.safe_dump_all([_vllm_deployment("decode", shell_style=True)], sort_keys=False)

    first, _ = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)
    second, _ = configuration_manifest.prepare_graceful_shutdown_manifest(first)

    assert second == first


def test_graceful_shutdown_manifest_covers_leader_worker_templates():
    manifest = yaml.safe_dump_all(
        [
            {
                "apiVersion": "llm-d.ai/v1alpha1",
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "wide-ep"},
                "spec": {
                    "leaderWorkerTemplate": {
                        "leaderTemplate": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "modelserver-leader",
                                        "image": "vllm/vllm-openai:v1",
                                        "command": ["/bin/sh", "-c"],
                                        "args": ["vllm serve model"],
                                    }
                                ]
                            }
                        },
                        "workerTemplate": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "modelserver-worker",
                                        "image": "vllm/vllm-openai:v1",
                                        "command": ["/bin/sh", "-c"],
                                        "args": ["bash worker.sh"],
                                    }
                                ]
                            }
                        },
                    }
                },
            }
        ],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    document = yaml.safe_load(rendered)
    templates = document["spec"]["leaderWorkerTemplate"]
    for key in ("leaderTemplate", "workerTemplate"):
        assert templates[key]["spec"]["terminationGracePeriodSeconds"] == 120
    leader = templates["leaderTemplate"]["spec"]["containers"][0]
    worker = templates["workerTemplate"]["spec"]["containers"][0]
    assert leader["args"][0].endswith("--shutdown-timeout=90")
    assert not any("--shutdown-timeout" in str(item) for item in worker["args"])
    assert len(report["containers"]) == 2


def test_graceful_shutdown_manifest_covers_disaggregated_set_roles():
    def role(name):
        return {
            "name": name,
            "spec": {
                "leaderWorkerTemplate": {
                    "size": 1,
                    "workerTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "modelserver",
                                    "image": "vllm/vllm-openai:v1",
                                    "command": ["vllm", "serve"],
                                    "args": ["model"],
                                }
                            ]
                        }
                    },
                }
            },
        }

    manifest = yaml.safe_dump(
        {
            "apiVersion": "disaggregatedset.x-k8s.io/v1",
            "kind": "DisaggregatedSet",
            "metadata": {"name": "pd-disagg"},
            "spec": {"roles": [role("prefill"), role("decode")]},
        },
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    document = yaml.safe_load(rendered)
    for item in document["spec"]["roles"]:
        pod = item["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
        assert pod["terminationGracePeriodSeconds"] == 120
        assert pod["containers"][0]["args"][-1] == "--shutdown-timeout=90"
    assert [entry["container"] for entry in report["containers"]] == ["modelserver", "modelserver"]


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_render_applies_graceful_shutdown_to_delegate_manifest(monkeypatch, tmp_path, directory):
    manifest = yaml.safe_dump_all([_vllm_deployment("decode")], sort_keys=False)
    source = tmp_path / "source.yaml"
    source.write_text(manifest, encoding="utf-8")
    if directory:
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        (overlay / "manifest.yaml").write_text(manifest, encoding="utf-8")
        (overlay / "kustomization.yaml").write_text("resources:\n- manifest.yaml\n", encoding="utf-8")
        manifest_ref = str(overlay)
    else:
        manifest_ref = str(source)

    delegate = SimpleNamespace(
        render=AsyncMock(
            return_value=GuideDeploymentArtifact(
                guide_id="baseline-vllm",
                artifact_hash="source",
                manifest_ref=manifest_ref,
                manifest_checksum="sha256:source",
            )
        )
    )
    monkeypatch.setattr(configuration_manifest, "configured_target_node", lambda: None)
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llmd-", 5, tmp_path / "output")

    artifact = await adapter.render(GuideDefinition("baseline-vllm", "source", "hash", "stable"), {})

    path = Path(artifact.manifest_ref)
    if directory:
        path = path / "manifest.yaml"
    documents = [item for item in yaml.safe_load_all(path.read_text()) if isinstance(item, dict)]
    deployment = next(item for item in documents if item.get("kind") == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] == 120
    assert "--shutdown-timeout=90" in pod["containers"][0]["args"]
    assert artifact.deployment_contract["gracefulShutdown"]["containers"]
    if directory:
        assert (overlay / "manifest.yaml").read_text(encoding="utf-8") == manifest
    else:
        assert source.read_text(encoding="utf-8") == manifest


def test_graceful_shutdown_sets_kv_load_failure_policy_in_structured_args():
    transfer = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"backends": ["UCX"]},
    }
    manifest = yaml.safe_dump_all(
        [
            _vllm_deployment(
                "decode",
                args=["model", "--kv-transfer-config", json.dumps(transfer)],
            )
        ],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    index = container["args"].index("--kv-transfer-config")
    updated = json.loads(container["args"][index + 1])
    assert updated["kv_load_failure_policy"] == "recompute"
    assert updated["kv_connector"] == "NixlConnector"
    assert updated["kv_role"] == "kv_both"
    assert updated["kv_connector_extra_config"] == {"backends": ["UCX"]}
    assert report["kvTransferConfigsDetected"] == 1
    assert report["kvTransferConfigsUpdated"] == 1
    assert report["warnings"] == []


def test_graceful_shutdown_sets_kv_load_failure_policy_in_shell_args():
    manifest = yaml.safe_dump_all(
        [
            _vllm_deployment(
                "prefill",
                shell_style=True,
                args=['vllm serve model --kv-transfer-config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\''],
            )
        ],
        sort_keys=False,
    )

    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)

    command = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "kv_load_failure_policy" in command
    assert "recompute" in command
    assert report["kvTransferConfigsDetected"] == 1
    assert report["kvTransferConfigsUpdated"] == 1


def test_graceful_shutdown_kv_policy_is_idempotent_and_preserved():
    transfer = {"kv_connector": "NixlConnector", "kv_load_failure_policy": "recompute"}
    manifest = yaml.safe_dump_all(
        [
            _vllm_deployment(
                "decode",
                args=["model", "--kv-transfer-config", json.dumps(transfer)],
            )
        ],
        sort_keys=False,
    )

    first, first_report = configuration_manifest.prepare_graceful_shutdown_manifest(manifest)
    second, second_report = configuration_manifest.prepare_graceful_shutdown_manifest(first)

    assert second == first
    assert first_report["kvTransferConfigsDetected"] == 1
    assert first_report["kvTransferConfigsUpdated"] == 0
    assert second_report["kvTransferConfigsUpdated"] == 0


def test_graceful_shutdown_ignores_non_kv_and_invalid_transfer_configs():
    without_kv = yaml.safe_dump_all([_vllm_deployment("decode")], sort_keys=False)
    _, no_kv_report = configuration_manifest.prepare_graceful_shutdown_manifest(without_kv)
    assert no_kv_report["kvTransferConfigsDetected"] == 0

    invalid = yaml.safe_dump_all(
        [
            _vllm_deployment(
                "decode",
                args=["model", "--kv-transfer-config", "{not-json}"],
            )
        ],
        sort_keys=False,
    )
    rendered, report = configuration_manifest.prepare_graceful_shutdown_manifest(invalid)
    container = yaml.safe_load(rendered)["spec"]["template"]["spec"]["containers"][0]
    index = container["args"].index("--kv-transfer-config")
    assert container["args"][index + 1] == "{not-json}"
    assert report["kvTransferConfigsDetected"] == 1
    assert report["kvTransferConfigsUpdated"] == 0
    assert "not a JSON object" in report["warnings"][0]


def _disaggregated_set(slices):
    return {
        "apiVersion": "disaggregatedset.x-k8s.io/v1",
        "kind": "DisaggregatedSet",
        "metadata": {"name": "pd-disagg-xpu"},
        "spec": {
            "slices": slices,
            "roles": [
                {
                    "name": "prefill",
                    "spec": {
                        "replicas": 4,
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {"containers": [{"name": "modelserver", "image": "vllm/vllm-openai:v1"}]}
                            }
                        },
                    },
                },
                {
                    "name": "decode",
                    "spec": {
                        "replicas": 1,
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {"containers": [{"name": "modelserver", "image": "vllm/vllm-openai:v1"}]}
                            }
                        },
                    },
                },
            ],
        },
    }


def test_single_slice_disaggregated_set_manifest_forces_slices_to_one():
    manifest = yaml.safe_dump_all([_disaggregated_set(2)], sort_keys=False)

    rendered, report = configuration_manifest.prepare_single_slice_disaggregated_set_manifest(manifest)

    document = yaml.safe_load(rendered)
    assert document["spec"]["slices"] == 1
    assert report["disaggregatedSets"] == 1
    assert report["adjusted"] == 1


def test_single_slice_disaggregated_set_manifest_leaves_other_kinds_unchanged():
    manifest = yaml.safe_dump_all([_vllm_deployment("decode")], sort_keys=False)

    rendered, report = configuration_manifest.prepare_single_slice_disaggregated_set_manifest(manifest)

    assert rendered == manifest
    assert report["disaggregatedSets"] == 0
    assert report["adjusted"] == 0


def test_single_slice_disaggregated_set_manifest_is_idempotent():
    manifest = yaml.safe_dump_all([_disaggregated_set(3)], sort_keys=False)

    first, first_report = configuration_manifest.prepare_single_slice_disaggregated_set_manifest(manifest)
    second, second_report = configuration_manifest.prepare_single_slice_disaggregated_set_manifest(first)

    assert second == first
    assert first_report["adjusted"] == 1
    assert second_report["adjusted"] == 0


@pytest.mark.asyncio
async def test_render_enforces_single_slice_disaggregated_set(monkeypatch, tmp_path):
    manifest = yaml.safe_dump_all([_disaggregated_set(2)], sort_keys=False)
    source = tmp_path / "manifest.yaml"
    source.write_text(manifest, encoding="utf-8")
    delegate = SimpleNamespace(
        render=AsyncMock(
            return_value=GuideDeploymentArtifact(
                guide_id="pd-disaggregation",
                artifact_hash="source",
                manifest_ref=str(source),
                manifest_checksum="sha256:source",
            )
        )
    )
    monkeypatch.setattr(configuration_manifest, "configured_target_node", lambda: None)
    adapter = ConfigurationManifestAdapter(delegate, _Runner(), "llmd-", 5, tmp_path / "output")

    artifact = await adapter.render(GuideDefinition("pd-disaggregation", "source", "hash", "stable"), {})

    document = yaml.safe_load(Path(artifact.manifest_ref).read_text())
    assert document["spec"]["slices"] == 1
    assert artifact.deployment_contract["disaggregatedSet"]["slices"] == 1
