from copy import deepcopy

import pytest
import yaml

from llm_d_lens.deploy import guide_bundle
from llm_d_lens.deploy import guide_planning as guide
from llm_d_lens.deploy import guide_runtime as runtime
from llm_d_lens.deploy import guide_sources as sources


def deployment(name="model", replicas=1, count=1):
    return {
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {
            "replicas": replicas,
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "modelserver",
                            "args": ["example/model", f"--tensor-parallel-size={count}"],
                            "resources": {"limits": {"nvidia.com/gpu": count}},
                        }
                    ],
                }
            },
        },
    }


def disaggregated_set():
    return {
        "apiVersion": "disaggregatedset.x-k8s.io/v1",
        "kind": "DisaggregatedSet",
        "metadata": {"name": "pd-disagg", "labels": {"llm-d.ai/guide": "pd-disaggregation"}},
        "spec": {
            "slices": 1,
            "roles": [
                {
                    "name": role,
                    "spec": {
                        "replicas": 1,
                        "leaderWorkerTemplate": {
                            "size": 1,
                            "workerTemplate": {
                                "metadata": {"labels": {"llm-d.ai/role": role}},
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "modelserver",
                                            "args": ["example/model", "--tensor-parallel-size=1"],
                                            "ports": [{"name": "modelserver", "containerPort": 8000}],
                                        }
                                    ],
                                },
                            },
                        },
                    },
                }
                for role in ("prefill", "decode")
            ],
        },
    }


def test_pd_plan_keeps_v1_sidecar_arguments_without_legacy_vllm_port():
    document = disaggregated_set()
    decode = document["spec"]["roles"][1]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    decode["initContainers"] = [
        {
            "name": "routing-proxy",
            "image": "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0",
            "args": ["--port=8000", "--kv-connector=nixlv2", "--secure-serving=false"],
        }
    ]
    plan = guide.plan_documents(
        [document],
        "Qwen/Qwen3-0.6B",
        {"cpu": {"logicalCores": 32}, "memoryGiB": 128, "accelerator": {"count": 8}},
        None,
        {
            "guide": "pd-disaggregation",
            "prefillReplicas": 1,
            "decodeReplicas": 1,
            "prefillTensorParallelSize": 1,
            "decodeTensorParallelSize": 1,
        },
    )

    planned = plan["documents"][0]["spec"]["roles"][1]["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    assert planned["initContainers"][0]["args"] == [
        "--port=8000",
        "--kv-connector=nixlv2",
        "--secure-serving=false",
    ]


def test_disaggregated_set_is_planned_as_native_roles():
    machine = {"cpu": {"logicalCores": 32}, "memoryGiB": 128, "accelerator": {"count": 8}}
    plan = guide.plan_documents(
        [disaggregated_set()],
        "Qwen/Qwen3-0.6B",
        machine,
        None,
        {
            "guide": "pd-disaggregation",
            "prefillReplicas": 2,
            "decodeReplicas": 3,
            "prefillTensorParallelSize": 1,
            "decodeTensorParallelSize": 2,
        },
    )

    assert not plan["validation"]["errors"]
    document = plan["documents"][0]
    assert [role["spec"]["replicas"] for role in document["spec"]["roles"]] == [2, 3]
    assert plan["summary"]["guideReplicas"] == 2
    assert plan["summary"]["plannedReplicas"] == 5

    contract = guide.deployment_contract(plan["documents"], "pd-disaggregation")
    assert contract["readinessDeployments"] == ["pd-disagg"]
    assert contract["endpoint"]["serviceName"] == "pd-disaggregation-modelserver"
    assert any(
        item.get("kind") == "Service" and item["metadata"]["name"] == "pd-disaggregation-modelserver"
        for item in plan["documents"]
    )


def test_variant_normalization_and_parent_rejection():
    paths = [
        "guides/guide/modelserver/gpu/vllm/base/kustomization.yaml",
        "guides/guide/modelserver/gpu/vllm/base/deployment.yaml",
    ]
    selected = guide.select_manifest_paths(paths, "guide", "gpu", "vllm", "/" * 10000 + "base" + "/" * 10000)
    assert selected["selected"] == paths
    assert selected["variant"] == "base"
    with pytest.raises(guide.GuideError, match="invalid"):
        guide.select_manifest_paths(paths, "guide", "gpu", "vllm", "../other")


def test_patch_matches_by_name_without_mutating_source():
    original = deployment()
    documents = [deepcopy(original)]
    errors, warnings = [], []
    applied = guide.apply_custom_patches(
        documents,
        [{"path": "replicas.yaml", "content": "kind: Deployment\nmetadata:\n  name: model\nspec:\n  replicas: 3\n"}],
        errors,
        warnings,
    )
    assert documents[0]["spec"]["replicas"] == 3
    assert original["spec"]["replicas"] == 1
    assert applied == [{"path": "replicas.yaml", "kind": "Deployment", "name": "model", "action": "merged"}]
    assert not errors


def test_ambiguous_patch_does_not_modify_documents():
    documents = [deployment("first"), deployment("second")]
    original = deepcopy(documents)
    errors = []
    assert (
        guide.apply_custom_patches(documents, [{"content": "kind: Deployment\nspec: {replicas: 3}"}], errors, []) == []
    )
    assert "Add metadata.name to disambiguate" in errors[0]
    assert documents == original


def test_capacity_counts_replicas_and_replaces_stale_errors():
    errors = ["Insufficient resources: stale", "other validation"]
    assert guide.validate_manifest_capacity([deployment(replicas=3, count=2)], 4, "available", errors) == 6
    assert len(errors) == 2
    assert "requires 6 accelerators" in errors[1]
    assert "currently available budget is 4" in errors[1]


def test_capacity_includes_restartable_init_and_init_peak():
    document = deployment(count=1)
    document["spec"]["template"]["spec"]["initContainers"] = [
        {"name": "sidecar", "restartPolicy": "Always", "resources": {"limits": {"nvidia.com/gpu": 1}}},
        {"name": "init", "resources": {"limits": {"nvidia.com/gpu": 4}}},
    ]
    errors = []
    assert guide.validate_manifest_capacity([document], 4, "total", errors) == 5
    assert errors


@pytest.mark.parametrize("accelerator", [None, {}, {"devices": None}, {"devices": []}])
def test_hardware_variant_and_runtime_use_registered_profiles(accelerator):
    cluster = {
        "allocatable": {"accelerators": {"nvidia.com/gpu": 4}},
        "accelerator": accelerator,
        "runtimeClasses": [{"name": "nvidia", "handler": "nvidia"}],
    }
    assert guide.resolve_accelerator_variant("xpu", cluster) == "gpu"
    assert guide.runtime_class_for_accelerator(cluster, "gpu") == "nvidia"
    assert guide.runtime_class_for_accelerator(cluster, "xpu") is None
    assert (
        guide.resolve_accelerator_variant(
            "xpu", {"allocatable": {"accelerators": {"unknown/gpu": 4}}, "accelerator": accelerator}
        )
        == "xpu"
    )


def test_device_plugin_cluster_without_dra_inventory_keeps_accelerator_resources():
    cluster = sources.cluster_profile(
        {"serverVersion": {"gitVersion": "v1.35.0"}},
        {"items": [{"metadata": {"name": "worker"}, "status": {"allocatable": {"nvidia.com/gpu": "4"}}}]},
    )
    assert cluster["accelerator"] is None
    assert cluster["allocatable"]["accelerators"] == {"nvidia.com/gpu": 4}
    assert guide.resolve_accelerator_variant("xpu", cluster) == "gpu"


def test_shell_model_rewrite_preserves_unmodified_expansions():
    container = {
        "command": ["bash", "-lc"],
        "args": [
            'vllm serve old/model --host "$HOST_IP" --tensor-parallel-size=1 --max-num-seqs=32',
        ],
    }
    runtime.configure_model_server(container, "/model-cache", 4, "new/model", False)
    assert '--host "$HOST_IP"' in container["args"][0]
    assert runtime.runtime_argument(container, "tensor-parallel-size") == "4"
    assert runtime.runtime_argument(container, "served-model-name") == "new/model"


@pytest.mark.parametrize("script", ["vllm serve model; touch /tmp/file", "vllm serve model\necho second"])
def test_compound_shell_commands_are_not_rewritten(script):
    container = {"command": ["bash", "-c"], "args": [script]}
    with pytest.raises(ValueError):
        runtime.configure_model_server(container, "new/model", 2, "new/model", False)


def test_runtime_overrides_preserve_admin_managed_environment():
    with pytest.raises(ValueError, match="must be an array"):
        runtime.runtime_overrides({"customParameters": {}})
    with pytest.raises(ValueError, match="managed by the platform administrator"):
        runtime.runtime_overrides(
            {
                "customParameters": [
                    {"kind": "environment", "target": "both", "name": "HF_TOKEN", "value": "not-a-real-token"},
                ]
            }
        )
    with pytest.raises(ValueError, match="Overlapping"):
        runtime.runtime_overrides(
            {
                "maxModelLen": 4096,
                "customParameters": [
                    {"kind": "argument", "target": "decode", "name": "max-model-len", "value": 8192},
                ],
            }
        )


def test_cpu_cache_connector_and_precise_event_identity():
    import json

    container = {
        "args": [
            "old/model",
            '--kv-transfer-config={"kv_connector":"OffloadingConnector"}',
            '--kv-events-config={"topic":"kv@host:5557@old/model"}',
        ]
    }
    runtime.configure_model_server(container, "new/model", 1, "new/model", True)
    runtime.configure_cpu_cache(container, 2)
    assert json.loads(runtime.runtime_argument(container, "kv-transfer-config"))["kv_connector_extra_config"] == {
        "cpu_bytes_to_use": 2 * 1024**3,
    }
    assert json.loads(runtime.runtime_argument(container, "kv-events-config"))["topic"] == "kv@host:5557@new/model"


def test_full_plan_preserves_explicit_topology_and_validates_budget(monkeypatch):
    monkeypatch.delenv("PRISM_GPU_PCI_ALLOWLIST", raising=False)
    monkeypatch.delenv("PRISM_K8S_TARGET_NODE", raising=False)
    machine = {"cpu": {"logicalCores": 64}, "memoryGiB": 512, "accelerator": {"count": 2}}
    result = guide.plan_documents(
        [deployment()],
        "example/model",
        machine,
        None,
        {
            "guide": "optimized-baseline",
            "accelerator": "gpu",
            "replicas": 2,
            "tensorParallelSize": 2,
        },
    )
    assert result["summary"]["requestedAccelerators"] == 4
    assert result["summary"]["plannedReplicas"] == 2
    assert result["validation"]["status"] == "invalid"
    assert any("Insufficient resources:" in error for error in result["validation"]["errors"])


def test_full_plan_mounts_cached_pvc_and_keeps_hub_offline(monkeypatch):
    monkeypatch.delenv("PRISM_GPU_PCI_ALLOWLIST", raising=False)
    machine = {"cpu": {"logicalCores": 64}, "memoryGiB": 512, "accelerator": {"count": 2}}
    result = guide.plan_documents(
        [deployment()],
        "example/model",
        machine,
        None,
        {
            "guide": "optimized-baseline",
            "accelerator": "gpu",
            "replicas": 1,
            "tensorParallelSize": 1,
            "modelSource": "auto-cache",
            "modelPvcClaimName": "cache-pvc",
        },
    )
    pod = result["documents"][0]["spec"]["template"]["spec"]
    assert pod["volumes"] == [{"name": "model-cache", "persistentVolumeClaim": {"claimName": "cache-pvc"}}]
    environment = {entry["name"]: entry.get("value") for entry in pod["containers"][0]["env"]}
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_HOME"] == "/model-cache"
    assert result["validation"]["errors"] == []


def test_catalog_uses_hardware_overlay_templates():
    catalog = sources.catalog_from_paths(
        [
            "guides/optimized-baseline/modelserver/gpu/vllm/base/kustomization.yaml",
            "guides/optimized-baseline/modelserver/gpu/vllm/custom/kustomization.yaml",
            "guides/optimized-baseline/modelserver/unknown/vllm/base/kustomization.yaml",
        ]
    )
    assert catalog == [
        {
            "id": "optimized-baseline",
            "label": "Optimized Baseline",
            "accelerators": [
                {"id": "gpu", "modelServers": [{"id": "vllm", "variants": ["base", "custom"]}]},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_local_source_cannot_escape_allowed_root(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("kind: Deployment")
    (allowed / "link.yaml").symlink_to(outside)
    monkeypatch.setenv("PRISM_GUIDE_LOCAL_ROOTS", str(allowed))
    with pytest.raises(guide.GuideError, match="allowed root"):
        await sources.load_selected_manifests(None, {"source": {"mode": "local", "path": str(allowed / "link.yaml")}})


@pytest.mark.parametrize("kubernetes_mode", ["disabled", "required"])
def test_plan_route_emits_validated_bundle_and_preserves_response(monkeypatch, tmp_path, kubernetes_mode):
    from unittest.mock import AsyncMock

    import yaml
    from fastapi.testclient import TestClient

    from llm_d_lens.api.main import app
    from llm_d_lens.deploy.providers.deployment_bundle import validate_deployment_bundle

    monkeypatch.delenv("PRISM_GPU_PCI_ALLOWLIST", raising=False)
    manifest_path = tmp_path / "model.yaml"
    manifest_path.write_text(yaml.safe_dump(deployment()))
    router_root = tmp_path / "guides/optimized-baseline/router"
    router_root.mkdir(parents=True)
    (router_root / "optimized-baseline.values.yaml").write_text("router: {epp: {replicas: 1}}")
    base_root = tmp_path / "guides/recipes/router"
    base_root.mkdir(parents=True)
    (base_root / "base.values.yaml").write_text("router: {}")
    monkeypatch.setenv("PRISM_GUIDE_LOCAL_ROOTS", str(tmp_path))
    monkeypatch.setattr(sources, "command_output", AsyncMock(return_value=""))
    monkeypatch.setattr(
        sources,
        "discover_local_cluster",
        AsyncMock(
            return_value=sources.cluster_profile(
                {"serverVersion": {"gitVersion": "v1.35.0"}},
                {"items": [{"metadata": {"name": "worker"}, "status": {"allocatable": {"nvidia.com/gpu": "4"}}}]},
            )
        ),
    )
    monkeypatch.setattr(
        sources,
        "machine_profile",
        AsyncMock(
            return_value={
                "cpu": {"logicalCores": 64},
                "memoryGiB": 512,
                "accelerator": {"count": 4},
            }
        ),
    )
    response = TestClient(app).post(
        "/api/guide-planning/plan",
        json={
            "guide": "optimized-baseline",
            "accelerator": "gpu",
            "modelServer": "vllm",
            "model": "example/model",
            "environment": {"kubernetesMode": kubernetes_mode},
            "replicas": 1,
            "tensorParallelSize": 1,
            "source": {"mode": "local", "path": str(manifest_path)},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == ("VERIFIED" if kubernetes_mode == "required" else "PLANNED")
    if kubernetes_mode == "required":
        assert result["clusterProfile"]["accelerator"] is None
        assert result["summary"]["availableAccelerators"] == 4
    assert result["validation"]["errors"] == []
    assert result["plannedDeployment"]["format"] == "yaml"
    assert "total;dur=" in response.headers["server-timing"]
    validate_deployment_bundle(result["deploymentBundle"], "optimized-baseline", result["source"]["commit"])


def test_plan_rejects_expired_explicit_session_with_discovery_disabled(monkeypatch):
    from fastapi.testclient import TestClient

    from llm_d_lens.api.main import app

    def expired(_session_id):
        raise ValueError("expired")

    monkeypatch.setattr(sources, "require_active_session", expired)
    response = TestClient(app).post(
        "/api/guide-planning/plan",
        json={
            "guide": "optimized-baseline",
            "accelerator": "gpu",
            "modelServer": "vllm",
            "model": "model",
            "clusterSessionId": "expired",
            "environment": {"kubernetesMode": "disabled"},
        },
    )
    assert response.status_code == 409, response.text


@pytest.mark.asyncio
async def test_remote_bundle_falls_back_to_flat_router_path(monkeypatch):
    import httpx

    from llm_d_lens.deploy import guide_bundle

    fetched = []

    async def fetch(url, _accept):
        fetched.append(url)
        if "/single-host/" in url:
            response = httpx.Response(404, request=httpx.Request("GET", url))
            response.raise_for_status()
        return "router: {}"

    monkeypatch.setattr(guide_bundle, "fetch_source", fetch)
    manifest = {"source": {"repository": "example/repository", "commit": "a" * 40}}
    bundle = await guide_bundle.build_guide_deployment_bundle("optimized-baseline", manifest, "model", "gpu")
    assert bundle["sourceCommit"] == "a" * 40
    assert any("/router/optimized-baseline.values.yaml" in url for url in fetched)


@pytest.mark.parametrize(
    ("operation", "loader", "status", "message"),
    [
        ("catalog", "load_guide_paths", 502, "Guide catalog could not be loaded"),
        ("prepare", "load_manifests", 502, "Guide preparation failed; retry when generating configuration"),
        ("plan", "load_selected_manifests", 500, "Guide planning failed"),
    ],
)
def test_guide_internal_errors_are_logged_not_exposed(monkeypatch, caplog, operation, loader, status, message):
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from llm_d_lens.api.main import app

    detail = "internal traceback /private/guide/config credential-value"
    monkeypatch.setattr(sources, loader, AsyncMock(side_effect=OSError(detail)))
    client = TestClient(app)
    response = (
        client.get("/api/guide-planning/catalog")
        if operation == "catalog"
        else client.post(
            f"/api/guide-planning/{operation}",
            json={
                "guide": "optimized-baseline",
                "accelerator": "gpu",
                "modelServer": "vllm",
                "model": "example/model",
                "environment": {"kubernetesMode": "disabled"},
            },
        )
    )
    assert response.status_code == status
    assert response.json() == {"error": message}
    assert detail not in response.text
    assert detail in caplog.text
    if operation == "plan":
        assert "total;dur=" in response.headers["server-timing"]


def test_guide_validation_keeps_actionable_message_and_status(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from llm_d_lens.api.main import app

    monkeypatch.setattr(
        sources,
        "load_selected_manifests",
        AsyncMock(side_effect=guide.GuideError("Remote branch/ref could not be resolved to a commit", 404)),
    )
    response = TestClient(app).post(
        "/api/guide-planning/plan",
        json={
            "guide": "optimized-baseline",
            "accelerator": "gpu",
            "modelServer": "vllm",
            "model": "example/model",
            "environment": {"kubernetesMode": "disabled"},
        },
    )
    assert response.status_code == 404
    assert response.json() == {"error": "Remote branch/ref could not be resolved to a commit"}


def test_patch_parser_exception_is_not_in_validation_errors(monkeypatch, caplog):
    import yaml

    detail = "internal parser detail /private/config credential-value"

    def fail(_content):
        raise yaml.YAMLError(detail)

    monkeypatch.setattr(guide.yaml, "safe_load_all", fail)
    errors = []
    assert guide.apply_custom_patches([deployment()], [{"content": "broken", "path": "patch.yaml"}], errors, []) == []
    assert errors == ["Custom patch patch.yaml could not be parsed as YAML."]
    assert detail in caplog.text


@pytest.mark.parametrize("helper", ["normalize_guide_settings", "runtime_overrides"])
def test_plan_validation_does_not_expose_internal_exceptions(monkeypatch, caplog, helper):
    detail = "internal validation detail /private/config credential-value"

    def fail(*_args):
        raise ValueError(detail)

    monkeypatch.setattr(guide, helper, fail)
    result = guide.plan_documents(
        [deployment()],
        "example/model",
        {"cpu": {"logicalCores": 64}, "memoryGiB": 512, "accelerator": {"count": 2}},
        None,
        {"guide": "optimized-baseline", "accelerator": "gpu"},
    )
    assert result["validation"]["status"] == "invalid"
    assert detail not in str(result)
    assert detail in caplog.text


def test_precise_v1_endpoint_discovery_does_not_need_a_render_pool():
    values = {
        "router": {
            "epp": {
                "pluginsConfigFile": "plugins.yaml",
                "pluginsCustomConfig": {
                    "plugins.yaml": yaml.safe_dump(
                        {
                            "plugins": [
                                {
                                    "type": "token-producer",
                                    "parameters": {"vllm": {"endpointDiscovery": {"attemptTimeout": "2s"}}},
                                }
                            ]
                        }
                    )
                },
            }
        }
    }
    assert guide_bundle._needs_render_service(values) is False


def test_precise_render_url_still_needs_a_render_pool():
    values = {
        "router": {
            "epp": {
                "pluginsConfigFile": "plugins.yaml",
                "pluginsCustomConfig": {
                    "plugins.yaml": yaml.safe_dump(
                        {
                            "plugins": [
                                {
                                    "type": "token-producer",
                                    "parameters": {"vllm": {"url": "http://render:8000"}},
                                }
                            ]
                        }
                    )
                },
            }
        }
    }
    assert guide_bundle._needs_render_service(values) is True
