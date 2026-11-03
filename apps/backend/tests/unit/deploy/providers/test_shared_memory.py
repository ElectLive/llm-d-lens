"""Shared-memory policy across rendered and published deployment manifests."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml

from llm_d_lens.deploy.providers.configuration_manifest import ConfigurationManifestAdapter
from llm_d_lens.deploy.providers.guide_adapter import GuideDefinition, GuideDeploymentArtifact
from llm_d_lens.deploy.providers.shared_memory import prepare_shared_memory, prepare_shared_memory_manifest
from llm_d_lens.infrastructure.artifacts.manifests import text_checksum


def _pod():
    return {
        "containers": [
            {
                "name": "modelserver",
                "command": ["vllm", "serve"],
                "volumeMounts": [{"name": "cache", "mountPath": "/cache"}],
            },
            {"name": "router", "image": "router"},
        ],
        "volumes": [{"name": "cache", "emptyDir": {}}],
    }


def _manifest(pod):
    return yaml.safe_dump({"kind": "Deployment", "metadata": {"name": "model"}, "spec": {"template": {"spec": pod}}})


@pytest.mark.parametrize("limit", ["64Mi", "16Gi", "20Gi", "250Gi", None])
def test_explicit_mount_is_preserved_byte_for_byte_and_unbounded_mount_warns(limit):
    pod = _pod()
    volume = {"name": "shm", "emptyDir": {"medium": "Memory"}}
    if limit is not None:
        volume["emptyDir"]["sizeLimit"] = limit
    pod["volumes"].append(volume)
    pod["containers"][0]["volumeMounts"].append({"name": "shm", "mountPath": "/dev/shm"})
    manifest = _manifest(pod)

    rendered, report = prepare_shared_memory_manifest(manifest)

    assert rendered == manifest
    assert report["addedMounts"] == 0
    assert report["mounts"][0]["sizeLimit"] == limit
    assert bool(report["warnings"]) == (limit is None)


def test_missing_mount_is_added_once_without_overwriting_volumes_or_sidecars():
    pod = _pod()
    pod["volumes"].append({"name": "lens-shm", "configMap": {"name": "existing"}})
    manifest = _manifest(pod)

    rendered, report = prepare_shared_memory_manifest(manifest)
    result = yaml.safe_load(rendered)["spec"]["template"]["spec"]

    assert report["addedMounts"] == 1
    assert result["volumes"][:2] == pod["volumes"]
    assert result["volumes"][2] == {
        "name": "lens-shm-1",
        "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"},
    }
    assert result["containers"][0]["volumeMounts"] == [
        *pod["containers"][0]["volumeMounts"],
        {"name": "lens-shm-1", "mountPath": "/dev/shm"},
    ]
    assert result["containers"][1] == pod["containers"][1]
    assert not result.get("hostIPC")
    assert prepare_shared_memory_manifest(rendered)[0] == rendered


@pytest.mark.parametrize("exact,host_ipc", [(True, False), (False, True), (True, True)])
def test_exact_manifest_and_host_ipc_are_not_silently_changed(exact, host_ipc):
    pod = _pod()
    if host_ipc:
        pod["hostIPC"] = True
    manifest = _manifest(pod)

    rendered, report = prepare_shared_memory_manifest(manifest, exact=exact)

    assert rendered == manifest
    assert report["addedMounts"] == 0
    assert report["mounts"][0]["mode"] == ("hostIPC" if host_ipc else "missing")
    assert report["warnings"]


def test_existing_alternative_mount_is_preserved_and_warned():
    pod = _pod()
    pod["volumes"].append({"name": "shm", "hostPath": {"path": "/dev/shm"}})
    pod["containers"][0]["volumeMounts"].append({"name": "shm", "mountPath": "/dev/shm", "readOnly": True})
    manifest = _manifest(pod)
    rendered, report = prepare_shared_memory_manifest(manifest)
    assert rendered == manifest
    assert len(report["warnings"]) == 2


@pytest.mark.parametrize("error", ["missing-volume", "duplicate-mount"])
def test_invalid_shm_binding_fails_explicitly(error):
    pod = _pod()
    pod["containers"][0]["volumeMounts"].append({"name": "missing", "mountPath": "/dev/shm"})
    if error == "duplicate-mount":
        pod["containers"][0]["volumeMounts"].append({"name": "other", "mountPath": "/dev/shm"})
    with pytest.raises(ValueError, match="/dev/shm"):
        prepare_shared_memory_manifest(_manifest(pod))


@pytest.mark.parametrize("kind", ["Pod", "Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob", "LeaderWorkerSet"])
def test_all_pod_templates_receive_independent_mounts(kind):
    pod = _pod()
    document = {"kind": kind, "metadata": {"name": "model"}}
    if kind == "Pod":
        document["spec"] = pod
    elif kind == "LeaderWorkerSet":
        document["spec"] = {
            "leaderWorkerTemplate": {
                "leaderTemplate": {"spec": pod},
                "workerTemplate": {"spec": deepcopy(pod)},
            }
        }
    elif kind == "CronJob":
        document["spec"] = {"jobTemplate": {"spec": {"template": {"spec": pod}}}}
    else:
        document["spec"] = {"template": {"spec": pod}}
    rendered, report = prepare_shared_memory_manifest(yaml.safe_dump({"kind": "List", "items": [document]}))
    assert report["addedMounts"] == (2 if kind == "LeaderWorkerSet" else 1)
    assert "sizeLimit: 4Gi" in rendered


def test_multiple_model_containers_share_one_dedicated_pod_volume():
    pod = _pod()
    pod["containers"].append({"name": "vllm"})
    rendered, report = prepare_shared_memory_manifest(_manifest(pod))
    result = yaml.safe_load(rendered)["spec"]["template"]["spec"]
    assert report["addedMounts"] == 2
    assert len(result["volumes"]) == 2
    assert result["containers"][0]["volumeMounts"][-1] == result["containers"][2]["volumeMounts"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["baseline-vllm", "optimized-baseline", "pd-disaggregation", "precise-prefix-cache-routing", "tiered-prefix-cache"],
)
@pytest.mark.parametrize("directory", [False, True])
async def test_wrapper_prepares_the_manifest_consumed_by_provider(monkeypatch, tmp_path, provider, directory):
    source = tmp_path / "provider"
    source.mkdir()
    manifest = source / "manifest.yaml"
    original = _manifest(_pod())
    manifest.write_text(original)
    if directory:
        (source / "kustomization.yaml").write_text("resources:\n- manifest.yaml\n")
    delegate = AsyncMock()
    delegate.render.return_value = GuideDeploymentArtifact(
        provider, "hash", manifest_ref=str(source if directory else manifest), manifest_checksum="source-checksum"
    )
    delegate.deploy.return_value = {"namespace": "llmd-test"}
    delegate.diagnostics.return_value = {"pods": "model Pending"}
    monkeypatch.setattr(
        "llm_d_lens.deploy.providers.configuration_manifest.instrument_manifest", lambda artifact, _: artifact
    )
    monkeypatch.setattr("llm_d_lens.deploy.providers.configuration_manifest.configured_target_node", lambda: None)
    wrapper = ConfigurationManifestAdapter(delegate, None, "llmd-", 30, tmp_path / "output")

    artifact = await wrapper.render(GuideDefinition(provider, "source", "hash", "supported"), {})
    reference = Path(artifact.manifest_ref)
    deployed_manifest = reference / "manifest.yaml" if directory else reference
    rendered = deployed_manifest.read_text()
    assert "sizeLimit: 4Gi" in rendered
    assert artifact.deployment_contract["sharedMemory"]["addedMounts"] == 1
    assert artifact.deployment_contract["sharedMemory"]["manifestChecksum"] == text_checksum(rendered)
    assert manifest.read_text() == original
    if directory:
        assert reference != source
        assert reference.parent == source.parent
        assert (reference / "kustomization.yaml").read_text() == "resources:\n- manifest.yaml\n"
        assert artifact.manifest_checksum == "source-checksum"
    else:
        assert artifact.manifest_checksum == text_checksum(rendered)
    context = {"namespace": "llmd-test", "_artifact": artifact}
    await wrapper.deploy(artifact, context)
    delegate.deploy.assert_awaited_once_with(artifact, context)
    diagnostics = await wrapper.diagnostics(context)
    assert diagnostics["shared_memory"] == artifact.deployment_contract["sharedMemory"]


def test_exact_artifact_warns_and_keeps_source_bytes_and_checksum(tmp_path, caplog):
    source = tmp_path / "published.yaml"
    original = _manifest(_pod())
    source.write_text(original)
    artifact = GuideDeploymentArtifact(
        "optimized-baseline", "hash", manifest_ref=str(source), manifest_checksum="saved"
    )
    prepared = prepare_shared_memory(artifact, tmp_path / "output", exact=True)
    assert prepared.manifest_ref == artifact.manifest_ref
    assert prepared.manifest_checksum == "saved"
    assert source.read_text() == original
    assert prepared.deployment_contract["sharedMemory"]["warnings"]
    assert "Exact manifest preserved" in caplog.text


@pytest.mark.asyncio
async def test_published_wrapper_checks_without_adding_shared_memory(monkeypatch, tmp_path):
    manifest = _manifest(_pod())
    monkeypatch.setattr(
        "llm_d_lens.deploy.providers.configuration_manifest.validate_configuration_manifest_content",
        lambda *_: (
            manifest,
            text_checksum(manifest),
            {"readinessDeployments": ["model"], "endpoint": {"serviceName": "model", "port": 8000}},
            {"commit": "saved"},
        ),
    )
    monkeypatch.setattr(
        "llm_d_lens.deploy.providers.configuration_manifest.instrument_manifest", lambda artifact, _: artifact
    )
    monkeypatch.setattr("llm_d_lens.deploy.providers.configuration_manifest.configured_target_node", lambda: None)
    wrapper = ConfigurationManifestAdapter(AsyncMock(), AsyncMock(), "llmd-", 30, tmp_path)
    artifact = await wrapper.render(
        GuideDefinition("optimized-baseline", "source", "hash", "supported"),
        {"officialGuide": {}, "_deployment_namespace": "llmd-test"},
    )
    rendered = Path(artifact.manifest_ref).read_text()
    assert "lens-shm" not in rendered
    assert artifact.manifest_checksum == text_checksum(rendered)
    assert artifact.deployment_contract["sharedMemory"]["warnings"]
    assert artifact.deployment_contract["sharedMemory"]["addedMounts"] == 0
    assert artifact.deployment_contract["gracefulShutdown"]["containers"]


@pytest.mark.asyncio
async def test_optimized_directory_validation_and_apply_use_prepared_output(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from tests.unit.deploy.providers.test_optimized_baseline_profiles import (
        _adapter,
        _deployment_parameters,
        _runtime,
        _write_model_overlay,
    )

    delegate, overlay = _adapter(tmp_path)
    _write_model_overlay(overlay)
    source_before = {path: path.read_bytes() for path in overlay.iterdir() if path.is_file()}
    monkeypatch.setattr(
        delegate, "_deployment_parameters", Mock(return_value=(_deployment_parameters(), "org/model", _runtime()))
    )
    wrapper = ConfigurationManifestAdapter(delegate, None, "test", 30, tmp_path / "output")
    artifact = await wrapper.render(delegate.discover(), {})
    delegate._validate_artifact(artifact)
    consumed = []

    async def runner(command):
        if "--kustomize" in command:
            import subprocess

            output = subprocess.run(
                ["kubectl", "kustomize", command[-1]], check=True, capture_output=True, text=True, timeout=30
            )
            consumed.append(output.stdout)
        return 0, "", ""

    delegate._command_runner = runner
    await wrapper.deploy(artifact, {"namespace": "test-run"})
    assert len(consumed) == 1
    assert "sizeLimit: 4Gi" in consumed[0]
    assert {path: path.read_bytes() for path in source_before} == source_before
