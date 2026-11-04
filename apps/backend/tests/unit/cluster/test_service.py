"""Tests for cluster hardware capacity accounting."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_d_lens.cluster import registry, service
from llm_d_lens.cluster.errors import ClusterOverviewError
from llm_d_lens.cluster.service import _active_gpu_allocations
from llm_d_lens.cluster.settings import ClusterSettings
from llm_d_lens.infrastructure.process import CommandResult


def _claim(*results: dict) -> dict:
    return {"status": {"allocation": {"devices": {"results": list(results)}}}}


def test_gateway_namespace_is_stable_unique_and_visible_on_cluster_records():
    first = registry.Cluster(id="dcea6ee0", name="first", description="", created_at="")
    second = registry.Cluster(id="1234abcd", name="second", description="", created_at="")
    assert first.gateway_namespace == "lens-gateway-dcea6ee0"
    assert second.gateway_namespace == "lens-gateway-1234abcd"
    assert first.to_dict()["gatewayNamespace"] == first.gateway_namespace
    assert registry.gateway_namespace_for(first.id, first.gateway_namespace) == first.gateway_namespace
    assert registry.gateway_namespace_for(first.id, "lens-gateway") == first.gateway_namespace
    assert registry.gateway_namespace_for(first.id, "custom") == "custom-dcea6ee0"
    assert len(registry.gateway_namespace_for(first.id, "x" * 63)) == 63
    with pytest.raises(ValueError, match="cluster id"):
        registry.gateway_namespace_for("")


@pytest.mark.asyncio
async def test_lws_status_reports_crd_and_controller(monkeypatch):
    commands = []

    async def fake_run_kubectl(argv, **kwargs):
        commands.append(argv)
        stdout = '{"status": {"availableReplicas": 1}}' if "deployment" in argv else "ok"
        return CommandResult(argv=tuple(argv), returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(service, "run_kubectl", fake_run_kubectl)
    status = await service.lws_status("cluster-a")

    assert status["ready"] is True
    assert status["crd"] == "ready"
    assert status["controller"] == "ready"
    assert status["version"] == service.lws_version()
    assert commands[0][:3] == ["get", "crd", "disaggregatedsets.disaggregatedset.x-k8s.io"]


@pytest.mark.asyncio
async def test_lws_status_reports_missing_crd(monkeypatch):
    async def fake_run_kubectl(argv, **kwargs):
        return CommandResult(argv=tuple(argv), returncode=1, stdout="", stderr="not found")

    monkeypatch.setattr(service, "run_kubectl", fake_run_kubectl)
    status = await service.lws_status("cluster-a")

    assert status["ready"] is False
    assert status["crd"] == "missing"
    assert status["controller"] == "missing"


@pytest.mark.asyncio
async def test_delete_cluster_rejects_when_storage_volumes_registered(monkeypatch):
    """A cluster must not be deletable while it still has Storage volumes
    registered against it -- reproduced live: deleting a cluster first, then
    trying to delete one of its ``local-disk`` volumes, raised
    ``FileNotFoundError`` (the volume's ``cluster_id`` no longer resolved to
    a kubeconfig) and surfaced as a 500 with no way to recover the volume.
    """
    cluster = registry.Cluster(id="cluster-1", name="c1", description="", created_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda cluster_id: False)
    monkeypatch.setattr(
        "llm_d_lens.storage.store.default_store",
        lambda: SimpleNamespace(list=lambda: [SimpleNamespace(cluster_id="cluster-1")]),
    )

    with pytest.raises(ClusterOverviewError) as excinfo:
        await service.delete_cluster("cluster-1")
    assert excinfo.value.code == "cluster_has_storage_volumes"


@pytest.mark.asyncio
async def test_delete_cluster_succeeds_when_no_storage_volumes(monkeypatch):
    cluster = registry.Cluster(id="cluster-1", name="c1", description="", created_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda cluster_id: False)
    monkeypatch.setattr(
        "llm_d_lens.storage.store.default_store",
        lambda: SimpleNamespace(list=lambda: [SimpleNamespace(cluster_id="cluster-other")]),
    )
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    deleted_namespaces: list[str] = []

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        deleted_namespaces.append(namespace)
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)
    deleted: list[str] = []
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: deleted.append(cluster_id))

    await service.delete_cluster("cluster-1")
    assert deleted == ["cluster-1"]
    assert deleted_namespaces == [cluster.gateway_namespace, "llm-d-bench-storage-cluster-1"]


@pytest.mark.asyncio
async def test_delete_cluster_deletes_secret_without_dropping_a_shared_namespace(monkeypatch):
    """A cluster may point at an HF_TOKEN Secret in a namespace Lens does not
    own (chosen in the wizard or by a later rotation). Deleting that whole
    namespace would destroy unrelated workloads, so only the Secret goes."""
    cluster = registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        hf_token_secret_namespace="inference-scheduling",  # noqa: S106 -- Kubernetes resource name, not a credential.
        hf_token_secret_name="llm-d-hf-token",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda cluster_id: False)
    monkeypatch.setattr(
        "llm_d_lens.storage.store.default_store",
        lambda: SimpleNamespace(list=lambda: []),
    )
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: None)
    deleted_namespaces: list[str] = []
    deleted_secrets: list[tuple[str, str]] = []

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        deleted_namespaces.append(namespace)
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    async def fake_delete_secret(name, *, namespace, cluster_id=None, timeout=None):
        deleted_secrets.append((namespace, name))
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_secret", fake_delete_secret)
    monkeypatch.setattr(
        service,
        "scoped_runner",
        _namespace_inspection_runner(labels={"kubernetes.io/metadata.name": "inference-scheduling"}, items=_BUSY_ITEMS),
    )

    await service.delete_cluster("cluster-1")

    assert deleted_secrets == [("inference-scheduling", "llm-d-hf-token")]
    # The Lens-owned namespace is still removed; the busy shared one is never deleted.
    assert deleted_namespaces == [cluster.gateway_namespace, "llm-d-bench-storage-cluster-1"]


_BUSY_ITEMS = [
    {"kind": "ConfigMap", "metadata": {"name": "kube-root-ca.crt"}},
    {"kind": "Deployment", "metadata": {"name": "epp"}},
]

_INJECTED_ONLY_ITEMS = [
    {"kind": "ConfigMap", "metadata": {"name": "kube-root-ca.crt"}},
    {"kind": "ConfigMap", "metadata": {"name": "istio-ca-root-cert"}},
    {"kind": "ServiceAccount", "metadata": {"name": "default"}},
    {"kind": "Secret", "metadata": {"name": "default-token-x"}, "type": "kubernetes.io/service-account-token"},
]


def _namespace_inspection_runner(*, labels, items, api_resources_rc=0):
    """Fake ``scoped_runner`` answering the namespace-emptiness probes."""

    async def run(command, *, input=None, timeout=None):  # noqa: A002 - mirrors the real signature
        if command[:3] == ["kubectl", "get", "namespace"]:
            return CommandResult(argv=(), returncode=0, stdout=json.dumps({"metadata": {"labels": labels}}), stderr="")
        if command[1] == "api-resources":
            return CommandResult(argv=(), returncode=api_resources_rc, stdout="configmaps\nsecrets\npods\n", stderr="")
        if command[1] == "get":
            return CommandResult(argv=(), returncode=0, stdout=json.dumps({"items": items}), stderr="")
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    return lambda cluster_id: SimpleNamespace(run=run)


def _cluster_with_external_secret():
    return registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        hf_token_secret_namespace="team-tokens",  # noqa: S106 -- Kubernetes resource name, not a credential.
        hf_token_secret_name="hf-token-1",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )


def _stub_delete_cluster_dependencies(monkeypatch, cluster, deleted_namespaces, deleted_secrets):
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda cluster_id: False)
    monkeypatch.setattr("llm_d_lens.storage.store.default_store", lambda: SimpleNamespace(list=lambda: []))
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: None)

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        deleted_namespaces.append(namespace)
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    async def fake_delete_secret(name, *, namespace, cluster_id=None, timeout=None):
        deleted_secrets.append((namespace, name))
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_secret", fake_delete_secret)


@pytest.mark.asyncio
async def test_delete_cluster_drops_a_namespace_left_holding_only_the_hf_token_secret(monkeypatch):
    """Once the Secret is gone the namespace holds nothing but objects every
    namespace gets for free, so keeping it serves no purpose."""
    deleted_namespaces: list[str] = []
    deleted_secrets: list[tuple[str, str]] = []
    _stub_delete_cluster_dependencies(monkeypatch, _cluster_with_external_secret(), deleted_namespaces, deleted_secrets)
    monkeypatch.setattr(service, "scoped_runner", _namespace_inspection_runner(labels={}, items=_INJECTED_ONLY_ITEMS))

    await service.delete_cluster("cluster-1")

    assert deleted_secrets == [("team-tokens", "hf-token-1")]
    assert deleted_namespaces == [
        registry.gateway_namespace_for("cluster-1"),
        "team-tokens",
        "llm-d-bench-storage-cluster-1",
    ]


@pytest.mark.asyncio
async def test_delete_cluster_drops_a_namespace_lens_created_for_the_secret(monkeypatch):
    """A namespace Lens created is removed on its label alone, without needing
    the content probe to agree."""
    deleted_namespaces: list[str] = []
    deleted_secrets: list[tuple[str, str]] = []
    _stub_delete_cluster_dependencies(monkeypatch, _cluster_with_external_secret(), deleted_namespaces, deleted_secrets)
    monkeypatch.setattr(
        service,
        "scoped_runner",
        _namespace_inspection_runner(
            labels={"app.kubernetes.io/managed-by": "llm-d-lens", "prism.ai/cluster-id": "cluster-1"},
            items=_BUSY_ITEMS,
        ),
    )

    await service.delete_cluster("cluster-1")

    assert deleted_namespaces == [
        registry.gateway_namespace_for("cluster-1"),
        "team-tokens",
        "llm-d-bench-storage-cluster-1",
    ]


@pytest.mark.asyncio
async def test_delete_cluster_keeps_a_namespace_whose_contents_cannot_be_listed(monkeypatch):
    """An uncertain answer must never authorise deleting a namespace."""
    deleted_namespaces: list[str] = []
    deleted_secrets: list[tuple[str, str]] = []
    _stub_delete_cluster_dependencies(monkeypatch, _cluster_with_external_secret(), deleted_namespaces, deleted_secrets)
    monkeypatch.setattr(
        service,
        "scoped_runner",
        _namespace_inspection_runner(labels={}, items=_INJECTED_ONLY_ITEMS, api_resources_rc=1),
    )

    await service.delete_cluster("cluster-1")

    assert deleted_namespaces == [registry.gateway_namespace_for("cluster-1"), "llm-d-bench-storage-cluster-1"]


@pytest.mark.asyncio
async def test_delete_cluster_never_deletes_a_protected_namespace(monkeypatch):
    """``default`` looks empty under the injected-object rule on some clusters;
    it is still never a candidate for deletion."""
    cluster = registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        hf_token_secret_namespace="default",  # noqa: S106 -- Kubernetes resource name, not a credential.
        hf_token_secret_name="hf-token-1",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    deleted_namespaces: list[str] = []
    deleted_secrets: list[tuple[str, str]] = []
    _stub_delete_cluster_dependencies(monkeypatch, cluster, deleted_namespaces, deleted_secrets)
    monkeypatch.setattr(
        service,
        "scoped_runner",
        _namespace_inspection_runner(labels={"app.kubernetes.io/managed-by": "llm-d-lens"}, items=_INJECTED_ONLY_ITEMS),
    )

    await service.delete_cluster("cluster-1")

    assert deleted_secrets == [("default", "hf-token-1")]
    assert deleted_namespaces == [cluster.gateway_namespace, "llm-d-bench-storage-cluster-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["returncode", "timeout"])
async def test_gateway_namespace_delete_failure_keeps_cluster_and_session(monkeypatch, failure):
    cluster = registry.Cluster(id="1234abcd", name="c", description="", created_at="")
    monkeypatch.setattr(registry, "require_cluster", lambda _: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda _: False)
    monkeypatch.setattr(service, "_has_storage_volumes", lambda _: False)
    delete_record, close_session, delete_hf = Mock(), Mock(), AsyncMock()
    monkeypatch.setattr(registry, "delete_cluster", delete_record)
    monkeypatch.setattr(service.sessions, "close_session_for_server", close_session)
    monkeypatch.setattr(service, "_delete_hf_token_resources", delete_hf)
    delete_ns = AsyncMock(
        side_effect=TimeoutError("namespace still terminating") if failure == "timeout" else None,
        return_value=CommandResult(argv=(), returncode=1, stdout="", stderr="forbidden"),
    )
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", delete_ns)

    with pytest.raises(ClusterOverviewError) as error:
        await service.delete_cluster(cluster.id)

    assert error.value.code == "cluster_gateway_cleanup_failed"
    assert error.value.status_code == 502
    delete_ns.assert_awaited_once_with("lens-gateway-1234abcd", cluster_id=cluster.id, timeout=120)
    delete_record.assert_not_called()
    close_session.assert_not_called()
    delete_hf.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_deletion_waits_for_gateway_work_and_cleans_namespace_before_record(monkeypatch):
    cluster = registry.Cluster(id="a1b2c3d4", name="c", description="", created_at="", gateway_namespace="custom")
    monkeypatch.setattr(registry, "require_cluster", lambda _: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda _: False)
    monkeypatch.setattr(service, "_has_storage_volumes", lambda _: False)
    monkeypatch.setattr(service, "_delete_hf_token_resources", AsyncMock())
    events = []

    async def delete_ns(namespace, *, cluster_id, timeout):
        events.append(("namespace", namespace))
        assert registry.gateway_lock(cluster_id).locked()
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", delete_ns)
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda _: events.append(("session", None)))
    monkeypatch.setattr(registry, "delete_cluster", lambda _: events.append(("record", None)))
    async with registry.gateway_lock(cluster.id):
        deletion = asyncio.create_task(service.delete_cluster(cluster.id))
        await asyncio.sleep(0)
        assert not events
    await deletion
    assert events == [("namespace", "custom-a1b2c3d4"), ("session", None), ("record", None)]


@pytest.mark.asyncio
async def test_delete_cluster_does_not_delete_the_secret_twice_in_its_own_namespace(monkeypatch):
    """When the Secret lives in the Lens-owned namespace, deleting the
    namespace already removes it; no separate Secret delete is issued."""
    cluster = registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        hf_token_secret_namespace="llm-d-bench-storage-cluster-1",  # noqa: S106 -- Resource name, not a credential.
        hf_token_secret_name="hf-token-abc",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service, "_has_available_deployments", lambda cluster_id: False)
    monkeypatch.setattr("llm_d_lens.storage.store.default_store", lambda: SimpleNamespace(list=lambda: []))
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: None)
    deleted_secrets: list[tuple[str, str]] = []

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    async def fake_delete_secret(name, *, namespace, cluster_id=None, timeout=None):
        deleted_secrets.append((namespace, name))
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_secret", fake_delete_secret)

    await service.delete_cluster("cluster-1")

    assert deleted_secrets == []


@pytest.mark.asyncio
async def test_delete_draft_cluster_deletes_storage_pvc_and_pv_not_only_the_record(monkeypatch):
    """Cancelling the creation wizard must not leave the volume's PVC/PV bound
    on the cluster with no Lens record left to reach them from."""
    cluster = registry.Cluster(
        id="cluster-draft", name="c", description="", created_at="2026-01-01T00:00:00Z", draft=True
    )
    volume = SimpleNamespace(id="storage-abc", cluster_id="cluster-draft")
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: None)
    removed_records: list[str] = []
    monkeypatch.setattr(
        "llm_d_lens.storage.store.default_store",
        lambda: SimpleNamespace(
            list=lambda: [volume, SimpleNamespace(id="storage-other", cluster_id="cluster-other")],
            delete=removed_records.append,
        ),
    )
    objects_deleted: list[tuple[str, str]] = []
    cache_cascades: list[tuple[str, bool]] = []

    async def fake_delete_volume_objects(vol, *, cluster_id):
        objects_deleted.append((vol.id, cluster_id))

    async def fake_delete_entries(volume_id, *, keep_model_files):
        cache_cascades.append((volume_id, keep_model_files))

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.storage.service.delete_volume_objects", fake_delete_volume_objects)
    monkeypatch.setattr("llm_d_lens.storage.service.delete_model_cache_entries_for_volume", fake_delete_entries)
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)

    await service.delete_cluster("cluster-draft")

    assert objects_deleted == [("storage-abc", "cluster-draft")]
    # Records are dropped but downloaded weights are preserved.
    assert cache_cascades == [("storage-abc", True)]
    assert removed_records == ["storage-abc"]


@pytest.mark.asyncio
async def test_delete_draft_cluster_still_removes_record_when_cluster_is_unreachable(monkeypatch):
    """An unreachable cluster must not leave the volume permanently undeletable."""
    cluster = registry.Cluster(
        id="cluster-draft", name="c", description="", created_at="2026-01-01T00:00:00Z", draft=True
    )
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(service.sessions, "close_session_for_server", lambda cluster_id: None)
    monkeypatch.setattr(registry, "delete_cluster", lambda cluster_id: None)
    removed_records: list[str] = []
    monkeypatch.setattr(
        "llm_d_lens.storage.store.default_store",
        lambda: SimpleNamespace(
            list=lambda: [SimpleNamespace(id="storage-abc", cluster_id="cluster-draft")],
            delete=removed_records.append,
        ),
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("cluster unreachable")

    async def fake_delete_namespace(namespace, *, cluster_id=None, timeout=None):
        return CommandResult(argv=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("llm_d_lens.storage.service.delete_volume_objects", boom)
    monkeypatch.setattr("llm_d_lens.storage.service.delete_model_cache_entries_for_volume", boom)
    monkeypatch.setattr("llm_d_lens.infrastructure.kubernetes.delete_namespace", fake_delete_namespace)

    await service.delete_cluster("cluster-draft")

    assert removed_records == ["storage-abc"]


def test_node_summary_reports_scheduling_disabled_from_spec_unschedulable():
    cordoned = {
        "metadata": {"name": "worker-1", "labels": {}},
        "spec": {"unschedulable": True},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }
    schedulable = {
        "metadata": {"name": "worker-2", "labels": {}},
        "spec": {},
        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
    }

    assert service._node_summary(cordoned)["schedulingDisabled"] is True
    assert service._node_summary(schedulable)["schedulingDisabled"] is False


@pytest.mark.asyncio
async def test_set_node_maintenance_cordons_worker_but_rejects_control_plane(monkeypatch):
    nodes_payload = {
        "items": [
            {
                "metadata": {"name": "worker-1", "labels": {}},
                "spec": {"taints": []},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
            {
                "metadata": {"name": "cp-1", "labels": {"node-role.kubernetes.io/control-plane": ""}},
                "spec": {"taints": [{"key": "node-role.kubernetes.io/control-plane", "effect": "NoSchedule"}]},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            },
        ]
    }

    calls: list[list[str]] = []

    async def fake_kubectl(command, *_args, **_kwargs):
        if command[:2] == ["get", "nodes"]:
            return CommandResult(("kubectl",), 0, json.dumps(nodes_payload), "")
        calls.append(command)
        return CommandResult(("kubectl",), 0, "node/worker-1 cordoned", "")

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)

    results = await service.set_node_maintenance(
        SimpleNamespace(id="cluster"), ["worker-1", "cp-1", "missing-node"], disabled=True
    )

    by_name = {item["name"]: item for item in results}
    assert by_name["worker-1"]["ok"] is True
    assert by_name["worker-1"]["schedulingDisabled"] is True
    assert by_name["cp-1"]["ok"] is False
    assert "control-plane" in by_name["cp-1"]["error"]
    assert by_name["missing-node"]["ok"] is False
    assert calls == [["cordon", "worker-1"]]


def test_active_gpu_allocations_ignore_admin_access_monitoring_claims():
    claims = [
        _claim(
            {"driver": "gpu.intel.com", "device": "gpu-1", "adminAccess": True},
            {"driver": "gpu.intel.com", "device": "gpu-2", "adminAccess": True},
        ),
        _claim({"driver": "gpu.intel.com", "device": "gpu-3"}),
        _claim({"driver": "example.com", "device": "other-device"}),
    ]

    assert _active_gpu_allocations([], claims) == 1


def _slice(node: str, *devices: tuple[str, str]) -> dict:
    return {
        "spec": {
            "driver": "gpu.intel.com",
            "nodeName": node,
            "pool": {"name": node},
            "devices": [{"name": name, "attributes": {"pciAddress": {"string": pci}}} for name, pci in devices],
        }
    }


def _memory_slice(node: str, *memory_values: str) -> dict:
    return {
        "spec": {
            "driver": "gpu.intel.com",
            "nodeName": node,
            "pool": {"name": node},
            "devices": [
                {"name": f"gpu-{index}", "capacity": {"memory": {"value": value}}}
                for index, value in enumerate(memory_values, start=1)
            ],
        }
    }


def test_hardware_summary_counts_only_unallocated_devices_in_configured_scope(monkeypatch):
    slices = [
        _slice("node-a", ("gpu-a1", "0000:01:00.0"), ("gpu-a2", "0000:02:00.0")),
        _slice("node-b", ("gpu-b1", "0000:01:00.0"), ("gpu-b2", "0000:02:00.0")),
    ]
    devices = service._resource_slice_gpu_devices(slices)
    claims = [
        _claim({"driver": "gpu.intel.com", "pool": "node-a", "device": "gpu-a1"}),
        _claim({"driver": "gpu.intel.com", "pool": "node-b", "device": "gpu-b2"}),
    ]
    nodes = [{"gpuCount": 2, "gpu": True}, {"gpuCount": 2, "gpu": True}]
    monkeypatch.setenv("PRISM_K8S_TARGET_NODE", "node-a")
    monkeypatch.setenv("PRISM_GPU_PCI_ALLOWLIST", "0000:01:00.0,0000:02:00.0")

    hardware = service._hardware_summary(
        nodes,
        allocated_gpu_count=2,
        gpu_devices=devices,
        allocated_gpu_devices=service._active_gpu_device_keys(claims),
    )

    assert hardware["gpuCount"] == 4
    assert hardware["availableGpuCount"] == 2
    assert hardware["configuredGpuLimit"] == 2
    assert hardware["usableGpuCount"] == 1


def test_hardware_summary_reports_usage_and_vram():
    nodes = [{"name": "node-a", "cpu": 16, "memoryBytes": 64 * 1024**3, "gpuCount": 4, "gpu": True}]
    hardware = service._hardware_summary(
        nodes,
        allocated_gpu_count=1,
        vram_bytes=192 * 1024**3,
        node_usage={"cpuMillicores": 4000, "memoryBytes": 16 * 1024**3},
    )

    assert hardware["cpuUsagePercent"] == 25
    assert hardware["memoryUsagePercent"] == 25
    assert hardware["gpuUsagePercent"] == 25
    assert hardware["vramBytes"] == 192 * 1024**3
    assert hardware["vramUsagePercent"] is None


def test_node_summary_with_gpus_validates_in_pydantic():
    from llm_d_lens.cluster.models import KubernetesSummary

    node = {
        "name": "worker-1",
        "version": "v1.28.0",
        "osImage": "Ubuntu 22.04",
        "ready": True,
        "cpu": 16.0,
        "cpuUsagePercent": 25.0,
        "memoryBytes": 64 * 1024**3,
        "memoryUsagePercent": 50.0,
        "gpu": True,
        "gpuCount": 2,
        "gpuUsagePercent": 80.0,
        "vramBytes": 96 * 1024**3,
        "vramUsagePercent": 40.0,
        "role": "worker",
        "cachedImages": ["ghcr.io/llm-d/llm-d-xpu:v0.9.0"],
        "gpus": [
            {
                "id": "gpu-0",
                "index": 0,
                "name": "GPU #0",
                "pciAddress": "0000:03:00.0",
                "computeUsagePercent": 80.0,
                "vramTotalBytes": 48 * 1024**3,
                "vramUsedBytes": 20 * 1024**3,
                "vramUsagePercent": 41.6,
                "vramReadThroughputBytes": 10000,
                "vramWriteThroughputBytes": 20000,
                "vramTotalThroughputBytes": 30000,
                "vramBandwidthPercent": 15.0,
            }
        ],
    }
    overview = {
        "components": {
            "intelDevicePlugin": {"installed": True, "ready": 1, "desired": 1, "status": "ready"},
            "monitoring": {"installed": True, "ready": 5, "desired": 5, "status": "ready"},
        },
        "nodes": [node],
        "namespaces": [],
        "hardware": {},
    }
    parsed = KubernetesSummary.model_validate(overview)
    assert len(parsed.nodes) == 1
    assert parsed.nodes[0].gpus[0].pci_address == "0000:03:00.0"
    assert parsed.nodes[0].gpus[0].vram_bandwidth_percent == 15.0
    assert parsed.nodes[0].cached_images == ["ghcr.io/llm-d/llm-d-xpu:v0.9.0"]


def test_hardware_summary_prefers_prometheus_gpu_and_vram_metrics():
    hardware = service._hardware_summary(
        [{"name": "node-a", "cpu": 16, "memoryBytes": 64 * 1024**3, "gpuCount": 4, "gpu": True}],
        allocated_gpu_count=1,
        vram_bytes=0,
        prometheus_metrics={
            "gpuUsagePercent": 72.34,
            "vramBytes": 96 * 1024**3,
            "vramUsagePercent": 55.55,
        },
    )

    assert hardware["gpuUsagePercent"] == 72.3
    assert hardware["vramBytes"] == 96 * 1024**3
    assert hardware["vramUsagePercent"] == 55.5


def test_hardware_summary_prefers_free_style_prometheus_memory_usage():
    hardware = service._hardware_summary(
        [{"name": "node-a", "cpu": 16, "memoryBytes": 64 * 1024**3, "gpuCount": 0, "gpu": False}],
        node_usage={"memoryBytes": 16 * 1024**3},
        prometheus_metrics={"memoryUsagePercent": 6.25},
    )

    assert hardware["memoryUsagePercent"] == 6.2


def test_hardware_summary_prefers_top_style_prometheus_cpu_usage():
    hardware = service._hardware_summary(
        [{"name": "node-a", "cpu": 16, "memoryBytes": 64 * 1024**3, "gpuCount": 0, "gpu": False}],
        node_usage={"cpuMillicores": 4000},
        prometheus_metrics={"cpuUsagePercent": 6.25},
    )

    assert hardware["cpuUsagePercent"] == 6.2


def test_resource_slice_gpu_vram_bytes():
    assert service._resource_slice_gpu_vram_bytes([_memory_slice("node-a", "48Gi", "512Mi")]) == int(48.5 * 1024**3)


def test_resource_slice_gpu_vram_bytes_counts_only_worker_nodes():
    slices = [_memory_slice("worker-a", "48Gi"), _memory_slice("control-plane", "48Gi")]

    assert service._resource_slice_gpu_vram_bytes(slices, {"worker-a"}) == 48 * 1024**3


def test_prometheus_device_values_filter_workers_and_dedupe_by_device():
    samples = [
        {"metric": {"node": "worker-a", "pci_bdf": "0000:01:00.0", "job": "xpumd-a"}, "value": [1, "51539607552"]},
        {"metric": {"node": "worker-a", "pci_bdf": "0000:01:00.0", "job": "xpumd-b"}, "value": [1, "51539607552"]},
        {"metric": {"node": "control-plane", "pci_bdf": "0000:02:00.0", "job": "xpumd"}, "value": [1, "51539607552"]},
    ]

    values = service._latest_device_values(samples, {"worker-a"}, {})

    assert values == {("worker-a", "0000:01:00.0"): 51539607552.0}


@pytest.mark.asyncio
async def test_prometheus_hardware_metrics_uses_compute_engine_utilization(monkeypatch):
    queries = []

    async def fake_local_port(cluster_id):
        assert cluster_id == "cluster-1"
        return 19090

    async def fake_query(_client, query):
        queries.append(query)
        if query == 'hw_gpu_utilization_ratio{hw_gpu_task="compute-all"}':
            return [
                {
                    "metric": {"node": "worker-a", "pci_bdf": "0000:01:00.0"},
                    "value": [1, "0.75"],
                }
            ]
        return []

    monkeypatch.setattr(service, "_prometheus_local_port", fake_local_port)
    monkeypatch.setattr(service, "_prometheus_query", fake_query)

    metrics = await service._prometheus_hardware_metrics(
        "cluster-1",
        {"worker-a"},
        {("worker-a", "gpu-a"): {"node": "worker-a", "pciAddress": "0000:01:00.0"}},
    )

    assert 'hw_gpu_utilization_ratio{hw_gpu_task="compute-all"}' in queries
    assert 'hw_gpu_utilization_ratio{hw_gpu_task="all"}' not in queries
    assert metrics["gpuUsagePercent"] == 75
    assert metrics["nodeGpu"] == {"worker-a": 75}


def test_instance_to_node_maps_prometheus_instance_label_via_node_uname_info():
    samples = [
        {"metric": {"instance": "172.19.0.2:9100", "nodename": "worker-a"}, "value": [1, "1"]},
        {"metric": {"instance": "172.19.0.3:9100", "node": "worker-b"}, "value": [1, "1"]},
        {"metric": {"instance": "172.19.0.4:9100"}, "value": [1, "1"]},
    ]

    mapping = service._instance_to_node(samples)

    assert mapping == {"172.19.0.2:9100": "worker-a", "172.19.0.3:9100": "worker-b"}


def test_latest_node_percent_values_falls_back_to_instance_mapping_and_filters_workers():
    samples = [
        {"metric": {"instance": "172.19.0.2:9100"}, "value": [1, "38.4"]},
        {"metric": {"instance": "172.19.0.9:9100"}, "value": [1, "99.9"]},
        {"metric": {"node": "worker-b"}, "value": [1, "150"]},
    ]
    instance_to_node = {"172.19.0.2:9100": "worker-a", "172.19.0.9:9100": "control-plane"}

    values = service._latest_node_percent_values(samples, instance_to_node, {"worker-a", "worker-b"})

    assert values == {"worker-a": 38.4, "worker-b": 100.0}


def test_annotate_node_usage_falls_back_to_prometheus_cpu_and_memory_when_metrics_api_missing():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 0, "gpu": False}]

    annotated = service._annotate_node_usage(
        nodes,
        {},
        {"nodeCpu": {"worker-a": 38.4}, "nodeMemory": {"worker-a": 21.2}},
    )

    assert annotated[0]["cpuUsagePercent"] == 38.4
    assert annotated[0]["memoryUsagePercent"] == 21.2


def test_annotate_node_usage_prefers_free_style_prometheus_memory_usage():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 0, "gpu": False}]

    annotated = service._annotate_node_usage(
        nodes,
        {"worker-a": {"memoryBytes": 8 * 1024**3}},
        {"nodeMemory": {"worker-a": 6.2}},
    )

    assert annotated[0]["memoryUsagePercent"] == 6.2


def test_annotate_node_usage_prefers_top_style_prometheus_cpu_usage():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 0, "gpu": False}]

    annotated = service._annotate_node_usage(
        nodes,
        {"worker-a": {"cpuMillicores": 2000}},
        {"nodeCpu": {"worker-a": 6.2}},
    )

    assert annotated[0]["cpuUsagePercent"] == 6.2


def test_annotate_node_usage_adds_per_worker_utilization():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 2, "gpu": True}]
    annotated = service._annotate_node_usage(
        nodes,
        {"worker-a": {"cpuMillicores": 2000, "memoryBytes": 8 * 1024**3}},
        {
            "nodeGpu": {"worker-a": 62.34},
            "nodeVram": {"worker-a": {"totalBytes": 48 * 1024**3, "usedBytes": 12 * 1024**3}},
        },
    )

    assert annotated[0]["cpuUsagePercent"] == 25
    assert annotated[0]["memoryUsagePercent"] == 25
    assert annotated[0]["gpuUsagePercent"] == 62.3
    assert annotated[0]["vramBytes"] == 48 * 1024**3
    assert annotated[0]["vramUsagePercent"] == 25


def test_annotate_node_usage_adds_kubelet_disk_stats():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 0, "gpu": False}]

    annotated = service._annotate_node_usage(
        nodes,
        {},
        {},
        disk_usage_by_node={"worker-a": {"capacityBytes": 470 * 1024**3, "usedBytes": 470 * 1024**3}},
    )

    assert annotated[0]["diskBytes"] == 470 * 1024**3
    assert annotated[0]["diskUsedBytes"] == 470 * 1024**3
    assert annotated[0]["diskUsagePercent"] == 100.0


def test_annotate_node_usage_disk_defaults_when_kubelet_unreachable():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 32 * 1024**3, "gpuCount": 0, "gpu": False}]

    annotated = service._annotate_node_usage(nodes, {}, {})

    assert annotated[0]["diskBytes"] == 0
    assert annotated[0]["diskUsedBytes"] is None
    assert annotated[0]["diskUsagePercent"] is None


def test_hardware_summary_aggregates_disk_usage_across_nodes():
    nodes = [
        {
            "name": "worker-a",
            "cpu": 8,
            "memoryBytes": 0,
            "gpuCount": 0,
            "gpu": False,
            "diskBytes": 100,
            "diskUsedBytes": 100,
            "diskUsagePercent": 100.0,
        },
        {
            "name": "worker-b",
            "cpu": 8,
            "memoryBytes": 0,
            "gpuCount": 0,
            "gpu": False,
            "diskBytes": 100,
            "diskUsedBytes": 0,
            "diskUsagePercent": 0.0,
        },
    ]

    hardware = service._hardware_summary(nodes)

    assert hardware["diskBytes"] == 200
    assert hardware["diskUsagePercent"] == 50.0
    # Free disk is the exact remainder of the kubelet's own used-bytes figures,
    # so the Clusters card can lead with capacity left instead of usage.
    assert hardware["diskUsedBytes"] == 100
    assert hardware["diskAvailableBytes"] == 100


def test_hardware_summary_disk_usage_percent_none_when_no_node_reports_disk():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 0, "gpuCount": 0, "gpu": False}]

    hardware = service._hardware_summary(nodes)

    assert hardware["diskBytes"] == 0
    assert hardware["diskUsagePercent"] is None
    assert hardware["diskUsedBytes"] is None
    assert hardware["diskAvailableBytes"] is None


def test_hardware_summary_reports_free_memory_from_the_usage_reading():
    nodes = [
        {"name": "worker-a", "cpu": 8, "memoryBytes": 600, "gpuCount": 0, "gpu": False},
        {"name": "worker-b", "cpu": 8, "memoryBytes": 400, "gpuCount": 0, "gpu": False},
    ]

    hardware = service._hardware_summary(nodes, prometheus_metrics={"memoryUsagePercent": 25.0})

    assert hardware["memoryBytes"] == 1000
    assert hardware["memoryUsagePercent"] == 25.0
    assert hardware["memoryAvailableBytes"] == 750


def test_hardware_summary_free_memory_is_unknown_without_a_usage_reading():
    nodes = [{"name": "worker-a", "cpu": 8, "memoryBytes": 600, "gpuCount": 0, "gpu": False}]

    hardware = service._hardware_summary(nodes)

    assert hardware["memoryBytes"] == 600
    assert hardware["memoryAvailableBytes"] is None


@pytest.mark.asyncio
async def test_node_disk_usage_parses_kubelet_summary(monkeypatch):
    async def fake_kubectl(command, timeout=10, cluster_id=None):
        assert command[-1] == "/api/v1/nodes/worker-a/proxy/stats/summary"
        return CommandResult(
            argv=command,
            returncode=0,
            stdout=json.dumps({"node": {"fs": {"capacityBytes": 1000, "usedBytes": 1000}}}),
            stderr="",
        )

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)

    result = await service._node_disk_usage("cluster-1", {"worker-a"})

    assert result == {"worker-a": {"capacityBytes": 1000, "usedBytes": 1000}}


@pytest.mark.asyncio
async def test_node_disk_usage_skips_node_when_kubelet_call_fails(monkeypatch):
    async def fake_kubectl(command, timeout=10, cluster_id=None):
        raise TimeoutError("kubectl timed out")

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)

    result = await service._node_disk_usage("cluster-1", {"worker-a"})

    assert result == {}


def test_hardware_summary_uses_node_with_pci_identity_not_pci_alone(monkeypatch):
    slices = [
        _slice("node-a", ("gpu-a1", "0000:01:00.0")),
        _slice("node-b", ("gpu-b1", "0000:01:00.0")),
    ]
    devices = service._resource_slice_gpu_devices(slices)
    monkeypatch.delenv("PRISM_K8S_TARGET_NODE", raising=False)
    monkeypatch.setenv("PRISM_GPU_PCI_ALLOWLIST", "0000:01:00.0")

    hardware = service._hardware_summary(
        [{"gpuCount": 1, "gpu": True}, {"gpuCount": 1, "gpu": True}],
        gpu_devices=devices,
        allocated_gpu_devices={("node-a", "gpu-a1")},
    )

    assert hardware["configuredGpuLimit"] == 2
    assert hardware["usableGpuCount"] == 1


@pytest.mark.asyncio
async def test_model_secret_discovery_returns_external_coordinates_without_values(monkeypatch):
    """Every namespace's HF_TOKEN secret is returned regardless of its name
    -- namespace names are user-chosen (including via a deployment's custom
    ``namespace_policy.prefix``, see ``build_namespace_factory``), so
    filtering by name/prefix is unreliable and previously hid legitimate
    secrets (reproduced live: a secret in ``llm-d-bench-storage-<cluster
    id>``, created by the Create Cluster wizard's Model Cache step, did not
    show up when choosing a token to prewarm a model). Secret values must
    never be included in the response, and secrets without HF_TOKEN must be
    skipped."""
    payload = {
        "items": [
            {
                "metadata": {"namespace": "models", "name": "huggingface", "creationTimestamp": "now"},
                "data": {"HF_TOKEN": "must-not-be-returned"},
            },
            {
                "metadata": {
                    "namespace": f"{service.cluster_settings.namespace_prefix}failed-deployment",
                    "name": "llm-d-hf-token",
                },
                "data": {"HF_TOKEN": "copied-deployment-token"},
            },
            {
                "metadata": {
                    "namespace": "llm-d-bench-storage-e97867eb",
                    "name": "hf-token-544aeb47",
                    "creationTimestamp": "now",
                },
                "data": {"HF_TOKEN": "value"},
            },
            {"metadata": {"namespace": "models", "name": "unrelated"}, "data": {"OTHER": "value"}},
        ]
    }

    async def fake_kubectl(*_args, **_kwargs):
        return CommandResult(("kubectl",), 0, json.dumps(payload), "")

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)
    result = await service.list_model_secrets(SimpleNamespace(id="cluster"))

    assert result == [
        {"namespace": "llm-d-bench-failed-deployment", "name": "llm-d-hf-token", "createdAt": ""},
        {"namespace": "llm-d-bench-storage-e97867eb", "name": "hf-token-544aeb47", "createdAt": "now"},
        {"namespace": "models", "name": "huggingface", "createdAt": "now"},
    ]


# --- Create Cluster wizard: proxy settings + HF token secret creation ---


def test_registry_round_trips_proxy_and_version_refs(tmp_path, monkeypatch):
    """Cluster metadata should persist proxy/llm-d ref fields across a save/load cycle."""
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    cluster = registry.create_cluster(
        "c1",
        "",
        "kubeconfig: {}",
        proxy_mode="custom",
        http_proxy="http://proxy:3128",
        https_proxy="http://proxy:3128",
        no_proxy="localhost",
        llm_d_ref="v0.2.0",
        llm_d_lensmark_ref="abc1234",
    )
    reloaded = registry.get_cluster(cluster.id)
    assert reloaded.proxy_mode == "custom"
    assert reloaded.http_proxy == "http://proxy:3128"
    assert reloaded.no_proxy == "localhost"
    assert reloaded.llm_d_ref == "v0.2.0"
    assert reloaded.llm_d_lensmark_ref == "abc1234"


def test_registry_update_cluster_patches_only_given_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    cluster = registry.create_cluster("c1", "desc", "kubeconfig: {}")
    updated = registry.update_cluster(cluster.id, llm_d_ref="main")
    assert updated.llm_d_ref == "main"
    assert updated.name == "c1"
    assert updated.description == "desc"
    assert updated.proxy_mode == "auto"
    reloaded = registry.get_cluster(cluster.id)
    assert reloaded.llm_d_ref == "main"


def test_create_cluster_rejects_empty_custom_proxy():
    with pytest.raises(ClusterOverviewError) as excinfo:
        service.create_cluster("c1", "", "kubeconfig: {}", proxy_mode="custom")
    assert excinfo.value.code == "invalid_proxy_config"


def test_update_cluster_settings_rejects_empty_custom_proxy(monkeypatch):
    cluster = registry.Cluster(id="cluster-1", name="c1", description="", created_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    with pytest.raises(ClusterOverviewError) as excinfo:
        service.update_cluster_settings("cluster-1", proxy_mode="custom")
    assert excinfo.value.code == "invalid_proxy_config"


def test_update_cluster_settings_rejects_model_gateway_provider_change(monkeypatch):
    cluster = registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        gateway_provider="istio",
    )
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)
    monkeypatch.setattr(registry, "update_cluster", lambda *args, **kwargs: cluster)
    with pytest.raises(ClusterOverviewError) as excinfo:
        service.update_cluster_settings("cluster-1", gateway_provider="envoy-ai-gateway")
    assert excinfo.value.code == "gateway_provider_immutable"
    service.update_cluster_settings("cluster-1", gateway_provider="istio")


def test_update_cluster_settings_allows_first_model_gateway_provider_assignment(monkeypatch):
    cluster = registry.Cluster(id="cluster-1", name="c1", description="", created_at="2026-01-01T00:00:00Z")
    captured = {}
    monkeypatch.setattr(registry, "require_cluster", lambda cluster_id: cluster)

    def fake_update(cluster_id, **kwargs):
        captured.update(kwargs)
        return cluster

    monkeypatch.setattr(registry, "update_cluster", fake_update)
    service.update_cluster_settings("cluster-1", gateway_provider="istio")
    assert captured["gateway_provider"] == "istio"


def test_create_cluster_rejects_duplicate_name(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    registry.create_cluster("prod-cluster", "", "kubeconfig: {}")
    with pytest.raises(ClusterOverviewError) as excinfo:
        registry.create_cluster("Prod-Cluster", "", "kubeconfig: {}")  # different case, still a duplicate
    assert excinfo.value.code == "cluster_name_conflict"
    assert excinfo.value.status_code == 409


def test_create_cluster_allows_duplicate_name_against_a_draft(tmp_path, monkeypatch):
    """A name collision with an in-progress (or abandoned) draft cluster must
    NOT be rejected: drafts are invisible scratch state that may never be
    cleaned up (e.g. the wizard tab was closed before its delete request
    landed), so counting them here would let a stale draft permanently
    squat a name and block real recreation with a spurious conflict."""
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    registry.create_cluster("wip-cluster", "", "kubeconfig: {}", draft=True)
    created = registry.create_cluster("wip-cluster", "", "kubeconfig: {}")
    assert created.name == "wip-cluster"


def test_update_cluster_rejects_finalizing_draft_into_duplicate_name(tmp_path, monkeypatch):
    """If two drafts raced to the same name, the second one to finalize
    (draft False) must still be rejected instead of silently producing two
    real clusters with the same name."""
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    first = registry.create_cluster("wip-cluster", "", "kubeconfig: {}", draft=True)
    second = registry.create_cluster("wip-cluster", "", "kubeconfig: {}", draft=True)
    registry.update_cluster(first.id, draft=False)
    with pytest.raises(ClusterOverviewError) as excinfo:
        registry.update_cluster(second.id, draft=False)
    assert excinfo.value.code == "cluster_name_conflict"


def test_update_cluster_rejects_rename_to_duplicate_name(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    registry.create_cluster("existing", "", "kubeconfig: {}")
    other = registry.create_cluster("other", "", "kubeconfig: {}")
    with pytest.raises(ClusterOverviewError) as excinfo:
        registry.update_cluster(other.id, name="existing")
    assert excinfo.value.code == "cluster_name_conflict"


def test_update_cluster_allows_renaming_to_its_own_current_name(tmp_path, monkeypatch):
    """Re-saving a cluster's own unchanged name must not trip the uniqueness
    check against itself (e.g. an Edit modal that always sends `name`)."""
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    cluster = registry.create_cluster("my-cluster", "", "kubeconfig: {}")
    updated = registry.update_cluster(cluster.id, name="my-cluster", description="new description")
    assert updated.name == "my-cluster"
    assert updated.description == "new description"


def _clear_proxy_env(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_resolve_proxy_env_uses_custom_cluster_values(monkeypatch):
    cluster = registry.Cluster(
        id="cluster-1",
        name="c1",
        description="",
        created_at="2026-01-01T00:00:00Z",
        proxy_mode="custom",
        http_proxy="http://custom:3128",
        https_proxy=None,
        no_proxy=None,
    )
    monkeypatch.setattr(registry, "get_cluster", lambda cluster_id: cluster)
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://ambient:3128")
    assert service.resolve_proxy_env("cluster-1") == {"HTTP_PROXY": "http://custom:3128"}


def test_resolve_proxy_env_falls_back_to_process_env_when_auto(monkeypatch):
    cluster = registry.Cluster(id="cluster-1", name="c1", description="", created_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(registry, "get_cluster", lambda cluster_id: cluster)
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://ambient:3128")
    assert service.resolve_proxy_env("cluster-1") == {"HTTP_PROXY": "http://ambient:3128"}


def test_resolve_proxy_env_returns_empty_for_unknown_cluster(monkeypatch):
    monkeypatch.setattr(registry, "get_cluster", lambda cluster_id: None)
    _clear_proxy_env(monkeypatch)
    assert service.resolve_proxy_env("unknown") == {}


@pytest.mark.asyncio
async def test_create_hf_token_secret_rejects_duplicate_name(monkeypatch):
    cluster = SimpleNamespace(id="cluster-1")

    async def fake_kubectl(command, *, cluster_id, timeout):
        assert command[:2] == ["get", "secret"]
        return CommandResult(("kubectl",), 0, "secret/hf-token-existing", "")

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)

    with pytest.raises(ClusterOverviewError) as excinfo:
        await service.create_hf_token_secret(cluster, "default", "hf-token-existing", "hf_xxx")
    assert excinfo.value.code == "secret_already_exists"


@pytest.mark.asyncio
async def test_create_hf_token_secret_creates_when_absent(monkeypatch):
    cluster = registry.create_cluster("cluster-1", "", "kubeconfig: {}")
    created: dict[str, object] = {}

    async def fake_kubectl(command, *, cluster_id, timeout):
        return CommandResult(("kubectl",), 0, "", "")

    class FakeRunner:
        async def run(self, argv, *, input=None, timeout=None):  # noqa: A002 - mirrors CommandRunner's public keyword
            if argv[1:3] == ["get", "namespace"]:
                return CommandResult(("kubectl",), 1, "", 'namespaces "default" not found')
            if argv[1] == "apply":
                created["namespace_manifest"] = json.loads(input)
                return CommandResult(("kubectl",), 0, "namespace/default created", "")
            if argv[1] == "create" and argv[2] == "-f":
                created["manifest"] = json.loads(input)
                return CommandResult(("kubectl",), 0, "secret/hf-token-new created", "")
            raise AssertionError(f"unexpected kubectl invocation: {argv}")

    monkeypatch.setattr(service, "run_kubectl", fake_kubectl)
    monkeypatch.setattr(service, "scoped_runner", lambda cluster_id: FakeRunner())

    ref = await service.create_hf_token_secret(cluster, "default", "hf-token-new", "hf_secret_value")

    assert ref == {"namespace": "default", "name": "hf-token-new"}
    assert created["manifest"]["metadata"] == {"name": "hf-token-new", "namespace": "default"}
    assert created["manifest"]["stringData"] == {"HF_TOKEN": "hf_secret_value"}
    # A namespace Lens creates is labelled, so deleting the cluster can remove it.
    assert created["namespace_manifest"]["metadata"]["labels"] == {
        "app.kubernetes.io/managed-by": "llm-d-lens",
        "prism.ai/cluster-id": cluster.id,
    }
    # Recorded on the cluster so Deploy/Evaluate can default to reusing it later.
    reloaded = registry.get_cluster(cluster.id)
    assert reloaded.hf_token_secret_namespace == "default"  # noqa: S105
    assert reloaded.hf_token_secret_name == "hf-token-new"  # noqa: S105 -- k8s Secret name, not a credential


def _apply_only_runner(captured: dict[str, object]):
    class FakeRunner:
        async def run(self, argv, *, input=None, timeout=None):  # noqa: A002 - mirrors CommandRunner's public keyword
            if argv[1:3] == ["get", "namespace"]:
                # The namespace already exists, so it is left exactly as is.
                return CommandResult(("kubectl",), 0, "namespace/llm-d-bench-storage-cluster-1", "")
            if argv[1] == "apply" and argv[2] == "-f":
                captured.setdefault("applied", []).append(json.loads(input or ""))
                return CommandResult(("kubectl",), 0, "secret/hf-token configured", "")
            raise AssertionError(f"unexpected kubectl invocation: {argv}")

    return FakeRunner()


@pytest.mark.asyncio
async def test_update_hf_token_secret_rewrites_recorded_secret_in_place(monkeypatch):
    """Rotating a token needs only the token: coordinates default to the ones
    already recorded on the cluster, and the write must use ``apply`` so an
    existing Secret is replaced instead of failing with a name collision."""
    cluster = registry.create_cluster("cluster-rotate", "", "kubeconfig: {}")
    registry.update_cluster(
        cluster.id,
        hf_token_secret_namespace="llm-d-bench-storage-x",  # noqa: S106 -- Resource name, not a credential.
        hf_token_secret_name="hf-token-old",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    cluster = registry.get_cluster(cluster.id)
    captured: dict[str, object] = {}
    monkeypatch.setattr(service, "scoped_runner", lambda cluster_id: _apply_only_runner(captured))

    ref = await service.update_hf_token_secret(cluster, "hf_rotated")

    assert ref == {"namespace": "llm-d-bench-storage-x", "name": "hf-token-old"}
    secret = [item for item in captured["applied"] if item and item.get("kind") == "Secret"][0]
    assert secret["metadata"] == {"name": "hf-token-old", "namespace": "llm-d-bench-storage-x"}
    assert secret["stringData"] == {"HF_TOKEN": "hf_rotated"}


@pytest.mark.asyncio
async def test_update_hf_token_secret_repoints_cluster_to_new_coordinates(monkeypatch):
    """Supplying namespace/name moves the cluster's pointer, so later deploys
    default to the new Secret rather than the one created by the wizard."""
    cluster = registry.create_cluster("cluster-repoint", "", "kubeconfig: {}")
    registry.update_cluster(
        cluster.id,
        hf_token_secret_namespace="old-ns",  # noqa: S106 -- Kubernetes resource name, not a credential.
        hf_token_secret_name="hf-token-old",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    cluster = registry.get_cluster(cluster.id)
    captured: dict[str, object] = {}
    monkeypatch.setattr(service, "scoped_runner", lambda cluster_id: _apply_only_runner(captured))

    ref = await service.update_hf_token_secret(cluster, "hf_new", namespace="new-ns", name="hf-token-new")

    assert ref == {"namespace": "new-ns", "name": "hf-token-new"}
    reloaded = registry.get_cluster(cluster.id)
    assert reloaded.hf_token_secret_namespace == "new-ns"  # noqa: S105
    assert reloaded.hf_token_secret_name == "hf-token-new"  # noqa: S105 -- k8s Secret name, not a credential


@pytest.mark.asyncio
async def test_update_hf_token_secret_requires_coordinates_when_cluster_has_none(monkeypatch):
    """A cluster created before the wizard's secret step has no recorded
    coordinates, so the caller must supply them instead of writing a Secret
    into an arbitrary namespace."""
    cluster = registry.create_cluster("cluster-no-secret", "", "kubeconfig: {}")
    monkeypatch.setattr(service, "scoped_runner", lambda cluster_id: _apply_only_runner({}))

    with pytest.raises(ClusterOverviewError) as excinfo:
        await service.update_hf_token_secret(cluster, "hf_new")
    assert excinfo.value.code == "invalid_secret_namespace"


@pytest.mark.asyncio
async def test_update_hf_token_secret_rejects_blank_token():
    cluster = registry.create_cluster("cluster-blank-token", "", "kubeconfig: {}")

    with pytest.raises(ClusterOverviewError) as excinfo:
        await service.update_hf_token_secret(cluster, "   ")
    assert excinfo.value.code == "invalid_secret_token"  # noqa: S105 -- error code, not a credential


def test_cluster_record_exposes_hf_token_secret_coordinates():
    """The Edit cluster drawer needs the recorded coordinates to prefill its
    rotation fields, so the API record must expose them (never the token)."""
    from llm_d_lens.cluster.router import _record

    cluster = registry.create_cluster("cluster-record-secret", "", "kubeconfig: {}")
    assert _record(cluster).hf_token_secret is None

    registry.update_cluster(
        cluster.id,
        hf_token_secret_namespace="ns-a",  # noqa: S106 -- Kubernetes resource name, not a credential.
        hf_token_secret_name="hf-token-a",  # noqa: S106 -- Kubernetes resource name, not a credential.
    )
    record = _record(registry.get_cluster(cluster.id))

    assert record.hf_token_secret is not None
    payload = record.model_dump(by_alias=True)
    assert payload["hfTokenSecret"] == {"namespace": "ns-a", "name": "hf-token-a"}
    assert "token" not in payload["hfTokenSecret"]


@pytest.mark.asyncio
async def test_monitoring_component_evaluates_expected_dashboards(monkeypatch):
    cluster = SimpleNamespace(id="cluster-a")
    captured = {}

    async def fake_expected_dashboard_names(cluster_id=None):
        captured["cluster_id"] = cluster_id
        return ["dashboard-a", "dashboard-b"]

    async def fake_discover(namespace, runner, *, expected_dashboards=None):
        captured["namespace"] = namespace
        captured["expected_dashboards"] = expected_dashboards
        return SimpleNamespace(
            status="degraded",
            components=[
                SimpleNamespace(name="prometheus", status="ready"),
                SimpleNamespace(name="llm_d_dashboards", status="missing"),
            ],
            message="Monitoring stack requires attention",
        )

    monkeypatch.setattr(
        "llm_d_lens.monitoring.cluster_stack.service.expected_dashboard_names", fake_expected_dashboard_names
    )
    monkeypatch.setattr(service, "discover_cluster_stack", fake_discover)
    monkeypatch.setattr(service, "scoped_runner", lambda cluster_id: object())
    service._MONITORING_CACHE.clear()

    result = await service.monitoring_component(cluster)

    assert result["status"] == "degraded"
    assert result["dashboardsReady"] is False
    assert result["dashboardOnly"] is True
    assert captured == {
        "cluster_id": "cluster-a",
        "namespace": "llm-d-monitoring",
        "expected_dashboards": ["dashboard-a", "dashboard-b"],
    }


def test_component_status_exposes_per_profile_device_plugins():
    resources = [
        {
            "kind": "DaemonSet",
            "metadata": {"name": "intel-gpu-plugin"},
            "status": {"desiredNumberScheduled": 1, "numberReady": 1},
        },
        {
            "kind": "DaemonSet",
            "metadata": {"name": "nvidia-device-plugin-daemonset"},
            "status": {"desiredNumberScheduled": 1, "numberReady": 1},
        },
    ]
    status = service.component_status(resources)
    assert status["intelDevicePlugin"]["installed"] is True
    assert status["acceleratorDevicePlugins"]["intel-xpu"]["ready"] == 1
    assert status["acceleratorDevicePlugins"]["nvidia"]["ready"] == 1

    from llm_d_lens.cluster.models import ComponentsSummary

    ComponentsSummary.model_validate(
        {
            **status,
            "monitoring": {"installed": False, "ready": 0, "desired": 0, "status": "missing"},
        }
    )


def test_hardware_summary_exposes_per_vendor_buckets():
    nodes = [
        {
            "name": "node-a",
            "cpu": 16,
            "memoryBytes": 64 * 1024**3,
            "gpuCount": 4,
            "gpu": True,
            "gpuModelsByProfile": {"intel-xpu": ["Intel Data Center GPU Flex B60"]},
        }
    ]
    hardware = service._hardware_summary(nodes, gpu_by_profile={"intel-xpu": 4})
    assert hardware["gpuCount"] == 4
    assert hardware["totalGpuCount"] == 4
    assert hardware["accelerators"] == [
        {"id": "intel-xpu", "gpuCount": 4, "models": ["Intel Data Center GPU Flex B60"]}
    ]


def test_node_summary_reads_accelerator_models_from_profile_labels():
    from llm_d_lens.cluster.models import NodeSummary

    node = {
        "metadata": {
            "name": "node-a",
            "labels": {
                "feature.node.kubernetes.io/pci-10de.present": "true",
                "nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB",
            },
        },
        "spec": {},
        "status": {"capacity": {"nvidia.com/gpu": "8"}, "conditions": []},
    }
    summary = service._node_summary(node)
    assert summary["gpuModelsByProfile"] == {"nvidia": ["NVIDIA-A100-SXM4-80GB"]}
    assert NodeSummary.model_validate(summary).gpu_models_by_profile == {"nvidia": ["NVIDIA-A100-SXM4-80GB"]}


@pytest.mark.parametrize(
    ("memory", "count", "expected"),
    [
        ("81920", "7", 7 * 80 * 1024**3),
        (None, "7", 0),
        ("invalid", "7", 0),
        ("0", "7", 0),
        ("-1", "7", 0),
        ("NaN", "7", 0),
        ("Infinity", "7", 0),
        ("81920", None, 0),
        ("81920", "70", 0),
    ],
)
def test_node_summary_reads_verified_device_memory_labels(memory, count, expected):
    labels = {"nvidia.com/gpu.product": "NVIDIA-A100-80GB-PCIe"}
    if memory is not None:
        labels["nvidia.com/gpu.memory"] = memory
    if count is not None:
        labels["nvidia.com/gpu.count"] = count
    node = {
        "metadata": {"name": "worker-a", "labels": labels},
        "status": {"capacity": {"nvidia.com/gpu": "7", "nvidia.com/gpu.shared": "70"}},
    }
    summary = service._node_summary(node)
    assert summary["gpuCount"] == 7
    assert summary["vramBytes"] == expected


def test_node_memory_label_discovery_uses_registered_profile_keys_and_units(monkeypatch):
    from llm_d_lens.hardware.models import HardwareProfile

    profile = HardwareProfile.from_dict(
        {
            "id": "test-gpu",
            "display_name": "Test GPU",
            "vendor": "test",
            "request_model": "extended-resource",
            "resource_prefixes": ["test.example/gpu"],
            "planning": {
                "node_memory": {
                    "per_device_label": "test.example/memory",
                    "count_label": "test.example/count",
                    "unit": "Gi",
                }
            },
        }
    )
    monkeypatch.setattr(service, "_hardware_profiles", lambda: [profile])
    monkeypatch.setattr(service, "get_profile", lambda profile_id: profile)
    summary = service._node_summary(
        {
            "metadata": {
                "name": "worker-a",
                "labels": {"test.example/memory": "24", "test.example/count": "2"},
            },
            "status": {"capacity": {"test.example/gpu": "2"}},
        }
    )
    assert summary["vramBytes"] == 48 * 1024**3


@pytest.mark.asyncio
@pytest.mark.parametrize(("source", "per_gpu_gib"), [("labels", 80), ("dra", 32), ("prometheus", 24)])
async def test_overview_prefers_prometheus_then_dra_then_node_label_vram(monkeypatch, tmp_path, source, per_gpu_gib):
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("")
    monkeypatch.setattr(registry, "kubeconfig_path", lambda _id: kubeconfig)
    node = {
        "metadata": {
            "name": "worker-a",
            "labels": {
                "nvidia.com/gpu.product": "NVIDIA-A100-80GB-PCIe",
                "nvidia.com/gpu.memory": "81920",
                "nvidia.com/gpu.count": "7",
            },
        },
        "status": {"capacity": {"nvidia.com/gpu": "7"}},
    }
    slices = [
        {
            "spec": {
                "driver": "gpu.nvidia.com",
                "nodeName": "worker-a",
                "devices": [{"name": f"gpu-{index}", "capacity": {"memory": {"value": "32Gi"}}} for index in range(7)],
            }
        }
    ]

    async def query(args, **_kwargs):
        items = [node] if args[1] == "nodes" else slices if args[1] == "resourceslices.resource.k8s.io" else []
        if source == "labels" and args[1] == "resourceslices.resource.k8s.io":
            items = []
        return CommandResult(argv=(), returncode=0, stdout=json.dumps({"items": items}), stderr="")

    metrics = (
        {
            "vramBytes": 7 * 24 * 1024**3,
            "nodeVram": {"worker-a": {"totalBytes": 7 * 24 * 1024**3}},
        }
        if source == "prometheus"
        else {}
    )
    monkeypatch.setattr(service, "run_kubectl", query)
    monkeypatch.setattr(service, "monitoring_component", AsyncMock(return_value={}))
    monkeypatch.setattr(service, "_prometheus_hardware_metrics", AsyncMock(return_value=metrics))
    monkeypatch.setattr(service, "_node_disk_usage", AsyncMock(return_value={}))
    overview = await service.build_overview(SimpleNamespace(id="cluster-a"))
    assert overview["hardware"]["gpuCount"] == 7
    assert overview["hardware"]["vramBytes"] == 7 * per_gpu_gib * 1024**3
    assert overview["nodes"][0]["vramBytes"] == 7 * per_gpu_gib * 1024**3
    assert all(gpu["vramTotalBytes"] == per_gpu_gib * 1024**3 for gpu in overview["nodes"][0]["gpus"])
    if source == "labels":
        assert overview["hardware"]["vramUsagePercent"] is None
        assert overview["nodes"][0]["vramUsagePercent"] is None
        assert all(gpu["vramUsedBytes"] is None for gpu in overview["nodes"][0]["gpus"])
        assert all(gpu["vramUsagePercent"] is None for gpu in overview["nodes"][0]["gpus"])


def test_node_summary_counts_extended_resources_per_profile_and_excludes_monitor_marker():
    node = {
        "metadata": {"name": "node-a", "labels": {}},
        "spec": {},
        "status": {
            "capacity": {"gpu.intel.com/xe": "8", "gpu.intel.com/monitoring": "1"},
            "conditions": [],
        },
    }
    summary = service._node_summary(node)
    assert summary["gpuCount"] == 8
    assert summary["gpuByProfile"] == {"intel-xpu": 8}


def test_node_summary_attributes_dra_slices_to_their_profile():
    node = {"metadata": {"name": "node-a", "labels": {}}, "spec": {}, "status": {"conditions": []}}
    summary = service._node_summary(node, {"node-a": 3}, {"node-a": {"intel-xpu": 3}})
    assert summary["gpuCount"] == 3
    assert summary["gpuByProfile"] == {"intel-xpu": 3}


def test_node_summary_does_not_double_count_nvidia_time_sliced_shared_resource():
    """Reproduced live: NVIDIA GPU time-slicing (without ``renameByDefault``)
    keeps the physical ``nvidia.com/gpu`` resource unchanged and adds a
    separate ``nvidia.com/gpu.shared`` resource exposing the same cards'
    oversubscribed replica pool (7 physical A100s, 10x time-sliced ->
    "nvidia.com/gpu.shared": "70"). A naive ``nvidia.com/gpu*`` prefix sweep
    summed both and read 77 GPUs instead of 7, breaking both the cluster
    overview and deployment capacity planning.
    """
    node = {
        "metadata": {"name": "node-a", "labels": {}},
        "spec": {},
        "status": {
            "capacity": {"nvidia.com/gpu": "7", "nvidia.com/gpu.shared": "70"},
            "conditions": [],
        },
    }
    summary = service._node_summary(node)
    assert summary["gpuCount"] == 7
    assert summary["gpuByProfile"] == {"nvidia": 7}


def test_node_summary_counts_renamed_only_nvidia_shared_resource_as_fallback():
    """``renameByDefault: true`` replaces ``nvidia.com/gpu`` entirely with the
    renamed resource; when the canonical name is absent, the broader prefix
    sweep still detects the renamed resource rather than reading zero GPUs.
    """
    node = {
        "metadata": {"name": "node-a", "labels": {}},
        "spec": {},
        "status": {"capacity": {"nvidia.com/gpu.shared": "70"}, "conditions": []},
    }
    summary = service._node_summary(node)
    assert summary["gpuCount"] == 70
    assert summary["gpuByProfile"] == {"nvidia": 70}


def test_node_summary_uses_dra_model_when_node_has_no_product_label():
    node = {"metadata": {"name": "node-a", "labels": {}}, "spec": {}, "status": {"conditions": []}}
    slices = [
        {
            "spec": {
                "nodeName": "node-a",
                "driver": "gpu.intel.com",
                "devices": [
                    {"attributes": {"model": {"string": "B60"}}},
                    {"attributes": {"model": {"string": "B60"}}},
                ],
            }
        }
    ]
    summary = service._node_summary(
        node,
        service._resource_slice_gpu_counts(slices),
        service._resource_slice_gpu_counts_by_profile(slices),
        service._resource_slice_gpu_models_by_profile(slices),
    )
    assert summary["gpuModelsByProfile"] == {"intel-xpu": ["B60"]}
    assert service._hardware_summary([summary], gpu_by_profile={"intel-xpu": 2})["accelerators"] == [
        {"id": "intel-xpu", "gpuCount": 2, "models": ["B60"]}
    ]
