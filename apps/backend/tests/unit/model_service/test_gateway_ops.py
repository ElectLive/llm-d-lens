"""Tests for model-service gateway operations (shared Gateway, model routes)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_d_lens.cluster.registry import gateway_namespace_for
from llm_d_lens.db.dao.model_service_gateway_operation import GatewayOperationDao
from llm_d_lens.db.dao.model_service_group import ModelServiceGroupDao
from llm_d_lens.db.dao.model_service_member import ModelServiceMemberDao
from llm_d_lens.model_service import gateway_ops as gateway_ops_module
from llm_d_lens.model_service.contracts import ModelServiceGroup, ModelServiceMember
from llm_d_lens.model_service.gateway_contracts import GatewayOperation
from llm_d_lens.model_service.gateway_ops import GatewayOpsService
from llm_d_lens.model_service.gateway_providers import (
    CLUSTER_LABEL,
    MANAGED_LABEL,
    MANAGED_VALUE,
    MODEL_SERVICE_LABEL,
    REWRITE_MANAGED_LABEL,
)
from llm_d_lens.model_service.targets import ExecutionMissingError


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["none", "unmanaged", "delete", "still-present", "missing"])
async def test_remove_model_route_deletes_exactly_the_models_named_resources(monkeypatch, failure):
    from llm_d_lens.model_service.gateway_ops import GatewayOpsError
    from llm_d_lens.model_service.gateway_providers import MANAGED_LABEL, MANAGED_VALUE, resource_name

    group = ModelServiceGroup(name="public", model_ref="provider", cluster_id="cluster-a")
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _: SimpleNamespace(gateway_namespace="lens-gateway"))
    ops = GatewayOpsService()
    deleted = set()
    calls = []

    async def kubectl(cluster_id, argv, **kwargs):
        assert cluster_id == group.cluster_id
        calls.append(argv)
        kind = argv[1]
        if argv[0] == "get":
            if failure == "missing" or (kind in deleted and failure != "still-present"):
                return _Result()
            return _Result(
                stdout=json.dumps(
                    {
                        "metadata": {
                            "labels": {
                                MANAGED_LABEL: "other" if failure == "unmanaged" else MANAGED_VALUE,
                            }
                        }
                    }
                )
            )
        assert argv[0] == "delete"
        name = resource_name(f"{group.name}-{group.id}") + ("-model-map" if kind == "configmap" else "")
        if failure == "unmanaged":
            assert name in argv
            assert "-l" not in argv
        else:
            assert argv[argv.index("--field-selector") + 1] == f"metadata.name={name}"
            assert argv[argv.index("-l") + 1] == f"{MANAGED_LABEL}={MANAGED_VALUE}"
        assert "--wait=true" in argv and "--timeout=60s" in argv
        deleted.add(kind)
        return _Result(
            returncode=1 if failure == "delete" else 0, stderr="deletion failed" if failure == "delete" else ""
        )

    monkeypatch.setattr(ops, "_kubectl", kubectl)
    if failure in ("none", "missing", "unmanaged"):
        await ops.remove_model_route(group)
        assert deleted == ({"httproute", "configmap"} if failure in ("none", "unmanaged") else set())
    else:
        with pytest.raises(GatewayOpsError):
            await ops.remove_model_route(group)
        assert not any(call[1] == "configmap" for call in calls)


@pytest.mark.asyncio
async def test_remove_model_route_treats_sdk_empty_list_as_absent(monkeypatch):
    group = ModelServiceGroup(name="public", model_ref="provider", cluster_id="cluster-a")
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _: SimpleNamespace(gateway_namespace="lens-gateway"))
    ops = GatewayOpsService()
    calls = []

    async def kubectl(cluster_id, argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "get"
        return _Result(stdout='{"items": []}')

    monkeypatch.setattr(ops, "_kubectl", kubectl)
    await ops.remove_model_route(group)

    assert len(calls) == 3
    assert all(call[0] == "get" for call in calls)


class _FakeRunner:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if any("infrastructure.parametersRef.name" in str(arg) for arg in argv):
            return _Result(stdout="")
        if list(argv[:3]) == ["kubectl", "get", "envoyproxy"]:
            return _Result(stdout="")
        if list(argv[:3]) == ["kubectl", "get", "httproute"]:
            return _Result(stdout="")
        if list(argv[:4]) == ["kubectl", "get", "gateway", "lens-inference-gateway"]:
            return _Result(stdout="True")
        if list(argv[:3]) == ["kubectl", "get", "gateway"]:
            return _Result(stdout="True")
        return _Result(stdout="applied")


class _ExistingRouteRunner(_FakeRunner):
    """Replays one Lens-managed HTTPRoute already on the cluster."""

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "httproute"]:
            return _Result(stdout="qwen-empty")
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


class _ExistingServiceTypeRunner(_FakeRunner):
    """Replays a Gateway already bound to a NodePort EnvoyProxy."""

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if any("infrastructure.parametersRef.name" in str(arg) for arg in argv):
            return _Result(stdout="lens-inference-gateway-proxy")
        if any("envoyService.type" in str(arg) for arg in argv):
            return _Result(stdout="NodePort")
        return _Result(stdout="True")


class _BusyPortRunner(_FakeRunner):
    """Gateway service plus another Service already using a nodePort."""

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        if list(argv[:3]) == ["kubectl", "get", "svc"] and "-l" in argv:
            self.calls.append((argv, input))
            return _Result(stdout="gw-svc")
        if list(argv[:3]) == ["kubectl", "get", "svc"] and "-A" in argv:
            self.calls.append((argv, input))
            return _Result(
                stdout='{"items":[{"metadata":{"namespace":"other","name":"other-svc"},'
                '"spec":{"ports":[{"nodePort":30999}]}}]}'
            )
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


def _seed_member(cluster_id: str = "cluster-a") -> ModelServiceMember:
    group = ModelServiceGroupDao().create(
        ModelServiceGroup(
            name=f"qwen3-8b-{cluster_id}",
            model_ref="Qwen/Qwen3-8B",
            cluster_id=cluster_id,
            base_model="Qwen/Qwen3-8B",
        )
    )
    return ModelServiceMemberDao().create(
        ModelServiceMember(
            group_id=group.id,
            execution_id=f"exec-{cluster_id}",
            cluster_id=cluster_id,
            target_namespace="ns",
            target_service="qwen-epp",
            target_port=8000,
            pool_name="qwen",
            endpoint_kind="llm-d-epp",
        )
    )


def _service(*, calls: list, runner_cls=_FakeRunner) -> GatewayOpsService:
    return GatewayOpsService(
        operations=GatewayOperationDao(),
        members=ModelServiceMemberDao(),
        groups=ModelServiceGroupDao(),
        runner_factory=lambda _cluster_id: runner_cls(calls),
    )


@pytest.mark.asyncio
async def test_preflight_missing_cluster(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: None)
    operation = await service.preflight("cluster-a")
    assert operation.status == "failed"
    assert "not found" in operation.message


@pytest.mark.asyncio
async def test_install_inference_gateway_applies_shared_gateway(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: object())

    operation = await service.install_inference_gateway("cluster-a", "istio", install_prerequisites=False)
    assert operation.status == "succeeded"
    assert operation.detail_json["provider"] == "istio"
    applies = [c for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]]
    assert applies
    manifest = applies[-1][1]
    assert "kind: Gateway" in manifest
    assert "controllerName: istio.io/gateway-controller" in manifest
    assert "kind: HTTPRoute" not in manifest
    assert f"namespace: {gateway_namespace_for('cluster-a')}" in manifest


@pytest.mark.asyncio
async def test_uninstall_only_removes_this_registration_gateway_resources(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_namespace="lens-gateway",
            gateway_provider="istio",
            gateway_name="lens-inference-gateway",
            gateway_port=None,
        ),
    )
    operation = await _service(calls=calls).uninstall_inference_gateway("cluster-a")
    assert operation.status == "succeeded"
    deletions = [argv for argv, _ in calls if list(argv[:2]) == ["kubectl", "delete"]]
    assert len(deletions) == 1
    argv = deletions[0]
    assert argv[argv.index("-n") + 1] == gateway_namespace_for("cluster-a")
    assert "-A" not in argv
    assert not {"gatewayclass", "inferencepool", "namespace"} & set(argv[2].split(","))


@pytest.mark.asyncio
async def test_gateway_install_does_not_register_a_global_istio_auth_provider(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(gateway_authz_host="10.0.0.9", gateway_namespace="lens-gateway"),
    )
    monkeypatch.setattr(gateway_ops_module.hostinfo, "current_serving_port", lambda: 8084)
    monkeypatch.setattr(gateway_ops_module, "apply_gateway_crds", AsyncMock(return_value={}))
    monkeypatch.setattr(gateway_ops_module, "ensure_istioctl", lambda _: "istioctl")
    operation = await _service(calls=calls).install_inference_gateway("cluster-a", "istio")
    assert operation.status == "succeeded"
    installs = [argv for argv, _ in calls if argv[0] == "istioctl"]
    assert installs
    assert all("-f" not in argv for argv in installs)
    manifest = [body for argv, body in calls if list(argv[:3]) == ["kubectl", "apply", "-f"]][-1]
    assert "kind: EnvoyFilter" in manifest
    assert "meshConfig" not in manifest


@pytest.mark.asyncio
async def test_install_preserves_existing_envoy_service_type(monkeypatch):
    calls: list = []
    service = _service(calls=calls, runner_cls=_ExistingServiceTypeRunner)
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: object())

    operation = await service.install_inference_gateway("cluster-a", "envoy-ai-gateway", install_prerequisites=False)
    assert operation.status == "succeeded"
    manifest = [c for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]][-1][1]
    assert "kind: EnvoyProxy" in manifest
    assert "type: NodePort" in manifest


@pytest.mark.asyncio
async def test_reconcile_routes_weighted_pools_per_service(monkeypatch):
    import yaml

    group = ModelServiceGroupDao().create(
        ModelServiceGroup(
            name="qwen3-8b", model_ref="Qwen/Qwen3-8B", cluster_id="cluster-a", base_model="Qwen/Qwen3-8B"
        )
    )
    for index, (service_name, namespace) in enumerate(
        [("optimized-baseline-epp", "ns-a"), ("optimized-baseline-b-epp", "ns-b")]
    ):
        ModelServiceMemberDao().create(
            ModelServiceMember(
                group_id=group.id,
                execution_id=f"exec-{index}",
                cluster_id="cluster-a",
                target_namespace=namespace,
                target_service=service_name,
                target_port=80,
                epp_ref=service_name,
                endpoint_kind="llm-d-epp",
                weight=90 if index == 0 else 10,
            )
        )
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.reconcile_cluster_gateway("cluster-a")
    assert operation.status == "succeeded"
    assert operation.detail_json["routes"] == 1
    manifest = [c for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]][-1][1]
    route = next(doc for doc in yaml.safe_load_all(manifest) if doc and doc["kind"] == "HTTPRoute")
    refs = route["spec"]["rules"][0]["backendRefs"]
    assert len(refs) == 2
    assert {ref["namespace"]: ref["weight"] for ref in refs} == {"ns-a": 90, "ns-b": 10}
    assert all("port" not in ref for ref in refs)
    grants = [doc for doc in yaml.safe_load_all(manifest) if doc and doc["kind"] == "ReferenceGrant"]
    assert {grant["metadata"]["namespace"] for grant in grants} == {"ns-a", "ns-b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("excluded", ["zero", "disabled"])
async def test_reconcile_omits_non_routable_providers(monkeypatch, excluded):
    import yaml

    member = _seed_member()
    ModelServiceMemberDao().create(
        member.model_copy(
            update={
                "id": "other-member",
                "execution_id": "other-exec",
                "pool_name": "other",
                "weight": 0 if excluded == "zero" else 100,
                "status": "active" if excluded == "zero" else "disabled",
            }
        )
    )
    calls = []
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )
    operation = await _service(calls=calls).reconcile_cluster_gateway("cluster-a")
    assert operation.status == "succeeded"
    manifest = [call for call in calls if list(call[0][:3]) == ["kubectl", "apply", "-f"]][-1][1]
    route = next(doc for doc in yaml.safe_load_all(manifest) if doc and doc["kind"] == "HTTPRoute")
    assert [ref["name"] for ref in route["spec"]["rules"][0]["backendRefs"]] == ["qwen"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_model", [True, False])
async def test_stop_and_resume_remove_and_restore_route_without_scaling(monkeypatch, stop_model):
    member = _seed_member()
    service = _service(calls=[], runner_cls=_ExistingRouteRunner)
    group = service.groups.get(member.group_id)
    if stop_model:
        group.status = "disabled"
        service.groups.save(group)
    else:
        member.status = "disabled"
        service.members.save(member)
    calls = []
    service = _service(calls=calls, runner_cls=_ExistingRouteRunner)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )
    stopped = await service.reconcile_cluster_gateway(member.cluster_id)
    assert stopped.status == "succeeded"
    assert stopped.detail_json["routes"] == 0
    assert any(argv[:4] == ["kubectl", "delete", "httproute", "qwen-empty"] for argv, _ in calls)
    assert not any("kind: HTTPRoute" in (manifest or "") for _, manifest in calls)
    assert not any("scale" in argv for argv, _ in calls)
    assert service.members.get(member.id).weight == 1

    if stop_model:
        group.status = "active"
        service.groups.save(group)
    else:
        member.status = "active"
        service.members.save(member)
    calls.clear()
    resumed = await service.reconcile_cluster_gateway(member.cluster_id)
    assert resumed.status == "succeeded"
    assert resumed.detail_json["routes"] == 1
    assert any("kind: HTTPRoute" in (manifest or "") for _, manifest in calls)
    assert not any("scale" in argv for argv, _ in calls)


@pytest.mark.asyncio
async def test_route_discovery_failure_does_not_report_stop_applied(monkeypatch):
    member = _seed_member()
    service = _service(calls=[])
    group = service.groups.get(member.group_id)
    group.status = "disabled"
    service.groups.save(group)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )
    original = service._kubectl

    async def fail_route_list(cluster_id, argv, **kwargs):
        if argv[:2] == ["get", "httproute"]:
            return _Result(returncode=1, stderr="route listing forbidden")
        return await original(cluster_id, argv, **kwargs)

    monkeypatch.setattr(service, "_kubectl", fail_route_list)
    operation = await service.reconcile_cluster_gateway(member.cluster_id)
    assert operation.status == "failed"
    assert operation.message == "route listing forbidden"


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_model", [True, False])
async def test_intentionally_stopped_publication_is_not_probed_or_healed(monkeypatch, stop_model):
    member = _seed_member()
    service = _service(calls=[])
    if stop_model:
        group = service.groups.get(member.group_id)
        group.status = "disabled"
        service.groups.save(group)
    else:
        member.status = "disabled"
        service.members.save(member)
    health, install = AsyncMock(), AsyncMock()
    monkeypatch.setattr(service, "_member_health", health)
    monkeypatch.setattr(service, "_heal_data_plane_locked", install)
    assert await service.probe_members() == 0
    await service._heal_data_plane(member.cluster_id)
    health.assert_not_awaited()
    install.assert_not_awaited()


@pytest.mark.asyncio
async def test_offline_during_probe_is_preserved_and_does_not_trigger_healing(monkeypatch):
    member = _seed_member()
    service = _service(calls=[])
    service._target_resolver = lambda _: SimpleNamespace(status="ready")

    async def health(_):
        offline = service.members.get(member.id)
        offline.status = "disabled"
        service.members.save(offline)
        return False, {"components": {"httpRoute": "missing"}}

    monkeypatch.setattr(service, "_member_health", health)
    heal = AsyncMock()
    monkeypatch.setattr(service, "_heal_data_plane", heal)
    await service.probe_members()
    assert service.members.get(member.id).status == "disabled"
    heal.assert_not_awaited()


@pytest.mark.asyncio
async def test_probe_prunes_member_whose_execution_is_gone():
    group = ModelServiceGroupDao().create(
        ModelServiceGroup(
            name="orphan-model", model_ref="Qwen/Qwen3-8B", cluster_id="cluster-a", base_model="Qwen/Qwen3-8B"
        )
    )
    member = ModelServiceMemberDao().create(
        ModelServiceMember(
            group_id=group.id,
            execution_id="exec-gone",
            cluster_id="cluster-a",
            target_namespace="ns",
            target_service="qwen-epp",
            target_port=80,
            epp_ref="qwen-epp",
            endpoint_kind="llm-d-epp",
        )
    )

    def resolver(_execution_id: str):
        raise ExecutionMissingError("execution not found")

    calls: list = []
    service = GatewayOpsService(
        operations=GatewayOperationDao(),
        members=ModelServiceMemberDao(),
        groups=ModelServiceGroupDao(),
        runner_factory=lambda _cluster_id: _FakeRunner(calls),
        target_resolver=resolver,
    )
    await service.probe_members()

    assert ModelServiceMemberDao().get(member.id) is None


@pytest.mark.asyncio
async def test_reconcile_removes_route_for_memberless_group(monkeypatch):
    calls: list = []
    service = _service(calls=calls, runner_cls=_ExistingRouteRunner)
    cluster = SimpleNamespace(
        gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.reconcile_cluster_gateway("cluster-a")
    assert operation.status == "succeeded"
    assert operation.detail_json["removed"] == 1
    deletes = [c for c in calls if list(c[0][:3]) == ["kubectl", "delete", "httproute"]]
    assert deletes and deletes[0][0][3] == "qwen-empty"


@pytest.mark.asyncio
async def test_install_rejects_exposed_port_out_of_range(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(
            gateway_provider="envoy-ai-gateway",
            gateway_namespace="lens-gateway",
            gateway_name="lens-inference-gateway",
            gateway_port=80,
        ),
    )
    operation = await service.install_inference_gateway("cluster-a", "envoy-ai-gateway", install_prerequisites=False)
    assert operation.status == "failed"
    assert "30000" in operation.message


@pytest.mark.asyncio
async def test_install_rejects_busy_exposed_port(monkeypatch):
    calls: list = []
    service = _service(calls=calls, runner_cls=_BusyPortRunner)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(
            gateway_provider="envoy-ai-gateway",
            gateway_namespace="lens-gateway",
            gateway_name="lens-inference-gateway",
            gateway_port=30999,
        ),
    )
    operation = await service.install_inference_gateway("cluster-a", "envoy-ai-gateway", install_prerequisites=False)
    assert operation.status == "failed"
    assert "already in use" in operation.message


@pytest.mark.asyncio
async def test_install_falls_back_to_nodeport_without_loadbalancer(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: object())
    waits = [False, True]

    async def fake_wait(*_args, **_kwargs):
        return waits.pop(0)

    async def fake_unassigned(*_args, **_kwargs):
        return True

    async def fake_existing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(service, "_wait_gateway_programmed", fake_wait)
    monkeypatch.setattr(service, "_gateway_address_unassigned", fake_unassigned)
    monkeypatch.setattr(service, "_existing_gateway_service_type", fake_existing)

    operation = await service.install_inference_gateway("cluster-a", "envoy-ai-gateway", install_prerequisites=False)
    assert operation.status == "succeeded"
    assert operation.detail_json["programmed"] is True
    assert operation.detail_json["serviceType"] == "NodePort"
    manifests = [c[1] for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]]
    assert "kind: EnvoyProxy" in manifests[-1]


@pytest.mark.asyncio
async def test_scale_gateway_scales_data_plane_deployment(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(
            gateway_provider="istio",
            gateway_namespace="lens-gateway",
            gateway_name="lens-inference-gateway",
            gateway_port=None,
        ),
    )
    operation = await service.scale_gateway("cluster-a", 0)
    assert operation.status == "succeeded"
    scales = [c for c in calls if list(c[0][:2]) == ["kubectl", "scale"]]
    assert scales and scales[0][0][2] == "deployment/lens-inference-gateway-istio"


@pytest.mark.asyncio
async def test_resolve_log_target_for_ipp(monkeypatch):
    service = _service(calls=[])
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(
            gateway_provider="istio",
            gateway_namespace="lens-gateway",
            gateway_name="lens-inference-gateway",
            gateway_port=None,
        ),
    )
    namespace, name = await service.resolve_log_target("cluster-a", component="ipp", namespace=None, name=None)
    assert (namespace, name) == (gateway_namespace_for("cluster-a"), "lens-ipp")


@pytest.mark.asyncio
async def test_install_inference_gateway_rejects_unknown_provider(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: object())

    operation = await service.install_inference_gateway("cluster-a", "traefik")
    assert operation.status == "failed"
    assert "unsupported gateway provider" in operation.message


@pytest.mark.asyncio
async def test_reconcile_cluster_gateway_applies_model_routes(monkeypatch):
    _seed_member(cluster_id="cluster-a")
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.reconcile_cluster_gateway("cluster-a")
    assert operation.status == "succeeded"
    assert operation.detail_json["routes"] == 1
    applies = [c for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]]
    route_manifest = applies[-1][1]
    assert "kind: HTTPRoute" in route_manifest
    assert "kind: ConfigMap" in route_manifest
    assert "name: qwen" in route_manifest
    assert "X-Gateway-Base-Model-Name" in route_manifest


@pytest.mark.asyncio
async def test_reconcile_derives_pool_name_for_legacy_member(monkeypatch):
    group = ModelServiceGroupDao().create(
        ModelServiceGroup(
            name="legacy-model",
            model_ref="Qwen/Qwen3-0.6B",
            cluster_id="cluster-a",
            base_model="Qwen/Qwen3-0.6B",
        )
    )
    ModelServiceMemberDao().create(
        ModelServiceMember(
            group_id=group.id,
            execution_id="exec-legacy",
            cluster_id="cluster-a",
            target_namespace="ns",
            target_service="optimized-baseline-epp",
            target_port=8000,
            epp_ref="optimized-baseline-epp",
            endpoint_kind="llm-d-epp",
        )
    )
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.reconcile_cluster_gateway("cluster-a")
    assert operation.status == "succeeded"
    assert operation.detail_json["routes"] == 1
    route_manifest = [c for c in calls if list(c[0][:3]) == ["kubectl", "apply", "-f"]][-1][1]
    assert "name: optimized-baseline" in route_manifest


@pytest.mark.asyncio
async def test_status_reports_gateway_state(monkeypatch):
    _seed_member(cluster_id="cluster-a")
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        name="Cluster A",
        gateway_provider="istio",
        gateway_namespace="lens-gateway",
        gateway_name="lens-inference-gateway",
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)
    monkeypatch.setattr(
        "llm_d_lens.cluster.registry.list_clusters", lambda: [SimpleNamespace(id="cluster-a", name="Cluster A")]
    )

    status = await service.status()
    assert len(status.clusters) == 1
    cluster_status = status.clusters[0]
    assert cluster_status.gateway_provider == "istio"
    assert cluster_status.gateway_state == "installed"
    assert cluster_status.gateway_ready is True
    assert cluster_status.members[0].pool_name == "qwen"
    # A gateway operation is not created merely by reading status.
    assert cluster_status.last_operation is None


@pytest.mark.asyncio
@pytest.mark.parametrize("disappeared", [False, True])
async def test_status_keeps_registered_empty_models_but_never_probes_orphan_clusters(monkeypatch, disappeared):
    ModelServiceGroupDao().create(ModelServiceGroup(name="empty-model", model_ref="provider", cluster_id="registered"))
    ModelServiceGroupDao().create(ModelServiceGroup(name="orphan-empty", model_ref="provider", cluster_id="deleted"))
    _seed_member(cluster_id="deleted-with-members")
    cluster = SimpleNamespace(
        id="registered",
        name="Registered cluster",
        gateway_provider=None,
        gateway_namespace="lens-gateway",
        gateway_name="lens-inference-gateway",
    )
    monkeypatch.setattr("llm_d_lens.cluster.registry.list_clusters", lambda: [cluster])
    looked_up = []

    def get_cluster(cluster_id):
        looked_up.append(cluster_id)
        assert cluster_id == "registered"
        return None if disappeared else cluster

    monkeypatch.setattr(gateway_ops_module, "get_cluster", get_cluster)
    calls = []
    service = _service(calls=calls)
    status = await service.status()
    assert [item.cluster_id for item in status.clusters] == ([] if disappeared else ["registered"])
    assert looked_up == ["registered"]
    if disappeared:
        assert calls == []
    else:
        assert status.clusters[0].members == []


class _DriftedProviderRunner(_FakeRunner):
    """Gateway is istio-classed, but no provider-specific Service lookup finds it
    (provider mislabeled as envoy-ai-gateway, and the `-istio` Service is absent)."""

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "svc"] and "-l" in argv:
            return _Result(stdout="")  # envoy labelled lookup: not found
        if list(argv[:3]) == ["kubectl", "get", "svc"] and argv[3].endswith("-istio"):
            return _Result(returncode=1, stderr="not found")
        if any("status.addresses[0].value" in str(arg) for arg in argv):
            return _Result(stdout="lens-inference-gateway-istio.lens-gateway.svc.cluster.local")
        if any("spec.gatewayClassName" in str(arg) for arg in argv):
            return _Result(stdout="istio")
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


@pytest.mark.asyncio
async def test_gateway_service_falls_back_to_discovery_on_provider_drift():
    """Regression: a wrong stored provider must not skip the public exposure."""
    calls: list = []
    service = _service(calls=calls, runner_cls=_DriftedProviderRunner)
    target = await service._gateway_service("cluster-a", "lens-gateway", "lens-inference-gateway", "envoy-ai-gateway")
    assert target == ("lens-gateway", "lens-inference-gateway-istio")


@pytest.mark.asyncio
async def test_actual_gateway_provider_maps_gateway_class():
    calls: list = []
    service = _service(calls=calls, runner_cls=_DriftedProviderRunner)
    assert await service._actual_gateway_provider("cluster-a", "lens-gateway", "lens-inference-gateway") == "istio"


@pytest.mark.asyncio
async def test_gateway_service_falls_back_when_istio_service_is_absent():
    """Regression: provider says istio but no such Service -> discover the real one."""
    calls: list = []
    service = _service(calls=calls, runner_cls=_DriftedProviderRunner)
    target = await service._gateway_service("cluster-a", "lens-gateway", "lens-inference-gateway", "istio")
    assert target == ("lens-gateway", "lens-inference-gateway-istio")


class _IppConfigRunner(_FakeRunner):
    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "configmap"]:
            return _Result(
                stdout=(
                    '{"custom-ipp-config.yaml":"plugins: []\\n",'
                    '"default-ipp-config.yaml":"plugins:\\n- type: body-field-to-header\\n"}'
                )
            )
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


@pytest.mark.asyncio
async def test_get_ipp_config_reads_live_configmap(monkeypatch):
    """IPP config is read live from the cluster ConfigMap (not Lens storage)."""
    calls: list = []
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(gateway_namespace="lens-gateway"),
    )
    service = _service(calls=calls, runner_cls=_IppConfigRunner)
    cfg = await service.get_ipp_config("cluster-a")
    assert cfg["custom"] is True
    assert cfg["config"] == "plugins: []\n"
    assert "body-field-to-header" in cfg["default"]


@pytest.mark.asyncio
async def test_get_ipp_config_falls_back_to_default_when_no_custom(monkeypatch):
    calls: list = []

    class R(_IppConfigRunner):
        async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
            if list(argv[:3]) == ["kubectl", "get", "configmap"]:
                return _Result(stdout='{"default-ipp-config.yaml":"plugins:\\n- type: x\\n"}')
            return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)

    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(gateway_namespace="lens-gateway"),
    )
    service = _service(calls=calls, runner_cls=R)
    cfg = await service.get_ipp_config("cluster-a")
    assert cfg["custom"] is False
    assert cfg["config"] == cfg["default"]


class _TlsEppRunner(_FakeRunner):
    """An EPP Deployment left on the default secure-serving=true (TLS)."""

    def __init__(self, calls):
        super().__init__(calls)
        self.patched = None

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "deploy"] and "-o" in argv and "json" in argv:
            return _Result(
                stdout=(
                    '{"spec":{"template":{"spec":{"containers":['
                    '{"name":"envoy-proxy","args":["--log-level","warn"]},'
                    '{"name":"epp","args":["--pool-name","qwen","--secure-serving=true"]}'
                    "]}}}}"
                )
            )
        if list(argv[:3]) == ["kubectl", "patch", "deploy"]:
            self.patched = argv
            return _Result(stdout="patched")
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


@pytest.mark.asyncio
async def test_ensure_epp_plaintext_rewrites_secure_serving():
    """Regression: an EPP on TLS (secure-serving=true) must be switched to plaintext."""
    calls: list = []
    runner = _TlsEppRunner(calls)
    service = GatewayOpsService(runner_factory=lambda _cid: runner)
    changed = await service._ensure_epp_plaintext("cluster-a", "ns", "qwen-epp")
    assert changed is True
    assert runner.patched is not None
    assert "--secure-serving=false" in runner.patched[-1]
    assert "/spec/template/spec/containers/1/args/2" in runner.patched[-1]


@pytest.mark.asyncio
async def test_ensure_epp_plaintext_is_noop_when_already_plaintext():
    calls: list = []

    class R(_FakeRunner):
        async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
            if list(argv[:3]) == ["kubectl", "get", "deploy"]:
                return _Result(
                    stdout=(
                        '{"spec":{"template":{"spec":{"containers":['
                        '{"name":"epp","args":["--pool-name","qwen","--secure-serving=false"]}'
                        "]}}}}"
                    )
                )
            if list(argv[:3]) == ["kubectl", "patch", "deploy"]:
                raise AssertionError("must not patch an already-plaintext EPP")
            return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)

    service = GatewayOpsService(runner_factory=lambda _cid: R(calls))
    assert await service._ensure_epp_plaintext("cluster-a", "ns", "qwen-epp") is False


@pytest.mark.asyncio
async def test_install_ipp_restarts_rollout_to_apply_config(monkeypatch):
    """A config change forces a restart so it is loaded even if inotify fails."""
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio",
        gateway_namespace="lens-gateway",
        gateway_name="lens-inference-gateway",
        ipp_version=None,
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.install_ipp("cluster-a", config="plugins: []")

    assert operation.status == "succeeded"
    assert any(list(call[0][:3]) == ["kubectl", "rollout", "restart"] for call in calls)


@pytest.mark.asyncio
async def test_apply_node_inotify_limit_applies_daemonset(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio",
        gateway_namespace="lens-gateway",
        gateway_name="lens-inference-gateway",
        ipp_version=None,
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.apply_node_inotify_limit("cluster-a", 8192)

    assert operation.status == "succeeded"
    apply_calls = [call for call in calls if list(call[0][:3]) == ["kubectl", "apply", "-f"]]
    assert apply_calls
    manifest = apply_calls[-1][1]
    assert "kind: DaemonSet" in manifest
    assert "fs.inotify.max_user_instances=8192" in manifest


@pytest.mark.asyncio
async def test_apply_node_inotify_limit_rejects_zero(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    cluster = SimpleNamespace(
        gateway_provider="istio",
        gateway_namespace="lens-gateway",
        gateway_name="lens-inference-gateway",
        ipp_version=None,
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _cid: cluster)

    operation = await service.apply_node_inotify_limit("cluster-a", 0)
    assert operation.status == "failed"


class _ComponentRunner(_FakeRunner):
    """Canned HTTPRoute/InferencePool/Service/EndpointSlice reads."""

    def __init__(
        self, calls: list, *, model_server_ready: bool = True, pool_statuses: dict[str, dict] | None = None
    ) -> None:
        super().__init__(calls)
        self._model_server_ready = model_server_ready
        self._pool_statuses = pool_statuses or {}

    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "endpoints"]:
            return _Result(stdout='{"subsets":[{"addresses":[{"ip":"1.2.3.4"}]}]}')
        if list(argv[:3]) == ["kubectl", "get", "httproute"]:
            return _Result(stdout='{"status":{"parents":[{"conditions":[{"type":"Accepted","status":"True"}]}]}}')
        if list(argv[:3]) == ["kubectl", "get", "inferencepool"]:
            namespace = argv[argv.index("-n") + 1]
            status = self._pool_statuses.get(
                namespace, {"parents": [{"conditions": [{"type": "Accepted", "status": "True"}]}]}
            )
            return _Result(
                stdout=json.dumps({"spec": {"selector": {"matchLabels": {"llm-d.ai/guide": "pool"}}}, "status": status})
            )
        if list(argv[:3]) == ["kubectl", "get", "pods"]:
            ready = "True" if self._model_server_ready else "False"
            return _Result(
                stdout=json.dumps(
                    {
                        "items": [
                            {
                                "status": {
                                    "phase": "Running",
                                    "conditions": [{"type": "Ready", "status": ready}],
                                }
                            }
                        ]
                    }
                )
            )
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


def _seed_component_member(name: str, *, weight: int = 1) -> ModelServiceMember:
    group = ModelServiceGroupDao().create(
        ModelServiceGroup(name=name, model_ref=name, cluster_id="cluster-a", base_model=name)
    )
    return ModelServiceMemberDao().create(
        ModelServiceMember(
            group_id=group.id,
            execution_id=f"exec-{name}",
            cluster_id="cluster-a",
            target_namespace="ns",
            target_service="pool-epp",
            target_port=80,
            pool_name="ns/pool",
            epp_ref="pool-epp",
            endpoint_kind="llm-d-epp",
            weight=weight,
        )
    )


@pytest.mark.asyncio
async def test_probe_members_reports_per_component_health(monkeypatch):
    member = _seed_component_member("percomp-ready")
    calls: list = []
    service = GatewayOpsService(
        operations=GatewayOperationDao(),
        members=ModelServiceMemberDao(),
        groups=ModelServiceGroupDao(),
        runner_factory=lambda _cid: _ComponentRunner(calls),
        target_resolver=lambda _eid: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"),
    )
    await service.probe_members()

    saved = ModelServiceMemberDao().get(member.id)
    assert saved.status == "active"
    assert saved.health_json["components"] == {
        "epp": "ready",
        "httpRoute": "ready",
        "inferencePool": "ready",
        "modelServer": "ready",
        "ipp": "ready",
        "serving": "n/a",
    }


@pytest.mark.asyncio
async def test_serving_probe_uses_public_service_name_not_model_ref(monkeypatch):
    payloads: list[dict] = []

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, json):  # noqa: A002
            payloads.append(json)
            return SimpleNamespace(status_code=401)

    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"),
    )
    monkeypatch.setattr(gateway_ops_module, "gateway_base_url", lambda *_args: "http://gateway")
    monkeypatch.setattr(gateway_ops_module, "local_public_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(gateway_ops_module.httpx, "AsyncClient", _Client)
    group = SimpleNamespace(name="qwen3-0.6b", model_ref="Qwen/Qwen3-0.6B")

    state = await GatewayOpsService()._gateway_serving_state(SimpleNamespace(cluster_id="cluster-a"), group)

    assert state == "ready"
    assert payloads[0]["model"] == "qwen3-0.6b"


@pytest.mark.asyncio
async def test_standby_provider_does_not_require_route_when_last_positive_provider_is_offline(monkeypatch):
    standby = _seed_component_member("no-routed-provider", weight=0)
    members = ModelServiceMemberDao()
    members.create(
        standby.model_copy(
            update={
                "id": "offline-provider",
                "execution_id": "offline-execution",
                "status": "disabled",
                "weight": 90,
            }
        )
    )
    calls = []
    service = GatewayOpsService(
        runner_factory=lambda _: _ComponentRunner(calls, pool_statuses={"ns": {}}),
        target_resolver=lambda _: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(gateway_ops_module, "get_cluster", lambda _: SimpleNamespace(gateway_namespace="lens-gateway"))
    heal = AsyncMock()
    monkeypatch.setattr(service, "_heal_data_plane", heal)
    await service.probe_members()
    saved = members.get(standby.id)
    assert saved.status == "active"
    assert saved.health_json["components"]["httpRoute"] == "standby"
    assert saved.health_json["components"]["inferencePool"] == "standby"
    assert saved.health_json["components"]["serving"] == "n/a"
    assert members.get("offline-provider").status == "disabled"
    assert not any(argv[:3] == ["kubectl", "get", "httproute"] for argv, _ in calls)
    heal.assert_not_awaited()


def test_member_is_routable_excludes_only_fully_unavailable_providers():
    from llm_d_lens.model_service.gateway_ops import member_is_routable

    base = SimpleNamespace(
        status="active",
        weight=1,
        endpoint_kind="llm-d-epp",
        pool_name="ns/pool",
        epp_ref="pool-epp",
        target_namespace="ns",
        target_service="pool-epp",
        health_json={
            "components": {"epp": "ready", "inferencePool": "ready", "modelServer": "ready", "serving": "ready"}
        },
    )
    assert member_is_routable(base)
    for component in ("epp", "inferencePool", "modelServer"):
        assert not member_is_routable(
            SimpleNamespace(
                **{
                    **vars(base),
                    "health_json": {
                        **base.health_json,
                        "components": {**base.health_json["components"], component: "missing"},
                    },
                }
            )
        )
    # A missing route is not a reason to drop the provider: the next reconcile
    # must be able to recreate that route and heal the provider.
    assert member_is_routable(
        SimpleNamespace(
            **{
                **vars(base),
                "health_json": {
                    **base.health_json,
                    "components": {**base.health_json["components"], "httpRoute": "missing"},
                },
            }
        )
    )
    assert not member_is_routable(SimpleNamespace(**{**vars(base), "weight": 0}))
    assert not member_is_routable(SimpleNamespace(**{**vars(base), "status": "disabled"}))


def test_data_plane_degraded_detects_broken_paths():
    assert gateway_ops_module._data_plane_degraded({"ipp": "missing"})
    assert gateway_ops_module._data_plane_degraded({"serving": "degraded"})
    assert gateway_ops_module._data_plane_degraded({"serving": "unreachable"})
    assert gateway_ops_module._data_plane_degraded({"httpRoute": "missing"})
    assert gateway_ops_module._data_plane_degraded({"inferencePool": "degraded"})
    assert not gateway_ops_module._data_plane_degraded({"epp": "ready", "ipp": "ready", "serving": "ready"})
    # A model server still starting is not a data-plane failure.
    assert not gateway_ops_module._data_plane_degraded({"modelServer": "missing"})
    assert not gateway_ops_module._data_plane_degraded({"inferencePool": "standby"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("weight", "pool_status", "model_ready", "expected_pool", "healthy", "heals"),
    [
        (0, {}, True, "standby", True, False),
        (0, {"parents": []}, False, "standby", False, False),
        (1, {}, True, "degraded", False, True),
        (
            0,
            {"parents": [{"conditions": [{"type": "Accepted", "status": "False"}]}]},
            True,
            "degraded",
            False,
            True,
        ),
        (
            0,
            {"parents": [{"conditions": [{"type": "Accepted", "status": "True"}]}]},
            True,
            "ready",
            True,
            False,
        ),
    ],
)
async def test_probe_distinguishes_standby_pools_from_failures(
    monkeypatch, weight, pool_status, model_ready, expected_pool, healthy, heals
):
    member = _seed_component_member("standby", weight=weight)
    calls: list = []
    service = GatewayOpsService(
        runner_factory=lambda _: _ComponentRunner(
            calls, model_server_ready=model_ready, pool_statuses={"ns": pool_status}
        ),
        target_resolver=lambda _: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"),
    )
    heal = AsyncMock()
    monkeypatch.setattr(service, "_heal_data_plane", heal)

    await service.probe_members()

    saved = ModelServiceMemberDao().get(member.id)
    assert saved.weight == weight
    assert saved.status == ("active" if healthy else "unhealthy")
    assert saved.health_json["ready"] == healthy
    assert saved.health_json["components"]["inferencePool"] == expected_pool
    assert saved.health_json["components"]["modelServer"] == ("ready" if model_ready else "missing")
    if heals:
        heal.assert_awaited_once_with("cluster-a")
    else:
        heal.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "state"),
    [(_Result(returncode=1), "missing"), (_Result(stdout="not JSON"), "degraded")],
)
async def test_zero_weight_does_not_mask_missing_or_invalid_pool(monkeypatch, result, state):
    service = _service(calls=[])
    monkeypatch.setattr(service, "_kubectl", AsyncMock(return_value=result))
    assert await service._pool_state("cluster-a", "ns", "pool", routed=False) == (state, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", [(1, 0), (0, 1), (90, 10)])
async def test_repeated_probe_and_reconcile_preserve_namespaced_pool_weights(monkeypatch, weights):
    import yaml

    first = _seed_component_member("shared-pool-name", weight=weights[0])
    second = ModelServiceMemberDao().create(
        first.model_copy(
            update={
                "id": "second-member",
                "execution_id": "second-execution",
                "target_namespace": "ns-b",
                "pool_name": "ns-b/pool",
                "weight": weights[1],
            }
        )
    )
    calls: list = []
    pool_statuses = {member.target_namespace: {} for member in (first, second) if member.weight == 0}
    service = GatewayOpsService(
        runner_factory=lambda _: _ComponentRunner(calls, pool_statuses=pool_statuses),
        target_resolver=lambda _: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )
    heal = AsyncMock()
    monkeypatch.setattr(service, "_heal_data_plane", heal)
    expected = {(member.target_namespace, "pool"): member.weight for member in (first, second) if member.weight > 0}
    for _ in range(2):
        await service.probe_members()
        operation = await service.reconcile_cluster_gateway("cluster-a")
        assert operation.status == "succeeded"
        manifest = [call[1] for call in calls if list(call[0][:3]) == ["kubectl", "apply", "-f"]][-1]
        routes = [doc for doc in yaml.safe_load_all(manifest) if doc and doc["kind"] == "HTTPRoute"]
        assert len(routes) == 1
        refs = routes[0]["spec"]["rules"][0]["backendRefs"]
        assert {(ref["namespace"], ref["name"]): ref["weight"] for ref in refs} == expected
        for member in (first, second):
            saved = ModelServiceMemberDao().get(member.id)
            assert saved.status == "active"
            assert saved.weight == member.weight
    heal.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_heals_secure_serving_on_an_unhealthy_member(monkeypatch):
    """An unhealthy probe may be caused by TLS on the EPP, so status must not suppress healing."""
    member = _seed_member("cluster-unhealthy")
    ModelServiceMemberDao().save_health(member.model_copy(update={"status": "unhealthy"}))
    service = _service(calls=[])
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )
    heal = AsyncMock(return_value=True)
    monkeypatch.setattr(service, "_ensure_epp_plaintext", heal)

    operation = await service.reconcile_cluster_gateway("cluster-unhealthy")

    assert operation.status == "succeeded"
    heal.assert_awaited_once_with("cluster-unhealthy", "ns", "qwen-epp", on_log=None)


@pytest.mark.asyncio
async def test_heal_data_plane_uses_no_actor_so_the_operation_can_be_recorded(monkeypatch):
    """`created_by_user_id` is a foreign key into `users`; a literal placeholder
    actor (for example "auto-heal") violates it and silently breaks the
    self-heal path, so automatic runs must pass actor=None like every other
    background reconcile (see GatewayOpsService._finish's `is_automatic` check).
    """
    _seed_member("cluster-heal")
    calls: list[tuple[str, object]] = []
    service = GatewayOpsService(
        operations=GatewayOperationDao(),
        members=ModelServiceMemberDao(),
        groups=ModelServiceGroupDao(),
    )

    async def fake_install_cluster_gateway(cluster_id, *, actor=None, install_prerequisites=True):
        assert install_prerequisites is False
        calls.append(("gateway", actor))

    async def fake_install_ipp(cluster_id, *, actor=None):
        calls.append(("ipp", actor))

    monkeypatch.setattr(service, "install_cluster_gateway", fake_install_cluster_gateway)
    monkeypatch.setattr(service, "install_ipp", fake_install_ipp)

    await service._heal_data_plane("cluster-heal")

    assert calls == [("gateway", None), ("ipp", None)]


@pytest.mark.asyncio
async def test_probe_members_flags_missing_model_server(monkeypatch):
    member = _seed_component_member("percomp-modelserver-down")
    calls: list = []
    service = GatewayOpsService(
        operations=GatewayOperationDao(),
        members=ModelServiceMemberDao(),
        groups=ModelServiceGroupDao(),
        runner_factory=lambda _cid: _ComponentRunner(calls, model_server_ready=False),
        target_resolver=lambda _eid: SimpleNamespace(status="ready"),
    )
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"),
    )
    reconcile = AsyncMock()
    monkeypatch.setattr(service, "reconcile_cluster", reconcile)
    await service.probe_members()

    reconcile.assert_awaited_once_with("cluster-a")
    saved = ModelServiceMemberDao().get(member.id)
    assert saved.status == "unhealthy"
    assert saved.health_json["components"]["modelServer"] == "missing"
    assert saved.health_json["components"]["epp"] == "ready"


def test_resolve_authz_endpoint_uses_live_serving_port(monkeypatch):
    monkeypatch.setattr(gateway_ops_module.hostinfo, "current_serving_port", lambda: 8084)
    assert gateway_ops_module.resolve_authz_endpoint(None) == (None, None)
    assert gateway_ops_module.resolve_authz_endpoint("  ") == (None, None)
    assert gateway_ops_module.resolve_authz_endpoint("10.0.0.5") == ("10.0.0.5", 8084)
    # A stray user-supplied port is ignored: Lens always uses its own.
    assert gateway_ops_module.resolve_authz_endpoint("10.0.0.5:9999") == ("10.0.0.5", 8084)


def test_resolve_authz_endpoint_does_not_guess_before_first_request(monkeypatch):
    monkeypatch.setattr(gateway_ops_module.hostinfo, "current_serving_port", lambda: None)
    assert gateway_ops_module.resolve_authz_endpoint("10.0.0.5") == ("10.0.0.5", None)


@pytest.mark.asyncio
async def test_install_refuses_to_write_authz_endpoint_without_live_port(monkeypatch):
    calls: list = []
    service = _service(calls=calls)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _: SimpleNamespace(gateway_authz_host="10.0.0.9", gateway_namespace="lens-gateway"),
    )
    monkeypatch.setattr(gateway_ops_module.hostinfo, "current_serving_port", lambda: None)

    operation = await service.install_inference_gateway("cluster-a", "istio", install_prerequisites=False)

    assert operation.status == "failed"
    assert "serving port is unknown" in operation.message
    assert not [argv for argv, _ in calls if list(argv[:3]) == ["kubectl", "apply", "-f"]]


@pytest.mark.asyncio
async def test_model_server_health_checks_pods_selected_by_the_inference_pool():
    calls: list[list[str]] = []
    service = GatewayOpsService(operations=GatewayOperationDao())

    async def kubectl(_cluster_id, argv, **_kwargs):
        calls.append(argv)
        return _Result(
            stdout=json.dumps(
                {
                    "items": [
                        {
                            "status": {
                                "phase": "Running",
                                "conditions": [{"type": "Ready", "status": "True"}],
                            }
                        }
                    ]
                }
            )
        )

    service._kubectl = kubectl

    assert await service._model_server_state("cluster-a", "ns", "llm-d.ai/guide=pool") == "ready"
    assert calls == [["get", "pods", "-n", "ns", "-l", "llm-d.ai/guide=pool", "-o", "json"]]


@pytest.mark.asyncio
async def test_managed_rewrites_are_scoped_to_this_lens_cluster():
    service = GatewayOpsService(operations=GatewayOperationDao())
    items = [
        {
            "metadata": {
                "name": "new-own",
                "namespace": "ns-own",
                "labels": {
                    REWRITE_MANAGED_LABEL: "true",
                    MODEL_SERVICE_LABEL: "ours",
                    CLUSTER_LABEL: "cluster-a",
                },
            }
        },
        {
            "metadata": {
                "name": "foreign-labeled",
                "namespace": "ns-other",
                "labels": {
                    REWRITE_MANAGED_LABEL: "true",
                    MODEL_SERVICE_LABEL: "theirs",
                    CLUSTER_LABEL: "cluster-b",
                },
            }
        },
        {
            "metadata": {
                "name": "legacy-foreign",
                "namespace": "ns-other",
                "labels": {MANAGED_LABEL: MANAGED_VALUE, MODEL_SERVICE_LABEL: "theirs"},
            }
        },
    ]

    async def kubectl(_cluster_id, argv, **_kwargs):
        assert list(argv[:2]) == ["get", "inferencemodelrewrite"]
        return _Result(stdout=json.dumps({"items": items}))

    service._kubectl = kubectl

    assert await service._managed_model_rewrites("cluster-a") == {("ns-own", "new-own")}


def test_automatic_gateway_operations_do_not_write_audit_events():
    service = GatewayOpsService(operations=GatewayOperationDao())
    audit = Mock()
    service._auth = SimpleNamespace(audit=audit)

    operation = service._record(GatewayOperation(kind="install-gateway", cluster_id="cluster-a"))
    service._finish(operation, ok=True, message="reconciled")

    audit.assert_not_called()


class _ModelRewriteRunner(_FakeRunner):
    async def run(self, argv, *, cwd=None, env=None, input=None, timeout=10):  # noqa: A002
        self.calls.append((argv, input))
        if list(argv[:3]) == ["kubectl", "get", "crd"]:
            return _Result(stdout="inferencemodelrewrites.llm-d.ai\n")
        if list(argv[:3]) == ["kubectl", "get", "inferencemodelrewrite"]:
            return _Result(stdout='{"items": []}')
        return await super().run(argv, cwd=cwd, env=env, input=input, timeout=timeout)


@pytest.mark.asyncio
async def test_reconcile_maps_public_model_name_to_served_model_name(monkeypatch):
    group = ModelServiceGroupDao().create(
        ModelServiceGroup(
            name="qwen3-0.6b",
            model_ref="Qwen/Qwen3-0.6B",
            cluster_id="cluster-a",
            served_name="Qwen/Qwen3-0.6B",
            base_model="qwen3-0.6b",
        )
    )
    ModelServiceMemberDao().create(
        ModelServiceMember(
            group_id=group.id,
            execution_id="exec-alias",
            cluster_id="cluster-a",
            target_namespace="ns",
            target_service="qwen-epp",
            target_port=80,
            pool_name="ns/qwen-pool",
            epp_ref="qwen-epp",
            endpoint_kind="llm-d-epp",
        )
    )
    calls: list = []
    service = _service(calls=calls, runner_cls=_ModelRewriteRunner)
    monkeypatch.setattr(
        gateway_ops_module,
        "get_cluster",
        lambda _cid: SimpleNamespace(
            gateway_provider="istio", gateway_namespace="lens-gateway", gateway_name="lens-inference-gateway"
        ),
    )

    operation = await service.reconcile_cluster_gateway("cluster-a")

    assert operation.status == "succeeded"
    rewrites = [
        item[1]
        for item in calls
        if list(item[0][:3]) == ["kubectl", "apply", "-f"] and "kind: InferenceModelRewrite" in (item[1] or "")
    ]
    assert rewrites
    manifest = rewrites[-1]
    assert "value: qwen3-0.6b" in manifest
    assert "modelRewrite: Qwen/Qwen3-0.6B" in manifest
    assert "name: qwen-pool" in manifest
    assert "lens.llm-d.ai/cluster-id: cluster-a" in manifest
    assert "lens.llm-d.ai/rewrite-managed: 'true'" in manifest
