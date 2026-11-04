"""Polling must not finalize executions while their creator still owns them."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from llm_d_lens.deploy.contracts import ConfigurationArtifact, DeploymentCreateRequest, DeploymentStatus
from llm_d_lens.deploy.providers.guide_adapter import GuideDefinition, GuideDeploymentArtifact, ValidationResult
from llm_d_lens.deploy.providers.guide_catalog import GuideCatalog
from llm_d_lens.deploy.service import ExecutionLogArchive, GuideAdapterDeploymentService
from tests.unit.deploy.providers.test_storage_mount import cached_model as cached_model
from tests.unit.deploy.test_bundle_evidence import EvidenceAdapter, _router_crds_ready
from tests.unit.deploy.test_readiness_transient import startup_snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("deploy_fails", [False, True])
async def test_refresh_during_creation_does_not_fail_missing_resources(
    tmp_path, deploy_fails, cached_model, monkeypatch
):
    _router_crds_ready(monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    class DelayedAdapter(EvidenceAdapter):
        async def deploy(self, artifact, context):
            entered.set()
            await release.wait()
            if deploy_fails:
                raise RuntimeError("manifest apply rejected")
            return await super().deploy(artifact, context)

        async def readiness(self, context):
            if not release.is_set() or deploy_fails:
                return ValidationResult(False, reasons=['deployments.apps "modelserver" not found'])
            return ValidationResult(True)

    catalog = GuideCatalog([DelayedAdapter()])
    creator = GuideAdapterDeploymentService(catalog, lambda _: "ns", ExecutionLogArchive(tmp_path))
    poller = GuideAdapterDeploymentService(catalog, lambda _: "ns", ExecutionLogArchive(tmp_path))
    request = DeploymentCreateRequest(
        provenance={"cluster_server_id": "cluster"},
        configuration_artifacts=[
            ConfigurationArtifact(
                schema_version="v1",
                kind="helm-values",
                provider_ref="optimized-baseline",
                media_type="application/json",
                checksum="hash",
                content='{"model":{"name":"org/model"},"runtime":{"storageVolumeId":"volume"}}',
            )
        ],
    )
    initial = []
    task = asyncio.create_task(creator.create(request, on_initial_execution=initial.append))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        refreshed = await poller.refresh_diagnostics(initial[0])
        assert refreshed.status == DeploymentStatus.DEPLOYING
    finally:
        release.set()
        final = await task
    assert final.status == (DeploymentStatus.FAILED if deploy_fails else DeploymentStatus.READY)
    # Once creation settles, normal reconciliation must resume, including after errors.
    refreshed = await poller.refresh_diagnostics(final)
    assert refreshed.status == final.status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["baseline-vllm", "optimized-baseline", "pd-disaggregation", "precise-prefix-cache-routing", "tiered-prefix-cache"],
)
@pytest.mark.parametrize("content", ["{}", '{"model":{"name":"missing/model"},"runtime":{"storageVolumeId":"volume"}}'])
async def test_all_providers_reject_uncached_models_before_rendering(tmp_path, cached_model, provider, content):
    class CacheRequiredAdapter(EvidenceAdapter):
        def discover(self):
            return GuideDefinition(provider, "saved", "hash", "stable")

    adapter = CacheRequiredAdapter()
    adapter.render = AsyncMock()
    adapter.deploy = AsyncMock()
    creator = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))
    request = DeploymentCreateRequest(
        provenance={"cluster_server_id": "cluster"},
        configuration_artifacts=[
            ConfigurationArtifact(
                schema_version="v1",
                kind="helm-values",
                provider_ref=provider,
                media_type="application/json",
                checksum="hash",
                content=content,
            )
        ],
    )
    execution = await creator.create(request)
    assert execution.status == DeploymentStatus.FAILED
    assert "Model Cache" in execution.diagnostics.value["reason"]
    adapter.render.assert_not_awaited()
    adapter.deploy.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_failure_preserves_reason_on_case(tmp_path):
    from types import SimpleNamespace

    from llm_d_lens.deploy.contracts import (
        DeployableConfiguration,
        DeploymentCaseStatus,
        DeploymentRun,
        VersionedPayload,
    )
    from llm_d_lens.deploy.run_store import JsonDeploymentRunStore
    from llm_d_lens.deploy.worker import DeploymentRunWorker
    from tests.unit.deploy.test_deployment_management import _case, _execution

    store = JsonDeploymentRunStore(tmp_path)
    execution = _execution("execution-1", DeploymentStatus.DEPLOYING, "ns")
    case = _case("run-1", 0, execution.execution_id, DeploymentCaseStatus.DEPLOYING)
    run = DeploymentRun(
        id="run-1",
        cases=[case],
        source_configurations=[
            DeployableConfiguration(
                type="optimized-baseline", format="helm", provider_ref="optimized-baseline", checksum="hash"
            )
        ],
    )
    store.create_run(run)
    store.save_execution(execution)

    async def refresh(current):
        return current.model_copy(
            update={
                "status": DeploymentStatus.FAILED,
                "diagnostics": VersionedPayload(
                    schema_version="deploy-diagnostics.v1",
                    value={"readiness": ["modelserver exceeded progress deadline"]},
                ),
            }
        )

    await DeploymentRunWorker(store, SimpleNamespace(refresh_diagnostics=refresh)).refresh_run_executions(run.id)
    failed = store.get_case(run.id, case.id)
    assert failed.status == DeploymentCaseStatus.FAILED
    assert failed.failure is not None
    assert "modelserver exceeded progress deadline" in failed.failure.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_status", "pods", "events", "expected_status"),
    [
        (DeploymentStatus.DEPLOYING, "model-0 0/1 Pending", "cannot allocate all claims", DeploymentStatus.DEPLOYING),
        (DeploymentStatus.FAILED, "model-0 0/1 Pending", "cannot allocate all claims", DeploymentStatus.DEPLOYING),
        (DeploymentStatus.FAILED, "model-0 0/1 Pending", "Insufficient nvidia.com/gpu", DeploymentStatus.DEPLOYING),
        (DeploymentStatus.READY, "model-0 0/1 Pending", "cannot allocate all claims", DeploymentStatus.READY),
        (DeploymentStatus.FAILED, "model-0 0/1 Running", "cannot allocate all claims", DeploymentStatus.FAILED),
        (DeploymentStatus.FAILED, "model-0 0/1 Pending", "connection refused", DeploymentStatus.FAILED),
        (DeploymentStatus.CLEANED, "model-0 0/1 Pending", "cannot allocate all claims", DeploymentStatus.CLEANED),
        (
            DeploymentStatus.ROLLED_BACK,
            "model-0 0/1 Pending",
            "cannot allocate all claims",
            DeploymentStatus.ROLLED_BACK,
        ),
        (
            DeploymentStatus.DEPLOYING,
            "model-0 0/1 CrashLoopBackOff",
            "cannot allocate all claims",
            DeploymentStatus.FAILED,
        ),
    ],
)
async def test_refresh_reconciles_resource_wait_without_masking_other_states(
    tmp_path, initial_status, pods, events, expected_status
):
    from tests.unit.deploy.test_deployment_management import _execution

    adapter = EvidenceAdapter()
    adapter.readiness = AsyncMock(return_value=ValidationResult(False, ["deployment exceeded progress deadline"]))
    adapter.diagnostics = AsyncMock(return_value={"pods": pods, "events": events})
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))
    execution = _execution("execution-1", initial_status, "ns")
    service._provider_executions[execution.execution_id] = (
        adapter,
        GuideDeploymentArtifact("optimized-baseline", "hash"),
        {"namespace": "ns"},
    )

    refreshed = await service.refresh_diagnostics(execution)

    assert refreshed.status == expected_status
    assert refreshed.diagnostics.value["snapshot"] == {"pods": pods, "events": events}
    assert refreshed.diagnostics.value["readiness"] == ["deployment exceeded progress deadline"]
    assert (await service.get(execution.execution_id)).status == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize("starting", [False, True])
async def test_resource_wait_recovery_clears_case_failure_and_marks_run_running(tmp_path, starting):
    from llm_d_lens.deploy.contracts import (
        DeployableConfiguration,
        DeploymentCaseStatus,
        DeploymentRun,
        DeploymentRunStatus,
        ProblemDetails,
    )
    from llm_d_lens.deploy.run_store import JsonDeploymentRunStore
    from llm_d_lens.deploy.worker import DeploymentRunWorker
    from tests.unit.deploy.test_deployment_management import _case, _execution

    adapter = EvidenceAdapter()
    adapter.readiness = AsyncMock(return_value=ValidationResult(False, ["deployment exceeded progress deadline"]))
    adapter.diagnostics = AsyncMock(
        return_value=startup_snapshot()
        if starting
        else {"pods": "model-0 0/1 Pending", "events": "FailedScheduling cannot allocate all claims"}
    )
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))
    store = JsonDeploymentRunStore(tmp_path / "runs")
    execution = _execution("execution-1", DeploymentStatus.FAILED, "ns")
    case = _case("run-1", 0, execution.execution_id, DeploymentCaseStatus.FAILED)
    case.failure = ProblemDetails(status=500, title="deployment case failed", code="deployment_failed")
    run = DeploymentRun(
        id="run-1",
        status=DeploymentRunStatus.FAILED,
        cases=[case],
        source_configurations=[
            DeployableConfiguration(
                type="optimized-baseline", format="helm", provider_ref="optimized-baseline", checksum="hash"
            )
        ],
    )
    store.create_run(run)
    store.save_execution(execution)

    await DeploymentRunWorker(store, service).refresh_run_executions(run.id)

    assert store.get_execution(execution.execution_id).status == DeploymentStatus.DEPLOYING
    assert store.get_case(run.id, case.id).status == DeploymentCaseStatus.DEPLOYING
    assert store.get_case(run.id, case.id).failure is None
    assert store.get_run(run.id).status == DeploymentRunStatus.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize("starting", [False, True])
async def test_creation_keeps_resource_wait_or_startup_without_cleanup(tmp_path, cached_model, starting, monkeypatch):
    _router_crds_ready(monkeypatch)
    adapter = EvidenceAdapter()
    adapter.readiness = AsyncMock(return_value=ValidationResult(False, ["deployment exceeded progress deadline"]))
    adapter.diagnostics = AsyncMock(
        return_value=startup_snapshot()
        if starting
        else {"pods": "model-0 0/1 Pending", "events": "FailedScheduling cannot allocate all claims"}
    )
    adapter.cleanup = AsyncMock()
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))
    request = DeploymentCreateRequest(
        provenance={"cluster_server_id": "cluster"},
        configuration_artifacts=[
            ConfigurationArtifact(
                schema_version="v1",
                kind="helm-values",
                provider_ref="optimized-baseline",
                media_type="application/json",
                checksum="hash",
                content='{"model":{"name":"org/model"},"runtime":{"storageVolumeId":"volume"}}',
            )
        ],
    )

    execution = await service.create(request)

    assert execution.status == DeploymentStatus.DEPLOYING
    assert execution.diagnostics.value["reasons"] == ["deployment exceeded progress deadline"]
    adapter.cleanup.assert_not_awaited()

    adapter.readiness.return_value = ValidationResult(True)
    refreshed = await service.refresh_diagnostics(execution)
    assert refreshed.status == DeploymentStatus.READY
    assert refreshed.endpoint.url == "http://router:8000"
    adapter.cleanup.assert_not_awaited()


def _router_request(provider_ref: str) -> DeploymentCreateRequest:
    return DeploymentCreateRequest(
        provenance={"cluster_server_id": "cluster"},
        configuration_artifacts=[
            ConfigurationArtifact(
                schema_version="v1",
                kind="helm-values",
                provider_ref=provider_ref,
                media_type="application/json",
                checksum="hash",
                content='{"model":{"name":"org/model"},"runtime":{"storageVolumeId":"volume"}}',
            )
        ],
    )


class _ContractAdapter(EvidenceAdapter):
    def __init__(self, provider_ref: str, deployment_contract: dict | None = None) -> None:
        self._provider_ref = provider_ref
        self._deployment_contract = deployment_contract or {}

    def discover(self):
        return GuideDefinition(self._provider_ref, "saved", "hash", "stable")

    async def render(self, *_):
        return GuideDeploymentArtifact(self._provider_ref, "hash", deployment_contract=dict(self._deployment_contract))


@pytest.mark.asyncio
async def test_router_provider_installs_crds_before_deploy(tmp_path, cached_model, monkeypatch):
    order: list[tuple[str, str]] = []

    async def ensure(cluster_id):
        order.append(("crds", cluster_id))
        return {"ok": True, "applied": True, "message": ""}

    monkeypatch.setattr("llm_d_lens.deploy.service.ensure_router_crds", AsyncMock(side_effect=ensure))

    class OrderedAdapter(EvidenceAdapter):
        async def deploy(self, artifact, context):
            order.append(("deploy", context["namespace"]))
            return await super().deploy(artifact, context)

    service = GuideAdapterDeploymentService(
        GuideCatalog([OrderedAdapter()]), lambda _: "ns", ExecutionLogArchive(tmp_path)
    )

    execution = await service.create(_router_request("optimized-baseline"))

    assert execution.status == DeploymentStatus.READY
    assert [step[0] for step in order] == ["crds", "deploy"]
    assert order[0][1] == "cluster"


@pytest.mark.asyncio
async def test_direct_data_plane_skips_the_router_crd_preflight(tmp_path, cached_model, monkeypatch):
    ensure = AsyncMock(return_value={"ok": True, "applied": False, "message": "already present"})
    monkeypatch.setattr("llm_d_lens.deploy.service.ensure_router_crds", ensure)
    adapter = _ContractAdapter("baseline-vllm", {"data_plane_kind": "direct"})
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))

    execution = await service.create(_router_request("baseline-vllm"))

    assert execution.status == DeploymentStatus.READY
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_crd_preflight_failure_stops_the_deployment_before_deploy(tmp_path, cached_model, monkeypatch):
    ensure = AsyncMock(return_value={"ok": False, "applied": True, "message": "apply failed"})
    monkeypatch.setattr("llm_d_lens.deploy.service.ensure_router_crds", ensure)
    adapter = EvidenceAdapter()
    adapter.deploy = AsyncMock()
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))

    execution = await service.create(_router_request("optimized-baseline"))

    assert execution.status == DeploymentStatus.FAILED
    adapter.deploy.assert_not_awaited()
    assert "apply failed" in execution.diagnostics.value["reason"]


@pytest.mark.asyncio
async def test_saved_bundle_router_version_skips_the_crd_preflight(tmp_path, cached_model, monkeypatch):
    ensure = AsyncMock(return_value={"ok": True, "applied": True, "message": ""})
    monkeypatch.setattr("llm_d_lens.deploy.service.ensure_router_crds", ensure)
    adapter = _ContractAdapter("optimized-baseline", {"deploymentBundle": {"helm": {"version": "v0.9.0"}}})
    service = GuideAdapterDeploymentService(GuideCatalog([adapter]), lambda _: "ns", ExecutionLogArchive(tmp_path))

    execution = await service.create(_router_request("optimized-baseline"))

    assert execution.status == DeploymentStatus.READY
    ensure.assert_not_awaited()
