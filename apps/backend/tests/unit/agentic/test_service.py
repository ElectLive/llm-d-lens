"""Tests for Agentic Deploy's delegation to existing Prism services."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from llm_d_lens.agentic.facts import PlanningEvidence, ResolvedPlanningFacts
from llm_d_lens.agentic.generator import CandidateProposal
from llm_d_lens.agentic.historical_candidate_seeds import HistoricalCandidateSeed
from llm_d_lens.agentic.models import (
    AgenticCandidateRefinementRequest,
    AgenticDeploymentCreateRequest,
    AgenticDeploymentStatus,
)
from llm_d_lens.agentic.planner import (
    AICCandidatePrediction,
    DeterministicPlanner,
    OpenAICompatiblePlanner,
    OpenAIPlannerSettings,
    PerformanceScoreInput,
    PlanningFacts,
    WorkloadProfile,
)
from llm_d_lens.agentic.planning_snapshot import PlanningSnapshotStore
from llm_d_lens.agentic.service import AgenticDeploymentService

_ORIGINAL_VALIDATE_MODEL_CACHE = AgenticDeploymentService._validate_model_cache


async def _validate_model_cache(*_args, **_kwargs) -> None:
    pass


@pytest.fixture(autouse=True)
def validated_model_cache(monkeypatch, tmp_path):
    async def hardware(_cluster, _model, facts, _principal, *, hardware_snapshot=None):
        return facts, {"status": "unavailable"}

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_retrieval_hardware", hardware)
    monkeypatch.setattr(AgenticDeploymentService, "_validate_model_cache", staticmethod(_validate_model_cache))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.PlanningSnapshotStore",
        lambda: PlanningSnapshotStore(tmp_path),
    )


@pytest.mark.parametrize("target", ["concurrency", "request_rate"])
def test_load_target_uses_exact_aic_replicas_without_plain_estimates(monkeypatch, target):
    from llm_d_lens.agentic.planner import AICCandidatePrediction, WorkloadProfile

    facts = PlanningFacts(
        8,
        32,
        8,
        workload_profile=WorkloadProfile(**{target: 10}),
        aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=3),),
        evidence_ids=("aic:search",),
    )

    async def resolve(*args, **kwargs):
        return SimpleNamespace(facts=facts)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    service = AgenticDeploymentService()
    candidates, _, _ = asyncio.run(
        service._generate_candidates(
            None,
            "cluster-a",
            "Qwen/Qwen3-8B",
            facts,
            None,
            provider_requested=False,
        )
    )
    assert len(candidates) == 1
    assert candidates[0].provider_ref == "optimized-baseline"
    assert candidates[0].replicas == 3
    assert candidates[0].required_gpus == 3


def test_joint_load_targets_fail_before_cluster_or_cache_access(monkeypatch):
    from llm_d_lens.agentic.planner import WorkloadProfile

    request = SimpleNamespace(
        planning_facts=PlanningFacts(
            8,
            32,
            8,
            workload_profile=WorkloadProfile(concurrency=10, request_rate=2),
        )
    )
    with pytest.raises(ValueError, match="choose one and clear the other"):
        asyncio.run(AgenticDeploymentService().create(request))


def test_context_budget_failures_are_distinct_from_provider_and_response_failures():
    from llm_d_lens.agentic.generator import CandidateGenerationError
    from llm_d_lens.agentic.planner import OpenAIPlannerError

    assert (
        AgenticDeploymentService._generator_failure_reason(CandidateGenerationError("Context window budget exceeded"))
        == "context_budget_exceeded"
    )
    assert (
        AgenticDeploymentService._planner_failure_reason(OpenAIPlannerError("Context window budget exceeded"))
        == "context_budget_exceeded"
    )


def _request(**overrides) -> AgenticDeploymentCreateRequest:
    overrides.setdefault("storage_volume_id", "storage-model-cache")
    return AgenticDeploymentCreateRequest(
        model="Qwen/Qwen3-8B",
        deployment_name="agentic-qwen3",
        cluster_session_id="a" * 36,
        replicas=1,
        tensor_parallel_size=1,
        planning_facts={"model_weight_gib": 8, "vram_per_gpu_gib": 32, "free_gpu_count": 4},
        **overrides,
    )


@pytest.mark.parametrize("context_length", [40960, 262144, 1048576])
def test_model_card_context_reaches_effective_planning_facts(context_length):
    request = _request(max_model_len=context_length)
    facts = AgenticDeploymentService._planning_only_facts(
        request.planning_facts,
        max_model_len=request.max_model_len,
        gpu_memory_utilization=request.gpu_memory_utilization,
    )

    assert request.planning_facts.context_length == 4096
    assert facts.context_length == context_length


@pytest.mark.parametrize("context_length", [255, 1048577])
def test_deployment_context_rejects_values_outside_supported_range(context_length):
    with pytest.raises(ValidationError, match="max_model_len"):
        _request(max_model_len=context_length)


def _resolved_facts(*, free_gpu_count=4, aic_predictions=()) -> ResolvedPlanningFacts:
    facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=free_gpu_count,
        aic_predictions=aic_predictions,
    )
    return ResolvedPlanningFacts(
        facts=facts,
        cluster_snapshot={
            "hardware": {
                "availableGpuCount": free_gpu_count,
                "accelerators": [{"id": "nvidia", "gpuCount": free_gpu_count}],
            }
        },
        evidence=[PlanningEvidence(id="cluster-overview:cluster-a", source="cluster-overview", status="available")],
    )


async def _resolve_facts(*_args, **_kwargs) -> ResolvedPlanningFacts:
    return _resolved_facts()


def test_create_generates_an_approval_plan_without_deploying(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: None,
    )

    run = asyncio.run(service.create(_request()))

    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert run.deployment_run_id is None
    assert run.configuration_artifact_id is None
    assert run.selected_candidate.replicas == 1
    assert run.selected_candidate.tensor_parallel_size == 1
    assert run.planning_evidence[0].source == "cluster-overview"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "deterministic"
    assert "Relative rank" in run.decision_metadata.score_method
    assert [item["phase"] for item in run.planning_trace] == [
        "input",
        "public_rag",
        "configuration_cases",
        "generation",
        "candidate_validation",
        "deterministic_ranking",
        "scoring",
        "selection",
    ]
    assert run.planning_trace[-1]["candidate_id"] == run.selected_candidate.id
    ranking = next(item for item in run.planning_trace if item["phase"] == "deterministic_ranking")
    assert ranking["ranking"][0]["candidate_id"] == run.selected_candidate.id


def test_create_preserves_full_prompt_in_append_only_planning_snapshot(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    run = asyncio.run(service.create(_request(planner_prompt="persist this complete prompt")))

    assert run.planning_snapshot_id is not None
    assert run.planning_snapshot_path is not None
    payload = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert payload["event"] == "created"
    assert payload["run_id"] == run.id
    assert payload["request"]["planner_prompt_present"] is True
    assert payload["schema_version"] == 2
    assert payload["request"]["planner_prompt"] == "persist this complete prompt"
    assert payload["generator"]["requests"] == []
    assert payload["planner"]["requests"] == []
    assert payload["selected_candidate"]["id"] == run.selected_candidate.id


@pytest.mark.parametrize("fallback", [False, True])
def test_public_rag_is_queried_before_generation_and_requeried_on_refinement(monkeypatch, tmp_path, fallback):
    from llm_d_lens.agentic.generator import CandidateGenerationResult
    from llm_d_lens.agentic.planner import PlannerDecision
    from llm_d_lens.auth.contracts import Principal, RoleBinding

    calls = []
    observations = []
    principal = Principal("admin-user", "admin", bindings=(RoleBinding("admin"),))

    async def hardware(_cluster, _model, facts, _principal, *, hardware_snapshot=None):
        from llm_d_lens.agentic.planning_requirements import enrich_planning_requirements

        return enrich_planning_requirements(facts, model_name=_model, hardware_model="H100", profile_id="nvidia"), {
            "status": "available",
            "hardware": "H100",
        }

    for module in ("generator", "planner", "facts"):
        monkeypatch.setattr(f"llm_d_lens.agentic.{module}.load_model_config", lambda _model: None)
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_retrieval_hardware", hardware)

    async def query(model, caller, *, cluster_server_id, planning_facts, hardware_snapshot=None):
        assert model == "Qwen/Qwen3-8B"
        assert caller is principal
        assert cluster_server_id == "cluster-a"
        assert planning_facts.workload_profile.mean_input_tokens == (2048 if not observations else 1024)
        assert planning_facts.workload_profile.mean_output_tokens == 256
        assert planning_facts.ttft_slo_ms == (100 if not observations else 50)
        calls.append("rag")
        item_id = f"observation-{len(observations) + 1}"
        observation = {
            "item_id": item_id,
            "metric": "ttft",
            "statistic": "p95" if fallback else None,
            "value": "125.50",
            "unit": "ms",
            "citation": {"source_revision": f"rev-{len(observations) + 1}", "sha256": "a" * 64},
            "experiment": {
                "item_id": f"experiment-{item_id}",
                "origin_id": "public",
                "model": model,
                "hardware": "H100",
                "runtime": "vllm",
                "workload": {}
                if fallback
                else {"mean_input_tokens": planning_facts.workload_profile.mean_input_tokens},
                "configuration": {"tp": 1},
            },
        }
        observations.append(observation)
        return PlanningEvidence(
            id="public-rag:query",
            source="public-benchmark-rag",
            status="available",
            details={
                "response": {"evidence": [observation]},
                "message": "Public evidence precedes AIC.",
                "target_hardware": {"model": "H100", "profile_id": "nvidia"},
            },
        )

    async def generate(_self, _cluster, _model, facts, _prompt, **kwargs):
        calls.append("generation")
        assert facts.requirements.hardware.device_model == "H100"
        assert facts.public_rag_evidence[0]["item_id"] == observations[-1]["item_id"]
        assert facts.configuration_cases[0]["experiment_id"] == f"experiment-{observations[-1]['item_id']}"
        assert facts.configuration_cases[0]["ranking"]["queue"] == ("fallback" if fallback else "ranked")
        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits.",
                    }
                ]
            },
            resolved=_resolved_facts().model_copy(update={"facts": facts}),
        )

    class Provider:
        settings = SimpleNamespace(model="planner")

        async def recommend(self, facts, candidates, **kwargs):
            calls.append("ranking")
            assert facts.requirements.hardware.device_model == "H100"
            reference = f"public-rag:{observations[-1]['item_id']}"
            assert facts.public_rag_evidence[0]["evidence_id"] == reference
            return PlannerDecision(
                action="select_candidate",
                candidate_id=candidates[0].id,
                confidence=0.8,
                rationale="Published evidence is primary.",
                reference_material_ids=[reference],
            ), candidates[:1]

    async def estimate(_self, _model, candidates, facts, _trace):
        return candidates, facts

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_public_rag_evidence", query)
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import InternalCaseOutcome

    async def internal_query(*_args, **_kwargs):
        return InternalCaseOutcome(status="empty")

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_internal_cases", internal_query)
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr(AgenticDeploymentService, "_estimate_missing_topologies", estimate)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session", lambda _id: SimpleNamespace(server_id="cluster-a")
    )
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    events = []

    async def progress(event):
        events.append(event)

    create_request = _request(ai_provider_id="provider", planner_prompt="ISL 2048 OSL 256 TTFT 100")
    run = asyncio.run(
        service.create(
            create_request,
            principal=principal,
            on_progress=progress,
        )
    )
    created = json.loads((tmp_path / run.planning_snapshot_path).read_text())
    public = next(item for item in created["planning_evidence"] if item["source"] == "public-benchmark-rag")
    assert public["details"]["response"]["evidence"][0]["citation"]["source_revision"] == "rev-1"
    case_packet = next(item for item in created["planning_evidence"] if item["source"] == "configuration-cases")
    assert case_packet["details"]["cases"][0]["measurements"][0]["evidence_id"] == "public-rag:observation-1"
    assert case_packet["details"]["cases"][0]["ranking"]["queue"] == ("fallback" if fallback else "ranked")
    assert created["decision"]["reference_material_ids"] == ["public-rag:observation-1"]
    assert "public-rag:observation-1" not in created["decision_evidence_ids"]
    assert any("Public evidence" in event.get("message", "") for event in events)

    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="ISL 1024 OSL 256 TTFT 50 prefer latency"),
            principal=principal,
        )
    )
    assert calls == ["rag", "generation", "ranking", "rag", "generation", "ranking"]
    assert refined.decision.reference_material_ids == ["public-rag:observation-2"]
    assert "public-rag:observation-2" not in refined.decision_evidence_ids
    assert "public-rag:observation-1" not in refined.decision_evidence_ids
    snapshot = json.loads((tmp_path / refined.planning_snapshot_path).read_text())
    assert snapshot["resolved_planning_facts"]["public_rag_evidence"][0]["citation"]["source_revision"] == "rev-2"
    assert snapshot["resolved_planning_facts"]["configuration_cases"][0]["ranking"]["queue"] == (
        "fallback" if fallback else "ranked"
    )

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_d_lens.agentic import router

    monkeypatch.setattr(router, "service", service)
    monkeypatch.setattr(router, "current_principal", lambda _request: principal)
    app = FastAPI()
    app.include_router(router.router)
    with TestClient(app) as client:
        response = client.get(f"/api/agentic-deployments/{refined.id}")
    assert response.status_code == 200
    wire_packet = next(item for item in response.json()["planning_evidence"] if item["source"] == "configuration-cases")
    assert wire_packet["details"]["cases"][0]["ranking"]["queue"] == ("fallback" if fallback else "ranked")


def test_case_selection_preserves_authorized_internal_history():
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import InternalCaseOutcome, build_case_bundle

    facts = PlanningFacts(8, 32, 4)
    observation = {
        "item_id": "measurement",
        "evidence_id": "public-rag:measurement",
        "metric": "ttft",
        "statistic": "mean",
        "value": "100",
        "unit": "ms",
        "citation": {"origin_id": "public", "source_revision": "rev", "sha256": "a" * 64},
        "experiment": {
            "item_id": "experiment",
            "origin_id": "public",
            "model": "model",
            "hardware": "accelerator",
            "runtime": "vllm",
            "workload": {},
            "configuration": {"tp": 1},
        },
    }
    bundle, kept = build_case_bundle((observation,), facts, model="model", hardware="accelerator", branch_status={})
    assert bundle.cases
    public = PlanningEvidence(
        id="public-rag:query",
        source="public-benchmark-rag",
        status="available",
        details={"response": {"evidence": [observation]}, "target_hardware": {"model": "accelerator"}},
    )
    internal = InternalCaseOutcome()
    selected_facts, selected_public, selected_internal, selected = AgenticDeploymentService._apply_case_selection(
        facts, public, internal, bundle, kept
    )
    assert len(selected.cases) == 1
    assert selected_facts.configuration_cases[0]["case_id"] == selected.cases[0].case_id
    assert selected_public.details["response"]["evidence"][0]["item_id"] == "measurement"
    assert selected_internal is internal


def test_approve_publishes_configuration_then_starts_existing_deploy(monkeypatch):
    from llm_d_lens.configuration.models import ModelSecretConfiguration

    service = AgenticDeploymentService()
    cache_checks = []

    async def validate_cache(_request, _cluster_id, *, require_model_ready=True):
        cache_checks.append(require_model_ready)
        if require_model_ready:
            return ModelSecretConfiguration(mode="existing-secret", sourceNamespace="models", sourceName="hf-token")
        return None

    monkeypatch.setattr(service, "_validate_model_cache", validate_cache)
    monkeypatch.setenv("PRISM_DEPLOY_ACCELERATOR", "intel-xpu")
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    planning_lengths = []
    planning_utilizations = []

    async def resolve_facts(_cluster_id, _model, facts, **_kwargs):
        planning_lengths.append(facts.context_length)
        planning_utilizations.append(facts.gpu_memory_utilization)
        resolved = _resolved_facts()
        resolved.facts = replace(
            resolved.facts,
            context_length=facts.context_length,
            gpu_memory_utilization=facts.gpu_memory_utilization,
        )
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve_facts)
    saved = {}

    async def start_run(request, **_kwargs):
        saved["configuration"] = request.configurations[0]
        saved["deployment"] = request
        return SimpleNamespace(id="deploy-run-1")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)

    async def nodes(*_args, **_kwargs):
        return [{"metadata": {"labels": {"kubernetes.io/hostname": "smc-22"}}}]

    monkeypatch.setattr("llm_d_lens.agentic.service.list_resources", nodes)
    run = asyncio.run(
        service.create(
            _request(
                max_model_len=32768,
                gpu_memory_utilization=0.6,
                placement_replicas="same-node",
                placement_node_label={"key": "kubernetes.io/hostname", "value": "smc-22"},
            )
        )
    )

    approved = asyncio.run(service.approve(run.id))

    assert approved.status == AgenticDeploymentStatus.DEPLOYING
    assert approved.configuration_artifact_id is None
    assert approved.deployment_run_id == "deploy-run-1"
    assert cache_checks == [False, True]
    assert approved.model_secret.mode == "existing-secret"
    assert approved.model_secret.source_name == "hf-token"
    assert planning_lengths == [32768, 32768]
    assert planning_utilizations == [0.6, 0.6]
    assert run.selected_candidate.gpu_memory_utilization == 0.6
    assert saved["configuration"].content["decode"]["maxModelLen"] == 32768
    assert saved["configuration"].content["runtime"]["image"] == "docker.io/vllm/vllm-openai:v0.30.0"
    assert saved["configuration"].content["hardware_profile"] == "nvidia"
    assert saved["configuration"].content["decode"]["gpuMemoryUtilization"] == 0.6
    assert saved["configuration"].provenance["cluster_ref"]["session_id"] == session.id
    assert saved["deployment"].runtime_binding.placement == {
        "replicas": "same-node",
        "nodeLabel": {"key": "kubernetes.io/hostname", "value": "smc-22"},
    }
    assert saved["deployment"].provenance["agentic_deployment_run_id"] == run.id


@pytest.mark.parametrize("capacity_simulation", [False, True])
def test_simulated_plan_cannot_enter_deployment(monkeypatch, capacity_simulation):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    run.planning_evidence.append(
        PlanningEvidence(
            id="cluster-overview:cluster-a" if capacity_simulation else "public-rag:query",
            source="cluster-overview" if capacity_simulation else "public-benchmark-rag",
            status="no_matches",
            details={"simulation": True}
            if capacity_simulation
            else {"target_hardware": {"model": "H100", "simulation": True}},
        )
    )
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda _session_id: pytest.fail("Simulation must fail before deployment preflight or mutations"),
    )
    with pytest.raises(ValueError, match="validation-only and cannot be deployed"):
        asyncio.run(service.approve(run.id))
    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert run.deployment_run_id is None


def test_select_candidate_only_allows_a_validated_alternative(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    alternative = run.candidates[1]
    alternative.slo_status = "not_satisfied"
    alternative.slo_confidence = 0.9
    alternative.evidence_ids = ["cluster-overview:cluster-a", "benchmark:alternative"]
    run.decision.evidence_ids = ["benchmark:previous"]

    updated = service.select_candidate(run.id, alternative.id)

    assert updated.selected_candidate.id == alternative.id
    assert updated.decision is not None
    assert updated.decision.rationale.startswith("Selected by the user")
    assert updated.decision.slo_status == alternative.slo_status
    assert updated.decision.slo_confidence == alternative.slo_confidence
    assert updated.decision.evidence_ids == alternative.evidence_ids
    assert "benchmark:alternative" in updated.decision_evidence_ids
    assert "benchmark:previous" not in updated.decision_evidence_ids
    with pytest.raises(ValueError, match="not a validated alternative"):
        service.select_candidate(run.id, "not-a-candidate")


@pytest.mark.parametrize("failure", ["authorization", "snapshot"])
def test_failed_refinement_preserves_the_previous_plan(monkeypatch, failure):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    original = run.model_dump(mode="json")

    def fail(*_args, **_kwargs):
        raise OSError("refinement was not committed")

    if failure == "authorization":
        monkeypatch.setattr("llm_d_lens.agentic.service.assert_case_access", fail)
    else:
        monkeypatch.setattr(service._snapshot_store, "save", fail)
    with pytest.raises(OSError, match="not committed"):
        asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency")))
    assert service.get(run.id).model_dump(mode="json") == original


def test_refinement_does_not_overwrite_a_selection_made_during_generation(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    alternative = run.candidates[1]

    async def progress(event):
        if event["phase"] == "validation" and event["status"] == "complete":
            service.select_candidate(run.id, alternative.id)

    with pytest.raises(ValueError, match="changed"):
        asyncio.run(
            service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency"), progress)
        )
    assert service.get(run.id).selected_candidate.id == alternative.id
    assert service.get(run.id).request.planner_prompt is None


def test_approval_cannot_submit_the_same_plan_twice(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    submissions = []

    async def start_run(request, **_kwargs):
        submissions.append(request)
        if len(submissions) == 1:
            with pytest.raises(ValueError, match="awaiting approval"):
                await service.approve(run.id)
        return SimpleNamespace(id="deploy-run-1")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)
    asyncio.run(service.approve(run.id))
    assert len(submissions) == 1


def test_approval_rejects_a_selection_changed_during_preflight(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))
    alternative = run.candidates[1]

    async def resolve(*_args, **_kwargs):
        service.select_candidate(run.id, alternative.id)
        return _resolved_facts()

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.deployment_run_manager.start_run",
        lambda *_args, **_kwargs: pytest.fail("A changed selection must be reviewed before submission"),
    )
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(service.approve(run.id))
    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert run.selected_candidate.id == alternative.id


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_uncertain_deployment_submission_is_not_left_approvable(monkeypatch, error_type):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    run = asyncio.run(service.create(_request()))

    async def start_run(*_args, **_kwargs):
        raise error_type("submission interrupted")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)
    with pytest.raises(error_type):
        asyncio.run(service.approve(run.id))
    assert run.status == AgenticDeploymentStatus.FAILED
    assert run.error


def test_approve_carries_model_specific_vllm_arguments_to_deployment(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    saved = {}

    async def start_run(request, **_kwargs):
        saved["configuration"] = request.configurations[0]
        return SimpleNamespace(id="deploy-run-1")

    monkeypatch.setattr("llm_d_lens.agentic.service.deployment_run_manager.start_run", start_run)
    run = asyncio.run(
        service.create(
            _request(
                vllm_arguments=[
                    {"name": "enable-auto-tool-choice", "value": "true"},
                    {"name": "tool-call-parser", "value": "hermes"},
                ]
            )
        )
    )

    asyncio.run(service.approve(run.id))

    arguments = saved["configuration"].content["customParameters"]
    assert {item["name"] for item in arguments} >= {"enable-auto-tool-choice", "tool-call-parser"}


def test_planning_uses_deployment_context_length_and_preserves_workload_signals(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    captured = {}

    async def resolve(_cluster_id, _model, facts, **_kwargs):
        captured["facts"] = facts
        return _resolved_facts(
            aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1),)
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    request = _request(
        max_model_len=32768,
        gpu_memory_utilization=0.6,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        vllm_arguments=[{"name": "enforce-eager", "value": "true"}],
    ).model_copy(
        update={
            "planning_facts": PlanningFacts(
                model_weight_gib=8,
                vram_per_gpu_gib=32,
                free_gpu_count=4,
                context_length=4096,
                use_case="long-inputs",
                workload_profile=WorkloadProfile(mean_input_tokens=16384, concurrency=32),
                vllm_arguments=(("enforce-eager", "true"),),
            )
        }
    )
    asyncio.run(service.create(request))

    facts = captured["facts"]
    assert facts.context_length == 32768
    assert facts.gpu_memory_utilization == 0.6
    assert facts.use_case == "long-inputs"
    assert facts.workload_profile is not None
    assert facts.workload_profile.mean_input_tokens == 16384
    assert facts.vllm_arguments == ()


def test_automatic_execution_is_not_enabled(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    with pytest.raises(ValueError, match="automatic Agentic execution is not enabled"):
        asyncio.run(service.create(_request(execution_policy={"mode": "automatic"})))


def test_apply_mode_merges_only_validated_historical_candidate_seeds(monkeypatch):
    service = AgenticDeploymentService(historical_seed_mode="apply")
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=2)
    candidates = DeterministicPlanner().candidates(facts)
    monkeypatch.setattr(
        service._historical_seed_retriever,
        "retrieve",
        lambda *_args, **_kwargs: (
            HistoricalCandidateSeed(
                proposal=CandidateProposal(
                    provider_ref="precise-prefix-cache-routing",
                    replicas=1,
                    tensor_parallel_size=1,
                    rationale="Qualified history.",
                ),
                evidence_refs=("benchmark:seed-1",),
            ),
            HistoricalCandidateSeed(
                proposal=CandidateProposal(
                    provider_ref="optimized-baseline",
                    replicas=3,
                    tensor_parallel_size=1,
                    rationale="Exceeds available capacity.",
                ),
                evidence_refs=("benchmark:invalid",),
            ),
        ),
    )

    merged, seed_count = asyncio.run(service._merge_historical_seeds("Qwen/Qwen3-8B", facts, candidates))

    seeded = next(candidate for candidate in merged if candidate.id == "precise-prefix-cache-routing-tp1-r1")
    assert seed_count == 1
    assert "benchmark:seed-1" in seeded.evidence_ids
    assert all(candidate.id != "optimized-baseline-tp1-r3" for candidate in merged)


def test_create_records_selected_benchmark_refs_in_decision_snapshot(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    def score_inputs(_self, _model, _facts, candidates, *, records):
        assert records == []
        selected = DeterministicPlanner().decide(_facts, candidates).candidate_id
        return (
            PerformanceScoreInput(
                candidate_id=selected,
                source="benchmark",
                status="measured",
                evidence_refs=("benchmark:run-1", "benchmark:run-2"),
            ),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.BenchmarkPerformanceEvidence.score_inputs", score_inputs)
    run = asyncio.run(service.create(_request()))

    assert run.decision_evidence_ids.count("benchmark:run-1") == 1
    assert "benchmark:run-2" in run.decision_evidence_ids
    snapshot = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["decision_evidence_ids"] == run.decision_evidence_ids


def test_create_and_refine_exclude_failure_blocked_candidates_from_ai_ranking(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    blocked_id = "optimized-baseline-tp1-r1"
    ranked = []

    def score_inputs(_self, _model, _facts, candidates, *, records):
        assert blocked_id in {candidate.id for candidate in candidates}
        return (
            PerformanceScoreInput(
                candidate_id=blocked_id,
                source="benchmark",
                status="interpolated",
                has_blocking_negative_evidence=True,
                evidence_refs=("benchmark:oom-1", "benchmark:oom-2", "benchmark:oom-3"),
            ),
        )

    class Provider:
        async def recommend(self, facts, candidates, **_kwargs):
            ids = {candidate.id for candidate in candidates}
            assert ids and blocked_id not in ids
            ranked.append(ids)
            planner = DeterministicPlanner()
            return planner.decide(facts, candidates), planner.score(facts, candidates)

    monkeypatch.setattr("llm_d_lens.agentic.service.BenchmarkPerformanceEvidence.score_inputs", score_inputs)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())
    run = asyncio.run(service.create(_request()))
    assert run.planner == "openai-compatible"
    assert run.selected_candidate.id != blocked_id
    refined = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency")))
    assert refined.planner == "openai-compatible"
    assert refined.selected_candidate.id != blocked_id
    assert len(ranked) == 2


@pytest.fixture
def live_planning_boundaries(monkeypatch):
    from llm_d_lens.agentic.facts import resolve_retrieval_hardware
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import InternalCaseOutcome
    from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters, QueryResponse
    from llm_d_lens.auth.contracts import Principal, RoleBinding

    discoveries, queries, internal_facts = [], [], []

    async def overview(cluster):
        discoveries.append(cluster.id)
        return {
            "hardware": {
                "gpuCount": 4,
                "availableGpuCount": 4,
                "vramBytes": 4 * 80 * 1024**3,
                "accelerators": [{"id": "nvidia", "gpuCount": 4, "models": ["NVIDIA-A100-80GB-PCIe"]}],
            },
            "discovery": len(discoveries),
        }

    class PublicQuery:
        def __init__(self, *, dao):
            self.dao = dao

        def query(self, query, _principal, **_kwargs):
            queries.append((query, self.dao.planning_facts))
            return QueryResponse(
                question=query.question,
                intent="comparison",
                filters=QueryFilters(model=query.filters.model, hardware=query.filters.hardware),
                status="abstained",
                answer="No public matches.",
                evidence=(),
                matched_items=0,
                returned_observations=0,
                truncated=False,
            )

    records = []

    async def internal(_model, _principal, *, facts):
        internal_facts.append(facts)
        return InternalCaseOutcome(status="available", records=tuple(records))

    async def supplementary(*_args):
        return []

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_retrieval_hardware", resolve_retrieval_hardware)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr(
        "llm_d_lens.agentic.facts.registry.require_cluster", lambda cluster_id: SimpleNamespace(id=cluster_id)
    )
    monkeypatch.setattr("llm_d_lens.agentic.facts.build_overview", overview)
    monkeypatch.setattr("llm_d_lens.agentic.facts._supplementary_evidence", supplementary)
    monkeypatch.setattr(
        "llm_d_lens.agentic.facts.PublicBenchmarkRecordQueryDao",
        lambda: SimpleNamespace(hardware_options=lambda _filters: ("NVIDIA-A100-80GB-PCIe",)),
    )
    monkeypatch.setattr("llm_d_lens.agentic.facts.PublicBenchmarkRecordQueryService", PublicQuery)
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_internal_cases", internal)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: None)
    request = _request()
    request.planning_facts = replace(
        request.planning_facts,
        model_config_dict={"quantization": "bf16"},
        workload_profile=WorkloadProfile(
            mean_input_tokens=1000, p95_input_tokens=1000, mean_output_tokens=200, shared_prefix_ratio=0.25
        ),
    )
    return SimpleNamespace(
        discoveries=discoveries,
        queries=queries,
        internal_facts=internal_facts,
        records=records,
        request=request,
        principal=Principal("admin-user", "admin", bindings=(RoleBinding("admin"),)),
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("outcome", ["failed", "succeeded"])
def test_create_refine_real_history_backend_guardrails(
    live_planning_boundaries, tmp_path, monkeypatch, backend, outcome
):
    from llm_d_lens.agentic.benchmark_evidence import BenchmarkRecord

    boundary = live_planning_boundaries
    boundary.records.extend(
        BenchmarkRecord.model_validate(
            {
                "benchmark_id": f"oom-{index}",
                "timestamp": datetime.now(UTC),
                "model": {"model_id": boundary.request.model, "quantization": "bf16"},
                "hardware": {
                    "accelerator": "A100",
                    "profile_id": "nvidia",
                    "form_factor": "PCIE",
                    "vram_per_gpu_gib": 80 * 10**9 / 2**30,
                    "gpu_count": 4,
                },
                "runtime": {"backend": backend},
                "workload": {
                    "mean_input_tokens": 1000,
                    "p95_input_tokens": 1000,
                    "mean_output_tokens": 200,
                    "shared_prefix_ratio": 0.25,
                },
                "configuration": {
                    "topology": "optimized-baseline",
                    "decode_tp": 1,
                    "decode_replicas": 1,
                    "optimizations": [""],
                },
                "metrics": {},
                "outcome": outcome,
                "failure_reason": "OOM" if outcome == "failed" else None,
            }
        )
        for index in range(3)
    )
    service = AgenticDeploymentService(historical_seed_mode="apply", snapshot_store=PlanningSnapshotStore(tmp_path))
    retrieved_seeds = []
    retrieve = service._historical_seed_retriever.retrieve

    def seeds(*args, **kwargs):
        result = retrieve(*args, **kwargs)
        retrieved_seeds.append(result)
        return result

    monkeypatch.setattr(service._historical_seed_retriever, "retrieve", seeds)
    run = asyncio.run(service.create(boundary.request, principal=boundary.principal))
    refined = asyncio.run(
        service.refine(
            run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency"), principal=boundary.principal
        )
    )
    for result in (run, refined):
        payload = json.loads((tmp_path / result.planning_snapshot_path).read_text())
        blocked = outcome == "failed" and backend == "vllm"
        scores = payload["resolved_planning_facts"]["performance_score_inputs"]
        score = next(item for item in scores if item["candidate_id"] == "optimized-baseline-tp1-r1")
        assert score["has_blocking_negative_evidence"] is blocked
        if blocked:
            assert result.selected_candidate.id != "optimized-baseline-tp1-r1"
            assert "optimized-baseline-tp1-r1" not in {candidate.id for candidate in result.candidates}
        assert bool(score["evidence_refs"]) is (backend == "vllm")
    assert len(retrieved_seeds) == 2
    assert all(bool(seeds) is (outcome == "succeeded" and backend == "vllm") for seeds in retrieved_seeds)
    for facts in boundary.internal_facts:
        assert facts.requirements.runtime.backend == "vllm"
        assert facts.requirements.runtime.backend_version is None
        assert facts.requirements.runtime.accelerator_runtime is None
        assert facts.requirements.runtime.driver_version is None
        assert facts.requirements.runtime.container_digest is None


def test_create_refine_share_one_discovery_but_approval_refreshes(live_planning_boundaries, monkeypatch):
    boundary = live_planning_boundaries
    service = AgenticDeploymentService()
    run = asyncio.run(service.create(boundary.request, principal=boundary.principal))
    assert boundary.discoveries == ["cluster-a"]
    assert run.planning_cluster_snapshot["discovery"] == 1
    refined = asyncio.run(
        service.refine(
            run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency"), principal=boundary.principal
        )
    )
    assert boundary.discoveries == ["cluster-a", "cluster-a"]
    assert refined.planning_cluster_snapshot["discovery"] == 2
    assert len(boundary.queries) == 2
    for (_, public_facts), internal_facts in zip(boundary.queries, boundary.internal_facts, strict=True):
        assert public_facts.requirements.hardware == internal_facts.requirements.hardware
    for (query, facts), result in zip(boundary.queries, (run, refined), strict=True):
        models = result.planning_cluster_snapshot["hardware"]["accelerators"][0]["models"]
        assert query.filters.hardware == facts.requirements.hardware.raw == models[0]

    another = asyncio.run(service.create(boundary.request, principal=boundary.principal))
    assert another.planning_cluster_snapshot["discovery"] == 3

    async def unavailable(_cluster):
        boundary.discoveries.append("approval")
        raise ValueError("fresh approval discovery")

    monkeypatch.setattr("llm_d_lens.agentic.facts.build_overview", unavailable)
    with pytest.raises(ValueError, match="fresh approval discovery"):
        asyncio.run(service.approve(run.id))
    assert boundary.discoveries == ["cluster-a", "cluster-a", "cluster-a", "approval"]


def test_decision_evidence_includes_selected_ai_raw_samples_once():
    facts = PlanningFacts(model_weight_gib=8, vram_per_gpu_gib=32, free_gpu_count=2)
    candidate = (
        DeterministicPlanner()
        .candidates(facts)[0]
        .model_copy(
            update={
                "evidence_ids": ["benchmark:shared", "benchmark:seed"],
                "historical_benchmarks": [
                    {"benchmark_id": "shared"},
                    {"benchmark_id": "raw"},
                ],
            }
        )
    )

    assert AgenticDeploymentService._decision_evidence_ids(facts, candidate) == [
        "benchmark:shared",
        "benchmark:seed",
        "benchmark:raw",
    ]


def test_create_falls_back_to_deterministic_planning_when_provider_fails(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class FailingProvider:
        async def recommend(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: FailingProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "deterministic"
    assert run.planner_fallback_reason is not None
    assert run.decision is not None
    assert run.decision.action == "select_candidate"


def test_create_falls_back_when_provider_selects_an_unvalidated_candidate(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class InvalidProvider:
        async def recommend(self, *_args, **_kwargs):
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate",
                candidate_id="external-invented-candidate",
                confidence=0.8,
                rationale="provider choice",
            ), []

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: InvalidProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "deterministic"
    assert run.planner_fallback_reason == "invalid_response"
    assert run.selected_candidate.id != "external-invented-candidate"
    assert run.selected_candidate.id in {candidate.id for candidate in run.candidates}


def test_create_prefers_openai_compatible_planning_when_provider_is_available(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class AvailableProvider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = next(candidate for candidate in candidates if candidate.deployable)
            from llm_d_lens.agentic.planner import PlannerDecision

            decision = PlannerDecision(
                action="select_candidate",
                candidate_id=selected.id,
                confidence=0.8,
                rationale="provider choice",
            )
            scored = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates
                if candidate.deployable
            ]
            return decision, scored

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: AvailableProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "openai-compatible"
    assert run.decision is not None
    assert run.decision.rationale == "provider choice"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert run.decision_metadata.planner_model is None
    assert 1 <= len(run.candidates) <= 3
    assert all(candidate.deployable and candidate.score_source == "openai-compatible" for candidate in run.candidates)


def test_create_shows_three_of_provider_selected_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 1 - index / 10, "score_source": "openai-compatible"})
                for index, candidate in enumerate(candidates[:10])
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="provider choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())
    run = asyncio.run(service.create(_request()))
    assert len(run.candidates) == 3


def test_create_uses_valid_ai_generated_candidates_without_falling_back(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    async def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("successful AI generation must not pre-resolve facts internally")

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", unexpected_resolve)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits.",
                    },
                    {
                        "provider_ref": "pd-disaggregation",
                        "replicas": 2,
                        "tensor_parallel_size": 2,
                        "prefill_replicas": 2,
                        "prefill_tensor_parallel_size": 2,
                        "rationale": "Too large.",
                    },
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator == "ai-mcp"
    assert run.generator_model == "generator-model"
    assert run.generator_fallback_reason is None
    assert [proposal["provider_ref"] for proposal in run.generator_candidate_proposals] == [
        "optimized-baseline",
        "pd-disaggregation",
    ]
    assert run.generator_candidate_proposals[1]["rationale"] == "Too large."
    assert [candidate.id for candidate in run.candidates] == ["optimized-baseline-tp1-r1"]
    assert len(run.rejected_candidates) == 1
    assert run.rejected_candidates[0].rejection_reasons == ["insufficient free accelerator cards"]


def test_create_scores_successful_ai_generated_candidates_without_resource_policy(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    captured: dict = {}
    score_calls: list[list[str]] = []
    original_score = DeterministicPlanner.score

    class Response:
        status_code = 200

        @property
        def text(self):
            return json.dumps(self.json())

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"candidate_id":"optimized-baseline-tp2-r1",'
                                '"candidate_ids":["optimized-baseline-tp2-r1","optimized-baseline-tp1-r1"],'
                                '"scores":{"optimized-baseline-tp2-r1":0.9,"optimized-baseline-tp1-r1":0.8},'
                                '"confidence":0.9,"rationale":"AIC-backed choice"}'
                            )
                        }
                    }
                ]
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, _url, **kwargs):
            captured.update(kwargs)
            return Response()

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits.",
                    },
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 2,
                        "rationale": "Scales.",
                    },
                ]
            },
            resolved=_resolved_facts(
                free_gpu_count=2,
                aic_predictions=(
                    AICCandidatePrediction(
                        mode="agg", tensor_parallel_size=1, replicas=1, throughput_tokens_per_sec=100
                    ),
                    AICCandidatePrediction(
                        mode="agg", tensor_parallel_size=2, replicas=1, throughput_tokens_per_sec=200
                    ),
                ),
            ),
        )

    def score_spy(self, facts, candidates):
        score_calls.append([candidate.id for candidate in candidates])
        return original_score(self, facts, candidates)

    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id",
        lambda _id: OpenAICompatiblePlanner("http://provider", "planner"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.planner.DeterministicPlanner.score", score_spy)
    monkeypatch.setattr("llm_d_lens.ai_providers.client.httpx.AsyncClient", lambda **_kwargs: Client())

    async def validate_provider(_url):
        pass

    monkeypatch.setattr("llm_d_lens.ai_providers.service.validate_provider_endpoint", validate_provider)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    planner_context = (
        captured["json"]["messages"][1]["content"].split("Planning context (JSON):\n", 1)[1].partition("\n\n")[0]
    )
    supplied = json.loads(planner_context)["valid_candidates"]
    supplied_by_id = {candidate["id"]: candidate for candidate in supplied}
    assert run.generator == "ai-mcp"
    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert score_calls
    assert score_calls[0] == ["optimized-baseline-tp1-r1", "optimized-baseline-tp2-r1"]
    assert "resource_policy" not in supplied_by_id["optimized-baseline-tp1-r1"]
    assert "resource_policy" not in supplied_by_id["optimized-baseline-tp2-r1"]


@pytest.mark.parametrize("refine", [False, True])
def test_ai_ranking_omits_aggregate_inputs_but_honors_failure_blocks(monkeypatch, refine):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    captured = {}

    class Provider:
        async def recommend(self, facts, candidates, **_kwargs):
            captured["performance_score_inputs"] = facts.performance_score_inputs
            captured["candidate_ids"] = {candidate.id for candidate in candidates}
            planner = DeterministicPlanner()
            return planner.decide(facts, candidates), planner.score(facts, candidates)

    def score_inputs(_self, _model, _facts, candidates, *, records):
        assert records == []
        baseline = next(candidate for candidate in candidates if candidate.id == "optimized-baseline-tp1-r1")
        return (
            PerformanceScoreInput(
                candidate_id=baseline.id,
                source="benchmark",
                status="measured",
                confidence=1,
                effective_sample_size=3,
                has_blocking_negative_evidence=True,
            ),
        )

    deterministic_calls = []
    original_decide = DeterministicPlanner.decide

    def decide_spy(self, facts, candidates):
        deterministic_calls.append(facts.performance_score_inputs)
        return original_decide(self, facts, candidates)

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.BenchmarkPerformanceEvidence.score_inputs", score_inputs)
    monkeypatch.setattr("llm_d_lens.agentic.planner.DeterministicPlanner.decide", decide_spy)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))
    if refine:
        run = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer latency")))

    assert run.decision_metadata is not None
    assert run.decision_metadata.planner == "openai-compatible"
    assert deterministic_calls[0][0].has_blocking_negative_evidence
    assert captured["performance_score_inputs"] == ()
    assert "optimized-baseline-tp1-r1" not in captured["candidate_ids"]


def test_relaxed_candidate_gets_exact_estimate_before_scoring(monkeypatch):
    calls = []

    async def estimated(request):
        calls.append(request)
        return {"mode": "agg", "tp": 2, "replicas": 1, "ttft_ms": 180, "tpot_ms": 22}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)
    facts = replace(_resolved_facts().facts, ttft_slo_ms=30)
    candidate = (
        DeterministicPlanner()
        .candidates(facts)[0]
        .model_copy(
            update={
                "id": "optimized-baseline-tp2-r1",
                "provider_ref": "optimized-baseline",
                "tensor_parallel_size": 2,
                "replicas": 1,
                "required_gpus": 2,
            }
        )
    )
    candidates, enriched = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            [candidate],
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {
                        "workload": {"isl": 4096, "osl": 256, "ttftMs": 30, "tpotMs": 10},
                        "searchConfig": {"aicSystemName": "h100_sxm", "aicBackendName": "vllm"},
                    },
                }
            ],
        )
    )

    assert calls[0].gpu_count == 2
    assert calls[0].aic_system_name == "h100_sxm"
    assert calls[0].ttft_target_ms is None
    assert calls[0].tpot_target_ms is None
    assert candidates[0].performance_estimate["ttft_ms"] == 180
    assert candidates[0].performance_estimate_source == "aic_estimate"
    assert DeterministicPlanner()._slo_assessment(enriched, candidates[0])[0] == "not_satisfied"
    assert OpenAICompatiblePlanner._candidate_context(enriched, candidates[0])["performance_estimate"] == {
        "ttft_ms": 180,
        "tpot_ms": 22,
        "throughput_tokens_per_sec": None,
    }


def test_missing_aic_predictions_estimate_each_supported_candidate_for_ai_scoring(monkeypatch):
    calls = []

    async def estimated(request):
        calls.append(request)
        return {"ttft_ms": 180 + len(calls), "tpot_ms": 22, "throughput_tokens_per_sec": 1200}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)
    facts = _resolved_facts(free_gpu_count=4).facts
    candidates = DeterministicPlanner().candidates(facts)
    supported = [
        next(candidate for candidate in candidates if candidate.provider_ref == provider_ref)
        for provider_ref in ("optimized-baseline", "pd-disaggregation")
    ]
    enriched_candidates, enriched_facts = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            supported,
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {
                        "workload": {"isl": 1024, "osl": 256},
                        "searchConfig": {"aicSystemName": "h100_sxm"},
                    },
                }
            ],
        )
    )

    assert len(calls) == len(supported)
    for index, (candidate, request) in enumerate(zip(enriched_candidates, calls, strict=True), start=1):
        assert request.gpu_count == candidate.required_gpus
        assert request.tp == candidate.tensor_parallel_size
        assert request.replicas == candidate.replicas
        if candidate.provider_ref == "pd-disaggregation":
            assert request.prefill_tp == candidate.prefill_tensor_parallel_size
            assert request.prefill_replicas == candidate.prefill_replicas
        context = OpenAICompatiblePlanner._candidate_context(enriched_facts, candidate)
        assert context["performance_estimate_source"] == "aic_estimate"
        assert context["performance_estimate"] == {
            "ttft_ms": 180 + index,
            "tpot_ms": 22,
            "throughput_tokens_per_sec": 1200,
        }


def test_relaxed_candidate_keeps_unknown_performance_when_estimate_fails(monkeypatch):
    from llm_d_lens.aic.service import AICError

    async def unavailable(_request):
        raise AICError("no matching topology")

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", unavailable)
    facts = _resolved_facts().facts
    candidate = DeterministicPlanner().candidates(facts)[0]
    candidates, enriched = asyncio.run(
        AgenticDeploymentService._estimate_missing_topologies(
            "Qwen/Qwen3-8B",
            [candidate],
            facts,
            [
                {
                    "tool": "search_candidates",
                    "arguments": {"workload": {}, "searchConfig": {"aicSystemName": "h100_sxm"}},
                }
            ],
        )
    )

    assert candidates[0].performance_estimate is None
    assert enriched.aic_predictions == ()


def test_create_falls_back_only_when_all_ai_generated_candidates_are_invalid(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 8,
                        "tensor_parallel_size": 8,
                        "rationale": "Exceeds capacity.",
                    }
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    progress = []

    async def on_progress(event):
        progress.append(event)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1"), on_progress=on_progress))

    assert run.generator == "deterministic"
    assert run.generator_fallback_reason == "all_ai_candidates_invalid"
    assert run.candidates
    assert run.rejected_candidates[0].id == "optimized-baseline-tp8-r8"
    assert any(
        event["phase"] == "generation"
        and event["status"] == "fallback"
        and "deterministic generation" in event["message"]
        for event in progress
    )


@pytest.mark.parametrize(
    ("error_message", "reason", "detail"),
    [
        ('MCP tool "search_candidates" failed: unavailable', "mcp_unavailable", None),
        (
            "AI candidate generation returned too many tool calls in round 1: 5 (limit 4)",
            "invalid_response",
            "AI candidate generation returned too many tool calls in round 1: 5 (limit 4)",
        ),
    ],
)
def test_create_falls_back_when_ai_generator_mcp_is_unavailable(monkeypatch, tmp_path, error_message, reason, detail):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(*_args, **_kwargs):
        from llm_d_lens.agentic.generator import CandidateGenerationError

        raise CandidateGenerationError(error_message)

    estimate_requests = []

    async def estimated(request):
        estimate_requests.append(request)
        return {"ttft_ms": 150, "tpot_ms": 20, "throughput_tokens_per_sec": 100}

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)
    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator == "deterministic"
    assert run.generator_fallback_reason == reason
    assert run.generator_fallback_detail == detail
    assert estimate_requests == []
    assert all(candidate.performance_estimate_source != "aic_estimate" for candidate in run.candidates)
    snapshot = json.loads((tmp_path / run.planning_snapshot_path).read_text(encoding="utf-8"))
    assert snapshot["generator"]["fallback_detail"] == detail
    assert all(
        candidate["performance_estimate_source"] != "aic_estimate" for candidate in snapshot["candidate_catalog"]
    )


def test_missing_requested_provider_estimates_fallback_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: None)
    estimate_requests = []

    async def estimated(request):
        estimate_requests.append(request)
        return {"ttft_ms": 150, "tpot_ms": 20}

    monkeypatch.setattr("llm_d_lens.agentic.service.estimate", estimated)

    run = asyncio.run(service.create(_request(ai_provider_id="provider-1")))

    assert run.generator_fallback_reason == "provider_unavailable"
    assert estimate_requests == []
    assert all(candidate.performance_estimate_source != "aic_estimate" for candidate in run.candidates)


def test_generator_provider_http_error_is_not_classified_as_invalid_candidate_response():
    from llm_d_lens.agentic.generator import CandidateGenerationError

    error = CandidateGenerationError(
        "AI candidate generation failed: OpenAI-compatible provider request failed: "
        "Client error '400 Bad Request' for url 'https://api.deepseek.com/v1/chat/completions'"
    )

    assert AgenticDeploymentService._generator_failure_reason(error) == "provider_unavailable"


def test_generator_invalid_initial_tool_batch_is_classified_as_invalid_response():
    from llm_d_lens.agentic.generator import CandidateGenerationError

    error = CandidateGenerationError(
        "AI candidate generation must first call only get_cluster_overview and AIC search_candidates"
    )

    assert AgenticDeploymentService._generator_failure_reason(error) == "invalid_response"


def test_refine_keeps_provider_selected_candidates(monkeypatch, tmp_path):
    service = AgenticDeploymentService(snapshot_store=PlanningSnapshotStore(tmp_path))
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 1 - index / 10, "score_source": "openai-compatible"})
                for index, candidate in enumerate(candidates[:10])
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="provider choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())
    run = asyncio.run(service.create(_request()))
    created_path = tmp_path / run.planning_snapshot_path
    created_payload = created_path.read_text(encoding="utf-8")
    from llm_d_lens.agentic.models import AgenticCandidateRefinementRequest

    refined = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="more capacity")))
    assert 1 <= len(refined.candidates) <= 10
    refined_path = tmp_path / refined.planning_snapshot_path
    assert refined_path != created_path
    assert json.loads(created_payload)["event"] == "created"
    assert json.loads(refined_path.read_text(encoding="utf-8"))["event"] == "refined"
    assert created_path.read_text(encoding="utf-8") == created_payload


@pytest.mark.parametrize("planner_fails", [False, True])
def test_refine_regenerates_candidates_with_ai_and_mcp(monkeypatch, tmp_path, planner_fails):
    from llm_d_lens.agentic.planner import OpenAIPlannerError
    from llm_d_lens.ai_providers.client import OpenAICompatibleClient

    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    prompts = []
    planning_lengths = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, endpoint, *, json, headers):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "complete raw response"}}]},
                request=httpx.Request("POST", endpoint),
            )

    async def validate_endpoint(_url):
        pass

    monkeypatch.setattr("llm_d_lens.ai_providers.service.validate_provider_endpoint", validate_endpoint)
    monkeypatch.setattr("llm_d_lens.ai_providers.client.httpx.AsyncClient", lambda **_kwargs: Client())

    class Provider:
        settings = OpenAIPlannerSettings(base_url="http://provider", model="generator-model")

        async def recommend(self, facts, candidates, **_kwargs):
            from llm_d_lens.agentic.planner import DeterministicPlanner

            await OpenAICompatibleClient("https://api.openai.com/", "planner-model").complete(
                system_prompt="full planner system", user_content=_kwargs["operator_prompt"]
            )
            if planner_fails:
                raise OpenAIPlannerError("invalid recorded response")
            return DeterministicPlanner().decide(facts, candidates), DeterministicPlanner().score(facts, candidates)

    async def generate(_self, _cluster_id, _model, facts, operator_prompt, on_progress=None, *, hardware_snapshot=None):
        from llm_d_lens.agentic.generator import CandidateGenerationResult

        await OpenAICompatibleClient("https://api.openai.com/", "generator-model").complete(
            system_prompt="full generator system", user_content=operator_prompt
        )
        prompts.append(operator_prompt)
        planning_lengths.append(facts.context_length)
        return CandidateGenerationResult(
            proposals={
                "candidates": [
                    {
                        "provider_ref": "optimized-baseline",
                        "replicas": 1,
                        "tensor_parallel_size": 1,
                        "rationale": "Fits the refreshed preference.",
                    }
                ]
            },
            resolved=_resolved_facts(),
        )

    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_provider_id", lambda _id: Provider())
    monkeypatch.setattr("llm_d_lens.agentic.service.AICandidateGenerator.generate", generate)

    run = asyncio.run(
        service.create(_request(ai_provider_id="provider-1", planner_prompt="balanced", max_model_len=32768))
    )
    original_path = tmp_path / run.planning_snapshot_path
    original_payload = original_path.read_text()
    assert run.generator_requests[0].request["messages"][1]["content"] == "balanced"
    assert run.planner_requests[0].request["messages"][1]["content"] == "balanced"
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="prefer fewer replicas"),
        )
    )

    assert prompts == ["balanced", "prefer fewer replicas"]
    assert planning_lengths == [32768, 32768]
    assert refined.generator == "ai-mcp"
    assert refined.generator_model == "generator-model"
    assert refined.generator_fallback_reason is None
    assert refined.planning_trace[-1]["candidate_id"] == refined.selected_candidate.id
    assert next(item for item in refined.planning_trace if item["phase"] == "deterministic_ranking")["ranking"]
    payload = json.loads((tmp_path / refined.planning_snapshot_path).read_text())
    assert payload["request"]["planner_prompt"] == "prefer fewer replicas"
    for stage in ("generator", "planner"):
        record = payload[stage]["requests"][0]
        assert record["request"]["messages"][1]["content"] == "prefer fewer replicas"
        assert "complete raw response" in record["response_body"]
        assert record["response_status"] == 200
        assert record["finished_at"] is not None
    assert refined.planner_fallback_reason == ("invalid_response" if planner_fails else None)
    assert original_path.read_text() == original_payload
    assert len(list((tmp_path / run.id / "requests").glob("*.json"))) == 4


def test_refine_falls_back_to_deterministic_planning_when_provider_fails(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)

    class FailingProvider:
        async def recommend(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: FailingProvider()
    )
    run = asyncio.run(service.create(_request()))
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="distributed"),
        )
    )

    assert refined.planner == "deterministic"
    assert refined.planner_fallback_reason is not None
    assert refined.decision_metadata.planner == "deterministic"
    assert all(
        candidate.score_source == "deterministic" and candidate.score is not None for candidate in refined.candidates
    )


def test_refine_without_ai_provider_uses_deterministic_planning(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    async def resolve_facts(_cluster_id, _model, facts, **_kwargs):
        resolved = _resolved_facts()
        resolved.facts = replace(
            resolved.facts,
            operator_preference=facts.operator_preference,
            gpu_memory_utilization=facts.gpu_memory_utilization,
        )
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve_facts)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: None)

    run = asyncio.run(service.create(_request(planner_prompt="balanced", gpu_memory_utilization=0.6)))
    refined = asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="distributed")))

    assert refined.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert refined.generator == "deterministic"
    assert refined.generator_fallback_reason is None
    assert refined.planner == "deterministic"
    assert refined.planner_fallback_reason is None
    assert refined.request.planner_prompt == "distributed"
    assert refined.selected_candidate.id == refined.candidates[0].id
    assert refined.selected_candidate.provider_ref == "pd-disaggregation"
    assert refined.selected_candidate.gpu_memory_utilization == 0.6
    assert all(candidate.gpu_memory_utilization == 0.6 for candidate in refined.candidates)
    assert all(
        candidate.score_source == "deterministic" and candidate.score is not None for candidate in refined.candidates
    )


def test_refine_allows_external_ranking_of_exact_aic_candidates(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    resolved = _resolved_facts()
    resolved.facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        aic_predictions=(
            AICCandidatePrediction(
                mode="agg",
                tensor_parallel_size=1,
                replicas=1,
                ttft_ms=300,
                tpot_ms=25,
                throughput_tokens_per_sec=700,
            ),
        ),
    )

    async def resolve(*_args, **_kwargs):
        return resolved

    class Provider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates[:2]
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="preference choice"
            ), selected

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: Provider())

    run = asyncio.run(service.create(_request()))
    refined = asyncio.run(
        service.refine(
            run.id,
            AgenticCandidateRefinementRequest(planner_prompt="distributed"),
        )
    )

    assert refined.planner == "openai-compatible"
    assert refined.decision.rationale == "preference choice"


def test_create_allows_external_planning_when_an_exact_aic_recommendation_exists(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    resolved = _resolved_facts()
    resolved.facts = PlanningFacts(
        model_weight_gib=8,
        vram_per_gpu_gib=32,
        free_gpu_count=4,
        aic_predictions=(
            AICCandidatePrediction(
                mode="agg",
                tensor_parallel_size=1,
                replicas=1,
                ttft_ms=300,
                tpot_ms=25,
                throughput_tokens_per_sec=700,
            ),
        ),
    )

    async def resolve(*_args, **_kwargs):
        return resolved

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)

    class AvailableProvider:
        async def recommend(self, _facts, candidates, **_kwargs):
            selected = [
                candidate.model_copy(update={"score": 0.9, "score_source": "openai-compatible"})
                for candidate in candidates[:2]
            ]
            from llm_d_lens.agentic.planner import PlannerDecision

            return PlannerDecision(
                action="select_candidate", candidate_id=selected[0].id, confidence=0.8, rationale="preference choice"
            ), selected

    monkeypatch.setattr(
        "llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment",
        lambda: AvailableProvider(),
    )

    run = asyncio.run(service.create(_request()))

    assert run.planner == "openai-compatible"
    assert run.selected_candidate.id == "optimized-baseline-tp1-r1"
    assert all(candidate.score_source == "openai-compatible" for candidate in run.candidates)
    assert len(run.candidates) == 2


def test_create_requires_a_model_cache_volume(monkeypatch):
    service = AgenticDeploymentService()
    monkeypatch.setattr(
        service,
        "_validate_model_cache",
        _ORIGINAL_VALIDATE_MODEL_CACHE,
    )
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )

    with pytest.raises(ValidationError, match="storage_volume_id"):
        asyncio.run(service.create(_request(storage_volume_id="")))


@pytest.mark.parametrize("cache_status", [None, "downloading"])
def test_planning_defers_model_readiness_until_approval(monkeypatch, cache_status):
    from llm_d_lens.model_cache.contracts import ModelCacheEntry
    from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose

    service = AgenticDeploymentService()
    monkeypatch.setattr(service, "_validate_model_cache", _ORIGINAL_VALIDATE_MODEL_CACHE)
    monkeypatch.setattr(
        "llm_d_lens.agentic.service.require_active_session",
        lambda session_id: SimpleNamespace(id=session_id, server_id="cluster-a"),
    )
    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", _resolve_facts)
    monkeypatch.setattr("llm_d_lens.agentic.service.OpenAICompatiblePlanner.from_environment", lambda: None)
    calls = []

    async def get_ready_volume(*_args, **_kwargs):
        return SimpleNamespace(purposes=[StorageVolumePurpose.MODEL_CACHE], kind=StorageVolumeKind.NFS)

    class CacheService:
        async def list(self, **kwargs):
            calls.append(kwargs)
            if cache_status is None:
                return []
            return [
                ModelCacheEntry.model_validate(
                    {
                        "clusterId": "cluster-a",
                        "storageVolumeId": "storage-model-cache",
                        "status": cache_status,
                        "cachePath": "hub/models--Qwen--Qwen3-8B",
                        "source": {"kind": "huggingface", "huggingface": {"repoId": "Qwen/Qwen3-8B"}},
                    }
                )
            ]

    monkeypatch.setattr("llm_d_lens.storage.service.get_ready_volume", get_ready_volume)
    monkeypatch.setattr("llm_d_lens.model_cache.service.default_service", lambda: CacheService())
    run = asyncio.run(service.create(_request()))
    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert calls == []
    asyncio.run(service.refine(run.id, AgenticCandidateRefinementRequest(planner_prompt="prefer fewer GPUs")))
    assert calls == []
    with pytest.raises(ValueError, match="selected model is not ready"):
        asyncio.run(service.approve(run.id))
    assert calls == [{"cluster_id": "cluster-a", "storage_volume_id": "storage-model-cache"}]
    assert run.status == AgenticDeploymentStatus.AWAITING_APPROVAL
    assert run.deployment_run_id is None


@pytest.mark.parametrize(
    ("token_source", "expected"),
    [
        (
            {"mode": "existing-secret", "namespace": "models", "name": "hf-token"},
            {"mode": "existing-secret", "sourceNamespace": "models", "sourceName": "hf-token"},
        ),
        # Model Cache ``host`` is best-effort; Deploy ``host`` is not, and
        # mapping it would disable the cluster default Secret fallback.
        ({"mode": "host"}, None),
        ({"mode": "none"}, None),
    ],
    ids=["existing-secret", "host", "none"],
)
def test_validate_model_cache_maps_only_existing_secret_token_source(monkeypatch, token_source, expected):
    from llm_d_lens.model_cache.contracts import ModelCacheEntry
    from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose

    entry = ModelCacheEntry.model_validate(
        {
            "clusterId": "cluster-a",
            "storageVolumeId": "vol-1",
            "status": "ready",
            "cachePath": "hub/models--Qwen--Qwen3-8B",
            "source": {"kind": "huggingface", "huggingface": {"repoId": "Qwen/Qwen3-8B", "revision": "main"}},
            "tokenSource": token_source,
        }
    )
    volume = SimpleNamespace(purposes=[StorageVolumePurpose.MODEL_CACHE], kind=StorageVolumeKind.NFS)

    async def get_ready_volume(*_args, **_kwargs):
        return volume

    class _CacheService:
        async def list(self, **_kwargs):
            return [entry]

    monkeypatch.setattr("llm_d_lens.storage.service.get_ready_volume", get_ready_volume)
    monkeypatch.setattr("llm_d_lens.model_cache.service.default_service", lambda: _CacheService())

    result = asyncio.run(
        _ORIGINAL_VALIDATE_MODEL_CACHE(
            _request(storage_type="model-cache", storage_volume_id="vol-1"),
            "cluster-a",
        )
    )

    assert (result.model_dump(mode="json", by_alias=True) if result else None) == expected


def test_approve_rejects_a_candidate_when_live_cluster_capacity_has_changed(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)
    resolved = iter([_resolved_facts(free_gpu_count=4), _resolved_facts(free_gpu_count=0)])

    async def resolve(*_args, **_kwargs):
        return next(resolved)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_request()))

    with pytest.raises(ValueError, match="sufficient free accelerator cards"):
        asyncio.run(service.approve(run.id))


def test_approve_rejects_a_candidate_when_preflight_capacity_validation_fails(monkeypatch):
    service = AgenticDeploymentService()
    session = SimpleNamespace(id="a" * 36, server_id="cluster-a")
    monkeypatch.setattr("llm_d_lens.agentic.service.require_active_session", lambda _session_id: session)

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    }
    # First call: cluster has 80GB VRAM, fits
    normal = _resolved_facts(free_gpu_count=4)
    # Second call (during approve): cluster VRAM drops to 8GB, impossible to fit 20GB weights + activation
    oom_facts = _resolved_facts(free_gpu_count=4)
    oom_facts.facts = replace(
        oom_facts.facts,
        vram_per_gpu_gib=8.0,
        model_name="meta-llama/Llama-3-8B",
        model_config_dict=llama_cfg,
    )

    resolved = iter([normal, oom_facts])

    async def resolve(*_args, **_kwargs):
        return next(resolved)

    monkeypatch.setattr("llm_d_lens.agentic.service.resolve_planning_facts", resolve)
    run = asyncio.run(service.create(_request()))

    with pytest.raises(
        ValueError, match="(fails pre-flight capacity validation|satisfies current resource constraints)"
    ):
        asyncio.run(service.approve(run.id))
