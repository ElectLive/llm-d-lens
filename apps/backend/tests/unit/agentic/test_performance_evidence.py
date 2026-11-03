"""Tests for benchmark evidence adaptation into deterministic score inputs."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkMetrics,
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
)
from llm_d_lens.agentic.benchmark_evidence import (
    WorkloadProfile as BenchmarkWorkloadProfile,
)
from llm_d_lens.agentic.benchmark_store import JsonBenchmarkRecordStore
from llm_d_lens.agentic.performance_evidence import BenchmarkPerformanceEvidence
from llm_d_lens.agentic.planner import (
    DeterministicPlanner,
    OpenAICompatiblePlanner,
    PlanningFacts,
    WorkloadProfile,
)
from llm_d_lens.agentic.planning_requirements import (
    HardwareIdentity,
    ModelIdentity,
    PlanningRequirements,
    RuntimeIdentity,
    normalize_requirements,
)


def _requirements():
    return PlanningRequirements(
        hardware=HardwareIdentity(device_model="Gaudi3", memory_gib=96, form_factor="OAM", device_count=2),
        model=ModelIdentity(model_id="Qwen/Qwen3-8B", architecture="dense", quantization="bf16"),
        runtime=RuntimeIdentity(backend="vllm", backend_version="0.8.5", accelerator_runtime="1.20"),
    )


def test_benchmark_performance_evidence_maps_matching_records_to_candidate_inputs(tmp_path):
    facts = PlanningFacts(
        model_weight_gib=16,
        vram_per_gpu_gib=96,
        free_gpu_count=2,
        requirements=_requirements(),
        workload_profile=WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.7,
        ),
    )
    candidate = next(
        item for item in DeterministicPlanner().candidates(facts) if item.id == "optimized-baseline-tp1-r1"
    )
    record = BenchmarkRecord(
        benchmark_id="bench-1",
        timestamp=datetime.now(UTC),
        outcome="succeeded",
        model=ModelDescriptor(
            model_id="Qwen/Qwen3-8B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=8,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-8B",
        ),
        hardware=HardwareDescriptor(accelerator="Gaudi3", vram_per_gpu_gib=96, gpu_count=2, form_factor="OAM"),
        runtime=RuntimeFingerprint(backend="vllm", backend_version="0.8.5", accelerator_runtime="1.20"),
        workload=BenchmarkWorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.7,
            prefill_fraction=0,
        ),
        configuration=CandidateConfiguration(
            topology="optimized-baseline", decode_tp=1, decode_replicas=1, optimizations=("",)
        ),
        metrics=BenchmarkMetrics(ttft_p95_ms=300, tpot_p95_ms=25, throughput_tokens_per_s=700, error_rate=0.01),
    )
    evidence = BenchmarkPerformanceEvidence(JsonBenchmarkRecordStore(tmp_path)).score_inputs(
        "Qwen/Qwen3-8B",
        facts,
        [candidate],
        records=[record],
    )

    assert len(evidence) == 1
    assert evidence[0].candidate_id == candidate.id
    assert evidence[0].status == "interpolated"
    assert evidence[0].ttft_p95_ms == pytest.approx(300)
    assert evidence[0].evidence_refs == ("benchmark:bench-1",)
    sample = evidence[0].weighted_samples[0]
    assert sample.benchmark_id == "bench-1"
    assert sample.match_confidence == pytest.approx(0.897142857, abs=0.001)
    assert sample.ttft_p95_ms == 300
    assert sample.outcome == "succeeded"
    assert sample.topology == "optimized-baseline"


def test_benchmark_performance_evidence_blocks_repeated_matching_failures(tmp_path):
    facts = PlanningFacts(
        model_weight_gib=16,
        vram_per_gpu_gib=96,
        free_gpu_count=2,
        requirements=_requirements(),
        workload_profile=WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=4096,
            mean_output_tokens=256,
            concurrency=1,
            request_rate=1,
            shared_prefix_ratio=0,
        ),
    )
    candidate = next(
        item for item in DeterministicPlanner().candidates(facts) if item.id == "optimized-baseline-tp1-r1"
    )
    common = {
        "timestamp": datetime.now(UTC),
        "outcome": "failed",
        "model": ModelDescriptor(
            model_id="Qwen/Qwen3-8B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=8,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-8B",
        ),
        "hardware": HardwareDescriptor(accelerator="Gaudi3", vram_per_gpu_gib=96, gpu_count=2, form_factor="OAM"),
        "runtime": RuntimeFingerprint(backend="vllm", backend_version="0.8.5", accelerator_runtime="1.20"),
        "workload": BenchmarkWorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=4096,
            mean_output_tokens=256,
            concurrency=1,
            request_rate=1,
            shared_prefix_ratio=0,
            prefill_fraction=0,
        ),
        "configuration": CandidateConfiguration(
            topology="optimized-baseline", decode_tp=1, decode_replicas=1, optimizations=("",)
        ),
        "failure_reason": "OOM",
    }
    records = [BenchmarkRecord(benchmark_id=f"failed-{index}", **common) for index in range(3)]
    evidence = BenchmarkPerformanceEvidence(JsonBenchmarkRecordStore(tmp_path)).score_inputs(
        "Qwen/Qwen3-8B",
        facts,
        [candidate],
        records=records,
    )

    assert evidence[0].has_blocking_negative_evidence


def test_persisted_benchmark_history_flows_to_planner_and_ai_context(tmp_path):
    facts = PlanningFacts(
        model_weight_gib=16,
        vram_per_gpu_gib=96,
        free_gpu_count=2,
        requirements=_requirements(),
        ttft_slo_ms=500,
        tpot_slo_ms=50,
        workload_profile=WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.7,
        ),
    )
    deterministic_planner = DeterministicPlanner()
    candidates = deterministic_planner.candidates(facts)
    baseline = next(candidate for candidate in candidates if candidate.id == "optimized-baseline-tp1-r1")
    pd = next(candidate for candidate in candidates if candidate.provider_ref == "pd-disaggregation")
    store = JsonBenchmarkRecordStore(tmp_path)
    common = {
        "timestamp": datetime.now(UTC),
        "model": ModelDescriptor(
            model_id="Qwen/Qwen3-8B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=8,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-8B",
        ),
        "hardware": HardwareDescriptor(accelerator="Gaudi3", vram_per_gpu_gib=96, gpu_count=2, form_factor="OAM"),
        "runtime": RuntimeFingerprint(backend="vllm", backend_version="0.8.5", accelerator_runtime="1.20"),
        "workload": BenchmarkWorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.7,
            prefill_fraction=0,
        ),
    }
    baseline_configuration = CandidateConfiguration(
        topology=baseline.provider_ref,
        decode_tp=baseline.tensor_parallel_size,
        decode_replicas=baseline.replicas,
        optimizations=(baseline.guide_variant or "",),
    )
    pd_configuration = CandidateConfiguration(
        topology=pd.provider_ref,
        decode_tp=pd.tensor_parallel_size,
        decode_replicas=pd.replicas,
        prefill_tp=pd.prefill_tensor_parallel_size,
        prefill_replicas=pd.prefill_replicas,
        optimizations=(pd.guide_variant or "",),
    )
    for index in range(3):
        store.save(
            BenchmarkRecord(
                benchmark_id=f"baseline-success-{index}",
                outcome="succeeded",
                configuration=baseline_configuration,
                metrics=BenchmarkMetrics(
                    ttft_p95_ms=300,
                    tpot_p95_ms=25,
                    throughput_tokens_per_s=700,
                    error_rate=0.01,
                ),
                **common,
            )
        )
        store.save(
            BenchmarkRecord(
                benchmark_id=f"pd-failure-{index}",
                outcome="failed",
                configuration=pd_configuration,
                failure_reason="OOM",
                **common,
            )
        )
    store.save(
        BenchmarkRecord(
            benchmark_id="ignored-other-accelerator",
            outcome="succeeded",
            configuration=baseline_configuration,
            hardware=HardwareDescriptor(accelerator="other", vram_per_gpu_gib=96, gpu_count=2),
            metrics=BenchmarkMetrics(ttft_p95_ms=1, tpot_p95_ms=1, throughput_tokens_per_s=9999),
            **{key: value for key, value in common.items() if key != "hardware"},
        )
    )

    score_inputs = BenchmarkPerformanceEvidence(store).score_inputs(
        "Qwen/Qwen3-8B",
        facts,
        [baseline, pd],
    )
    inputs_by_candidate = {item.candidate_id: item for item in score_inputs}
    baseline_evidence = inputs_by_candidate[baseline.id]
    pd_evidence = inputs_by_candidate[pd.id]

    assert baseline_evidence.status == "interpolated"
    assert baseline_evidence.ttft_p95_ms == pytest.approx(300)
    assert baseline_evidence.tpot_p95_ms == pytest.approx(25)
    assert baseline_evidence.throughput_tokens_per_s == pytest.approx(700)
    assert baseline_evidence.evidence_refs == tuple(f"benchmark:baseline-success-{index}" for index in range(3))
    assert [sample.benchmark_id for sample in baseline_evidence.weighted_samples] == [
        f"baseline-success-{index}" for index in range(3)
    ]
    assert pd_evidence.has_blocking_negative_evidence
    assert pd_evidence.evidence_refs == tuple(f"benchmark:pd-failure-{index}" for index in range(3))

    facts_with_evidence = replace(
        facts,
        performance_score_inputs=score_inputs,
        requirements=facts.requirements.model_copy(
            update={
                "performance_targets": normalize_requirements(
                    "TTFT P95 <= 500 ms; TPOT P95 <= 50 ms"
                ).performance_targets
            }
        ),
    )
    candidates_with_evidence = deterministic_planner.candidates(facts_with_evidence)
    decision = deterministic_planner.decide(facts_with_evidence, candidates_with_evidence)
    assert decision.candidate_id == baseline.id
    blocked_pd = next(candidate for candidate in candidates_with_evidence if candidate.id == pd.id)
    assert not blocked_pd.deployable
    assert "blocking benchmark failure evidence" in blocked_pd.rejection_reasons
    assert [
        sample["benchmark_id"]
        for sample in next(
            candidate for candidate in candidates_with_evidence if candidate.id == baseline.id
        ).historical_benchmarks
    ] == [f"baseline-success-{index}" for index in range(3)]
    assert [sample["failure_reason"] for sample in blocked_pd.historical_benchmarks] == ["OOM"] * 3

    baseline_with_evidence = next(candidate for candidate in candidates_with_evidence if candidate.id == baseline.id)
    ai_context = OpenAICompatiblePlanner("http://provider", "planner")._candidate_context(
        facts_with_evidence,
        baseline_with_evidence,
    )
    assert "benchmark_evidence" not in ai_context
    assert len(ai_context["historical_benchmarks"]) == 3
    assert all("benchmark_id" not in sample for sample in ai_context["historical_benchmarks"])
    assert ai_context["historical_benchmarks"] == [
        {key: value for key, value in sample.items() if key != "benchmark_id"}
        for sample in baseline_with_evidence.historical_benchmarks
    ]
    assert all(
        sample["match_confidence"] == pytest.approx(0.897142857, abs=0.001)
        for sample in ai_context["historical_benchmarks"]
    )
