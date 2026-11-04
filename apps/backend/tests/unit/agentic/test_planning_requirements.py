"""Regression coverage for explicit, versioned planning requirements."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkEvidenceRetriever,
    BenchmarkMetrics,
    BenchmarkQuery,
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    historical_query_context,
)
from llm_d_lens.agentic.benchmark_evidence import WorkloadProfile as BenchmarkWorkloadProfile
from llm_d_lens.agentic.generator import AICandidateGenerator, _aic_prediction_satisfies_slo
from llm_d_lens.agentic.planner import (
    AICCandidatePrediction,
    DeterministicPlanner,
    PlanningFacts,
    WorkloadProfile,
    latency_target_passes,
)
from llm_d_lens.agentic.planning_requirements import (
    ModelIdentity,
    NumericRequirement,
    PlanningRequirements,
    Provenance,
    RuntimeIdentity,
    enrich_planning_requirements,
    normalize_hardware,
    normalize_requirements,
)


@pytest.mark.parametrize("text,operator", [("TTFT < 2 s", "<"), ("TTFT <= 2000 ms", "<=")])
def test_latency_units_and_operators(text, operator):
    requirements = normalize_requirements(text).require_valid()
    (target,) = requirements.performance_targets
    assert (target.value, target.unit, target.operator, target.statistic) == (2000, "ms", operator, "mean")
    assert target.provenance.raw == text
    assert text[target.provenance.start : target.provenance.end] == text
    assert PlanningRequirements.model_validate_json(requirements.model_dump_json()) == requirements


@pytest.mark.parametrize("unit,value", [("ms/token", "20"), ("s/token", "0.02"), ("ms/tokens", "20")])
def test_tpot_per_token_units_preserve_threshold_and_provenance(unit, value):
    text = f"TPOT <= {value} {unit}"
    requirements = normalize_requirements(text).require_valid()
    (target,) = requirements.performance_targets
    assert (target.field, target.value, target.unit, target.operator) == ("tpot", 20, "ms", "<=")
    assert target.provenance.raw == text
    assert text[target.provenance.start : target.provenance.end] == text
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(8, 80, 4), text)
    assert facts.tpot_slo_ms == 20


@pytest.mark.parametrize(
    "text,form_field,field",
    [
        ("TTFT < 2 s", "ttft_slo_ms", "ttft"),
        ("TPOT < 2 ms", "tpot_slo_ms", "tpot"),
        ("E2E < 2 s", "end_to_end_latency_slo_ms", "e2e"),
    ],
)
def test_latency_defaults_to_mean_for_text_form_and_serialized_requirements(text, form_field, field):
    assert normalize_requirements(text).require_valid().performance_targets[0].statistic == "mean"
    assert normalize_requirements(form={form_field: 2}).require_valid().performance_targets[0].statistic == "mean"
    target = NumericRequirement(field=field, value=2, unit="ms", provenance=Provenance(source="form", field=form_field))
    assert target.statistic == "mean"
    restored = NumericRequirement.model_validate({**target.model_dump(), "statistic": "unspecified"})
    assert restored.statistic == "mean"


@pytest.mark.parametrize("statistic", ["mean", "p50", "p90", "p95", "p99"])
def test_explicit_latency_statistics_override_default(statistic):
    target = normalize_requirements(f"TTFT {statistic} < 2 s").require_valid().performance_targets[0]
    assert target.statistic == statistic


def test_explicit_latency_statistic_applies_to_unlabelled_form_without_overwriting_values():
    requirements = normalize_requirements("TTFT P95 <= 2 s", form={"ttft_slo_ms": 2000}).require_valid()
    assert [item.statistic for item in requirements.performance_targets] == ["p95", "p95"]
    assert {item.provenance.source for item in requirements.performance_targets} == {"text", "form"}
    assert normalize_requirements("TTFT P95 <= 2 s", form={"ttft_slo_ms": 1000}).conflicts


def test_default_mean_latency_does_not_accept_percentile_predictions():
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), "TTFT < 2s")
    assert latency_target_passes(facts, "ttft", 1000, statistic="mean")
    assert not latency_target_passes(facts, "ttft", 1000, statistic="p95")
    assert not latency_target_passes(PlanningFacts(16, 80, 2, ttft_slo_ms=2000), "ttft", 1000, statistic="p95")


@pytest.mark.parametrize(
    "text",
    [
        "TTFT < 2 tokens",
        "TTFT < 2 minutes",
        "ISL 4096 ms",
        "TTFT -2 s",
        "QPS 10 tokens",
        "TTFT 2 ms/tokens",
        "E2E 2 s/token",
        "ISL 2 ms/token",
        "TPOT 2 tokens/s",
        "TPOT 2 ms/request",
        "TPOT 2 ms/token/second",
        "TPOT 2 ms/tokenish",
        "TTFT 2 squirrels",
        "ISL 1024 bananas OSL 256",
        "TTFT < 100\u00b5s",
        "TTFT < 100\u03bcs",
        "TTFT < 100\u5fae\u79d2",
        "TPOT < 100\u00b5s/token",
        "TTFT < 100 ms\u5fae\u79d2",
    ],
)
def test_illegal_values_and_units_are_field_errors(text):
    requirements = normalize_requirements(text)
    assert requirements.errors
    with pytest.raises(ValueError, match=requirements.errors[0].field):
        requirements.require_valid()


def test_typed_requirement_rejects_wrong_unit():
    with pytest.raises(ValidationError, match="expected ms"):
        NumericRequirement(field="ttft", value=2, unit="tokens", provenance=Provenance(source="form", field="ttft"))


def test_common_chinese_and_english_numeric_requirements():
    text = "\u9996\u5b57\u5ef6\u8fdf\u4e0d\u8d85\u8fc7 2 \u79d2\uff0c\u5e76\u53d1\u81f3\u5c11 32\uff0c\u8f93\u5165\u957f\u5ea6 4096; OSL 200; QPS 8"
    requirements = normalize_requirements(text).require_valid()
    assert requirements.performance_targets[0].value == 2000
    assert requirements.performance_targets[0].operator == "<="
    assert [(item.field, item.value, item.operator) for item in requirements.workload] == [
        ("concurrency", 32, ">="),
        ("input_tokens", 4096, "="),
        ("output_tokens", 200, "="),
        ("request_rate", 8, "="),
    ]


def test_statistics_are_independent_and_never_fabricated():
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), "TTFT < 2s; ISL 4096")
    assert facts.ttft_slo_ms == 2000
    assert facts.workload_profile.mean_input_tokens == 4096
    assert facts.workload_profile.p95_input_tokens is None
    assert facts.requirements.workload[0].statistic == "unspecified"
    stats = normalize_requirements("mean ISL 1000; P95 ISL 4096; ISL P99 8192").require_valid()
    assert [(item.statistic, item.value) for item in stats.workload] == [("mean", 1000), ("p95", 4096), ("p99", 8192)]


def test_mean_only_does_not_invent_p95():
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), "mean ISL 1000")
    assert facts.workload_profile.mean_input_tokens == 1000
    assert facts.workload_profile.p95_input_tokens is None


@pytest.mark.parametrize(
    "text,input_tokens,ttft,has_preference",
    [
        ("ISL 2048 OSL 256 TTFT 100", 2048, 100, False),
        ("ISL 1024 OSL 256 TTFT 50 prefer latency", 1024, 50, True),
    ],
)
def test_compact_multi_field_syntax_keeps_units_statistics_and_soft_preferences(
    text,
    input_tokens,
    ttft,
    has_preference,
):
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(8, 32, 4), text)
    assert facts.workload_profile.mean_input_tokens == input_tokens
    assert facts.workload_profile.mean_output_tokens == 256
    assert facts.workload_profile.p95_input_tokens is None
    assert facts.ttft_slo_ms == ttft
    assert [item.preference for item in facts.requirements.soft_preferences] == (["latency"] if has_preference else [])


def test_p95_only_does_not_invent_a_mean():
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), "P95 ISL 4096")
    assert facts.workload_profile.p95_input_tokens == 4096
    assert facts.workload_profile.mean_input_tokens is None


def test_explicit_p95_not_treated_as_an_unspecified_aic_statistic():
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), "P95 TTFT < 2s")
    prediction = AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1, ttft_ms=1000)
    assert not _aic_prediction_satisfies_slo(facts, prediction)
    assert latency_target_passes(facts, "ttft", 1000, statistic="p95")


def test_form_text_conflicts_have_both_sources_and_no_silent_overwrite():
    requirements = normalize_requirements("ISL 1000; TTFT < 2 s", form={"mean_input_tokens": 4096, "ttft_slo_ms": 1000})
    assert {issue.field for issue in requirements.conflicts} == {"input_tokens", "ttft"}
    assert {source.source for source in requirements.conflicts[0].sources} == {"form", "text"}
    with pytest.raises(ValueError, match="input_tokens"):
        AICandidateGenerator._apply_preference_constraints(
            PlanningFacts(16, 80, 2, workload_profile=WorkloadProfile(mean_input_tokens=4096)),
            "ISL 1000",
        )


def test_same_value_different_units_agree_but_contradictory_ranges_do_not():
    assert not normalize_requirements("TTFT <=2s", form={"ttft_slo_ms": 2000}).conflicts
    assert normalize_requirements("TTFT > 3s; TTFT <= 2s").conflicts
    assert normalize_requirements("ISL 1000; ISL 2000").conflicts


def test_strict_latency_boundary_and_shared_contract():
    requirements = normalize_requirements("TTFT < 2s").require_valid()
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(16, 80, 2, requirements=requirements),
        "Ignored unparsed text",
    )
    assert facts.requirements is requirements
    assert facts.ttft_slo_ms == 2000
    assert latency_target_passes(facts, "ttft", 1999)
    assert not latency_target_passes(facts, "ttft", 2000)


def test_end_to_end_latency_keeps_its_operator_and_cannot_be_proven_by_ttft():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(
            16,
            80,
            2,
            aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1, ttft_ms=100),),
        ),
        "TTFT < 2s; E2E < 1s",
    )
    assert latency_target_passes(facts, "e2e", 999)
    assert not latency_target_passes(facts, "e2e", 1000)
    planner = DeterministicPlanner()
    candidates = planner.candidates(facts)
    assert planner.decide(facts, candidates).slo_status == "unknown"
    assert all(candidate.slo_status == "unknown" for candidate in planner.score(facts, candidates))


def test_missing_aic_latency_is_unknown_instead_of_satisfying_a_requested_target():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(
            16,
            80,
            2,
            aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1, ttft_ms=100),),
        ),
        "TTFT < 2s; TPOT < 50ms",
    )
    planner = DeterministicPlanner()
    assert not planner._aic_slo_passes(facts, facts.aic_predictions[0])
    assert planner.decide(facts, planner.candidates(facts)).slo_status == "unknown"


@pytest.mark.parametrize("text", ["TTFT >= 2s", "TTFT = 2s", "TTFT < 100ms"])
def test_unmet_latency_ranking_honors_all_operators_without_empty_violations(text):
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(
            16,
            80,
            1,
            aic_predictions=(AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1, ttft_ms=100),),
        ),
        text,
    )
    planner = DeterministicPlanner()
    candidates = planner.candidates(facts)
    candidate = next(item for item in candidates if item.id == "optimized-baseline-tp1-r1")
    assert planner._slo_assessment(facts, candidate)[0] == "not_satisfied"
    assert planner._unmet_slo_priority(facts, candidate)[0] == 1
    assert planner.decide(facts, [candidate]).slo_status == "not_satisfied"


def test_shared_retrieval_public_context_and_generator_normalization_are_idempotent():
    preference = "ISL 2048; P95 ISL 4096; TTFT < 2s"
    initial = PlanningFacts(16, 80, 2, workload_profile=WorkloadProfile(mean_output_tokens=200))
    retrieval = AICandidateGenerator._apply_preference_constraints(initial, preference)
    public_context = AICandidateGenerator._apply_preference_constraints(retrieval, preference)
    generator = AICandidateGenerator._apply_preference_constraints(public_context, preference)
    assert retrieval == public_context == generator
    assert generator.requirements is retrieval.requirements
    assert generator.requirements.provenance == retrieval.requirements.provenance
    assert not generator.requirements.conflicts
    assert generator.workload_profile.mean_input_tokens == 2048
    assert generator.workload_profile.p95_input_tokens == 4096
    assert generator.ttft_slo_ms == 2000
    assert [item.provenance.source for item in generator.requirements.workload if item.field == "input_tokens"] == [
        "text",
        "text",
    ]


def test_soft_preferences_keep_order_and_only_explicit_weights():
    requirements = normalize_requirements("low latency weight 0.7; high throughput; low cost")
    assert [(item.preference, item.order, item.weight) for item in requirements.soft_preferences] == [
        ("latency", 0, 0.7),
        ("throughput", 1, None),
        ("cost", 2, None),
    ]


def test_hardware_equivalent_labels_counts_memory_and_form_factor():
    left = normalize_hardware("16x NVIDIA H100 80 GB SXM")
    right = normalize_hardware("16x H100 80GB SXM")
    assert left.model_dump(exclude={"raw"}) == right.model_dump(exclude={"raw"})
    assert left.device_count == 16
    assert left.device_model == "H100"
    assert left.profile_id == "nvidia"
    assert left.memory_gib == pytest.approx(80 * 10**9 / 2**30)
    assert left.form_factor == "SXM"
    assert normalize_hardware("H100").memory_gib is None
    assert normalize_hardware("H100").form_factor is None
    assert normalize_hardware("H100 80GB PCIe").form_factor == "PCIE"
    assert normalize_hardware("H100 94GB SXM").memory_gib != left.memory_gib
    assert normalize_hardware("H100 + A100").device_model is None
    assert normalize_hardware("NovelDevice").profile_id is None
    with pytest.raises(ValueError, match="device_count"):
        normalize_hardware("16x H100", device_count=8)


def test_historical_queries_keep_unknowns_and_do_not_estimate_model_from_weight():
    context = historical_query_context(
        "org/model",
        PlanningFacts(999, 80, 8, context_length=4096, workload_profile=WorkloadProfile(mean_input_tokens=1000)),
    )
    assert context["model"].architecture is None
    assert context["model"].parameter_count_b is None
    assert context["model"].quantization is None
    assert context["hardware"].accelerator is None
    assert context["hardware"].gpu_count is None
    assert context["runtime"].backend is None
    workload = context["workload"]
    assert workload.mean_input_tokens == 1000
    assert all(
        getattr(workload, field) is None
        for field in (
            "p95_input_tokens",
            "mean_output_tokens",
            "concurrency",
            "request_rate",
            "shared_prefix_ratio",
            "prefill_fraction",
        )
    )


def test_real_historical_identity_matches_but_unknown_and_variant_mismatches_do_not():
    requirements = normalize_requirements(
        hardware=normalize_hardware("2x H100 80GB SXM"),
        model=ModelIdentity(model_id="org/model", architecture="moe", quantization="fp8"),
        runtime=RuntimeIdentity(backend="vllm", backend_version="0.11.0", accelerator_runtime="12.8"),
    )
    facts = PlanningFacts(
        16,
        80,
        2,
        requirements=requirements,
        workload_profile=WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.7,
        ),
    )
    configuration = CandidateConfiguration(topology="optimized-baseline", decode_tp=1, decode_replicas=1)
    query = BenchmarkQuery(**historical_query_context("org/model", facts), candidate=configuration)
    record = BenchmarkRecord(
        benchmark_id="real",
        timestamp=datetime.now(UTC),
        outcome="succeeded",
        model=query.model,
        hardware=query.hardware,
        runtime=query.runtime,
        workload=query.workload,
        configuration=configuration,
        metrics=BenchmarkMetrics(ttft_p95_ms=300),
    )
    retriever = BenchmarkEvidenceRetriever()
    assert retriever.retrieve("candidate", query, [record]).benchmark_refs == ["real"]
    unknown = BenchmarkQuery(
        **historical_query_context("org/model", replace(facts, requirements=None)), candidate=configuration
    )
    assert not retriever.matching_benchmarks(unknown, [record])
    for hardware in (
        HardwareDescriptor(accelerator="2x H100 80GB PCIe"),
        HardwareDescriptor(accelerator="2x H100 94GB SXM"),
        HardwareDescriptor(accelerator="H100", gpu_count=2),
    ):
        assert not retriever.matching_benchmarks(query.model_copy(update={"hardware": hardware}), [record])
    sparse = query.model_copy(update={"workload": BenchmarkWorkloadProfile(mean_input_tokens=4096)})
    assert retriever.workload_similarity(record.workload, sparse.workload) < 0.2


def test_unknown_workload_fields_cannot_score_as_equal():
    assert BenchmarkEvidenceRetriever.workload_similarity(BenchmarkWorkloadProfile(), BenchmarkWorkloadProfile()) == 0
    assert (
        BenchmarkEvidenceRetriever.model_similarity(
            ModelDescriptor(model_id="m", quantization="unknown"),
            ModelDescriptor(model_id="m", quantization="unknown"),
        )
        == 0
    )
    assert BenchmarkEvidenceRetriever.runtime_similarity(RuntimeFingerprint(), RuntimeFingerprint()) == 0


def test_verified_discovery_enriches_identities_without_losing_requirements_or_runtime_unknowns():
    facts = AICandidateGenerator._apply_preference_constraints(
        PlanningFacts(16, 80, 4),
        "ISL 4096; TTFT <2s; low latency",
    )
    kwargs = {
        "model_name": "org/model",
        "hardware_model": "H100",
        "profile_id": "nvidia",
        "device_count": 4,
        "model_config": {
            "num_local_experts": 8,
            "quantization_config": {"quant_method": "fp8"},
            "_commit_hash": "verified-revision",
        },
    }
    enriched = enrich_planning_requirements(facts, **kwargs)
    requirements = enriched.requirements
    assert requirements.hardware.device_model == "H100"
    assert requirements.hardware.device_count == 4
    assert requirements.hardware.memory_gib == 80
    assert requirements.hardware.form_factor is None
    assert requirements.model == ModelIdentity(
        model_id="org/model",
        architecture="moe",
        quantization="fp8",
        revision="verified-revision",
    )
    assert requirements.runtime == RuntimeIdentity()
    assert requirements.workload == facts.requirements.workload
    assert requirements.performance_targets == facts.requirements.performance_targets
    assert requirements.soft_preferences == facts.requirements.soft_preferences
    assert all(item in requirements.provenance for item in facts.requirements.provenance)
    assert any(
        item.field == "hardware.memory_gib" and item.source == "verified_discovery" for item in requirements.provenance
    )
    assert enrich_planning_requirements(enriched, **kwargs) == enriched


def test_discovery_does_not_infer_dense_quantization_or_hardware_when_absent():
    enriched = enrich_planning_requirements(
        PlanningFacts(16, 80, 4),
        model_name="org/model",
        hardware_model=None,
        model_config={"architectures": ["SomeForCausalLM"], "torch_dtype": "bfloat16"},
    )
    assert enriched.requirements.hardware.device_model is None
    assert enriched.requirements.hardware.memory_gib is None
    assert enriched.requirements.model.architecture is None
    assert enriched.requirements.model.quantization is None
    assert enriched.requirements.runtime == RuntimeIdentity()


def test_discovery_reports_conflicting_user_identity_instead_of_overwriting():
    requirements = normalize_requirements(hardware=normalize_hardware("H100 80GB PCIe"))
    facts = PlanningFacts(16, 80, 4, requirements=requirements)
    with pytest.raises(ValueError, match="hardware.form_factor"):
        enrich_planning_requirements(facts, model_name="org/model", hardware_model="H100 80GB SXM")
    assert facts.requirements.hardware.form_factor == "PCIE"


@pytest.mark.parametrize("available_gpus", [2, 4])
def test_generator_enriches_verified_homogeneous_mcp_hardware_and_model(available_gpus):
    facts = PlanningFacts(
        16,
        80,
        4,
        model_name="org/model",
        model_config_dict={"architecture": "dense", "quantization_config": {"quant_method": "awq"}},
    )
    resolved = AICandidateGenerator._resolve_mcp_facts(
        "cluster",
        facts,
        {
            "get_cluster_overview": [
                {
                    "hardware": {
                        "gpuCount": 4,
                        "availableGpuCount": available_gpus,
                        "vramBytes": 4 * 80 * 1024**3,
                        "accelerators": [{"id": "nvidia", "gpuCount": 4, "models": ["H100 80GB SXM"]}],
                    },
                }
            ],
        },
    )
    assert resolved.facts.requirements.hardware.form_factor == "SXM"
    assert resolved.facts.requirements.hardware.device_count == available_gpus
    assert resolved.facts.requirements.model.architecture == "dense"
    assert resolved.facts.requirements.model.quantization == "awq"
    assert resolved.facts.requirements.runtime.backend is None


@pytest.mark.parametrize("operator,passes", [("<", False), ("<=", True)])
def test_strict_latency_operator_is_enforced_by_aic_and_planner(operator, passes):
    facts = AICandidateGenerator._apply_preference_constraints(PlanningFacts(16, 80, 2), f"TTFT {operator} 2s")
    prediction = AICCandidatePrediction(mode="agg", tensor_parallel_size=1, replicas=1, ttft_ms=2000)
    assert _aic_prediction_satisfies_slo(facts, prediction) is passes
    assert DeterministicPlanner._aic_slo_passes(facts, prediction) is passes
