"""Planning case boundaries, budgets, authorization and immutable metric scope."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_d_lens.agentic.planner import (
    DeterministicPlanner,
    PlannedCandidate,
    PlanningFacts,
    WorkloadProfile,
    normalize_planning_facts,
    public_rag_context,
)
from llm_d_lens.agentic.planning_requirements import (
    ModelIdentity,
    RuntimeIdentity,
    normalize_hardware,
    normalize_requirements,
)
from llm_d_lens.agentic.public_benchmark_record.case_retrieval import (
    MAX_EVIDENCE_BYTES,
    assert_case_access,
    build_case_bundle,
    candidate_evidence_results,
    compact_reference,
    configuration_technique_hints,
    model_reference_map,
    resolve_internal_cases,
    restore_model_references,
)
from llm_d_lens.core.exceptions import ForbiddenError


def observation(identity="one", *, experiment="run", series=None, **changes):
    entry = {
        "evidence_id": identity,
        "metric": "ttft",
        "statistic": "p50",
        "value": "2",
        "unit": "s",
        "normalized_value": "2000",
        "normalized_unit": "ms",
        "normalization_rule": "seconds-to-ms",
        "series": series,
        "citation": {"origin_id": "public", "source_revision": "rev", "sha256": "a" * 64},
        "experiment": {
            "item_id": experiment,
            "origin_id": "public",
            "model": "model",
            "hardware": "accelerator",
            "runtime": "vllm",
            "configuration": {"tp": 1},
            "workload": {"mean_input_tokens": 4096},
            "source_pointer": {},
        },
    }
    entry.update(changes)
    return entry


def bundle(*entries, budget=8000):
    return build_case_bundle(
        entries,
        PlanningFacts(8, 80, 8, workload_profile=WorkloadProfile(mean_input_tokens=4096)),
        model="model",
        hardware="accelerator",
        branch_status={},
        token_budget=budget,
    )


def test_metrics_preserve_experiment_series_statistics_and_units():
    result, kept = bundle(
        observation("first"),
        observation("second", statistic="p95", value="3"),
        observation("third", experiment="another"),
        observation("fourth", series={"concurrency": 4}),
    )
    # Each complete experiment/series is indivisible, even when the small budget excludes one.
    assert result.candidate_total == 3
    assert result.token_count_upper_bound <= 8000
    first = next(case for case in result.cases if case.measurements[0]["evidence_id"] == "first")
    assert [entry["statistic"] for entry in first.measurements] == ["p50", "p95"]
    assert first.measurements[0]["unit"] == "s"
    assert first.measurements[0]["normalized_unit"] == "ms"
    assert first.measurements[0]["normalization_rule"] == "seconds-to-ms"
    assert first.applicability == "reference_only"
    assert "model_descriptor" in first.unknown_fields
    assert {entry["evidence_id"] for entry in kept} >= {"first", "second"}


def test_incompatible_identity_is_excluded_before_model_context():
    entry = observation()
    entry["experiment"]["hardware"] = "different"
    result, kept = bundle(entry)
    assert not result.cases and not kept
    assert list(result.excluded.values()) == ["incompatible_identity"]


@pytest.mark.parametrize(
    ("requested", "reported", "matches"),
    [
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", True),
        ("qwen3-0.6b", "Qwen/Qwen3-0.6B", True),
        ("another-org/Qwen3-0.6B", "Qwen3-0.6B", True),
        ("Publisher/Falcon-7B", "Falcon-7B", True),
        ("falcon-7b", "Publisher/Falcon-7B", True),
        ("Publisher-Falcon-7B", "Publisher/Falcon-7B", True),
        ("Qwen/Qwen3-0.6B-FP8", "Qwen3-0.6B", False),
    ],
)
def test_model_alias_retrieval_survives_case_selection_without_rewriting_provenance(requested, reported, matches):
    entry = observation("public-rag:alias")
    entry["experiment"]["model"] = reported
    result, kept = build_case_bundle(
        (entry,),
        PlanningFacts(8, 80, 8),
        model=requested,
        hardware="accelerator",
        branch_status={},
        token_budget=8000,
    )
    if matches:
        assert len(result.cases) == 1
        assert "model" not in result.cases[0].known_differences
        assert result.cases[0].model == reported
        assert kept[0]["experiment"]["model"] == reported
    else:
        assert not result.cases and not kept
        assert list(result.excluded.values()) == ["incompatible_identity"]


def test_unknown_workload_is_not_a_match_or_current_slo_proof():
    entry = observation()
    entry["experiment"]["workload"] = {}
    result, _ = bundle(entry)
    assert "workload.mean_input_tokens" in result.cases[0].unknown_fields
    assert result.cases[0].validation_required
    assert "SLO compliance" in result.usage


@pytest.mark.parametrize(
    ("configuration", "score", "invalid_field"),
    [
        ({"tp": "8.0"}, 1 / 8, None),
        ({"tp": 8, "replicas": "2.0"}, 1 / 16, None),
        ({"decode_tp": "4.0", "decode_replicas": "2.0", "tp": 8, "replicas": 4}, 1 / 8, None),
        ({}, 1, None),
        ({"tp": None, "replicas": None}, 1, None),
        ({"tp": 0}, 0, "tp"),
        ({"tp": -1}, 0, "tp"),
        ({"tp": True}, 0, "tp"),
        ({"tp": "unknown"}, 0, "tp"),
        ({"tp": "NaN"}, 0, "tp"),
        ({"tp": "Infinity"}, 0, "tp"),
        ({"replicas": "0.0"}, 0, "replicas"),
        ({"replicas": -1}, 0, "replicas"),
        ({"replicas": False}, 0, "replicas"),
        ({"decode_tp": 0, "tp": 8}, 0, "decode_tp"),
        ({"decode_replicas": "invalid", "replicas": 2}, 0, "decode_replicas"),
        ({"tp": "1e308", "replicas": "1e308"}, 0, "device_count"),
        ({"tp": "1e-300", "replicas": "1e-300"}, 0, "device_count"),
    ],
)
def test_cost_preference_validates_counts_without_rewriting_source(configuration, score, invalid_field, caplog):
    entry = observation()
    entry["experiment"]["configuration"] = configuration
    result, _ = build_case_bundle(
        (entry,),
        PlanningFacts(8, 80, 8, requirements=normalize_requirements("prefer cost")),
        model="model",
        hardware="accelerator",
        branch_status={},
    )
    (case,) = result.cases
    assert case.preference_score == pytest.approx(score)
    assert case.configuration == configuration
    if invalid_field:
        assert f"configuration.{invalid_field}" in case.unknown_fields
        assert "reason=invalid_device_count" in caplog.text


@pytest.mark.parametrize(
    ("profile", "workload", "similarity", "coverage", "status"),
    [
        (None, {}, None, None, "not_requested"),
        (WorkloadProfile(), {}, None, None, "not_requested"),
        (WorkloadProfile(mean_input_tokens=4096), {}, None, 0, "unknown"),
        (WorkloadProfile(mean_input_tokens=0), {"mean_input_tokens": 4096}, 0, 1, "compared"),
    ],
)
def test_workload_comparison_preserves_unknown_and_explicit_zero(profile, workload, similarity, coverage, status):
    entry = observation()
    entry["experiment"]["workload"] = workload
    result, _ = build_case_bundle(
        (entry,),
        PlanningFacts(8, 80, 8, workload_profile=profile),
        model="model",
        hardware="accelerator",
        branch_status={},
    )
    case = result.cases[0]
    assert case.workload_similarity == similarity
    assert case.evidence_coverage == coverage
    assert case.workload_match_status == status
    assert case.model_dump(mode="json")["workload_similarity"] == similarity


@pytest.mark.parametrize(
    ("metric", "value", "quality"),
    [
        ("latency", "0", "nonpositive_performance"),
        ("tpot", "-1", "nonpositive_performance"),
        ("throughput", "10", "informative"),
        ("error_count", "0", "context_only"),
        ("ttft", "NaN", "nonfinite_or_missing_value"),
    ],
)
def test_measurement_quality_does_not_rewrite_values_or_zero_counters(metric, value, quality):
    entry = observation(metric=metric, value=value, normalized_value=value)
    result, kept = bundle(entry)
    measurement = result.cases[0].measurements[0]
    assert measurement["quality"] == quality
    assert measurement["value"] == measurement["normalized_value"] == value
    assert kept[0]["value"] == value
    assert "quality" not in entry


def test_informative_cases_precede_zero_only_cases_and_preserve_complete_series():
    zero = observation("zero", experiment="zero", metric="latency", statistic="p99", normalized_value="0", value="0")
    tpot = observation("tpot", experiment="useful", metric="tpot", normalized_value="20")
    throughput = observation("throughput", experiment="useful", metric="throughput", normalized_value="3000")
    result, kept = bundle(zero, tpot, throughput)
    assert result.cases[0].experiment_id == "useful"
    assert {measurement["evidence_id"] for measurement in result.cases[0].measurements} == {"tpot", "throughput"}
    assert {entry["evidence_id"] for entry in kept} >= {"tpot", "throughput"}
    assert result.cases[0].applicability == "reference_only"


def test_model_projection_omits_duplicate_import_metadata_but_preserves_conditions_and_provenance():
    entry = observation("public-rag:one")
    entry["experiment"]["configuration"].update(
        source_info={"raw_url": "x" * 8000},
        _fileSizeKb="5000",
        vllm_arguments=[["--enable-prefix-caching", "true"]],
        model_descriptor={"parameter_count_b": 7, "tokenizer_id": "org/tokenizer", "revision": "b" * 64},
        hardware_descriptor={"accelerator": "accelerator", "vram_per_gpu_gib": 80},
        runtime_fingerprint={"backend": "vllm", "backend_version": "0.10", "container_digest": "sha256:" + "c" * 64},
    )
    entry["citation"]["source_pointer"] = {
        "json_pointer": "/15/metrics/tpot",
        "original_source": {"raw_url": "x" * 8000},
    }
    result, kept = bundle(entry)
    assert len(result.cases) == 1
    case = result.cases[0].model_dump(mode="json")
    facts = PlanningFacts(8, 80, 8, configuration_cases=(case,), public_rag_evidence=kept)
    (projected,) = public_rag_context(facts)["configuration_cases"]
    assert "source_info" not in projected["configuration"] and "_fileSizeKb" not in projected["configuration"]
    assert projected["configuration"]["vllm_arguments"] == [["--enable-prefix-caching", "true"]]
    assert projected["configuration"]["model_descriptor"] == {"parameter_count_b": 7, "tokenizer_id": "org/tokenizer"}
    assert projected["configuration"]["hardware_descriptor"] == {"accelerator": "accelerator", "vram_per_gpu_gib": 80}
    assert projected["configuration"]["runtime_fingerprint"] == {"backend": "vllm", "backend_version": "0.10"}
    assert all(value not in json.dumps(projected) for value in ("a" * 64, "b" * 64, "c" * 64, "raw_url"))
    measurement = projected["measurements"][0]
    assert measurement["value"] == case["measurements"][0]["value"]
    assert measurement["normalized"] == {"value": "2000", "unit": "ms"}
    assert "provenance" not in projected and "provenance_sources" not in projected
    assert "evidence_id" not in measurement and "case_id" not in projected
    assert entry["evidence_id"] in result.model_reference_map.values()
    assert case["configuration"]["source_info"] == entry["experiment"]["configuration"]["source_info"]
    assert kept[0]["citation"] == entry["citation"]
    assert "original_source" in case["provenance"][0]["source_pointer"]


@pytest.mark.parametrize("technique", ["kv-scorer", "prefix-scorer", "precise_prefix_cache_aware"])
def test_model_projection_replaces_configuration_source_clues_with_semantic_hints(technique):
    entry = observation()
    path = f"Inference_Scheduling/{technique}/results/per_request_lifecycle_metrics.json"
    entry["experiment"]["configuration"] = {
        "run_id": "inference-perf-setup",
        "source_path": path,
        "source_info": {"file_identifier": path, "raw_url": "x" * 8000},
    }
    original = deepcopy(entry)
    result, kept = bundle(entry)
    facts = PlanningFacts(
        8, 80, 8, configuration_cases=(result.cases[0].model_dump(mode="json"),), public_rag_evidence=kept
    )
    configuration = public_rag_context(facts)["configuration_cases"][0]["configuration"]
    assert not {"run_id", "source_path", "source_info"} & configuration.keys()
    expected_provider = "optimized-baseline" if technique == "prefix-scorer" else "precise-prefix-cache-routing"
    assert configuration["technique_hints"] == [
        {
            "related_provider_ref": expected_provider,
            "technique_description": (
                "Prefix-affinity and load-aware routing"
                if expected_provider == "optimized-baseline"
                else "Precise prefix-cache routing"
            ),
            "mapping_basis": "source_label",
        }
    ]
    assert path not in json.dumps(configuration) and technique not in json.dumps(configuration)
    assert result.cases[0].configuration == original["experiment"]["configuration"]
    assert entry == original
    assert "not model-verified usage" in result.usage
    assert "establish technique-to-provider mappings" in result.usage


@pytest.mark.parametrize(
    ("label", "provider"),
    [
        ("precise_prefix_cache_aware.cache_estimate", "precise-prefix-cache-routing"),
        ("precise_prefix_cache_aware.cache_tracking", "precise-prefix-cache-routing"),
        ("Precise-Prefix-Cache-Routing", "precise-prefix-cache-routing"),
        ("kv_scorer", "precise-prefix-cache-routing"),
        ("prefix-scorer", "optimized-baseline"),
        ("pd_disaggregation", "pd-disaggregation"),
        ("tiered-prefix-cache", "tiered-prefix-cache"),
        ("baseline-vllm", "baseline-vllm"),
    ],
)
@pytest.mark.parametrize("source_field", ["run_id", "source_path", "source_info.file_identifier"])
def test_technique_hints_establish_mapping_without_asserting_configuration_equivalence(label, provider, source_field):
    source = f"experiment/{label}/results.json"
    configuration = {"provider_ref": "reported-provider", "guide_variant": "reported-guide"}
    if source_field == "source_info.file_identifier":
        configuration["source_info"] = {"file_identifier": source}
    else:
        configuration[source_field] = source
    original = deepcopy(configuration)
    (hint,) = configuration_technique_hints(configuration)
    assert hint["provider_ref"] == provider
    assert hint["source_field"] == source_field and hint["source_value"] == source
    assert hint["technique"] in source
    assert hint["mapping_basis"] == "source_label"
    assert "verification" not in hint
    assert configuration == original


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"run_id": None, "source_path": 3, "source_info": []},
        {"source_path": "generic-prefix-cache/cache_tracking/results.json"},
        {"source_path": "notprecise_prefix_cache_awareish/kv-scorerish/results.json"},
        {"metadata": {"technique": "precise_prefix_cache_aware"}},
        {"source_info": {"raw_url": "https://example.test/precise_prefix_cache_aware"}},
        {"technique_hints": [{"provider_ref": "precise-prefix-cache-routing", "verification": "verified"}]},
    ],
)
def test_technique_hints_do_not_infer_from_generic_labels_or_arbitrary_metadata(configuration):
    assert configuration_technique_hints(configuration) == []


def test_legacy_observation_context_also_extracts_technique_hints():
    entry = observation()
    path = "precise_prefix_cache_aware.cache_tracking/results.json"
    entry["experiment"]["configuration"]["source_info"] = {"file_identifier": path}
    original = deepcopy(entry)
    context = public_rag_context(PlanningFacts(8, 80, 8, public_rag_evidence=(entry,)))
    (hint,) = context["published_benchmark_evidence"][0]["experiment"]["configuration"]["technique_hints"]
    assert hint["related_provider_ref"] == "precise-prefix-cache-routing"
    assert hint["technique_description"] == "Precise prefix-cache routing"
    assert hint["mapping_basis"] == "source_label"
    assert "verification" not in hint
    assert path not in json.dumps(context)
    assert entry == original


def test_missing_statistic_defaults_to_mean_in_model_context_without_rewriting_audit():
    entry = observation(statistic=None)
    result, kept = bundle(entry)
    facts = PlanningFacts(8, 80, 8, configuration_cases=(result.cases[0].model_dump(mode="json"),))
    measurement = public_rag_context(facts)["configuration_cases"][0]["measurements"][0]
    assert measurement["statistic"] == "mean"
    assert measurement["statistic_original"] is None
    assert measurement["statistic_basis"] == "default_mean"
    assert result.cases[0].measurements[0]["statistic"] is None
    assert kept[0]["statistic"] is None


def test_semantic_projection_keeps_distinct_experiment_measurements_grouped_without_ids():
    first = observation("public-rag:first", experiment="first", value="1")
    second = observation("public-rag:second", experiment="second", value="2")
    second["experiment"]["configuration"]["tp"] = 2
    result, kept = bundle(first, second)
    facts = PlanningFacts(
        8,
        80,
        8,
        configuration_cases=tuple(case.model_dump(mode="json") for case in result.cases),
        public_rag_evidence=kept,
    )
    context = public_rag_context(facts)
    assert len(context["configuration_cases"]) == 2
    assert {
        case["configuration"]["tp"]: [measurement["value"] for measurement in case["measurements"]]
        for case in context["configuration_cases"]
    } == {1: ["1"], 2: ["2"]}
    serialized = json.dumps(context)
    assert "case_id" not in serialized and "evidence_id" not in serialized
    assert "public-rag:first" not in serialized and "public-rag:second" not in serialized
    assert {entry["evidence_id"] for entry in facts.public_rag_evidence} == {"public-rag:first", "public-rag:second"}


def test_model_measurements_remain_grouped_while_full_citations_stay_in_audit():
    first = observation("ttft", experiment="shared")
    second = observation("tpot", experiment="shared", metric="tpot")
    for entry in (first, second):
        entry["citation"]["source_pointer"] = {
            "record_id": "record",
            "json_pointer": f"/15/metrics/{entry['metric']}",
            "original_source": {"raw_url": "x" * 2000},
        }
    result, kept = bundle(first, second)
    facts = PlanningFacts(
        8,
        80,
        8,
        configuration_cases=tuple(case.model_dump(mode="json") for case in result.cases),
        public_rag_evidence=kept,
    )
    (projected,) = public_rag_context(facts)["configuration_cases"]
    assert "provenance_sources" not in projected and "provenance" not in projected
    for measurement, original in zip(projected["measurements"], (first, second), strict=True):
        assert "evidence_id" not in measurement
        assert measurement["metric"] == original["metric"]
        audit_measurement = next(
            item for item in result.cases[0].measurements if item["evidence_id"] == original["evidence_id"]
        )
        assert result.cases[0].provenance[audit_measurement["provenance_index"]] == original["citation"]
    assert result.cases[0].provenance[0] == first["citation"]


def test_complete_case_can_use_more_than_half_the_budget():
    entry = observation("public-rag:large")
    entry["experiment"]["configuration"]["vllm_arguments"] = [["--custom-options", "x" * 4000]]
    result, kept = build_case_bundle(
        (entry,),
        PlanningFacts(8, 80, 8),
        model="model",
        hardware="accelerator",
        branch_status={},
        token_budget=10_000,
    )
    assert len(result.cases) == 1 and len(kept) == 1
    assert 5000 < result.token_count_upper_bound <= 10_000


def test_budget_does_not_remove_units_or_partial_case():
    entries = [observation(str(index), experiment=str(index)) for index in range(50)]
    result, kept = bundle(*entries, budget=1800)
    assert result.truncated
    assert result.token_count_upper_bound <= 1800
    assert len(result.cases) <= 10
    assert all(entry["unit"] and entry["citation"] and entry["experiment"] for entry in kept)
    empty, kept = bundle(*entries, budget=0)
    assert empty.cases == () and kept == ()
    assert empty.token_count_upper_bound == 0


def test_model_context_deduplicates_measurements_without_losing_audit_evidence():
    entry = observation("public-rag:one")
    result, kept = bundle(entry)
    facts = PlanningFacts(
        8,
        80,
        8,
        configuration_cases=tuple(case.model_dump(mode="json") for case in result.cases),
        configuration_case_status=result.model_dump(mode="json", exclude={"cases"}),
        public_rag_evidence=kept,
    )
    context = public_rag_context(facts)
    assert not context.get("published_benchmark_evidence")
    assert "evidence_id" not in context["configuration_cases"][0]["measurements"][0]
    assert "provenance" not in context["configuration_cases"][0]
    assert facts.public_rag_evidence[0]["experiment"] == entry["experiment"]
    assert facts.public_rag_evidence[0]["unit"] == "s"
    standalone = observation("public-rag:standalone", experiment="other")
    mixed = public_rag_context(replace(facts, public_rag_evidence=(*kept, standalone)))
    assert len(mixed["published_benchmark_evidence"]) == 1
    assert "evidence_id" not in mixed["published_benchmark_evidence"][0]


def test_sql_budget_does_not_load_audit_exclusion_lists():
    incompatible = []
    for index in range(50):
        entry = observation(f"public-rag:excluded-{index}", experiment=f"other-{index}")
        entry["experiment"]["hardware"] = "different"
        incompatible.append(entry)
    result, kept = build_case_bundle(
        (observation("public-rag:one"), *incompatible),
        PlanningFacts(8, 80, 8),
        model="model",
        hardware="accelerator",
        branch_status={},
        token_budget=8000,
    )
    assert len(result.cases) == 1
    assert len(result.excluded) == 50
    facts = PlanningFacts(
        8,
        80,
        8,
        configuration_cases=tuple(case.model_dump(mode="json") for case in result.cases),
        configuration_case_status=result.model_dump(mode="json", exclude={"cases"}),
        public_rag_evidence=kept,
    )
    context = public_rag_context(facts)
    assert "configuration_case_status" not in context
    size = len(json.dumps(context, ensure_ascii=True).encode())
    assert size <= result.token_count_upper_bound <= 8000
    assert len(facts.configuration_case_status["excluded"]) == 50


def test_no_experiment_context_cannot_be_promoted_to_measurement_case():
    result, kept = bundle(observation(experiment=None))
    assert not kept
    assert result.excluded == {"one": "missing_experiment_context"}


def test_cross_source_identity_is_not_automatically_independent_corroboration():
    first = observation()
    first["experiment"]["source_pointer"] = {"record_id": "evaluate:one"}
    second = observation("two")
    second["experiment"] = {**first["experiment"], "origin_id": "agentic:evaluate"}
    second["citation"] = {"origin_id": "agentic:evaluate", "benchmark_id": "evaluate:one"}
    result, _ = bundle(first, second)
    assert len(result.cases) == 2
    assert {case.possible_duplicate_group for case in result.cases} == {"evaluate:one"}
    assert all(case.cross_source_independence == "not_established" for case in result.cases)
    assert sum(case.independent_experiment_count for case in result.cases) == 1
    assert result.independent_experiment_count == 1


@pytest.mark.parametrize("reported_model", ["model", "Publisher/model", "PUBLISHER/MODEL"])
def test_verified_cross_source_run_and_identical_conditions_merge_without_extra_weight(reported_model):
    first = observation("public-rag:first")
    first["verified_evaluate_run_id"] = "evaluate:one"
    first["experiment"]["configuration"]["model_descriptor"] = {
        "model_id": "model",
        "tokenizer_id": "model",
        "revision": "rev-a",
    }
    second = deepcopy(first)
    second["evidence_id"] = "internal:second"
    second["verified_evaluate_run_id"] = "one"
    second["experiment"].update(item_id="evaluate:one", origin_id="agentic:evaluate", model=reported_model)
    second["experiment"]["configuration"]["model_descriptor"].update(
        model_id=reported_model,
        tokenizer_id=reported_model,
    )
    second["citation"] = {"origin_id": "agentic:evaluate", "benchmark_id": "evaluate:one"}
    result, kept = bundle(first, second)
    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.identity_verification == "verified_run_and_conditions"
    assert set(case.experiment_ids) == {"run", "evaluate:one"}
    assert len(case.provenance) == 2
    assert len(kept) == 2
    assert result.independent_experiment_count == 1


@pytest.mark.parametrize("difference", ["workload", "series", "value"])
def test_same_run_with_different_conditions_or_values_is_not_merged(difference):
    first = observation()
    first["verified_evaluate_run_id"] = "evaluate:one"
    second = deepcopy(first)
    second["evidence_id"] = "two"
    second["experiment"]["origin_id"] = "agentic:evaluate"
    second["citation"] = {"origin_id": "agentic:evaluate", "benchmark_id": "evaluate:one"}
    if difference == "workload":
        second["experiment"]["workload"] = {"mean_input_tokens": 8192}
    elif difference == "series":
        second["series"] = {"rate": 2}
    else:
        second["normalized_value"] = "3000"
    result, _ = bundle(first, second)
    assert len(result.cases) == 2
    assert result.independent_experiment_count == 1


def test_ranking_prefers_matching_workload_before_identifier_order():
    far = observation("a", experiment="a")
    far["experiment"]["workload"] = {"mean_input_tokens": 8192}
    near = observation("z", experiment="z")
    result, _ = bundle(far, near)
    assert result.cases[0].experiment_id == "z"
    assert result.cases[0].workload_similarity == 1


def test_case_budget_preserves_lower_latency_ranking_not_slo_boundary_or_identifier():
    facts = normalize_planning_facts(
        PlanningFacts(8, 80, 8, workload_profile=WorkloadProfile(mean_input_tokens=4096)),
        "TTFT mean <= 1000 ms",
    )
    entries = [
        observation(
            "near-boundary",
            experiment="a-near",
            statistic="mean",
            value="990",
            unit="ms",
            normalized_value="990",
            normalized_unit="ms",
        ),
        observation(
            "fast",
            experiment="z-fast",
            statistic="mean",
            value="10",
            unit="ms",
            normalized_value="10",
            normalized_unit="ms",
        ),
    ]
    result, kept = build_case_bundle(
        tuple(entries), facts, model="model", hardware="accelerator", branch_status={}, token_budget=16000
    )
    assert [case.experiment_id for case in result.cases] == ["z-fast", "a-near"]
    assert [entry["evidence_id"] for entry in kept] == ["fast", "near-boundary"]
    assert result.cases[0].ranking["dominated_by"] == 0


def test_fallback_queue_is_last_and_labeled_in_audit_and_model_context():
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import configuration_case_context

    fallback = observation("fallback", experiment="a-fallback", metric="throughput", value="42")
    fallback["experiment"]["workload"] = {}
    ranked = observation("ranked", experiment="z-ranked")
    result, _ = build_case_bundle(
        (fallback, ranked), PlanningFacts(8, 80, 8), model="model", hardware="accelerator", branch_status={}
    )
    assert [case.experiment_id for case in result.cases] == ["z-ranked", "a-fallback"]
    audit = result.model_dump(mode="json")
    assert [case["ranking"]["queue"] for case in audit["cases"]] == ["ranked", "fallback"]
    context = configuration_case_context(audit["cases"], audit)
    assert context["configuration_cases"][1]["ranking"]["fallback_reason"] == (
        "no_comparable_workload_or_valid_latency"
    )
    assert "not low-latency evidence" in context["configuration_case_usage"]


def test_device_count_is_a_reported_difference_not_an_exact_match():
    entry = observation()
    entry["experiment"]["configuration"]["hardware_descriptor"] = {"gpu_count": 4}
    facts = normalize_planning_facts(PlanningFacts(8, 80, 8), "")
    facts = replace(
        facts,
        requirements=facts.requirements.model_copy(
            update={"hardware": facts.requirements.hardware.model_copy(update={"device_count": 8})}
        ),
    )
    result, _ = build_case_bundle((entry,), facts, model="model", hardware="accelerator", branch_status={})
    assert "hardware.device_count" in result.cases[0].known_differences
    assert result.cases[0].applicability == "reference_only"


def test_audit_branch_metadata_does_not_consume_model_budget():
    result, kept = build_case_bundle(
        (observation(),),
        PlanningFacts(8, 80, 8),
        model="model",
        hardware="accelerator",
        branch_status={"documents": {"status": "disabled", "reason": "provider_not_installed", "audit": "x" * 3000}},
        token_budget=4000,
    )
    assert not result.model_context_omitted and kept
    assert result.branch_status["documents"]["status"] == "disabled"
    assert "document_selection" not in result.branch_status
    assert result.token_count_upper_bound < 4000
    assert result.metadata_token_count < 100
    normal, _ = bundle(observation())
    assert normal.metadata_token_count > 0
    projected = public_rag_context(
        PlanningFacts(8, 80, 8, configuration_cases=(normal.cases[0].model_dump(mode="json"),))
    )
    assert normal.token_count_upper_bound >= len(json.dumps(projected))


def test_compact_references_are_stable_and_only_resolve_selected_evidence(monkeypatch):
    cases = [{"case_id": "case:first", "measurements": [{"evidence_id": "public-rag:one"}]}, {"case_id": "case:second"}]
    mapping = model_reference_map(cases)
    references = [
        compact_reference("case:first", "case"),
        compact_reference("public-rag:one", "measurement"),
    ]
    assert restore_model_references(references, mapping) == ["case:first", "public-rag:one"]
    assert restore_model_references(["case:first"], mapping) == ["case:first"]
    pruned = model_reference_map(cases[1:])
    assert pruned[compact_reference("case:second", "case")] == "case:second"
    with pytest.raises(ValueError, match="Unknown compact"):
        restore_model_references(references[:1], pruned)
    monkeypatch.setattr(
        "llm_d_lens.agentic.public_benchmark_record.case_retrieval.compact_reference",
        lambda _reference, _kind: "c:collision",
    )
    with pytest.raises(ValueError, match="collision"):
        model_reference_map(cases)


def test_candidate_contract_never_joins_metrics_from_separate_experiments_to_claim_slo():
    facts = normalize_planning_facts(PlanningFacts(8, 80, 8), "TTFT mean < 2 s; TPOT mean < 20 ms")
    first = observation("ttft", statistic="mean", value="1", normalized_value="1000")
    second = observation(
        "tpot",
        experiment="different",
        metric="tpot",
        statistic="mean",
        value="10",
        unit="ms",
        normalized_value="10",
        normalized_unit="ms",
    )
    result, _ = build_case_bundle((first, second), facts, model="model", hardware="accelerator", branch_status={})
    candidates = DeterministicPlanner().candidates(facts)
    mappings = candidate_evidence_results(candidates[:2], result, facts)
    assert {mapping.candidate_id for mapping in mappings} == {candidate.id for candidate in candidates[:2]}
    assert all(mapping.performance_status == "unknown" for mapping in mappings)
    assert all(mapping.validation_required for mapping in mappings)
    assert all(mapping.configuration["topology"] for mapping in mappings)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "satisfied"),
        ({"max_model_len": 8192}, "unknown"),
        ({"gpu_memory_utilization": 0.8}, "unknown"),
        ({"optimizations": ["prefix-cache"]}, "unknown"),
    ],
)
def test_exact_candidate_measurements_require_matching_serving_configuration(changes, expected):
    hardware = "NVIDIA H100 80GB SXM"
    facts = normalize_planning_facts(
        PlanningFacts(8, 80, 8, workload_profile=WorkloadProfile(mean_input_tokens=4096)), "TTFT < 2 s"
    )
    model = ModelIdentity(model_id="model", revision="rev", architecture="dense", quantization="bf16")
    runtime = RuntimeIdentity(backend="vllm", backend_version="0.8", accelerator_runtime="12")
    facts = replace(
        facts,
        requirements=facts.requirements.model_copy(
            update={
                "hardware": normalize_hardware(hardware),
                "model": model,
                "runtime": runtime,
            }
        ),
    )
    entry = observation(statistic="mean", value="1", normalized_value="1000")
    entry["experiment"].update(
        hardware=hardware,
        configuration={
            "topology": "optimized-baseline",
            "decode_tp": 1,
            "decode_replicas": 1,
            "max_model_len": facts.context_length,
            "gpu_memory_utilization": facts.gpu_memory_utilization,
            "vllm_arguments": [],
            "optimizations": [],
            "model_descriptor": model.model_dump(),
            "runtime_fingerprint": runtime.model_dump(),
            "hardware_descriptor": normalize_hardware(hardware).model_dump(),
            **changes,
        },
    )
    result, _ = build_case_bundle((entry,), facts, model="model", hardware=hardware, branch_status={})
    candidate = PlannedCandidate(
        id="validated",
        replicas=1,
        tensor_parallel_size=1,
        max_model_len=facts.context_length,
        gpu_memory_utilization=facts.gpu_memory_utilization,
        required_gpus=1,
        deployable=True,
    )
    (mapped,) = candidate_evidence_results([candidate], result, facts)
    assert mapped.performance_status == expected
    assert (mapped.applicability == "exact") == (expected == "satisfied")


def test_internal_authorization_precedes_payload_read_and_rechecks(monkeypatch):
    calls = []

    class Dao:
        def search(self, filters, principal, *, limit):
            calls.append(("sql-scope", filters.model, principal, limit))
            return SimpleNamespace(matches=(), total=0)

        def assert_current(self, matches, principal):
            calls.append(("recheck", matches, principal))

    principal = SimpleNamespace(user_id="caller")
    monkeypatch.setattr("llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", Dao)
    result = asyncio.run(resolve_internal_cases("model", principal))
    assert calls == [("sql-scope", "model", principal, 50), ("recheck", (), principal)]
    assert result.status == "empty"


def test_internal_denial_and_no_identity_are_explicit(monkeypatch):
    class Dao:
        def search(self, *_args, **_kwargs):
            raise ForbiddenError("denied")

    monkeypatch.setattr("llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", Dao)
    denied = asyncio.run(resolve_internal_cases("model", SimpleNamespace(user_id="caller")))
    anonymous = asyncio.run(resolve_internal_cases("model", None))
    assert denied.status == anonymous.status == "forbidden"
    assert not denied.records and not anonymous.records


def test_budget_rejects_invalid_limit():
    with pytest.raises(ValueError, match=f"between zero and {MAX_EVIDENCE_BYTES}"):
        bundle(observation(), budget=MAX_EVIDENCE_BYTES + 1)


def test_retained_internal_evidence_rechecks_access_before_plan_read(monkeypatch):
    class Dao:
        def search(self, filters, principal, *, limit):
            assert filters.record_id == "evaluate:one" and limit == 1
            return SimpleNamespace(matches=(), total=0)

    monkeypatch.setattr("llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", Dao)
    cases = ({"provenance": [{"origin_id": "agentic:evaluate", "benchmark_id": "evaluate:one"}]},)
    with pytest.raises(ForbiddenError, match="no longer accessible"):
        assert_case_access(cases, SimpleNamespace(user_id="caller"))
    with pytest.raises(ForbiddenError, match="authenticated"):
        assert_case_access(cases, None)


@pytest.mark.parametrize("enabled,manifest_id", [(False, "manifest"), (True, "new-version")])
def test_source_withdrawal_or_version_change_during_planning_blocks_delivery(monkeypatch, enabled, manifest_id):
    from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION
    from llm_d_lens.auth.contracts import Principal, RoleBinding

    class Dao:
        def head(self, origin):
            assert origin == "public"
            return SimpleNamespace(
                state=SimpleNamespace(enabled=enabled), manifest=SimpleNamespace(item_id=manifest_id)
            )

    monkeypatch.setattr("llm_d_lens.agentic.public_benchmark_record.case_retrieval.PublicBenchmarkRecordQueryDao", Dao)
    cases = ({"provenance": [{"origin_id": "public", "published_manifest_id": "manifest"}]},)
    principal = Principal(
        user_id="caller",
        username="caller",
        bindings=(RoleBinding("query"),),
        role_permissions={"query": frozenset({QUERY_PERMISSION})},
    )
    with pytest.raises(ForbiddenError, match="source changed"):
        assert_case_access(cases, principal, check_public_sources=True)


def test_public_cases_require_public_query_permission_before_source_reads():
    from llm_d_lens.auth.contracts import Principal

    cases = ({"provenance": [{"origin_id": "public", "published_manifest_id": "manifest"}]},)
    principal = Principal(user_id="caller", username="caller")
    with pytest.raises(ForbiddenError, match="permission denied"):
        assert_case_access(cases, principal, check_public_sources=True)
