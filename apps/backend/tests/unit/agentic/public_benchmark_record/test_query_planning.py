import pytest

from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters, QueryRequest
from llm_d_lens.agentic.public_benchmark_record.service import plan_query
from llm_d_lens.core.exceptions import DomainValidationError


@pytest.mark.parametrize(
    ("phrase", "statistic"),
    [
        ("p99", "p99"),
        ("P95", "p95"),
        ("99th percentile", "p99"),
        ("99.9th percentile", "p99.9"),
        ("mean", "mean"),
        ("average", "mean"),
        ("avg", "mean"),
        ("median", "p50"),
        ("minimum", "min"),
        ("maximum", "max"),
    ],
)
def test_question_statistics_are_required_filters(phrase, statistic):
    plan = plan_query(QueryRequest(question=f"What was the {phrase} latency for Qwen3-0.6B on H100?"))
    assert plan.filters.statistic == statistic
    assert plan.precondition is None


@pytest.mark.parametrize(
    "model",
    [
        "Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        "Mistral-7B",
        "DeepSeek-R1",
        "new-vendor/Unknown-8B",
        "Qwen/Qwen3-32B",
        "Falcon",
    ],
)
def test_model_names_are_not_limited_to_known_families(model):
    plan = plan_query(QueryRequest(question=f"What was the throughput for {model} on H100?"))
    assert plan.filters.model == model
    assert plan.precondition is None


def test_explicitly_named_model_without_family_or_size():
    plan = plan_query(QueryRequest(question='Show throughput on H100 for model "Custom Model".'))
    assert plan.filters.model == "Custom Model"


@pytest.mark.parametrize(
    ("question", "filters", "field"),
    [
        ("p99 latency for Qwen3-0.6B on H100", QueryFilters(statistic="mean"), "statistic"),
        ("Throughput for Llama-3.1-70B on H100", QueryFilters(model="Qwen3-0.6B"), "model"),
    ],
)
def test_question_filter_conflicts_are_rejected(question, filters, field):
    with pytest.raises(DomainValidationError, match=field):
        plan_query(QueryRequest(question=question, filters=filters))


def test_matching_explicit_filters_and_statistic_aliases_are_preserved():
    plan = plan_query(
        QueryRequest(
            question="Average latency for Llama-3.1-70B on H100",
            filters=QueryFilters(model="llama-3.1-70b", statistic="avg"),
        )
    )
    assert plan.filters.model == "llama-3.1-70b"
    assert plan.filters.statistic == "mean"
    assert plan.precondition is None


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("Compare p50 and p99 latency for Qwen3-0.6B on H100", "statistic_scope_required"),
        ("Compare Llama-3.1-70B and Qwen3-0.6B throughput on H100", "model_scope_required"),
        ("Throughput for an unspecified model on H100", "model_scope_required"),
    ],
)
def test_ambiguous_question_conditions_abstain_before_retrieval(question, code):
    plan = plan_query(QueryRequest(question=question))
    assert plan.precondition is not None
    assert plan.precondition.code == code


def test_hardware_and_record_ids_are_not_model_names():
    plan = plan_query(QueryRequest(question="Throughput for Prism ID 02c8f038-1cad-4ee8-a72e-724affa923f4 on H100"))
    assert plan.filters.model is None
    assert plan.filters.record_id == "02c8f038-1cad-4ee8-a72e-724affa923f4"
    assert plan.precondition is None


def test_hardware_only_question_does_not_invent_a_model_filter():
    plan = plan_query(QueryRequest(question="What was the mean throughput for H100?"))
    assert plan.filters.model is None
    assert plan.filters.hardware == "H100"
    assert plan.precondition is None


@pytest.mark.parametrize("hardware", ["Intel Gaudi 3", "Gaudi3", "MI355X", "H100"])
def test_explicit_hardware_condition_is_never_discarded(hardware):
    plan = plan_query(QueryRequest(question=f"What was throughput for Qwen3-0.6B on {hardware}?"))
    assert plan.filters.hardware == hardware
    assert plan.filters.model == "Qwen3-0.6B"
    assert plan.precondition is None


def test_unknown_hardware_requires_clarification_without_broadening_query():
    plan = plan_query(QueryRequest(question="Throughput for Qwen3-0.6B on Mystery-Accelerator9?"))
    assert plan.filters.hardware == "Mystery-Accelerator9"
    assert plan.precondition.code == "hardware_scope_required"
    explicit = plan_query(
        QueryRequest(
            question="Throughput for Qwen3-0.6B on Mystery-Accelerator9?",
            filters=QueryFilters(hardware="Mystery-Accelerator9"),
        )
    )
    assert explicit.filters.hardware == "Mystery-Accelerator9" and explicit.precondition is None


@pytest.mark.parametrize(
    "question",
    [
        "Throughput for Qwen3-0.6B on?",
        "Throughput for Qwen3-0.6B hardware: Unknown9?",
        "Throughput for Qwen3-0.6B on Unknown9?",
    ],
)
def test_unknown_or_missing_named_hardware_fails_closed(question):
    plan = plan_query(QueryRequest(question=question))
    assert plan.precondition.code == "hardware_scope_required"


def test_multiword_hardware_subject_is_not_mistaken_for_a_model():
    plan = plan_query(QueryRequest(question="What was throughput for Intel Gaudi 3?"))
    assert plan.filters.hardware == "Intel Gaudi 3"
    assert plan.filters.model is None and plan.precondition is None


def test_conflicting_or_compound_hardware_cannot_drop_conditions():
    with pytest.raises(DomainValidationError, match="hardware"):
        plan_query(QueryRequest(question="Throughput on MI355X", filters=QueryFilters(hardware="H100")))
    plan = plan_query(QueryRequest(question="Throughput for Qwen3-0.6B on H100 and MI355X?"))
    assert plan.precondition.code == "hardware_scope_required"


@pytest.mark.parametrize("suffix", [":stage:0", ":stage:2", ":matrix:1:stage:2"])
def test_internal_question_keeps_full_stage_identity(suffix):
    identity = f"evaluate:02c8f038-1cad-4ee8-a72e-724affa923f4{suffix}"
    plan = plan_query(QueryRequest(question=f"Throughput for {identity}?", include_internal=True))
    assert plan.filters.record_id == identity
    assert plan.precondition is None


@pytest.mark.parametrize("suffix", [":stage:-1", ":stage:01", ":matrix:1", ":matrix:1:stage:2:extra"])
def test_malformed_stage_question_cannot_fall_back_to_whole_run(suffix):
    plan = plan_query(
        QueryRequest(
            question=f"Throughput for evaluate:02c8f038-1cad-4ee8-a72e-724affa923f4{suffix}?", include_internal=True
        )
    )
    assert plan.precondition.code == "record_scope_required"


def test_distinct_stages_of_same_run_are_not_collapsed_to_one_record_query():
    identity = "evaluate:02c8f038-1cad-4ee8-a72e-724affa923f4"
    plan = plan_query(
        QueryRequest(
            question=f"Compare throughput for {identity}:stage:0 and {identity}:stage:1?", include_internal=True
        )
    )
    assert plan.precondition.code == "comparison_requires_explicit_scope"


def test_question_terms_do_not_override_an_explicit_prism_source_path():
    plan = plan_query(
        QueryRequest(
            question="Compare H200 gpt-oss-120b weight-load times",
            filters=QueryFilters(
                model="openai/gpt-oss-120b",
                source_path="archives/llm-d-benchmarks.json",
            ),
        )
    )
    assert plan.filters.model == "openai/gpt-oss-120b"
    assert plan.filters.source_path == "archives/llm-d-benchmarks.json"
    assert plan.precondition is None


@pytest.mark.parametrize(
    "question",
    [
        "Compare H200 gpt-oss-120b weight-load times",
        "What is a benchmark report intended to capture?",
        "Show Qwen3 Coder TPU PNG",
    ],
)
def test_legacy_guide_labels_do_not_route_queries_to_other_sources(question):
    plan = plan_query(QueryRequest(question=question))
    assert plan.filters.origins == ("prism:public-snapshot",)
    assert plan.filters.source_path is None


@pytest.mark.parametrize(
    "question",
    [
        "What is TTFT?",
        "What is throughput?",
        "What is the TPOT metric?",
        "Define request latency.",
        "Explain time to first token.",
        "What does TTFT mean?",
        "What does TPOT stand for?",
        "Compare the definitions of TTFT and TPOT.",
    ],
)
def test_metric_definitions_use_passage_search(question):
    plan = plan_query(QueryRequest(question=question))
    assert plan.intent == "definition"
    assert plan.filters.metric is None
    assert plan.precondition is None
    assert plan.text_query


@pytest.mark.parametrize(
    ("phrase", "metric"),
    [
        ("TTFT", "ttft"),
        ("time to first token", "ttft"),
        ("TPOT", "tpot"),
        ("time per output token", "tpot"),
        ("latency", "latency"),
        ("throughput", "throughput"),
        ("weight loading time", "weight_load"),
    ],
)
def test_metric_measurements_use_the_same_vocabulary_for_intent_and_filter(phrase, metric):
    plan = plan_query(QueryRequest(question=f"What is the {phrase} for Llama-3.1-70B on H100?"))
    assert plan.intent == "numeric"
    assert plan.filters.metric == metric
    assert plan.filters.model == "Llama-3.1-70B"
    assert plan.precondition is None


def test_explicit_intent_is_not_overridden_by_definition_words():
    plan = plan_query(
        QueryRequest(question="What is TTFT?", intent="numeric", filters=QueryFilters(model="Qwen3-0.6B"))
    )
    assert plan.intent == "numeric"
    assert plan.filters.metric == "ttft"


def test_explicit_measurement_scope_disambiguates_a_short_question():
    plan = plan_query(QueryRequest(question="What is TTFT?", filters=QueryFilters(model="Qwen3-0.6B")))
    assert plan.intent == "numeric"
    assert plan.filters.metric == "ttft"


def test_explicit_metric_filter_preserves_numeric_intent():
    plan = plan_query(QueryRequest(question="What is TTFT?", filters=QueryFilters(metric="ttft")))
    assert plan.intent == "numeric"
    assert plan.filters.metric == "ttft"
    assert plan.precondition is not None
    assert plan.precondition.code == "experiment_conditions_required"


@pytest.mark.parametrize("quote", ['"', "`"])
@pytest.mark.parametrize("model", ["Custom Model-8B", "Custom org/Model-8B"])
def test_quoted_model_is_one_identifier_not_its_embedded_tokens(quote, model):
    plan = plan_query(QueryRequest(question=f"Show throughput on H100 for model {quote}{model}{quote}."))
    assert plan.filters.model == model
    assert plan.precondition is None


@pytest.mark.parametrize(
    "question",
    [
        "Show throughput for model on H100",
        "Show throughput on H100 for model",
        "Show throughput for model: on H100",
        'Show throughput for model="" on H100',
        'Show throughput for model " " on H100',
    ],
)
def test_missing_labeled_model_abstains_instead_of_using_a_placeholder(question):
    plan = plan_query(QueryRequest(question=question))
    assert plan.precondition is not None
    assert plan.precondition.code == "model_scope_required"
    assert plan.filters.model is None


def test_model_identifier_containing_model_is_not_a_label():
    plan = plan_query(QueryRequest(question="Show throughput for model-8B on H100"))
    assert plan.filters.model == "model-8B"
    assert plan.precondition is None


def test_record_description_is_not_an_ambiguous_model_subject():
    plan = plan_query(
        QueryRequest(
            question="Give throughput for an unregistered baseline record",
            filters=QueryFilters(record_id="unregistered-baseline-record", metric="throughput"),
        )
    )
    assert plan.filters.record_id == "unregistered-baseline-record"
    assert plan.filters.model is None
    assert plan.precondition is None


def test_record_filter_does_not_discard_a_named_model_condition():
    plan = plan_query(
        QueryRequest(
            question="Give throughput for Llama-3.1-70B on H100",
            filters=QueryFilters(record_id="record-a"),
        )
    )
    assert plan.filters.record_id == "record-a"
    assert plan.filters.model == "Llama-3.1-70B"
    assert plan.precondition is None
