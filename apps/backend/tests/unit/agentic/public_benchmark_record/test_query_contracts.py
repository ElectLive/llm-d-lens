from decimal import Decimal

import pytest

from llm_d_lens.agentic.public_benchmark_record.contracts import model_identities_match, model_identity_aliases
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    Citation,
    ExperimentContext,
    QueryEvidence,
    QueryFilters,
    QueryRequest,
    QueryResponse,
)

CELL = " 30.3 s "
REPORT = "\nOriginal report with significant boundary whitespace.\n"
POINTER = {
    "cell": CELL,
    "subvalue": {"start": 1, "end": 5, "text": "30.3"},
    " raw key ": " raw value ",
}


@pytest.mark.parametrize(
    ("name", "aliases"),
    [
        ("Publisher/Falcon-7B", ("publisher/falcon-7b", "falcon-7b", "publisher-falcon-7b")),
        (" OtherOrg/CustomModel ", ("otherorg/custommodel", "custommodel", "otherorg-custommodel")),
        ("LOCAL/models/Widget-v2-FP8", ("local/models/widget-v2-fp8", "widget-v2-fp8", "local-models-widget-v2-fp8")),
        ("Publisher-Falcon-7B", ("publisher-falcon-7b",)),
        ("Falcon-7B", ("falcon-7b",)),
    ],
)
def test_model_aliases_are_generated_for_every_name_without_a_registry(name, aliases):
    assert model_identity_aliases(name) == aliases


@pytest.mark.parametrize(
    ("first", "second", "matches"),
    [
        ("Publisher/Falcon-7B", "falcon-7b", True),
        ("Publisher/Falcon-7B", "PUBLISHER-FALCON-7B", True),
        ("Publisher/Falcon-7B", "another-publisher/Falcon-7B", True),
        ("Publisher/Falcon-7B-FP8", "Falcon-7B", False),
        ("Publisher/Falcon-v2", "Falcon-v1", False),
        ("org/model%_v2", "model%_v2", True),
        ("Publisher-Falcon-7B", "Falcon-7B", False),
        (None, "model", False),
        ("", "model", False),
    ],
)
def test_model_alias_matching_compares_both_sets_symmetrically(first, second, matches):
    assert model_identities_match(first, second) is matches
    assert model_identities_match(second, first) is matches


def test_blank_model_name_is_not_a_search_alias():
    with pytest.raises(ValueError, match="must not be blank"):
        model_identity_aliases("  ")


def response() -> QueryResponse:
    citation = Citation(
        origin_id="https://github.com/llm-d/llm-d",
        source_revision="a" * 40,
        source_path="guides/example.md",
        sha256="b" * 64,
        artifact_uri=f"lens-artifact://benchmark-rag/{'b' * 64}/source",
        source_pointer=POINTER,
        published_manifest_id="c" * 64,
    )
    context = ExperimentContext(
        item_id="d" * 64,
        origin_id=citation.origin_id,
        model="Example",
        hardware=None,
        runtime=None,
        workload={" original key ": " original condition "},
        configuration={"published_report": REPORT},
        source_pointer=POINTER,
    )
    evidence = QueryEvidence(
        item_id="e" * 64,
        evidence_type="published_table",
        citation=citation,
        experiment=context,
        metric="weight_load_time",
        value=Decimal("30.3"),
        unit="s",
        text=REPORT,
    )
    return QueryResponse(
        question="Example question",
        intent="numeric",
        filters=QueryFilters(),
        status="answered",
        answer="Example answer",
        evidence=(evidence,),
        matched_items=1,
        truncated=False,
        returned_observations=1,
    )


@pytest.mark.parametrize("round_trip", [False, True])
def test_response_preserves_original_evidence_strings_and_offsets(round_trip: bool) -> None:
    result = response()
    if round_trip:
        result = QueryResponse.model_validate_json(result.model_dump_json())
    evidence = result.evidence[0]
    assert isinstance(evidence.citation, Citation)
    assert evidence.citation.source_pointer == POINTER
    cell = evidence.citation.source_pointer["cell"]
    assert isinstance(cell, str)
    assert Decimal(cell[1:5]) == evidence.value
    assert evidence.text == REPORT
    assert evidence.experiment is not None
    assert evidence.experiment.configuration == {"published_report": REPORT}
    assert evidence.experiment.workload == {" original key ": " original condition "}
    assert evidence.experiment.source_pointer == POINTER


def test_request_still_trims_question_and_filter_strings() -> None:
    request = QueryRequest.model_validate({"question": "  Find throughput  ", "filters": {"metric": " throughput "}})
    assert request.question == "Find throughput"
    assert request.filters.metric == "throughput"
