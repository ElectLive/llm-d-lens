from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from llm_d_lens.agentic.public_benchmark_record.evaluation import EvaluationCase, EvaluationSuite, run_evaluation
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    Citation,
    QueryEvidence,
    QueryFilters,
    QueryRequest,
    QueryResponse,
)
from llm_d_lens.agentic.public_benchmark_record.service import plan_query
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN
from llm_d_lens.core.exceptions import ForbiddenError


def _case(**updates):
    return EvaluationCase(
        case_id="exact-value",
        category="numeric",
        request=QueryRequest(question="Reported throughput", limit=1),
        expected_returned=1,
        required_values=(Decimal("42"),),
        **updates,
    )


def _response():
    return QueryResponse(
        question="Reported throughput",
        intent="numeric",
        filters={},
        status="answered",
        answer="[public] throughput (mean (default; source statistic unreported)): 42 tokens/s; fixture.",
        evidence=(
            QueryEvidence(
                item_id="measurement",
                evidence_type="raw_json",
                metric="throughput",
                value=Decimal("42"),
                unit="tokens/s",
                citation=Citation(
                    origin_id="fixture",
                    source_revision="a" * 40,
                    source_path="fixture.json",
                    sha256="b" * 64,
                    artifact_uri="lens-artifact://fixture",
                    source_pointer={"json_pointer": "/throughput"},
                    published_manifest_id="c" * 64,
                ),
            ),
        ),
        matched_items=1,
        returned_observations=1,
        truncated=False,
    )


def test_fixed_suite_roundtrip_and_exact_report():
    suite = EvaluationSuite(name="fixture-v1", source_pins={"fixture": {"revision": "a" * 40}}, cases=(_case(),))
    assert EvaluationSuite.model_validate_json(suite.model_dump_json()) == suite
    calls = []

    def verify(case, evidence):
        calls.append((case.case_id, evidence.item_id))
        return ()

    report = run_evaluation(suite, execute=lambda _: _response(), verify_evidence=verify)
    assert report.passed and report.total_cases == report.passed_cases == 1
    assert report.verified_citations == report.numeric_observations == 1
    assert report.suite_sha256 == suite.digest
    assert report.source_pins == suite.source_pins
    assert calls == [("exact-value", "measurement")]
    assert "fixture.json" not in report.model_dump_json()


@pytest.mark.parametrize(
    "mutation,expected_issue",
    [
        ("answer-value", "answer_measurements_disagree_with_evidence"),
        ("unsupported-extra-claim", "answer_measurements_disagree_with_evidence"),
        ("evidence-value", "missing_exact_measurement"),
        ("internal", "internal_evidence_without_opt_in"),
        ("count", "returned_observations_mismatch"),
        ("hidden-id", "forbidden_answer_or_evidence_fragment"),
    ],
)
def test_evaluator_catches_corrupt_numeric_and_access_results(mutation, expected_issue):
    case = _case(forbidden_fragments=("private-run-id",))
    response = _response()
    if mutation == "answer-value":
        response = response.model_copy(update={"answer": response.answer.replace(": 42 ", ": 999 ")})
    elif mutation == "unsupported-extra-claim":
        response = response.model_copy(update={"answer": response.answer + "\n" + response.answer.replace("42", "999")})
    elif mutation == "evidence-value":
        response = response.model_copy(
            update={"evidence": (response.evidence[0].model_copy(update={"value": Decimal("999")}),)}
        )
    elif mutation == "internal":
        response = response.model_copy(
            update={"evidence": (response.evidence[0].model_copy(update={"origin": "agentic"}),)}
        )
    elif mutation == "count":
        response = response.model_copy(update={"returned_observations": 0})
    else:
        response = response.model_copy(update={"answer": response.answer + " private-run-id"})
    report = run_evaluation(
        EvaluationSuite(name="fixture", cases=(case,)), execute=lambda _: response, verify_evidence=lambda *_: ()
    )
    assert not report.passed and report.failed_cases == 1
    assert expected_issue in report.results[0].issues


def test_provenance_failure_is_not_counted_as_verified():
    report = run_evaluation(
        EvaluationSuite(name="fixture", cases=(_case(),)),
        execute=lambda _: _response(),
        verify_evidence=lambda *_: ("source_hash_mismatch",),
    )
    assert not report.passed and report.verified_citations == 0
    assert report.results[0].issues == ("source_hash_mismatch",)


def test_expected_denial_must_be_an_actual_domain_error():
    def deny(_):
        raise ForbiddenError("sensitive error details must not be copied to reports")

    case = _case(expected_error="forbidden")
    suite = EvaluationSuite(name="denial", cases=(case,))
    denied = run_evaluation(suite, execute=deny, verify_evidence=lambda *_: ())
    assert denied.passed and denied.results[0].error_code == "forbidden"
    assert "sensitive" not in denied.model_dump_json()
    accepted = run_evaluation(suite, execute=lambda _: _response(), verify_evidence=lambda *_: ())
    assert not accepted.passed and "expected_domain_error_missing" in accepted.results[0].issues
    unexpected = run_evaluation(
        EvaluationSuite(name="unexpected", cases=(_case(),)), execute=deny, verify_evidence=lambda *_: ()
    )
    assert not unexpected.passed


def test_unexpected_execution_failure_is_not_swallowed():
    def broken(_):
        raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        run_evaluation(EvaluationSuite(name="fixture", cases=(_case(),)), execute=broken, verify_evidence=lambda *_: ())
    with pytest.raises(ValueError, match="tuple"):
        run_evaluation(
            EvaluationSuite(name="fixture", cases=(_case(),)),
            execute=lambda _: _response(),
            verify_evidence=lambda *_: None,
        )


def test_suite_rejects_empty_duplicate_and_incoherent_expectations():
    with pytest.raises(ValidationError):
        EvaluationSuite(name="empty", cases=())
    with pytest.raises(ValidationError, match="unique"):
        EvaluationSuite(name="duplicate", cases=(_case(), _case()))
    with pytest.raises(ValidationError):
        _case(minimum_evidence=2)


def test_replay_preserves_unset_filter_scope_for_question_planning():
    request = QueryRequest(
        question="What was the recorded throughput for Prism ID 02c8f038-1cad-4ee8-a72e-724affa923f4?"
    )
    case = EvaluationCase(case_id="implicit-prism-scope", category="numeric", request=request)
    original_fields = request.filters.model_fields_set

    def execute(replayed):
        assert replayed.request.filters.model_fields_set == original_fields
        assert plan_query(replayed.request).filters.origins == (PRISM_ORIGIN,)
        return _response().model_copy(update={"question": request.question})

    report = run_evaluation(
        EvaluationSuite(name="unset-scope", cases=(case,)), execute=execute, verify_evidence=lambda *_: ()
    )
    assert report.passed
    explicit = case.model_copy(
        update={"request": request.model_copy(update={"filters": QueryFilters(origins=request.filters.origins)})}
    )
    assert (
        EvaluationSuite(name="scope", cases=(case,)).digest != EvaluationSuite(name="scope", cases=(explicit,)).digest
    )
