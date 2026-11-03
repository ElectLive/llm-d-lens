"""Read-only, fixed-case SQL evidence evaluation with mandatory provenance checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationModel, content_digest
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    EvidenceType,
    QueryEvidence,
    QueryRequest,
    QueryResponse,
)
from llm_d_lens.core.exceptions import DomainError

EvaluationCategory = Literal[
    "numeric", "conditions", "definition", "figure_gap", "false_join", "comparison", "access", "provenance", "overlap"
]
_NUMERIC_CLAIM = re.compile(
    r"^\[(public|agentic|public_and_agentic)\] (.*?) \((.*?)\): "
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) ([^;\n]+);",
    re.MULTILINE,
)


class EvaluationCase(PublicationModel):
    case_id: str = Field(min_length=1)
    category: EvaluationCategory
    actor: str = "reader"
    request: QueryRequest
    expected_status: Literal["answered", "partial", "abstained"] | None = "answered"
    expected_error: str | None = None
    expected_matched_items: int | None = Field(default=None, ge=0)
    expected_returned: int | None = Field(default=None, ge=0)
    minimum_evidence: int = Field(default=0, ge=0)
    expected_truncated: bool | None = None
    allowed_origins: tuple[str, ...] = ()
    allowed_evidence_types: tuple[EvidenceType, ...] = ()
    required_values: tuple[Decimal, ...] = ()
    answer_contains: tuple[str, ...] = ()
    forbidden_fragments: tuple[str, ...] = ()
    expected_abstentions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def explicit_outcome(self) -> Self:
        if self.expected_error is None and self.expected_status is None:
            raise ValueError("a case needs an expected response status or domain error")
        if any(not value.is_finite() for value in self.required_values):
            raise ValueError("expected measurement values must be finite")
        if self.expected_returned is not None and self.minimum_evidence > self.expected_returned:
            raise ValueError("minimum evidence cannot exceed the exact expected count")
        return self


class EvaluationSourcePin(PublicationModel):
    revision: str = Field(min_length=1)
    inventory_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class EvaluationSuite(PublicationModel):
    contract: Literal["benchmark-rag.evaluation.v1"] = "benchmark-rag.evaluation.v1"
    name: str = Field(min_length=1)
    source_pins: dict[str, EvaluationSourcePin] = Field(default_factory=dict)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation case IDs must be unique")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json", exclude_unset=True))


class CaseResult(PublicationModel):
    case_id: str
    category: EvaluationCategory
    passed: bool
    issues: tuple[str, ...]
    evidence_count: int
    numeric_count: int
    verified_citations: int
    error_code: str | None = None


class EvaluationReport(PublicationModel):
    contract: Literal["benchmark-rag.evaluation-report.v1"] = "benchmark-rag.evaluation-report.v1"
    suite: str
    suite_sha256: str
    source_pins: dict[str, EvaluationSourcePin]
    started_at: datetime
    finished_at: datetime
    total_cases: int
    passed_cases: int
    failed_cases: int
    passed: bool
    category_counts: dict[EvaluationCategory, int]
    verified_citations: int
    numeric_observations: int
    results: tuple[CaseResult, ...]


def _response_issues(case: EvaluationCase, response: QueryResponse) -> list[str]:
    issues = []
    if case.expected_error is not None:
        issues.append("expected_domain_error_missing")
    if case.expected_status is not None and response.status != case.expected_status:
        issues.append("unexpected_response_status")
    if response.question != case.request.question:
        issues.append("response_question_mismatch")
    if response.internal_branch_enabled != case.request.include_internal:
        issues.append("internal_opt_in_mismatch")
    if response.returned_observations != len(response.evidence):
        issues.append("returned_observations_mismatch")
    if len(response.evidence) > case.request.limit:
        issues.append("response_exceeds_limit")
    if len({entry.item_id for entry in response.evidence}) != len(response.evidence):
        issues.append("duplicate_evidence_identity")
    if case.expected_matched_items is not None and response.matched_items != case.expected_matched_items:
        issues.append("unexpected_match_count")
    if case.expected_returned is not None and len(response.evidence) != case.expected_returned:
        issues.append("unexpected_evidence_count")
    if len(response.evidence) < case.minimum_evidence:
        issues.append("missing_required_evidence")
    if case.expected_truncated is not None and response.truncated != case.expected_truncated:
        issues.append("unexpected_truncation")
    values = Counter(entry.value for entry in response.evidence if entry.value is not None)
    if Counter(case.required_values) - values:
        issues.append("missing_exact_measurement")
    claims = Counter(
        (origin, metric, statistic, Decimal(value), unit)
        for origin, metric, statistic, value, unit in _NUMERIC_CLAIM.findall(response.answer)
    )
    supported = Counter(
        (
            entry.origin,
            entry.metric,
            entry.statistic if entry.statistic is not None else "mean (default; source statistic unreported)",
            entry.value,
            entry.unit or "[unit not reported]",
        )
        for entry in response.evidence
        if entry.value is not None
    )
    if claims != supported:
        issues.append("answer_measurements_disagree_with_evidence")
    if any(text.casefold() not in response.answer.casefold() for text in case.answer_contains):
        issues.append("missing_expected_answer_context")
    visible = response.model_dump_json(include={"answer", "evidence", "warnings", "abstentions"})
    if any(fragment in visible for fragment in case.forbidden_fragments):
        issues.append("forbidden_answer_or_evidence_fragment")
    if not set(case.expected_abstentions).issubset(abstention.code for abstention in response.abstentions):
        issues.append("missing_required_abstention")
    return issues


def run_evaluation(
    suite: EvaluationSuite,
    *,
    execute: Callable[[EvaluationCase], QueryResponse],
    verify_evidence: Callable[[EvaluationCase, QueryEvidence], tuple[str, ...]],
) -> EvaluationReport:
    """Run authorized SQL cases; unexpected errors propagate and cannot become passes.

    Callers independently verify every citation against retained bytes or authorized
    records. Reports contain only case IDs, counts and issue codes, never private responses.
    """
    suite = EvaluationSuite.model_validate(suite.model_dump(exclude_unset=True))
    started = datetime.now(UTC)
    results = []
    for case in suite.cases:
        try:
            response = execute(case)
        except DomainError as error:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    passed=error.code == case.expected_error,
                    issues=() if error.code == case.expected_error else ("unexpected_domain_error",),
                    evidence_count=0,
                    numeric_count=0,
                    verified_citations=0,
                    error_code=error.code,
                )
            )
            continue
        response = QueryResponse.model_validate(response.model_dump())
        issues = _response_issues(case, response)
        verified = 0
        for evidence in response.evidence:
            if case.allowed_origins and evidence.citation.origin_id not in case.allowed_origins:
                issues.append("unexpected_evidence_origin")
            if case.allowed_evidence_types and evidence.evidence_type not in case.allowed_evidence_types:
                issues.append("unexpected_evidence_type")
            if not case.request.include_internal and (evidence.origin != "public" or evidence.corroborating_citations):
                issues.append("internal_evidence_without_opt_in")
            citation_issues = verify_evidence(case, evidence)
            if not isinstance(citation_issues, tuple) or any(
                not isinstance(issue, str) or not issue for issue in citation_issues
            ):
                raise ValueError("evidence verifier must return a tuple of nonempty issue codes")
            if citation_issues:
                issues.extend(citation_issues)
            else:
                verified += 1 + len(evidence.corroborating_citations)
        results.append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                passed=not issues,
                issues=tuple(dict.fromkeys(issues)),
                evidence_count=len(response.evidence),
                numeric_count=sum(evidence.value is not None for evidence in response.evidence),
                verified_citations=verified,
            )
        )
    passed = sum(result.passed for result in results)
    return EvaluationReport(
        suite=suite.name,
        suite_sha256=suite.digest,
        source_pins=suite.source_pins,
        started_at=started,
        finished_at=datetime.now(UTC),
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=len(results) - passed,
        passed=passed == len(results),
        category_counts=dict(Counter(case.category for case in suite.cases)),
        verified_citations=sum(result.verified_citations for result in results),
        numeric_observations=sum(result.numeric_count for result in results),
        results=tuple(results),
    )
