from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkMetrics,
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    WorkloadProfile,
)
from llm_d_lens.agentic.public_benchmark_record.agentic_answers import (
    agentic_evidence,
    merge_verified,
    pinned_evaluate_identity,
)
from llm_d_lens.agentic.public_benchmark_record.agentic_metrics import measured_values
from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    QUERY_PERMISSION,
    AgenticCitation,
    Citation,
    QueryEvidence,
    QueryFilters,
    QueryRequest,
)
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService, plan_query
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN
from llm_d_lens.auth.contracts import Principal, RoleBinding
from llm_d_lens.core.exceptions import ForbiddenError
from llm_d_lens.db.dao.public_benchmark_record_query import QueryMatch, QuerySlice

RUN = "8321fc28-7c64-4ca1-9d3e-281350a0f386"
IDENTITY = f"evaluate:{RUN}"
RAW = json.dumps([{"id": IDENTITY, "throughput": 42}]).encode()
DIGEST = hashlib.sha256(RAW).hexdigest()
PATH = "archives/llm-d-benchmarks.json"


@dataclass(frozen=True)
class Match:
    record: BenchmarkRecord
    evaluate_run_id: str = RUN
    cluster_id: str = "cluster1"
    owner_user_id: str = "caller"
    version_id: int = 1


@dataclass(frozen=True)
class Slice:
    matches: tuple[Match, ...]
    total: int


def match() -> Match:
    return Match(
        BenchmarkRecord(
            benchmark_id=IDENTITY,
            timestamp=datetime(2026, 10, 19, tzinfo=UTC),
            outcome="succeeded",
            model=ModelDescriptor(
                model_id="Qwen3-0.6B",
                family="Qwen",
                architecture="dense",
                parameter_count_b=0.6,
                quantization="bf16",
                tokenizer_id="Qwen3-0.6B",
            ),
            hardware=HardwareDescriptor(accelerator="H100", vram_per_gpu_gib=80, gpu_count=1),
            runtime=RuntimeFingerprint(backend="vllm", backend_version="unknown", accelerator_runtime="unknown"),
            workload=WorkloadProfile(
                mean_input_tokens=100,
                p95_input_tokens=200,
                mean_output_tokens=50,
                concurrency=4,
                request_rate=1,
                shared_prefix_ratio=0,
                prefill_fraction=0,
            ),
            configuration=CandidateConfiguration(topology="baseline", decode_tp=1, decode_replicas=1),
            metrics=BenchmarkMetrics(throughput_tokens_per_s=42, ttft_p95_ms=10, error_rate=0),
        )
    )


def principal() -> Principal:
    return Principal(
        user_id="caller",
        username="caller",
        bindings=(RoleBinding("reader"),),
        role_permissions={"reader": frozenset({QUERY_PERMISSION, "evaluate:run:read"})},
    )


def public_match(identity: str = IDENTITY) -> QueryMatch:
    common = {
        "origin_id": PRISM_ORIGIN,
        "source_revision": "2026-10-13",
        "content_sha256": DIGEST,
        "access_classification": "verified_public",
        "source_path": PATH,
        "format": "json",
    }
    experiment = PublicBenchmarkRecord(
        **common,
        item_id="experiment",
        logical_key="experiment",
        kind="experiment",
        model="Qwen3-0.6B",
        hardware="H100",
        runtime="vllm",
        workload={"concurrency": 4},
        source_pointer={"record_id": identity, "json_pointer": "/0"},
    )
    item = PublicBenchmarkRecord(
        **common,
        item_id="measurement",
        logical_key="measurement",
        kind="measurement",
        parent_id=experiment.item_id,
        experiment_id=experiment.item_id,
        metric="throughput",
        value=Decimal("42"),
        original_value=Decimal("42"),
        unit="tokens/s",
        original_unit="tokens/s",
        evidence_type="raw_json",
        source_pointer={"json_pointer": "/0/throughput"},
    )
    source = PublicBenchmarkRecord(
        **common,
        item_id="file",
        logical_key="file",
        kind="source",
        source_role="file",
        source_details={"path": PATH},
    )
    return QueryMatch(item=item, experiment=experiment, source=source, manifest_id="manifest", generation=1)


def public_evidence() -> QueryEvidence:
    internal = agentic_evidence(match(), QueryFilters(metric="throughput"))[0]
    return internal.model_copy(
        update={
            "item_id": "public",
            "origin": "public",
            "evidence_type": "raw_json",
            "citation": Citation(
                origin_id=PRISM_ORIGIN,
                source_revision="2026-10-13",
                source_path=PATH,
                sha256=DIGEST,
                artifact_uri="lens-artifact://benchmark-rag/source",
                source_pointer={"json_pointer": "/0/throughput"},
                published_manifest_id="manifest",
            ),
        }
    )


def test_default_never_consults_internal_branch():
    dao = Mock(spec=["search", "assert_current"])
    dao.search.return_value = QuerySlice(matches=(), total=0)
    internal = Mock(spec=["search", "assert_current"])
    response = PublicBenchmarkRecordQueryService(dao=dao, agentic_dao=internal).query(
        QueryRequest(question="Throughput for Qwen3-0.6B"),
        principal(),
    )
    internal.search.assert_not_called()
    internal.assert_current.assert_not_called()
    assert response.internal_branch_enabled is False
    assert response.returned_observations == 0


def test_explicit_internal_query_has_stored_float_citation_and_bounded_output():
    dao = Mock(spec=["search", "assert_current"])
    dao.search.return_value = QuerySlice(matches=(), total=0)
    internal = Mock(spec=["search", "assert_current"])
    internal.search.return_value = Slice(matches=(match(),), total=1)
    response = PublicBenchmarkRecordQueryService(dao=dao, agentic_dao=internal).query(
        QueryRequest(
            question=f"Internal benchmark throughput for {IDENTITY}",
            include_internal=True,
        ),
        principal(),
    )
    assert response.status == "answered" and response.internal_branch_enabled is True
    assert response.returned_observations == response.matched_items == 1
    entry = response.evidence[0]
    assert entry.origin == "agentic" and entry.evidence_type == "agentic_measurement"
    assert isinstance(entry.citation, AgenticCitation)
    assert entry.citation.resource_uri.endswith(RUN)
    assert entry.citation.numeric_precision == "stored_float"
    assert entry.value == Decimal("42") and entry.unit == "tokens/s"
    assert "agentic" in response.answer
    internal.assert_current.assert_called_once()
    assert internal.search.call_args.args[0].record_id == IDENTITY


def test_opt_in_does_not_replace_feature_permission():
    dao = Mock(spec=["search", "assert_current"])
    internal = Mock(spec=["search", "assert_current"])
    with pytest.raises(ForbiddenError):
        PublicBenchmarkRecordQueryService(dao=dao, agentic_dao=internal).query(
            QueryRequest(question="Internal throughput", include_internal=True),
            replace(principal(), role_permissions={"reader": frozenset({"evaluate:run:read"})}),
        )
    dao.search.assert_not_called()
    internal.search.assert_not_called()


def test_limit_is_across_both_branches_not_per_metric_or_branch():
    dao = Mock(spec=["search", "assert_current"])
    dao.search.return_value = QuerySlice(matches=(), total=0)
    internal = Mock(spec=["search", "assert_current"])
    internal.search.return_value = Slice(matches=(match(),), total=1)
    response = PublicBenchmarkRecordQueryService(dao=dao, agentic_dao=internal).query(
        QueryRequest(
            question="Reported metrics",
            intent="numeric",
            include_internal=True,
            limit=1,
            filters=QueryFilters(record_id=IDENTITY),
        ),
        principal(),
    )
    assert len(response.evidence) == response.returned_observations == 1
    assert response.truncated is True and response.status == "partial"


def test_same_record_different_metrics_are_not_a_complete_comparison():
    dao = Mock(spec=["search", "assert_current"])
    dao.search.return_value = QuerySlice(matches=(), total=0)
    internal = Mock(spec=["search", "assert_current"])
    internal.search.return_value = Slice(matches=(match(),), total=1)
    response = PublicBenchmarkRecordQueryService(dao=dao, agentic_dao=internal).query(
        QueryRequest(
            question="Compare recorded metrics",
            intent="comparison",
            include_internal=True,
            filters=QueryFilters(record_id=IDENTITY),
        ),
        principal(),
    )
    assert response.status == "partial"
    assert any(item.code == "comparison_incomplete" for item in response.abstentions)


def test_verified_merge_keeps_both_citations_and_one_observation():
    public = public_evidence()
    internal = agentic_evidence(match(), QueryFilters(metric="throughput"))[0]
    assert pinned_evaluate_identity(public_match(), RAW) == IDENTITY
    merged = merge_verified([public], [internal], {"public": IDENTITY})
    assert len(merged) == 1 and merged[0].origin == "public_and_agentic"
    assert merged[0].corroborating_citations == (internal.citation,)
    assert merged[0].identity_verification == "canonical_evaluate_run_and_metric"
    assert merged[0].citation == public.citation


@pytest.mark.parametrize(
    "change",
    [
        {"value": Decimal("43")},
        {"unit": "output tokens/s"},
        {"statistic": "mean"},
        {"series": {"load": 2}},
    ],
)
def test_run_identity_without_same_metric_scope_does_not_merge(change):
    public = public_evidence().model_copy(update=change)
    internal = agentic_evidence(match(), QueryFilters(metric="throughput"))[0]
    assert len(merge_verified([public], [internal], {"public": IDENTITY})) == 2


def test_similarity_and_forged_source_pointer_do_not_prove_identity():
    internal = agentic_evidence(match(), QueryFilters(metric="throughput"))[0]
    assert len(merge_verified([public_evidence()], [internal], {})) == 2
    assert pinned_evaluate_identity(public_match(), b'[{"id":"different","throughput":42}]') is None
    assert pinned_evaluate_identity(public_match("evaluate:simulation-id"), RAW) is None
    assert pinned_evaluate_identity(public_match(RUN), RAW) is None


def test_measured_filters_default_missing_statistics_to_mean_without_guessing_denominator():
    record = match().record
    mean = measured_values(record, QueryFilters(metric="throughput", statistic="mean"))
    assert len(mean) == 1 and mean[0][1] == Decimal("42.0")
    assert mean[0][0].statistic is None
    assert not measured_values(record, QueryFilters(metric="throughput", statistic="p95"))
    assert not measured_values(record, QueryFilters(metric="throughput", unit="output tokens/s"))
    assert not measured_values(record, QueryFilters(metric="error_rate", normalized=True))
    values = measured_values(record, QueryFilters(metric="ttft", statistic="p95", unit="ms"))
    assert len(values) == 1 and values[0][1] == Decimal("10")


def test_question_default_preserves_internal_abstention_but_opt_in_allows_it():
    assert (
        plan_query(QueryRequest(question="Internal benchmark throughput")).precondition.code
        == "internal_branch_disabled"
    )
    plan = plan_query(
        QueryRequest(
            question=f"Internal benchmark throughput for {IDENTITY}",
            include_internal=True,
        )
    )
    assert plan.precondition is None and plan.filters.record_id == IDENTITY
    explicit = plan_query(
        QueryRequest(
            question=f"Throughput for {RUN}",
            include_internal=True,
            filters=QueryFilters(record_id=IDENTITY),
        )
    )
    assert explicit.filters.record_id == IDENTITY


def test_conflicting_runtime_or_workload_is_not_a_verified_duplicate():
    internal = agentic_evidence(match(), QueryFilters(metric="throughput"))[0]
    public = public_evidence()
    for changed in (
        public.experiment.model_copy(update={"runtime": "different-runtime"}),
        public.experiment.model_copy(update={"workload": {"concurrency": 100}}),
    ):
        candidate = public.model_copy(update={"experiment": changed})
        assert len(merge_verified([candidate], [internal], {"public": IDENTITY})) == 2
