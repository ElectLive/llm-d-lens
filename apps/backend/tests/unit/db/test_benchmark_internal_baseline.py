from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.public_benchmark_record.evaluation import EvaluationSuite, run_evaluation
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import item_id
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION, AgenticCitation, Citation
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService, _verify_measurement
from llm_d_lens.auth.contracts import Principal, RoleBinding
from llm_d_lens.db.dao.agentic_benchmark_record import BenchmarkRecordDao
from llm_d_lens.db.dao.benchmark_experiment_query import BenchmarkExperimentQueryDao
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao, QueryMatch
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow
from llm_d_lens.db.models.evaluate_run import EvaluateRunRow
from llm_d_lens.db.models.user import UserRow
from tests.unit.db.test_benchmark_experiment_service import publish, record

_RUNS = ("11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222")
_ORPHAN = "33333333-3333-4333-8333-333333333333"
_USERS = ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
_ADMIN = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
_CASES = Path(__file__).resolve().parents[2] / "fixtures/public_benchmark_record/internal_baseline_v1.json"


def test_fixed_authorized_internal_baseline(postgres_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "baseline-evidence"))
    factory = sessionmaker(bind=postgres_engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add_all(
            UserRow(id=identifier, username=f"baseline-{index}") for index, identifier in enumerate((*_USERS, _ADMIN))
        )
        session.flush()
        session.add_all(
            EvaluateRunRow(
                id=run_id,
                kind="benchmark",
                status="succeeded",
                cluster_id="cluster1",
                extra={"owner_user_id": owner},
            )
            for run_id, owner in zip(_RUNS, _USERS, strict=True)
        )
        session.add(EvaluateRunRow(id=_ORPHAN, kind="benchmark", status="succeeded", cluster_id="cluster1"))
    for run_id in (*_RUNS, _ORPHAN, "deadbeef"):
        BenchmarkRecordDao(factory).save(record(run_id))
    reader = Principal(
        user_id=_USERS[0],
        username="baseline-0",
        bindings=(RoleBinding("rag"), RoleBinding("self")),
        role_permissions={"rag": frozenset({QUERY_PERMISSION}), "self": frozenset({"evaluate:run:read"})},
        role_self_scoped={"self": frozenset({"evaluate:run:read"})},
    )
    actors = {
        "reader": reader,
        "second": replace(reader, user_id=_USERS[1], username="baseline-1"),
        "no-evaluate": replace(reader, bindings=(RoleBinding("rag"),)),
        "no-rag": replace(reader, bindings=(RoleBinding("self"),)),
        "anonymous": None,
        "stale": replace(reader, authz_version=reader.authz_version + 1),
        "admin": Principal(user_id=_ADMIN, username="baseline-2", bindings=(RoleBinding("admin"),)),
    }
    allowed = {"reader": {_RUNS[0]}, "second": {_RUNS[1]}, "admin": set(_RUNS), "no-evaluate": set()}
    service = PublicBenchmarkRecordQueryService(
        PublicBenchmarkRecordQueryDao(factory), agentic_dao=BenchmarkExperimentQueryDao(factory)
    )
    publish(service, _RUNS[0])
    suite = EvaluationSuite.model_validate_json(_CASES.read_bytes())
    writes = []
    with factory() as session:
        before = {row.benchmark_id: row.to_dto() for row in session.scalars(select(BenchmarkRecordRow))}

    def audit(_connection, _cursor, statement, *_args):
        if not statement.lstrip().upper().startswith(("SELECT", "WITH", "SET TRANSACTION")):
            writes.append(statement.split()[0])

    def verify_internal(case, evidence, citation):
        issues = []
        if citation.evaluate_run_id not in allowed.get(case.actor, set()):
            issues.append("unauthorized_internal_citation")
        with factory() as session:
            row = session.get(BenchmarkRecordRow, citation.benchmark_id)
            context = session.get(EvaluateRunRow, citation.evaluate_run_id)
            if row is None or context is None:
                return ("missing_internal_source",)
            stored = row.to_dto()
            owner = (context.extra or {}).get("owner_user_id")
            if case.actor != "admin" and owner != actors[case.actor].user_id:
                issues.append("wrong_internal_owner")
            digest = hashlib.sha256(
                json.dumps(stored.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if citation.record_sha256 != digest or citation.record_timestamp != stored.timestamp:
                issues.append("internal_source_hash_or_timestamp_mismatch")
            if (
                citation.metric_field != "throughput_tokens_per_s"
                or citation.cluster_id != context.cluster_id
                or citation.resource_uri != f"/api/v1/evaluate/runs/{context.id}"
                or evidence.metric != "throughput"
                or evidence.unit != "tokens/s"
                or evidence.value != Decimal(str(stored.metrics.throughput_tokens_per_s))
            ):
                issues.append("internal_metric_or_context_mismatch")
        return tuple(issues)

    def verify(case, evidence):
        issues = []
        if isinstance(evidence.citation, AgenticCitation):
            issues.extend(verify_internal(case, evidence, evidence.citation))
        else:
            citation = evidence.citation
            assert isinstance(citation, Citation)
            stored = service.dao.get(evidence.item_id)
            assert stored is not None
            experiment = service.dao.get(stored.experiment_id)
            source = service.dao.get(item_id(stored.origin_id, f"file:{stored.source_path}", stored.source_revision))
            assert source is not None
            data = service.archive.read(citation.sha256)
            head = service.dao.head(citation.origin_id)
            _verify_measurement(
                QueryMatch(stored, experiment, source, citation.published_manifest_id, head.state.generation), data
            )
            if (
                citation.sha256 != stored.content_sha256
                or citation.source_path != stored.source_path
                or citation.source_revision != stored.source_revision
                or citation.published_manifest_id != head.registry.published_manifest_id
                or evidence.value != stored.original_value
                or evidence.unit != stored.original_unit
                or citation.source_pointer != stored.source_pointer
            ):
                issues.append("public_metric_or_citation_mismatch")
        for citation in evidence.corroborating_citations:
            issues.extend(verify_internal(case, evidence, citation))
        if case.case_id == "owner-verified-overlap-keeps-both-citations" and (
            evidence.origin != "public_and_agentic"
            or len(evidence.corroborating_citations) != 1
            or evidence.identity_verification != "canonical_evaluate_run_and_metric"
        ):
            issues.append("missing_verified_dual_provenance")
        return tuple(issues)

    event.listen(postgres_engine, "before_cursor_execute", audit)
    try:
        report = run_evaluation(
            suite,
            execute=lambda case: service.query(case.request, actors[case.actor]),
            verify_evidence=verify,
        )
    finally:
        event.remove(postgres_engine, "before_cursor_execute", audit)
    assert not writes
    with factory() as session:
        assert {row.benchmark_id: row.to_dto() for row in session.scalars(select(BenchmarkRecordRow))} == before
    destination = os.environ.get("LENS_RAG_BASELINE_REPORT_DIR")
    if destination:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        (target / "internal-baseline.json").write_text(report.model_dump_json(indent=2) + "\n")
    assert report.passed, report.model_dump_json(indent=2)
    assert report.total_cases == 12 and report.failed_cases == 0
    assert report.verified_citations == report.numeric_observations + 1
    assert _USERS[0] not in report.model_dump_json()
