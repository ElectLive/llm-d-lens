from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkMetrics,
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    WorkloadProfile,
)
from llm_d_lens.agentic.planner import PlanningFacts
from llm_d_lens.agentic.planning_requirements import (
    ModelIdentity,
    NumericRequirement,
    PlanningRequirements,
    Provenance,
    RuntimeIdentity,
    normalize_hardware,
)
from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InventoryFile,
    ParsedFile,
    PinnedInventory,
    SourceApproval,
    item_id,
)
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    QUERY_PERMISSION,
    AgenticCitation,
    QueryFilters,
    QueryRequest,
    QueryResponse,
)
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, allowed_paths
from llm_d_lens.auth.contracts import Principal, RoleBinding
from llm_d_lens.core.exceptions import ConflictError
from llm_d_lens.db.dao.agentic_benchmark_record import BenchmarkRecordDao
from llm_d_lens.db.dao.benchmark_experiment_query import BenchmarkExperimentQueryDao
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow
from llm_d_lens.db.models.evaluate_run import EvaluateRunRow
from llm_d_lens.db.models.user import UserRow

PATH = "archives/llm-d-benchmarks.json"
REVISION = "2026-10-13"


def record(run_id: str, *, value: float = 42) -> BenchmarkRecord:
    return BenchmarkRecord(
        benchmark_id=f"evaluate:{run_id}",
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
        metrics=BenchmarkMetrics(throughput_tokens_per_s=value),
    )


@pytest.fixture
def branch(postgres_engine, tmp_path, monkeypatch):
    from llm_d_lens.db.dao.benchmark_experiment_query import BenchmarkExperimentQueryDao

    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "agentic-service"))
    factory = sessionmaker(bind=postgres_engine, expire_on_commit=False)
    user_ids = (str(uuid4()), str(uuid4()))
    run_ids = (str(uuid4()), str(uuid4()))
    with factory.begin() as session:
        session.add_all(UserRow(id=user_id, username=f"user-{index}") for index, user_id in enumerate(user_ids))
        session.flush()
        session.add_all(
            EvaluateRunRow(
                id=run_id,
                kind="benchmark",
                status="succeeded",
                cluster_id="cluster1",
                extra={"owner_user_id": owner_id},
            )
            for run_id, owner_id in zip(run_ids, user_ids, strict=True)
        )
    for run_id in run_ids:
        BenchmarkRecordDao(factory).save(record(run_id))
    principal = Principal(
        user_id=user_ids[0],
        username="user-0",
        bindings=(RoleBinding("rag"), RoleBinding("self")),
        role_permissions={"rag": frozenset({QUERY_PERMISSION}), "self": frozenset({"evaluate:run:read"})},
        role_self_scoped={"self": frozenset({"evaluate:run:read"})},
    )
    service = PublicBenchmarkRecordQueryService(
        dao=PublicBenchmarkRecordQueryDao(factory), agentic_dao=BenchmarkExperimentQueryDao(factory)
    )
    return service, principal, factory, run_ids, user_ids


def request(*, internal=True, **filters):
    return QueryRequest(
        question="Reported throughput",
        intent="numeric",
        include_internal=internal,
        filters=QueryFilters(model="Qwen3-0.6B", metric="throughput", **filters),
    )


@pytest.mark.parametrize("metric", [None, "latency", "e2e"])
def test_p95_latency_selects_and_projects_internal_e2e(branch, metric):
    service, principal, factory, run_ids, _ = branch
    with factory.begin() as session:
        for run_id in run_ids:
            session.get(BenchmarkRecordRow, f"evaluate:{run_id}").metrics_e2e_p95_ms = 123
    response = service.query(
        QueryRequest(
            question="p95 latency",
            include_internal=True,
            filters=QueryFilters(model="Qwen3-0.6B", metric=metric),
        ),
        principal,
    )
    assert response.status == "answered"
    assert len(response.evidence) == 1
    evidence = response.evidence[0]
    assert evidence.value == Decimal("123")
    assert evidence.statistic == "p95" and evidence.unit == "ms"
    assert evidence.citation.metric_field == "e2e_p95_ms"
    assert evidence.citation.evaluate_run_id == run_ids[0]


def publish(service, run_id: str):
    identity = f"evaluate:{run_id}"
    data = json.dumps([{"id": identity, "throughput": 42}]).encode()
    digest = service.archive.put(data)
    inventory = PinnedInventory(
        origin_id=PRISM_ORIGIN,
        revision=REVISION,
        complete=True,
        files=(InventoryFile(path=PATH, sha256=digest, format="json"),),
    )
    common = {
        "origin_id": PRISM_ORIGIN,
        "source_revision": REVISION,
        "source_path": PATH,
        "content_sha256": digest,
        "access_classification": "verified_public",
        "format": "json",
    }
    key = f"{PATH}#experiment"
    experiment = PublicBenchmarkRecord(
        **common,
        kind="experiment",
        item_id=item_id(PRISM_ORIGIN, key, REVISION),
        logical_key=key,
        parent_id=item_id(PRISM_ORIGIN, f"file:{PATH}", REVISION),
        model="Qwen3-0.6B",
        hardware="H100",
        runtime="vllm",
        workload={"concurrency": 4},
        source_pointer={"record_id": identity, "json_pointer": "/0"},
    )
    key = f"{PATH}#measurement"
    measurement = PublicBenchmarkRecord(
        **common,
        kind="measurement",
        item_id=item_id(PRISM_ORIGIN, key, REVISION),
        logical_key=key,
        parent_id=experiment.item_id,
        experiment_id=experiment.item_id,
        evidence_type="raw_json",
        metric="throughput",
        value=Decimal("42"),
        original_value=Decimal("42"),
        unit="tokens/s",
        original_unit="tokens/s",
        source_pointer={"json_pointer": "/0/throughput"},
    )
    publication = PublicBenchmarkRecordPublicationService(dao=service.dao, archive=service.archive)
    publication.register_source(
        PRISM_ORIGIN,
        SourceApproval(
            verified_public=True,
            approved_by="isolated service fixture",
            approval_reference="fixture only",
            license_reference="fixture only",
            parser_version="fixture-v1",
            allowed_paths=allowed_paths(PRISM_ORIGIN),
        ),
    )
    manifest = publication.stage(
        inventory,
        (ParsedFile(path=PATH, items=(experiment, measurement)),),
        expected_counts={"source": 1, "experiment": 1, "measurement": 1},
    )
    publication.activate(PRISM_ORIGIN, manifest.item_id)


def test_actual_service_principals_get_only_their_own_records(branch):
    service, owner, _factory, runs, users = branch
    first = service.query(request(), owner)
    second = service.query(request(), replace(owner, user_id=users[1], username="user-1"))
    assert first.matched_items == second.matched_items == 1
    assert first.returned_observations == second.returned_observations == 1
    assert first.evidence[0].citation.evaluate_run_id == runs[0]
    assert second.evidence[0].citation.evaluate_run_id == runs[1]
    assert runs[1] not in first.model_dump_json() and runs[0] not in second.model_dump_json()


@pytest.mark.parametrize(
    "question",
    [
        "What was the p99 throughput for Qwen3-0.6B on H100?",
        "What was the throughput for Llama-3.1-70B on H100?",
    ],
)
def test_question_constraints_also_apply_to_internal_measurements(branch, question):
    service, owner, _factory, _runs, _users = branch
    result = service.query(QueryRequest(question=question, include_internal=True), owner)
    assert result.internal_branch_enabled
    assert result.status == "abstained"
    assert result.matched_items == 0
    assert not result.evidence
    assert [item.code for item in result.abstentions] == ["no_exact_match"]


def test_hidden_exact_ids_do_not_leak_match_or_count(branch):
    service, owner, _factory, runs, _users = branch
    hidden = service.query(request(record_id=f"evaluate:{runs[1]}"), owner)
    missing = service.query(request(record_id=f"evaluate:{uuid4()}"), owner)
    assert hidden.matched_items == missing.matched_items == 0
    assert hidden.evidence == missing.evidence == ()
    assert hidden.abstentions == missing.abstentions
    assert hidden.status == missing.status == "abstained"
    assert runs[1] not in hidden.answer


def test_default_performs_no_agentic_or_evaluation_query(branch, postgres_engine):
    service, owner, _factory, _runs, _users = branch
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(postgres_engine, "before_cursor_execute", capture)
    try:
        response = service.query(request(internal=False), owner)
    finally:
        event.remove(postgres_engine, "before_cursor_execute", capture)
    assert not response.internal_branch_enabled and not response.evidence
    assert not any("agentic_benchmark_records" in statement or "evaluate_runs" in statement for statement in statements)


def test_actual_verified_overlap_is_one_envelope_with_both_provenances(branch):
    service, owner, _factory, runs, _users = branch
    publish(service, runs[0])
    response = service.query(request(), owner)
    assert response.matched_items == 2
    assert response.returned_observations == len(response.evidence) == 1
    assert response.evidence[0].origin == "public_and_agentic"
    assert len(response.evidence[0].corroborating_citations) == 1
    assert response.evidence[0].identity_verification == "canonical_evaluate_run_and_metric"
    assert "both provenances" in response.answer


def test_verified_overlap_accepts_model_aliases_without_changing_citations(branch):
    service, owner, factory, runs, _users = branch
    publish(service, runs[0])
    with factory.begin() as session:
        session.get(BenchmarkRecordRow, f"evaluate:{runs[0]}").model_model_id = "Qwen/Qwen3-0.6B"
    response = service.query(
        QueryRequest(
            question="Reported throughput",
            intent="numeric",
            include_internal=True,
            filters=QueryFilters(model="Qwen/Qwen3-0.6B", metric="throughput"),
        ),
        owner,
    )
    assert response.matched_items == 2
    assert len(response.evidence) == 1
    assert response.evidence[0].origin == "public_and_agentic"
    assert response.evidence[0].experiment.model == "Qwen3-0.6B"
    assert len(response.evidence[0].corroborating_citations) == 1


def test_inaccessible_overlap_never_adds_internal_status_or_provenance(branch):
    service, owner, _factory, runs, _users = branch
    publish(service, runs[1])
    no_eval = replace(owner, bindings=(RoleBinding("rag"),))
    response = service.query(request(), no_eval)
    assert response.matched_items == response.returned_observations == 1
    assert response.evidence[0].origin == "public"
    assert not response.evidence[0].corroborating_citations
    assert response.evidence[0].identity_verification is None
    assert "public_and_agentic" not in response.model_dump_json()


def test_same_values_but_unproven_identity_stay_separate(branch):
    service, owner, _factory, _runs, _users = branch
    publish(service, str(uuid4()))
    response = service.query(request(), owner)
    assert response.returned_observations == 2
    assert {entry.origin for entry in response.evidence} == {"public", "agentic"}
    assert all(not entry.corroborating_citations for entry in response.evidence)


def test_delete_or_revoke_during_public_bytes_discards_internal_response(branch):
    service, owner, factory, runs, _users = branch
    publish(service, runs[0])
    read = service.archive.read

    def revoke(digest):
        data = read(digest)
        with factory.begin() as session:
            session.delete(session.get(EvaluateRunRow, runs[0]))
        return data

    service.archive.read = revoke
    with pytest.raises(ConflictError):
        service.query(request(), owner)


def test_query_uses_no_writes_and_does_not_mutate_planner_record(branch, postgres_engine):
    service, owner, factory, _runs, _users = branch
    with factory() as session:
        before = [row.to_dto().model_dump(mode="json") for row in session.scalars(select(BenchmarkRecordRow))]
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(postgres_engine, "before_cursor_execute", capture)
    try:
        service.query(request(), owner)
    finally:
        event.remove(postgres_engine, "before_cursor_execute", capture)
    assert not any(statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for statement in statements)
    with factory() as session:
        after = [row.to_dto().model_dump(mode="json") for row in session.scalars(select(BenchmarkRecordRow))]
    assert before == after


def test_query_contract_preserves_authorized_internal_envelope(branch):
    service, owner, _factory, _runs, _users = branch
    payload = request()
    response = service.query(payload, owner)
    assert QueryResponse.model_validate_json(response.model_dump_json()) == response
    assert response.model_dump(mode="json")["evidence"][0]["value"] == "42.0"
    assert isinstance(service.query(payload, owner).evidence[0].citation, AgenticCitation)


def planning_facts(**changes):
    return PlanningFacts(
        model_weight_gib=1,
        vram_per_gpu_gib=80,
        free_gpu_count=8,
        requirements=PlanningRequirements(
            hardware=normalize_hardware("NVIDIA H100 80GiB"),
            model=ModelIdentity(quantization="bf16"),
            runtime=RuntimeIdentity(backend="vllm"),
        ),
        **changes,
    )


def test_internal_ingestion_and_both_query_paths_use_all_model_aliases_without_acl_expansion(branch):
    _service, owner, factory, runs, _users = branch
    dao = BenchmarkExperimentQueryDao(factory)
    store = BenchmarkRecordDao(factory)
    labels = ("Publisher/Falcon-7B", "Falcon-7B", "Publisher-Falcon-7B", "PUBLISHER/FALCON-7B")
    saved = {}
    for label in labels:
        run = str(uuid4())
        with factory.begin() as session:
            session.add(
                EvaluateRunRow(
                    id=run,
                    kind="benchmark",
                    status="succeeded",
                    cluster_id="cluster1",
                    extra={"owner_user_id": owner.user_id},
                )
            )
        item = record(run)
        item.model.model_id = label
        assert store.save(item).model.model_id == label
        assert store.save(item).model.model_id == label
        saved[run] = label
    with factory.begin() as session:
        session.get(BenchmarkRecordRow, f"evaluate:{runs[1]}").model_model_id = labels[0]
    for requested in ("Publisher/Falcon-7B", "Falcon-7B", "Publisher-Falcon-7B"):
        expected = (
            set(saved)
            if requested == "Publisher/Falcon-7B"
            else {
                run
                for run, label in saved.items()
                if label != ("Publisher-Falcon-7B" if requested == "Falcon-7B" else "Falcon-7B")
            }
        )
        filters = QueryFilters(model=requested, metric="throughput")
        for result in (
            dao.search(filters, owner, limit=50),
            dao.planning_search(requested, owner, planning_facts()),
        ):
            assert result.total == len(expected)
            assert {match.evaluate_run_id for match in result.matches} == expected
            assert all(match.record.model.model_id == saved[match.evaluate_run_id] for match in result.matches)
    assert len(store.list()) == 2 + len(labels)
    assert dao.search(QueryFilters(model="Publisher/Falcon-7B-FP8"), owner, limit=50).total == 0
    assert dao.planning_search("Publisher/Falcon-7B-FP8", owner, planning_facts()).total == 0
    no_evaluate = replace(owner, bindings=(RoleBinding("rag"),))
    assert dao.search(QueryFilters(model="falcon-7b"), no_evaluate, limit=50).total == 0
    assert dao.planning_search("falcon-7b", no_evaluate, planning_facts()).total == 0


def test_planning_ranks_before_limit_and_reads_only_authorized_selected_payloads(branch, monkeypatch):
    _service, owner, factory, _runs, users = branch
    expected_run = "ffffffff-ffff-4fff-8fff-fffffffffff0"
    hidden_run = "ffffffff-ffff-4fff-8fff-fffffffffff1"
    with factory.begin() as session:
        for index in range(104):
            run = (
                expected_run
                if index == 102
                else hidden_run
                if index == 103
                else f"{index:08x}-0000-4000-8000-000000000000"
            )
            session.add(
                EvaluateRunRow(
                    id=run,
                    kind="benchmark",
                    status="succeeded",
                    cluster_id="cluster1",
                    extra={"owner_user_id": users[1] if index == 103 else users[0]},
                )
            )
            item = record(run)
            if index < 51:
                item.hardware = HardwareDescriptor(accelerator="H200", vram_per_gpu_gib=80, gpu_count=1)
            if index >= 102:
                item.workload.mean_input_tokens = 1000
                item.configuration.topology = "pd-disaggregation"
            # Wrong workloads are fresher: freshness must not hide the useful row.
            item.timestamp = datetime(2026, 10, 25 if index < 102 else 20, tzinfo=UTC)
            session.add(BenchmarkRecordRow.from_dto(item))

    read_ids = []
    original = BenchmarkRecordRow.to_dto

    def checked_payload(row):
        read_ids.append(row.benchmark_id)
        assert row.benchmark_id == f"evaluate:{expected_run}"
        return original(row)

    monkeypatch.setattr(BenchmarkRecordRow, "to_dto", checked_payload)
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    engine = factory.kw["bind"]
    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = BenchmarkExperimentQueryDao(factory).planning_search(
            "Qwen3-0.6B", owner, planning_facts(workload_profile={"mean_input_tokens": 1000}), limit=1
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert result.total == 53  # 51 wrong workloads, original owner row, useful row.
    assert [match.evaluate_run_id for match in result.matches] == [expected_run]
    assert read_ids == [f"evaluate:{expected_run}"]
    selection = next(sql for sql in statements if "count(*) OVER" in sql)
    assert "LIMIT" in selection and "ORDER BY" in selection
    assert "agentic_benchmark_records.provenance," not in selection
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)
    no_evaluate = replace(owner, bindings=(RoleBinding("rag"),))
    assert BenchmarkExperimentQueryDao(factory).planning_search("Qwen3-0.6B", no_evaluate, planning_facts()).total == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_model_id", "qwen3-0.6b-fp8"),
        ("hardware_accelerator", "H200"),
        ("hardware_vram_per_gpu_gib", 40),
        ("runtime_backend", "sglang"),
        ("model_quantization", "int8"),
        ("provenance", {"hardware_identity": {"profile_id": "different-profile"}}),
        ("metrics_throughput_tokens_per_s", -1),
        ("outcome", "failed"),
        ("evaluate_run_id", None),
    ],
)
def test_planning_hard_compatibility_and_eligibility_precede_payload_reads(branch, monkeypatch, field, value):
    _service, owner, factory, runs, _users = branch
    with factory.begin() as session:
        setattr(session.get(BenchmarkRecordRow, f"evaluate:{runs[0]}"), field, value)

    def forbidden_payload(_row):
        pytest.fail("incompatible or inaccessible record payload was read")

    monkeypatch.setattr(BenchmarkRecordRow, "to_dto", forbidden_payload)
    result = BenchmarkExperimentQueryDao(factory).planning_search("Qwen3-0.6B", owner, planning_facts())
    assert result.total == 0 and not result.matches


def test_planning_unknown_identity_does_not_invent_constraints(branch):
    _service, owner, factory, runs, _users = branch
    with factory.begin() as session:
        row = session.get(BenchmarkRecordRow, f"evaluate:{runs[0]}")
        row.hardware_accelerator = None
        row.hardware_vram_per_gpu_gib = None
        row.runtime_backend = "unknown"
        row.model_quantization = None
        row.provenance = {}
    dao = BenchmarkExperimentQueryDao(factory)
    assert dao.planning_search("Qwen3-0.6B", owner, planning_facts()).total == 1
    unknown = PlanningFacts(model_weight_gib=1, vram_per_gpu_gib=80, free_gpu_count=8)
    with factory.begin() as session:
        row = session.get(BenchmarkRecordRow, f"evaluate:{runs[0]}")
        row.hardware_accelerator = "H200"
        row.hardware_vram_per_gpu_gib = 140
        row.runtime_backend = "other"
        row.model_quantization = "int8"
    assert dao.planning_search("Qwen3-0.6B", owner, unknown).total == 1


@pytest.mark.parametrize("statistic", ["mean", "p95"])
def test_planning_target_statistic_coverage_precedes_freshness_without_mean_substitution(branch, statistic):
    _service, owner, factory, runs, users = branch
    older_run = "ffffffff-ffff-4fff-8fff-fffffffffff2"
    with factory.begin() as session:
        session.add(
            EvaluateRunRow(
                id=older_run,
                kind="benchmark",
                status="succeeded",
                cluster_id="cluster1",
                extra={"owner_user_id": users[0]},
            )
        )
        older = record(older_run)
        older.timestamp = datetime(2026, 10, 18, tzinfo=UTC)
        older.metrics.ttft_p95_ms = 25
        session.add(BenchmarkRecordRow.from_dto(older))
    target = NumericRequirement(
        field="ttft",
        value=100,
        unit="ms",
        **({"statistic": statistic} if statistic == "p95" else {}),
        provenance=Provenance(source="form", field="ttft"),
    )
    assert target.statistic == statistic
    facts = planning_facts()
    facts = replace(facts, requirements=facts.requirements.model_copy(update={"performance_targets": (target,)}))
    result = BenchmarkExperimentQueryDao(factory).planning_search("Qwen3-0.6B", owner, facts, limit=1)
    assert result.matches[0].evaluate_run_id == (older_run if statistic == "p95" else runs[0])


def test_internal_planning_prefers_lower_latency_before_timestamp_and_limit(branch):
    _service, owner, factory, runs, users = branch
    older_run = "ffffffff-ffff-4fff-8fff-fffffffffff3"
    with factory.begin() as session:
        session.get(BenchmarkRecordRow, f"evaluate:{runs[0]}").metrics_ttft_p95_ms = 90
        session.add(
            EvaluateRunRow(
                id=older_run,
                kind="benchmark",
                status="succeeded",
                cluster_id="cluster1",
                extra={"owner_user_id": users[0]},
            )
        )
        older = record(older_run)
        older.timestamp = datetime(2026, 10, 18, tzinfo=UTC)
        older.metrics.ttft_p95_ms = 5
        session.add(BenchmarkRecordRow.from_dto(older))
    result = BenchmarkExperimentQueryDao(factory).planning_search("Qwen3-0.6B", owner, planning_facts(), limit=1)
    assert result.matches[0].evaluate_run_id == older_run
    assert result.total == 2


@pytest.mark.parametrize("limit", [0, 51])
def test_planning_limit_is_bounded(branch, limit):
    _service, owner, factory, _runs, _users = branch
    with pytest.raises(ValueError, match="between 1 and 50"):
        BenchmarkExperimentQueryDao(factory).planning_search("Qwen3-0.6B", owner, planning_facts(), limit=limit)
