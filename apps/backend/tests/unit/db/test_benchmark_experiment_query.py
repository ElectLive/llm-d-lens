from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import Engine, event, select, update
from sqlalchemy.orm import Session, sessionmaker

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkMetrics,
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    WorkloadProfile,
)
from llm_d_lens.agentic.facts import PlanningEvidence
from llm_d_lens.agentic.performance_evidence import BenchmarkPerformanceEvidence
from llm_d_lens.agentic.planner import DeterministicPlanner, PlanningFacts
from llm_d_lens.agentic.planner import WorkloadProfile as PlanningWorkloadProfile
from llm_d_lens.agentic.planning_requirements import (
    HardwareIdentity,
    ModelIdentity,
    PlanningRequirements,
    RuntimeIdentity,
)
from llm_d_lens.agentic.public_benchmark_record.agentic_metrics import AGENTIC_METRICS, measured_values
from llm_d_lens.agentic.public_benchmark_record.case_retrieval import build_case_bundle, resolve_internal_cases
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION, QueryFilters
from llm_d_lens.agentic.service import AgenticDeploymentService
from llm_d_lens.auth.contracts import Principal, ResourceRef, RoleBinding, ScopeType
from llm_d_lens.auth.policy import authorize
from llm_d_lens.core.exceptions import (
    ConflictError,
    ForbiddenError,
    PasswordChangeRequiredError,
    UnauthenticatedError,
)
from llm_d_lens.db.dao.benchmark_experiment_query import AgenticSlice, BenchmarkExperimentQueryDao
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow
from llm_d_lens.db.models.evaluate_run import EvaluateRunRow
from llm_d_lens.db.models.user import UserRow

_READ = "evaluate:run:read"
_QUERY = RoleBinding("query")


def _principal(*bindings: RoleBinding, self_scoped: bool = False, code: str = _READ) -> Principal:
    return Principal(
        user_id="alice",
        username="alice",
        bindings=(_QUERY, *bindings),
        role_permissions={"query": frozenset({QUERY_PERMISSION}), "reader": frozenset({code})},
        role_self_scoped={"reader": frozenset({_READ}) if self_scoped else frozenset()},
    )


def _record(run_id: str) -> BenchmarkRecord:
    return BenchmarkRecord(
        benchmark_id="evaluate:" + run_id,
        timestamp=datetime(2026, 10, 19, tzinfo=UTC),
        outcome="succeeded",
        model=ModelDescriptor(
            model_id="Qwen/Qwen3-8B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=8,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-8B",
        ),
        hardware=HardwareDescriptor(accelerator="Intel Gaudi 3", vram_per_gpu_gib=96, gpu_count=4),
        runtime=RuntimeFingerprint(backend="vllm", backend_version="0.10", accelerator_runtime="synapse"),
        workload=WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.5,
            prefill_fraction=0.5,
        ),
        configuration=CandidateConfiguration(topology="baseline-vllm", decode_tp=2, decode_replicas=2),
        metrics=BenchmarkMetrics(
            ttft_p95_ms=40, tpot_p95_ms=10, e2e_p95_ms=400, throughput_tokens_per_s=100, error_rate=0.02
        ),
    )


@dataclass
class Store:
    factory: sessionmaker[Session]
    dao: BenchmarkExperimentQueryDao

    def add(
        self,
        *,
        owner: str | None = "alice",
        cluster: str | None = "cluster1",
        extra: dict[str, object] | None = None,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or str(uuid4())
        with self.factory.begin() as session:
            session.add(
                EvaluateRunRow(
                    id=run_id,
                    kind="benchmark",
                    status="succeeded",
                    cluster_id=cluster,
                    owner_user_id=owner,
                    extra=extra or {},
                )
            )
            session.add(BenchmarkRecordRow.from_dto(_record(run_id)))
        return run_id


@pytest.fixture
def store(postgres_engine: Engine) -> Iterator[Store]:
    factory = sessionmaker(postgres_engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add_all(
            [
                UserRow(id="alice", username="alice", status="active"),
                UserRow(id="bob", username="bob", status="active"),
                UserRow(id="inactive", username="inactive", status="inactive"),
            ]
        )
    yield Store(factory, BenchmarkExperimentQueryDao(factory))


def test_sql_scope_matches_real_policy_for_effective_role_grants(store: Store) -> None:
    resources = [
        (store.add(owner=owner, cluster=cluster), cluster, owner)
        for owner in ("alice", "bob", "inactive")
        for cluster in ("cluster1", "cluster2")
    ]
    cases = [
        _principal(),
        _principal(RoleBinding("reader")),
        _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1")),
        _principal(RoleBinding("reader"), self_scoped=True),
        _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"), self_scoped=True),
        _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster3"), self_scoped=True),
        _principal(RoleBinding("reader"), code="evaluate:*:read"),
        _principal(RoleBinding("reader"), code="evaluate:run:*"),
        _principal(RoleBinding("reader"), code="evaluate:*:*"),
        _principal(RoleBinding("reader"), code="*:*:*"),
        _principal(RoleBinding("reader"), code="evaluate:simulation:read"),
        _principal(RoleBinding("admin")),
        _principal(RoleBinding("reader", ScopeType.RESOURCE, "cluster1", "evaluate_run", resources[0][0])),
        _principal(
            RoleBinding("reader", ScopeType.RESOURCE, "cluster1", "evaluate_run", resources[0][0]),
            RoleBinding("reach", ScopeType.CLUSTER, "cluster1"),
        ),
    ]
    # Group membership alone never grants evaluation ownership. Group-inherited
    # effective bindings have exactly the same policy semantics as direct ones.
    cases.extend(
        [
            replace(_principal(), group_ids=frozenset({"group1"})),
            replace(_principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster2")), group_ids=frozenset({"group1"})),
        ]
    )
    for principal in cases:
        expected = {
            run_id
            for run_id, cluster, owner in resources
            if authorize(
                principal,
                _READ,
                resource=ResourceRef("evaluate_run", run_id, cluster, owner_user_id=owner),
            ).allowed
        }
        result = store.dao.search(QueryFilters(), principal, limit=50)
        assert {match.evaluate_run_id for match in result.matches} == expected
        assert result.total == len(expected)


def test_verified_ownership_and_references_fail_closed_even_for_admin(store: Store) -> None:
    typed = store.add()
    json_owner = store.add(owner=None, extra={"owner_user_id": "bob"})
    agreeing = store.add(extra={"owner_user_id": "alice"})
    inactive = store.add(owner="inactive")
    for extra in (
        {"owner_user_id": "bob"},
        {"owner_user_id": 1},
        {"owner_user_id": None},
        {"owner_user_id": ""},
        {"owner_user_id": " alice "},
        {"id": "other"},
        {"id": None},
        {"cluster_id": "cluster2"},
        {"cluster_id": 5},
    ):
        store.add(extra=extra)
    store.add(owner=None)
    store.add(owner=None, extra={"owner_user_id": "deleted"})
    store.add(cluster=None)
    store.add(run_id="abcd1234")  # Actual legacy simulation task format.
    store.add(run_id="A1234567-1234-1234-1234-123456789ABC")
    missing = store.add()
    deleted_owner = store.add(owner=None, extra={"owner_user_id": "gone"})
    mismatched_reference = store.add()
    with store.factory.begin() as session:
        session.delete(session.get(EvaluateRunRow, missing))
        session.add(UserRow(id="gone", username="gone"))
        session.flush()
        session.delete(session.get(UserRow, "gone"))
        session.execute(
            update(BenchmarkRecordRow)
            .where(BenchmarkRecordRow.benchmark_id == "evaluate:" + mismatched_reference)
            .values(evaluate_run_id=typed)
        )
    result = store.dao.search(QueryFilters(), _principal(RoleBinding("admin")), limit=50)
    assert {match.evaluate_run_id for match in result.matches} == {typed, json_owner, agreeing, inactive}
    assert deleted_owner not in {match.evaluate_run_id for match in result.matches}


def test_authorization_precedes_count_order_and_limit(store: Store, postgres_engine: Engine) -> None:
    for number in range(1, 5):
        store.add(owner="bob", run_id=f"00000000-0000-0000-0000-{number:012d}")
    own = store.add(run_id="ffffffff-ffff-ffff-ffff-ffffffffffff")
    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _many) -> None:
        statements.append(statement)

    event.listen(postgres_engine, "before_cursor_execute", capture)
    try:
        result = store.dao.search(QueryFilters(), _principal(RoleBinding("reader"), self_scoped=True), limit=1)
    finally:
        event.remove(postgres_engine, "before_cursor_execute", capture)
    assert result.total == 1
    assert result.matches[0].evaluate_run_id == own
    retrieval = [sql for sql in statements if "count(*) OVER" in sql]
    assert len(retrieval) == 1
    assert "WHERE" in retrieval[0] and "LIMIT" in retrieval[0]
    assert "coalesce(evaluate_runs.owner_user_id" in retrieval[0]


@pytest.mark.parametrize(
    ("filters", "matched"),
    [
        ({"model": "qwen/qwen3-8b", "hardware": "INTEL GAUDI 3", "runtime": "VLLM"}, True),
        ({"model": "Qwen"}, False),
        ({"hardware": "Gaudi"}, False),
        ({"record_id": "dynamic"}, True),
        ({"experiment_id": "dynamic"}, True),
        ({"experiment_id": "uuid"}, False),
        ({"source_revision": "main"}, False),
        ({"source_path": "bench.json"}, False),
        ({"series": {"concurrency": 16}}, False),
        ({"workload": {"concurrency": 16, "mean_input_tokens": 4096}}, True),
        (
            {
                "workload": {
                    "mean_input_tokens": 4096,
                    "p95_input_tokens": 8192,
                    "mean_output_tokens": 512,
                    "concurrency": 16,
                    "request_rate": 4,
                    "shared_prefix_ratio": 0.5,
                    "prefill_fraction": 0.5,
                }
            },
            True,
        ),
        ({"workload": {"batch_size": 16}}, False),
        ({"workload": {"concurrency": True}}, False),
        ({"workload": {"concurrency": "16"}}, False),
        ({"workload": {"mean_output_tokens": 1}}, False),
        ({"metric": "ttft", "statistic": "p95", "unit": "ms", "value_min": 40, "value_max": 40}, True),
        ({"metric": "tpot", "unit": "ms/token", "value_max": 10}, True),
        ({"metric": "tpot", "unit": "ms/token", "normalized": True, "value_max": 10}, False),
        ({"metric": "tpot", "unit": "ms"}, False),
        ({"metric": "e2e", "statistic": "mean"}, False),
        ({"metric": "throughput", "unit": "tokens/s", "normalized": True, "value_min": 100}, True),
        ({"metric": "throughput", "statistic": "mean", "unit": "tokens/s"}, True),
        ({"metric": "throughput", "statistic": "p99", "unit": "tokens/s"}, False),
        ({"metric": "throughput", "statistic": "output"}, False),
        ({"metric": "error_rate", "unit": "fraction"}, True),
        ({"metric": "error_rate", "normalized": True}, False),
        ({"metric": "error_rate", "unit": "fraction", "value_max": 1}, True),
        ({"metric": "error_rate", "unit": "fraction", "value_min": 0.02, "value_max": 0.02}, True),
        ({"metric": "error_rate", "unit": "fraction", "value_min": 0.03}, False),
        ({"metric": "error_rate", "unit": "fraction", "normalized": True, "value_max": 1}, False),
        ({"metric": "ttft", "unit": "s"}, False),
        ({"metric": "unknown"}, False),
        ({"metric": "throughput", "value_min": 200, "unit": "tokens/s"}, False),
    ],
)
def test_exact_filters_and_metric_mapping(store: Store, filters: dict[str, object], matched: bool) -> None:
    run_id = store.add()
    filters = dict(filters)
    for field in ("record_id", "experiment_id"):
        if filters.get(field) == "dynamic":
            filters[field] = "evaluate:" + run_id
        elif filters.get(field) == "uuid":
            filters[field] = run_id
    result = store.dao.search(QueryFilters.model_validate(filters), _principal(RoleBinding("reader")), limit=10)
    assert bool(result.matches) is matched
    assert result.total == int(matched)
    by_uuid = store.dao.search(QueryFilters(record_id=run_id), _principal(RoleBinding("reader")), limit=10)
    assert by_uuid.total == 1


def test_failed_absent_and_nonfinite_metrics_never_escape(store: Store) -> None:
    good = store.add()
    failed = store.add()
    absent = store.add()
    bad = [store.add() for _ in range(3)]
    with store.factory.begin() as session:
        session.execute(
            update(BenchmarkRecordRow).where(BenchmarkRecordRow.evaluate_run_id == failed).values(outcome="failed")
        )
        session.execute(
            update(BenchmarkRecordRow)
            .where(BenchmarkRecordRow.evaluate_run_id == absent)
            .values(
                metrics_ttft_p95_ms=None,
                metrics_tpot_p95_ms=None,
                metrics_e2e_p95_ms=None,
                metrics_throughput_tokens_per_s=None,
                metrics_error_rate=None,
            )
        )
        for run_id, value in zip(bad, (float("nan"), float("inf"), float("-inf")), strict=True):
            session.execute(
                update(BenchmarkRecordRow)
                .where(BenchmarkRecordRow.evaluate_run_id == run_id)
                .values(metrics_ttft_p95_ms=value)
            )
    result = store.dao.search(QueryFilters(), _principal(RoleBinding("reader")), limit=10)
    assert result.total == 1  # Count records, not the good record's five metric leaves.
    assert result.matches[0].evaluate_run_id == good
    store.dao.assert_current(result.matches, _principal(RoleBinding("reader")))


def test_unicode_filters_preserve_existing_sql_lower_semantics(store: Store) -> None:
    run_id = store.add()
    with store.factory.begin() as session:
        session.execute(
            update(BenchmarkRecordRow)
            .where(BenchmarkRecordRow.evaluate_run_id == run_id)
            .values(model_model_id="Straße", hardware_accelerator="Σίσυφος", runtime_backend="ﬃ")
        )
    for filters in (
        QueryFilters(model="Straße"),
        QueryFilters(model="STRAßE"),
        QueryFilters(hardware="σίσυφος"),
        QueryFilters(runtime="ﬃ"),
        QueryFilters(model="Straße", hardware="Σίσυφος", runtime="ﬃ"),
    ):
        assert store.dao.search(filters, _principal(RoleBinding("reader")), limit=1).total == 1
    assert store.dao.search(QueryFilters(model="STRAS"), _principal(RoleBinding("reader")), limit=1).total == 0
    for filters in (
        QueryFilters(model="STRASSE"),
        QueryFilters(hardware="σίσυφοσ"),
        QueryFilters(runtime="ffi"),
    ):
        assert store.dao.search(filters, _principal(RoleBinding("reader")), limit=1).total == 0


def test_sql_metric_selection_agrees_with_shared_measured_fields(store: Store) -> None:
    run_id = store.add()
    record = _record(run_id)
    principal = _principal(RoleBinding("reader"))
    for spec in AGENTIC_METRICS:
        for normalized in (False, True):
            filters = QueryFilters(metric=spec.metric, statistic=spec.statistic, unit=spec.unit, normalized=normalized)
            result = store.dao.search(filters, principal, limit=1)
            assert bool(result.matches) == bool(measured_values(record, filters))
            assert result.total == int(bool(measured_values(record, filters)))
        raw_range = QueryFilters(metric=spec.metric, unit=spec.unit, value_min=0, value_max=1000)
        result = store.dao.search(raw_range, principal, limit=1)
        assert bool(result.matches) == bool(measured_values(record, raw_range))
        assert result.total == int(bool(measured_values(record, raw_range)))


def test_caller_authentication_and_persisted_freshness(store: Store) -> None:
    store.add()
    good = _principal(RoleBinding("reader"))
    with store.factory.begin() as session:
        session.add(UserRow(id="system", username="system", status="active"))
    for principal, error in (
        (None, UnauthenticatedError),
        (replace(good, user_id="system", authz_version=0), UnauthenticatedError),
        (replace(good, user_id="system"), UnauthenticatedError),
        (replace(good, must_change_password=True), PasswordChangeRequiredError),
        (replace(good, user_id="missing"), ConflictError),
        (replace(good, authz_version=2), ConflictError),
    ):
        with pytest.raises(error):
            store.dao.search(QueryFilters(), principal, limit=10)  # type: ignore[arg-type]
    for values in ({"status": "inactive"}, {"must_change_password": True}, {"principal_version": 2}):
        with store.factory.begin() as session:
            session.execute(update(UserRow).where(UserRow.id == "alice").values(**values))
        with pytest.raises(ConflictError):
            store.dao.search(QueryFilters(), good, limit=10)
        with pytest.raises(ConflictError):
            store.dao.assert_current((), good)
        with store.factory.begin() as session:
            session.execute(
                update(UserRow)
                .where(UserRow.id == "alice")
                .values(status="active", must_change_password=False, principal_version=1)
            )


def test_internal_case_access_needs_resource_access_not_public_query_grant(store, monkeypatch):
    from llm_d_lens.agentic.public_benchmark_record.agentic_answers import agentic_evidence
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import assert_case_access

    run_id = store.add()
    store.add(cluster="cluster2")
    principal = replace(
        _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1")),
        bindings=(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"),),
    )
    filters = QueryFilters()
    result = store.dao.search(filters, principal, limit=50)
    assert {match.evaluate_run_id for match in result.matches} == {run_id}
    store.dao.assert_current(result.matches, principal)
    citation = agentic_evidence(result.matches[0], filters)[0].citation.model_dump(mode="json")
    monkeypatch.setattr(
        "llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", lambda: store.dao
    )
    cases = ({"provenance": [citation]},)
    assert_case_access(cases, principal, check_public_sources=True)
    with pytest.raises(ForbiddenError, match="no longer accessible"):
        assert_case_access(cases, replace(principal, bindings=()))


@pytest.mark.parametrize(
    "mutation", ["owner", "cluster", "version", "outcome", "delete", "owner_delete", "json_conflict", "access"]
)
def test_revalidation_detects_provenance_and_access_races(store: Store, mutation: str) -> None:
    run_id = store.add(owner=None, extra={"owner_user_id": "bob"})
    principal = _principal(RoleBinding("reader"))
    before = store.dao.search(QueryFilters(), principal, limit=10)
    assert len(before.matches) == 1
    store.dao.assert_current(before.matches, principal)
    with store.factory.begin() as session:
        run = session.get(EvaluateRunRow, run_id)
        assert run is not None
        if mutation == "owner":
            run.extra = {"owner_user_id": "alice"}
        elif mutation == "cluster":
            run.cluster_id = "cluster2"
        elif mutation == "version":
            run.status = "failed"
        elif mutation == "outcome":
            session.execute(
                update(BenchmarkRecordRow).where(BenchmarkRecordRow.evaluate_run_id == run_id).values(outcome="failed")
            )
        elif mutation == "delete":
            session.delete(run)
        elif mutation == "owner_delete":
            session.delete(session.get(UserRow, "bob"))
        elif mutation == "json_conflict":
            run.extra = {"owner_user_id": "bob", "id": "wrong"}
    if mutation == "access":
        principal = _principal()
    with pytest.raises(ConflictError):
        store.dao.assert_current(before.matches, principal)


def test_read_only_api_and_bounded_limit(store: Store) -> None:
    run_id = store.add()
    principal = _principal(RoleBinding("reader"))
    with store.factory() as session:
        before = session.scalar(select(EvaluateRunRow.version_id).where(EvaluateRunRow.id == run_id))
    result = store.dao.search(QueryFilters(), principal, limit=1)
    assert isinstance(result, AgenticSlice)
    store.dao.assert_current(result.matches, principal)
    with store.factory() as session:
        assert session.scalar(select(EvaluateRunRow.version_id).where(EvaluateRunRow.id == run_id)) == before
    for limit in (0, 51):
        with pytest.raises(ValueError):
            store.dao.search(QueryFilters(), principal, limit=limit)


def test_stages_keep_complete_sample_identity_and_resource_authorization(store, monkeypatch):
    from llm_d_lens.agentic.public_benchmark_record.agentic_answers import agentic_evidence
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import assert_case_access

    run_id = store.add()
    identities = {f"evaluate:{run_id}{suffix}" for suffix in ("", ":stage:0", ":matrix:1:stage:2")}
    with store.factory.begin() as session:
        for identity in identities - {f"evaluate:{run_id}"}:
            session.add(BenchmarkRecordRow.from_dto(_record(run_id).model_copy(update={"benchmark_id": identity})))
    principal = replace(
        _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1")),
        bindings=(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"),),
    )
    filters = QueryFilters(record_id=run_id)
    result = store.dao.search(filters, principal, limit=50)
    assert {match.record.benchmark_id for match in result.matches} == identities
    assert result.total == 3
    facts = PlanningFacts(8, 96, 4)
    planned = store.dao.planning_search("Qwen/Qwen3-8B", principal, facts)
    assert {match.record.benchmark_id for match in planned.matches} == identities
    observations = tuple(
        {"evidence_id": f"internal:{item.item_id}", **item.model_dump(mode="json")}
        for match in planned.matches
        for item in agentic_evidence(match, QueryFilters())
    )
    bundle, kept = build_case_bundle(observations, facts, model="Qwen/Qwen3-8B", hardware=None, branch_status={})
    assert {case.experiment_id for case in bundle.cases} == identities
    assert len(kept) == len(observations)
    assert all(
        len(case.measurements) == len(agentic_evidence(result.matches[0], QueryFilters())) for case in bundle.cases
    )
    for identity in identities:
        exact = store.dao.search(QueryFilters(record_id=identity), principal, limit=50)
        assert {match.record.benchmark_id for match in exact.matches} == {identity}
    store.dao.assert_current(result.matches, principal)
    monkeypatch.setattr(
        "llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", lambda: store.dao
    )
    cases = tuple(case.model_dump(mode="json") for case in bundle.cases)
    assert_case_access(cases, principal)
    assert not store.dao.search(filters, replace(principal, bindings=()), limit=50).matches
    with pytest.raises(ConflictError):
        store.dao.assert_current(result.matches, replace(principal, bindings=()))


@pytest.mark.parametrize(
    "suffix",
    [
        ":stage:-1",
        ":stage:01",
        ":stage:",
        ":stage:1:extra",
        ":matrix:1",
        ":matrix:01:stage:0",
        ":stage:1.0",
        ":matrix:1:stage:-2",
    ],
)
def test_malformed_stage_id_cannot_borrow_a_live_run(store, suffix):
    run_id = store.add()
    with store.factory.begin() as session:
        row = BenchmarkRecordRow.from_dto(
            _record(run_id).model_copy(update={"benchmark_id": f"evaluate:{run_id}{suffix}"})
        )
        row.evaluate_run_id = run_id
        session.add(row)
    result = store.dao.search(QueryFilters(record_id=run_id), _principal(RoleBinding("reader")), limit=50)
    assert [match.record.benchmark_id for match in result.matches] == [f"evaluate:{run_id}"]


def test_failed_stage_is_retained_and_revalidated_as_negative_evidence(store):
    run_id = store.add()
    identity = f"evaluate:{run_id}:matrix:0:stage:1"
    with store.factory.begin() as session:
        session.add(
            BenchmarkRecordRow.from_dto(
                _record(run_id).model_copy(
                    update={"benchmark_id": identity, "outcome": "failed", "failure_reason": "measured stage failure"}
                )
            )
        )
    principal = _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"))
    failures = store.dao.planning_failures("Qwen/Qwen3-8B", principal, PlanningFacts(8, 96, 4))
    assert [match.record.benchmark_id for match in failures.matches] == [identity]
    store.dao.assert_current(failures.matches, principal)
    with store.factory.begin() as session:
        session.execute(update(EvaluateRunRow).where(EvaluateRunRow.id == run_id).values(version_id=2))
    with pytest.raises(ConflictError):
        store.dao.assert_current(failures.matches, principal)


@pytest.mark.parametrize("token_budget", [0, 8000])
def test_authorized_failures_block_planning_independently_of_prompt_budget(store, monkeypatch, token_budget):
    facts = PlanningFacts(
        16,
        96,
        2,
        requirements=PlanningRequirements(
            hardware=HardwareIdentity(device_model="Gaudi3", memory_gib=96, form_factor="OAM", device_count=2),
            model=ModelIdentity(model_id="Qwen/Qwen3-8B", architecture="dense", quantization="bf16"),
            runtime=RuntimeIdentity(backend="vllm", backend_version="0.10", accelerator_runtime="synapse"),
        ),
        workload_profile=PlanningWorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.5,
        ),
    )
    ids = []
    for cluster in ("cluster1", "cluster2"):
        for _ in range(3):
            run_id = store.add(cluster=cluster)
            record = _record(run_id).model_copy(
                update={
                    "timestamp": datetime.now(UTC),
                    "outcome": "failed",
                    "failure_reason": "OOM",
                    "metrics": BenchmarkMetrics(),
                    "hardware": HardwareDescriptor(
                        accelerator="Gaudi3", vram_per_gpu_gib=96, gpu_count=2, form_factor="OAM"
                    ),
                    "workload": _record(run_id).workload.model_copy(update={"prefill_fraction": 0}),
                    "configuration": CandidateConfiguration(
                        topology="optimized-baseline", decode_tp=1, decode_replicas=1, optimizations=("",)
                    ),
                }
            )
            with store.factory.begin() as session:
                session.merge(BenchmarkRecordRow.from_dto(BenchmarkRecord.model_validate(record.model_dump())))
            if cluster == "cluster1":
                ids.append(record.benchmark_id)
    successful_id = "evaluate:" + store.add()
    ids.append(successful_id)
    successful_record = BenchmarkRecord.model_validate(
        {
            **record.model_dump(),
            "benchmark_id": successful_id,
            "outcome": "succeeded",
            "failure_reason": None,
            "metrics": _record(successful_id.removeprefix("evaluate:")).metrics.model_dump(),
        }
    )
    with store.factory.begin() as session:
        session.merge(BenchmarkRecordRow.from_dto(successful_record))

    principal = _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"))
    monkeypatch.setattr(
        "llm_d_lens.agentic.public_benchmark_record.case_retrieval.BenchmarkExperimentQueryDao", lambda: store.dao
    )
    internal = asyncio.run(resolve_internal_cases("Qwen/Qwen3-8B", principal, facts=facts))
    assert {record.benchmark_id for record in internal.records} == set(ids)
    assert internal.status == "available" and internal.total == 4
    assert {item.experiment.item_id for item in internal.observations} == {successful_id}
    assert store.dao.search(QueryFilters(), principal, limit=50).total == 1

    observations = tuple(
        {"evidence_id": f"internal:{item.item_id}", **item.model_dump(mode="json")} for item in internal.observations
    )
    bundle, kept = build_case_bundle(
        observations, facts, model="Qwen/Qwen3-8B", hardware="Gaudi3", branch_status={}, token_budget=token_budget
    )
    *_, retained, _bundle = AgenticDeploymentService._apply_case_selection(
        facts,
        PlanningEvidence(id="public", source="public-benchmark-rag", status="empty"),
        internal,
        bundle,
        kept,
    )
    assert retained.records == internal.records
    candidate = next(
        item for item in DeterministicPlanner().candidates(facts) if item.id == "optimized-baseline-tp1-r1"
    )
    (score,) = BenchmarkPerformanceEvidence().score_inputs(
        "Qwen/Qwen3-8B", facts, [candidate], records=list(retained.records)
    )
    assert score.has_blocking_negative_evidence
    assert set(score.evidence_refs) == {f"benchmark:{identity}" for identity in ids}
    scored_facts = replace(facts, performance_score_inputs=(score,))
    assert not DeterministicPlanner()._attach_performance_evidence(scored_facts, candidate).deployable

    denied = asyncio.run(resolve_internal_cases("Qwen/Qwen3-8B", _principal(), facts=facts))
    assert not denied.records


def test_planning_failures_have_independent_limits_and_revalidate_outcomes(store):
    principal = _principal(RoleBinding("reader", ScopeType.CLUSTER, "cluster1"))
    facts = PlanningFacts(8, 96, 4)
    failed_ids = [store.add() for _ in range(3)]
    foreign_id = store.add(cluster="cluster2")
    for _ in range(3):
        store.add()
    with store.factory.begin() as session:
        session.execute(
            update(BenchmarkRecordRow)
            .where(BenchmarkRecordRow.evaluate_run_id.in_([*failed_ids, foreign_id]))
            .values(outcome="failed", failure_reason="OOM")
        )
    failures = store.dao.planning_failures("Qwen/Qwen3-8B", principal, facts, limit=1)
    assert failures.total == 3 and len(failures.matches) == 1
    assert failures.matches[0].evaluate_run_id in failed_ids
    assert store.dao.planning_search("Qwen/Qwen3-8B", principal, facts, limit=1).total == 3
    assert not store.dao.planning_failures("other-model", principal, facts).matches
    assert not store.dao.planning_failures("Qwen/Qwen3-8B", _principal(), facts).matches
    store.dao.assert_current(failures.matches, principal)
    with store.factory.begin() as session:
        session.execute(
            update(BenchmarkRecordRow)
            .where(BenchmarkRecordRow.benchmark_id == failures.matches[0].record.benchmark_id)
            .values(outcome="succeeded")
        )
    with pytest.raises(ConflictError):
        store.dao.assert_current(failures.matches, principal)
    for limit in (0, 51):
        with pytest.raises(ValueError):
            store.dao.planning_failures("Qwen/Qwen3-8B", principal, facts, limit=limit)
