"""Read-only, caller-scoped measured Agentic evidence.

Access uses evaluation resource permissions, not public-corpus query grants.
Only canonical ``evaluate:<uuid>`` records and individually identified stage or
matrix-stage samples with a live, unambiguous evaluation and owner are eligible.
Legacy simulation task references are not evaluations.
TTFT/E2E milliseconds and throughput tokens/s have identity normalization; the
throughput denominator does not establish output tokens. TPOT ms/token and
fraction normalization are unsupported; raw fraction ranges are supported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlalchemy import Float, Select, and_, case, false, func, literal, or_, select, text, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkRecord,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    WorkloadProfile,
    historical_query_context,
)
from llm_d_lens.agentic.public_benchmark_record.agentic_metrics import AGENTIC_METRICS, metric_matches
from llm_d_lens.agentic.public_benchmark_record.contracts import planning_latency_axes
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters
from llm_d_lens.auth.contracts import Principal, ScopeType
from llm_d_lens.auth.permissions import permission_granted
from llm_d_lens.auth.scope import reachable_clusters
from llm_d_lens.core.exceptions import ConflictError, PasswordChangeRequiredError, UnauthenticatedError
from llm_d_lens.db.dao.base import BaseDao
from llm_d_lens.db.dao.public_benchmark_record_query import latency_dominator_count, model_identity_condition
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow
from llm_d_lens.db.models.evaluate_run import EvaluateRunRow
from llm_d_lens.db.models.user import UserRow

if TYPE_CHECKING:
    from llm_d_lens.agentic.planner import PlanningFacts

_READ = "evaluate:run:read"
_UUID = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_WORKLOAD = frozenset(
    {
        "mean_input_tokens",
        "p95_input_tokens",
        "mean_output_tokens",
        "concurrency",
        "request_rate",
        "shared_prefix_ratio",
        "prefill_fraction",
    }
)


@dataclass(frozen=True)
class AgenticMatch:
    record: BenchmarkRecord
    evaluate_run_id: str
    cluster_id: str
    owner_user_id: str
    version_id: int


@dataclass(frozen=True)
class AgenticSlice:
    matches: tuple[AgenticMatch, ...]
    total: int


def _owner() -> ColumnElement[str]:
    return func.coalesce(EvaluateRunRow.owner_user_id, EvaluateRunRow.extra["owner_user_id"].as_string())


def _authorization(principal: Principal) -> ColumnElement[bool]:
    """SQL equivalent of authorize(..., ResourceRef('evaluate_run', ...)).

    Evaluate resources have no resource shares or owner-group inheritance.
    Effective bindings already contain group-inherited role grants.
    """
    if principal.is_global_admin():
        return true()
    action_scopes: list[ColumnElement[bool]] = []
    broad_scopes: list[ColumnElement[bool]] = []
    for binding in principal.bindings:
        if binding.scope_type is ScopeType.RESOURCE:
            continue
        if not permission_granted(principal.permissions_for(binding.role), _READ):
            continue
        scope = (
            true() if binding.scope_type is ScopeType.GLOBAL else EvaluateRunRow.cluster_id == binding.scope_cluster_id
        )
        action_scopes.append(scope)
        if _READ not in principal.self_scoped_for(binding.role):
            broad_scopes.append(scope)
    clusters = reachable_clusters(principal)
    reachable = true() if clusters is None else EvaluateRunRow.cluster_id.in_(clusters)
    return and_(
        or_(false(), *action_scopes),
        or_(false(), *broad_scopes, and_(_owner() == principal.user_id, reachable)),
    )


def _verified() -> list[ColumnElement[bool]]:
    extra = EvaluateRunRow.extra.cast(JSONB)
    owner = extra["owner_user_id"].astext
    return [
        EvaluateRunRow.id.op("~")(_UUID),
        BenchmarkRecordRow.evaluate_run_id == EvaluateRunRow.id,
        BenchmarkRecordRow.benchmark_id.regexp_match(
            func.concat(
                "^evaluate:",
                EvaluateRunRow.id,
                "(:stage:(0|[1-9][0-9]*)|:matrix:(0|[1-9][0-9]*):stage:(0|[1-9][0-9]*))?$",
            )
        ),
        EvaluateRunRow.kind == "benchmark",
        EvaluateRunRow.cluster_id.is_not(None),
        func.length(func.trim(EvaluateRunRow.cluster_id)) > 0,
        # A present JSON projection must agree, including its JSON type.
        *[
            or_(
                ~extra.has_key(key),
                and_(func.jsonb_typeof(extra[key]) == "string", extra[key].astext == column),
            )
            for key, column in (("id", EvaluateRunRow.id), ("cluster_id", EvaluateRunRow.cluster_id))
        ],
        or_(
            ~extra.has_key("owner_user_id"),
            and_(
                func.jsonb_typeof(extra["owner_user_id"]) == "string",
                func.length(func.trim(owner)) > 0,
                owner == func.trim(owner),
                or_(EvaluateRunRow.owner_user_id.is_(None), EvaluateRunRow.owner_user_id == owner),
            ),
        ),
        func.length(func.trim(_owner())) > 0,
    ]


def _metric_conditions(filters: QueryFilters) -> list[ColumnElement[bool]]:
    available: list[ColumnElement[bool]] = []
    valid: list[ColumnElement[bool]] = []
    for spec in AGENTIC_METRICS:
        column = getattr(BenchmarkRecordRow, "metrics_" + spec.field)
        finite = and_(column > float("-inf"), column < float("inf"))
        # Validate every persisted metric before constructing the strict DTO.
        domain = and_(column >= 0, column <= 1) if spec.unit == "fraction" else column > 0
        valid.append(or_(column.is_(None), and_(finite, domain)))
        if not metric_matches(spec, filters):
            continue
        bounds: list[ColumnElement[bool]] = [column.is_not(None), finite, domain]
        if filters.value_min is not None:
            bounds.append(column >= filters.value_min)
        if filters.value_max is not None:
            bounds.append(column <= filters.value_max)
        available.append(and_(*bounds))
    return [*valid, or_(false(), *available)]


def _known_identity(column: ColumnElement[str]) -> ColumnElement[bool]:
    return and_(
        column.is_not(None),
        func.lower(func.trim(column)).not_in(("", "unknown", "unavailable", "cluster", "none")),
    )


def _planning_selection(
    model: str, facts: PlanningFacts
) -> tuple[list[ColumnElement[bool]], list[ColumnElement], ColumnElement, ColumnElement]:
    context = historical_query_context(model, facts)
    hardware = cast(HardwareDescriptor, context["hardware"])
    runtime = cast(RuntimeFingerprint, context["runtime"])
    requested_model = cast(ModelDescriptor, context["model"])
    workload = cast(WorkloadProfile, context["workload"])
    row = BenchmarkRecordRow
    identity = row.provenance.cast(JSONB)["hardware_identity"]
    conditions: list[ColumnElement[bool]] = [model_identity_condition(row.model_model_id, model)]
    identity_scores: list[ColumnElement[int]] = []
    # HardwareDescriptor normalizes accelerator names at persistence and query
    # boundaries; variant identity survives in the existing provenance projection.
    for column, requested in (
        (row.hardware_accelerator, hardware.accelerator),
        (identity["profile_id"].astext, hardware.profile_id),
        (identity["form_factor"].astext, hardware.form_factor),
        (identity["interconnect"].astext, hardware.interconnect),
        (row.runtime_backend, runtime.backend),
        (row.model_quantization, requested_model.quantization),
    ):
        if requested and requested.strip().lower() not in {"unknown", "unavailable", "cluster", "none"}:
            known = _known_identity(column)
            same = func.lower(func.trim(column)) == requested.strip().lower()
            conditions.append(or_(~known, same))
            identity_scores.append(case((and_(known, same), 1), else_=0))
    if hardware.vram_per_gpu_gib is not None:
        conditions.append(
            or_(row.hardware_vram_per_gpu_gib.is_(None), row.hardware_vram_per_gpu_gib == hardware.vram_per_gpu_gib)
        )
        identity_scores.append(case((row.hardware_vram_per_gpu_gib == hardware.vram_per_gpu_gib, 1), else_=0))

    similarity: list[ColumnElement[float]] = []
    workload_coverage: list[ColumnElement[float]] = []
    for field in sorted(_WORKLOAD):
        requested = getattr(workload, field)
        if requested is None:
            continue
        column = getattr(row, "workload_" + field)
        if field in {"shared_prefix_ratio", "prefill_fraction"}:
            score = 1.0 - func.abs(column - requested)
        else:
            # min/max is exp(-abs(log(actual/requested))) without log(0).
            score = case(
                (column == requested, 1.0),
                else_=func.least(column, requested) / func.nullif(func.greatest(column, requested), 0),
            )
        similarity.append(case((column.is_not(None), score), else_=0.0))
        workload_coverage.append(case((column.is_not(None), 1.0), else_=0.0))

    order: list[ColumnElement] = []
    order.append(row.timestamp.desc())
    if identity_scores:
        order.append(sum(identity_scores).desc())
    if hardware.gpu_count is not None:
        order.append(
            case(
                (
                    row.hardware_gpu_count.is_not(None),
                    func.least(row.hardware_gpu_count, hardware.gpu_count)
                    / func.greatest(row.hardware_gpu_count, hardware.gpu_count),
                ),
                else_=0.0,
            ).desc()
        )
    if runtime.backend_version:
        order.append(case((row.runtime_backend_version == runtime.backend_version, 1), else_=0).desc())
    order.append(row.benchmark_id.asc())
    score = (
        case((sum(workload_coverage) > 0, sum(similarity) / len(similarity)), else_=None)
        if similarity
        else literal(None, Float)
    )
    coverage = sum(workload_coverage) / len(workload_coverage) if workload_coverage else literal(None, Float)
    return conditions, order, score, coverage


class BenchmarkExperimentQueryDao(BaseDao):
    @staticmethod
    def _caller(session: Session, principal: Principal) -> None:
        if principal is None or not principal.user_id or principal.user_id == "system":
            raise UnauthenticatedError("authentication required")
        if principal.must_change_password:
            raise PasswordChangeRequiredError("password change required before continuing")
        caller = session.get(UserRow, principal.user_id)
        if (
            caller is None
            or caller.status != "active"
            or caller.must_change_password
            or caller.principal_version != principal.authz_version
        ):
            raise ConflictError("caller identity changed; authenticate again")

    @staticmethod
    def _statement(principal: Principal) -> Select[tuple[BenchmarkRecordRow, EvaluateRunRow, str]]:
        return (
            select(BenchmarkRecordRow, EvaluateRunRow, _owner().label("verified_owner"))
            .join(EvaluateRunRow, BenchmarkRecordRow.evaluate_run_id == EvaluateRunRow.id)
            .join(UserRow, UserRow.id == _owner())
            .where(*_verified(), _authorization(principal))
        )

    @staticmethod
    def _read_transaction(session: Session) -> None:
        if session.get_bind().dialect.name != "postgresql":
            raise ValueError("Agentic benchmark retrieval requires PostgreSQL")
        session.execute(text("SET TRANSACTION READ ONLY"))

    def search(self, filters: QueryFilters, principal: Principal, *, limit: int) -> AgenticSlice:
        if not 1 <= limit <= 50:
            raise ValueError("Agentic retrieval needs a limit between 1 and 50")
        with self._read_only() as session:
            self._read_transaction(session)
            self._caller(session, principal)
            if filters.source_revision is not None or filters.source_path is not None or filters.series:
                return AgenticSlice((), 0)
            statement = self._statement(principal).where(
                BenchmarkRecordRow.outcome == "succeeded", *_metric_conditions(filters)
            )
            if filters.record_id is not None:
                statement = statement.where(
                    or_(
                        BenchmarkRecordRow.benchmark_id == filters.record_id,
                        EvaluateRunRow.id == filters.record_id,
                    )
                )
            if filters.experiment_id is not None:
                statement = statement.where(BenchmarkRecordRow.benchmark_id == filters.experiment_id)
            if filters.model is not None:
                statement = statement.where(model_identity_condition(BenchmarkRecordRow.model_model_id, filters.model))
            for field, name in (
                ("hardware", "hardware_accelerator"),
                ("runtime", "runtime_backend"),
            ):
                value = getattr(filters, field)
                if value is not None:
                    statement = statement.where(func.lower(getattr(BenchmarkRecordRow, name)) == value.lower())
            for field, value in filters.workload.items():
                if field not in _WORKLOAD or isinstance(value, bool) or not isinstance(value, (int, float)):
                    return AgenticSlice((), 0)
                statement = statement.where(getattr(BenchmarkRecordRow, "workload_" + field) == value)
            rows = session.execute(
                statement.add_columns(func.count().over()).order_by(BenchmarkRecordRow.benchmark_id).limit(limit)
            ).all()
            matches = tuple(
                AgenticMatch(record.to_dto(), run.id, run.cluster_id, owner, run.version_id)
                for record, run, owner, _count in rows
            )
            return AgenticSlice(matches, rows[0][3] if rows else 0)

    def planning_search(
        self, model: str, principal: Principal, facts: PlanningFacts, *, limit: int = 50
    ) -> AgenticSlice:
        """Rank authorized, compatible history in SQL before bounded payload reads.

        Unknown identities remain neutral, not exact matches. No candidate topology
        exists at this stage, and no topology or GPU-count constraint is inferred.
        """
        if not 1 <= limit <= 50:
            raise ValueError("Agentic retrieval needs a limit between 1 and 50")
        conditions, order, workload_score, workload_coverage = _planning_selection(model, facts)
        with self._read_only() as session:
            self._read_transaction(session)
            self._caller(session, principal)
            eligible = self._statement(principal).where(
                BenchmarkRecordRow.outcome == "succeeded", *_metric_conditions(QueryFilters()), *conditions
            )
            axes = planning_latency_axes(facts)
            latency_columns = []
            for index, (metric, statistic) in enumerate(axes):
                spec = next(
                    (
                        spec
                        for spec in AGENTIC_METRICS
                        if ("latency" if spec.metric == "e2e" else spec.metric) == metric
                        and spec.statistic == statistic
                    ),
                    None,
                )
                latency_columns.append(
                    (getattr(BenchmarkRecordRow, "metrics_" + spec.field) if spec else literal(None, Float)).label(
                        f"latency_{index}"
                    )
                )
            profiles = eligible.with_only_columns(
                BenchmarkRecordRow.benchmark_id,
                workload_score.label("workload_score"),
                workload_coverage.label("workload_coverage"),
                *latency_columns,
            ).cte("authorized_planning_profiles")
            coverage = sum(
                case((profiles.c[f"latency_{index}"].is_not(None), 1), else_=0) for index in range(len(axes))
            )
            dominated = latency_dominator_count(profiles, tuple(f"latency_{index}" for index in range(len(axes))))
            selected_profiles = select(
                profiles.c.benchmark_id,
                profiles.c.workload_score,
                profiles.c.workload_coverage,
                coverage.label("latency_coverage"),
                dominated.label("dominated_by"),
            ).cte("internal_experiment_ranking")
            eligible = eligible.join(
                selected_profiles, selected_profiles.c.benchmark_id == BenchmarkRecordRow.benchmark_id
            )
            order = [
                selected_profiles.c.workload_score.desc().nullslast(),
                selected_profiles.c.workload_coverage.desc().nullslast(),
                selected_profiles.c.latency_coverage.desc(),
                selected_profiles.c.dominated_by,
                *order,
            ]
            selected = session.execute(
                eligible.with_only_columns(BenchmarkRecordRow.benchmark_id, func.count().over())
                .order_by(*order)
                .limit(limit)
            ).all()
            if not selected:
                return AgenticSlice((), 0)
            rows = session.execute(
                eligible.where(BenchmarkRecordRow.benchmark_id.in_([key for key, _count in selected])).order_by(*order)
            ).all()
            matches = tuple(
                AgenticMatch(record.to_dto(), run.id, run.cluster_id, owner, run.version_id)
                for record, run, owner in rows
            )
            return AgenticSlice(matches, selected[0][1])

    def planning_failures(
        self, model: str, principal: Principal, facts: PlanningFacts, *, limit: int = 50
    ) -> AgenticSlice:
        """Retrieve bounded negative evidence without competing with measured-case limits."""
        if not 1 <= limit <= 50:
            raise ValueError("Agentic retrieval needs a limit between 1 and 50")
        conditions, order, workload_score, workload_coverage = _planning_selection(model, facts)
        with self._read_only() as session:
            self._read_transaction(session)
            self._caller(session, principal)
            rows = session.execute(
                self._statement(principal)
                .where(BenchmarkRecordRow.outcome == "failed", *conditions)
                .add_columns(func.count().over())
                .order_by(workload_score.desc().nullslast(), workload_coverage.desc().nullslast(), *order)
                .limit(limit)
            ).all()
            matches = tuple(
                AgenticMatch(record.to_dto(), run.id, run.cluster_id, owner, run.version_id)
                for record, run, owner, _count in rows
            )
            return AgenticSlice(matches, rows[0][3] if rows else 0)

    def assert_current(self, matches: Sequence[AgenticMatch], principal: Principal) -> None:
        """Recheck authorization and provenance immediately before emission."""
        with self._read_only() as session:
            self._read_transaction(session)
            self._caller(session, principal)
            if not matches:
                return
            rows = session.execute(
                self._statement(principal).where(
                    BenchmarkRecordRow.benchmark_id.in_([match.record.benchmark_id for match in matches])
                )
            ).all()
            current = {
                record.benchmark_id: (run.id, run.cluster_id, owner, run.version_id, record.outcome)
                for record, run, owner in rows
            }
            if any(
                current.get(match.record.benchmark_id)
                != (
                    match.evaluate_run_id,
                    match.cluster_id,
                    match.owner_user_id,
                    match.version_id,
                    match.record.outcome,
                )
                for match in matches
            ):
                raise ConflictError("Agentic evaluation evidence or access changed; retry the query")
