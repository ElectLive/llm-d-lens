"""Published-only PostgreSQL retrieval; callers must enforce the feature permission."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from sqlalchemy import Float, Numeric, and_, case, cast, func, literal, or_, select, union_all
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import aliased
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.sql.elements import ColumnElement

from llm_d_lens.agentic.public_benchmark_record.contracts import (
    DEFAULT_PUBLIC_RUNTIME,
    UNKNOWN_RUNTIME_LABELS,
    WORKLOAD_ALIASES,
    WORKLOAD_NUMBER_PATTERN,
    PublicBenchmarkRecord,
    effective_runtime,
    metric_aliases,
    model_identity_aliases,
    planning_latency_axes,
    statistic_aliases,
    workload_number,
)
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationError, registry_id
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters
from llm_d_lens.core.exceptions import ConflictError
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao
from llm_d_lens.db.models.public_benchmark_record import PublicBenchmarkRecordRow

if TYPE_CHECKING:
    from llm_d_lens.agentic.planner import PlanningFacts


@dataclass(frozen=True)
class QueryMatch:
    item: PublicBenchmarkRecord
    experiment: PublicBenchmarkRecord | None
    source: PublicBenchmarkRecord
    manifest_id: str
    generation: int
    related_experiments: tuple[PublicBenchmarkRecord, ...] = ()


@dataclass(frozen=True)
class QuerySlice:
    matches: tuple[QueryMatch, ...]
    total: int
    experiment_total: int | None = None


def model_identity_condition(column: ColumnElement[str], model: str) -> ColumnElement[bool]:
    """Derive stored aliases too, so existing rows support reverse alias lookup without a backfill."""
    name = func.lower(func.trim(column))
    aliases = model_identity_aliases(model)
    return or_(
        name.in_(aliases),
        func.regexp_replace(name, "^.*/", "").in_(aliases),
        func.replace(name, "/", "-").in_(aliases),
    )


def runtime_condition(column: ColumnElement[str], runtime: str) -> ColumnElement[bool]:
    name = func.lower(func.trim(column))
    backend = case(
        (or_(name.is_(None), name.in_(UNKNOWN_RUNTIME_LABELS)), DEFAULT_PUBLIC_RUNTIME),
        else_=name,
    )
    return backend == effective_runtime(runtime).strip().casefold()


def latency_dominator_count(profiles, axis_names: tuple[str, ...]):
    peer = profiles.alias("latency_peer")
    return (
        select(func.count())
        .select_from(peer)
        .where(
            peer.c.workload_score.is_not_distinct_from(profiles.c.workload_score),
            peer.c.workload_coverage.is_not_distinct_from(profiles.c.workload_coverage),
            *(profiles.c[name].is_(None) == peer.c[name].is_(None) for name in axis_names),
            *(or_(profiles.c[name].is_(None), peer.c[name] <= profiles.c[name]) for name in axis_names),
            or_(*(peer.c[name] < profiles.c[name] for name in axis_names)),
        )
        .correlate(profiles)
        .scalar_subquery()
    )


class PublicBenchmarkRecordQueryDao(PublicBenchmarkRecordDao):
    planning_facts: PlanningFacts | None = None

    def hardware_options(self, filters: QueryFilters) -> tuple[str, ...]:
        """Published model-specific hardware labels for authorized query orchestration."""
        with self._read_only() as session:
            self._require_postgres(session)
            published = union_all(*(self._published_members(origin) for origin in filters.origins)).cte()
            experiment = aliased(PublicBenchmarkRecordRow, published)
            statement = select(experiment.hardware).where(
                experiment.kind == "experiment",
                experiment.hardware.is_not(None),
            )
            if filters.model is not None:
                statement = statement.where(model_identity_condition(experiment.model, filters.model))
            return tuple(name for name in session.scalars(statement.distinct().order_by(experiment.hardware)) if name)

    @staticmethod
    def _conditions(
        experiment: AliasedClass[PublicBenchmarkRecordRow], filters: QueryFilters
    ) -> list[ColumnElement[bool]]:
        conditions = []
        if filters.experiment_id is not None:
            conditions.append(experiment.item_id == filters.experiment_id)
        if filters.record_id is not None:
            conditions.append(experiment.source_pointer["record_id"].as_string() == filters.record_id)
        if filters.model is not None:
            conditions.append(model_identity_condition(experiment.model, filters.model))
        if filters.hardware is not None:
            conditions.append(func.lower(experiment.hardware) == filters.hardware.lower())
        if filters.runtime is not None:
            conditions.append(runtime_condition(experiment.runtime, filters.runtime))
        for field, requested in filters.workload.items():
            number = workload_number(requested)
            if number is None:
                conditions.append(experiment.workload.cast(JSONB).contains({field: requested}))
                continue
            raw = experiment.workload.cast(JSONB)[field]
            numeric = case(
                (
                    and_(
                        func.jsonb_typeof(raw).in_(("number", "string")),
                        raw.astext.op("~")(WORKLOAD_NUMBER_PATTERN),
                    ),
                    cast(raw.astext, Numeric),
                ),
                else_=None,
            )
            conditions.append(numeric == number)
        return conditions

    def _planning_statement(self, statement, item, experiment, source, registry, *, experiment_limit):
        facts = self.planning_facts
        assert facts is not None
        requested = (
            {key: value for key, value in asdict(facts.workload_profile).items() if value is not None}
            if facts.workload_profile is not None
            else {}
        )
        similarities = []
        coverage = []
        for field, target in requested.items():
            numbers = []
            for name in WORKLOAD_ALIASES.get(field, (field,)):
                raw = experiment.workload.cast(JSONB)[name]
                text = raw.astext
                numbers.append(
                    case(
                        (
                            and_(
                                func.jsonb_typeof(raw).in_(("number", "string")),
                                text.op("~")(r"^[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"),
                            ),
                            cast(text, Float),
                        ),
                        else_=None,
                    )
                )
            value = func.coalesce(*numbers, literal(None))
            known = and_(value >= 0, value < float("inf"))
            coverage.append(case((known, 1.0), else_=0.0))
            score = (
                1 - func.abs(value - target)
                if field == "shared_prefix_ratio"
                else case(
                    (value == target, 1.0),
                    else_=func.least(value, target) / func.nullif(func.greatest(value, target), 0),
                )
            )
            similarities.append(case((known, score), else_=0.0))
        workload_score = (
            case((sum(coverage) > 0, sum(similarities) / len(requested)), else_=None)
            if requested
            else literal(None, Float)
        )
        workload_coverage = sum(coverage) / len(requested) if requested else literal(None, Float)
        number = case((item.normalized_unit.is_not(None), item.normalized_value), else_=item.value)
        unit = func.coalesce(item.normalized_unit, item.unit)
        axes = planning_latency_axes(facts)
        latency_columns = []
        for index, (metric, statistic) in enumerate(axes):
            milliseconds = "ms/token" if metric == "tpot" else "ms"
            seconds = "s/token" if metric == "tpot" else "s"
            latency_columns.append(
                case(
                    (
                        and_(
                            item.metric.in_(metric_aliases(metric)),
                            func.coalesce(item.statistic, "mean").in_(statistic_aliases(statistic))
                            if statistic is not None
                            else item.statistic.is_(None),
                            number > 0,
                            number < float("inf"),
                        ),
                        case((unit == milliseconds, number), (unit == seconds, number * 1000), else_=None),
                    ),
                    else_=None,
                ).label(f"latency_{index}")
            )
        base = statement.with_only_columns(
            item.item_id.label("measurement_id"),
            experiment.item_id.label("experiment_id"),
            source.item_id.label("source_id"),
            registry.item_id.label("registry_id"),
            item.origin_id.label("origin_id"),
            item.content_sha256.label("source_sha256"),
            func.coalesce(item.series.cast(JSONB), cast(literal("{}"), JSONB)).label("series"),
            experiment.timestamp.label("experiment_time"),
            workload_score.label("workload_score"),
            workload_coverage.label("workload_coverage"),
            case((item.evidence_type == "raw_json", 0), else_=1).label("source_priority"),
            *latency_columns,
        ).cte("planning_measurements")
        group_keys = ("origin_id", "experiment_id", "source_sha256", "series")
        grouped = (
            select(
                *(base.c[key] for key in group_keys),
                func.max(base.c.experiment_time).label("experiment_time"),
                func.max(base.c.workload_score).label("workload_score"),
                func.max(base.c.workload_coverage).label("workload_coverage"),
                func.min(base.c.source_priority).label("source_priority"),
                func.count().label("observation_count"),
                *(func.max(base.c[f"latency_{index}"]).label(f"latency_{index}") for index in range(len(axes))),
            )
            .group_by(*(base.c[key] for key in group_keys))
            .cte("planning_experiments")
        )
        dominated = latency_dominator_count(grouped, tuple(f"latency_{index}" for index in range(len(axes))))
        profiles = select(
            grouped,
            sum(case((grouped.c[f"latency_{index}"].is_not(None), 1), else_=0) for index in range(len(axes))).label(
                "latency_coverage"
            ),
            dominated.label("dominated_by"),
            func.sum(grouped.c.observation_count).over().label("total_observations"),
            func.count().over().label("total_experiments"),
        ).cte("planning_profiles")

        def order(columns):
            return (
                columns.workload_score.desc().nullslast(),
                columns.workload_coverage.desc().nullslast(),
                columns.latency_coverage.desc(),
                columns.dominated_by,
                columns.experiment_time.desc().nullslast(),
                columns.source_priority,
                columns.experiment_id,
                columns.source_sha256,
                cast(columns.series, JSONB),
            )

        chosen = select(profiles).order_by(*order(profiles.c)).limit(experiment_limit).cte("planning_selected")
        selected = (
            select(base.c.measurement_id, base.c.source_id, base.c.registry_id, chosen)
            .join(chosen, and_(*(base.c[key] == chosen.c[key] for key in group_keys)))
            .cte("selected_planning_measurements")
        )
        selected_item = aliased(PublicBenchmarkRecordRow, name="selected_measurement")
        selected_experiment = aliased(PublicBenchmarkRecordRow, name="selected_experiment")
        selected_source = aliased(PublicBenchmarkRecordRow, name="selected_source")
        selected_registry = aliased(PublicBenchmarkRecordRow, name="selected_registry")
        return (
            select(
                selected_item,
                selected_experiment,
                selected_source,
                selected_registry,
                selected.c.total_observations,
                selected.c.total_experiments,
            )
            .select_from(selected)
            .join(selected_item, selected_item.item_id == selected.c.measurement_id)
            .join(selected_experiment, selected_experiment.item_id == selected.c.experiment_id)
            .join(selected_source, selected_source.item_id == selected.c.source_id)
            .join(selected_registry, selected_registry.item_id == selected.c.registry_id)
            .order_by(*order(selected.c), selected_item.item_id)
        )

    def search(
        self,
        filters: QueryFilters,
        *,
        intent: str,
        text_query: str,
        limit: int | None,
        experiment_limit: int | None = None,
    ) -> QuerySlice:
        if intent not in ("numeric", "comparison", "figure"):
            raise PublicationError("retrieval needs a resolved intent")
        if limit is None:
            if intent not in ("numeric", "comparison") or not filters.model or not filters.hardware:
                raise PublicationError("complete retrieval requires exact model and hardware scope")
        elif not 1 <= limit <= 50:
            raise PublicationError("retrieval needs a bounded limit")
        if experiment_limit is not None and (
            not 1 <= experiment_limit <= 50
            or self.planning_facts is None
            or intent not in ("numeric", "comparison")
            or not filters.model
            or not filters.hardware
        ):
            raise PublicationError("experiment retrieval requires planning facts and exact model/hardware scope")
        with self._read_only() as session:
            self._require_postgres(session)
            published = union_all(*(self._published_members(origin) for origin in filters.origins)).cte("query_members")
            item = aliased(PublicBenchmarkRecordRow, published)
            # Repeated wide manifest scans otherwise multiply with every measurement.
            experiment_members = (
                select(published)
                .where(published.c.kind == "experiment")
                .cte("published_experiments")
                .prefix_with("MATERIALIZED")
            )
            source_members = (
                select(published)
                .where(published.c.kind == "source", published.c.source_role == "file")
                .cte("published_files")
                .prefix_with("MATERIALIZED")
            )
            experiment = aliased(PublicBenchmarkRecordRow, experiment_members)
            source = aliased(PublicBenchmarkRecordRow, source_members)
            registry = aliased(PublicBenchmarkRecordRow, name="query_registry")
            member_ids = select(published.c.item_id).correlate(None)
            statement = (
                select(item, experiment, source, registry, func.count().over())
                .select_from(item)
                .outerjoin(
                    experiment,
                    and_(
                        experiment.item_id == item.experiment_id,
                        experiment.kind == "experiment",
                        experiment.origin_id == item.origin_id,
                    ),
                )
                .join(
                    source,
                    and_(
                        source.kind == "source",
                        source.source_role == "file",
                        source.source_path == item.source_path,
                        source.source_revision == item.source_revision,
                        source.content_sha256 == item.content_sha256,
                        source.origin_id == item.origin_id,
                    ),
                )
                .join(
                    registry,
                    and_(
                        registry.item_id.in_([registry_id(origin) for origin in filters.origins]),
                        registry.origin_id == item.origin_id,
                    ),
                )
            )
            kind = "measurement" if intent in ("numeric", "comparison") else "figure"
            statement = statement.where(item.kind == kind)
            conditions = self._conditions(experiment, filters)
            link = aliased(PublicBenchmarkRecordRow, name="query_figure_link")
            linked = (
                select(link.parent_id, experiment)
                .select_from(link)
                .join(experiment, experiment.item_id == link.referenced_item_id)
                .where(
                    link.kind == "figure_link",
                    link.item_id.in_(member_ids),
                    experiment.kind == "experiment",
                    experiment.item_id.in_(member_ids),
                    link.origin_id == experiment.origin_id,
                    *conditions,
                )
            )
            if kind == "figure" and conditions:
                statement = statement.where(
                    linked.where(link.parent_id == item.item_id, link.origin_id == item.origin_id).exists()
                )
            elif kind != "figure":
                statement = statement.where(*conditions)
            if intent in ("numeric", "comparison"):
                statement = statement.where(experiment.item_id.is_not(None))
            for field in ("source_revision", "source_path"):
                value = getattr(filters, field)
                if value is not None:
                    statement = statement.where(getattr(item, field) == value)
            if filters.series:
                statement = statement.where(item.series.cast(JSONB).contains(filters.series))
            for field in ("metric", "statistic"):
                value = getattr(filters, field)
                if value is not None:
                    column = func.coalesce(item.statistic, "mean") if field == "statistic" else item.metric
                    aliases = statistic_aliases(value) if field == "statistic" else metric_aliases(value)
                    statement = statement.where(column.in_(aliases))
            value_column = item.normalized_value if filters.normalized else item.value
            unit_column = item.normalized_unit if filters.normalized else item.unit
            if filters.unit is not None:
                statement = statement.where(unit_column == filters.unit)
            if filters.value_min is not None:
                statement = statement.where(value_column >= filters.value_min)
            if filters.value_max is not None:
                statement = statement.where(value_column <= filters.value_max)
            facts = self.planning_facts
            if facts is not None and kind == "measurement":
                if facts.requirements is not None and facts.requirements.runtime.backend:
                    statement = statement.where(
                        runtime_condition(experiment.runtime, facts.requirements.runtime.backend)
                    )
                statement = self._planning_statement(
                    statement, item, experiment, source, registry, experiment_limit=experiment_limit
                )
            else:
                statement = statement.order_by(case((item.evidence_type == "raw_json", 0), else_=1), item.item_id)
            rows = session.execute(statement.limit(10_001 if experiment_limit is not None else limit)).all()
            if experiment_limit is not None and len(rows) > 10_000:
                raise PublicationError("selected experiments exceed the 10000-measurement planning safety limit")
            related: dict[str, list[PublicBenchmarkRecord]] = {}
            if kind == "figure" and rows:
                for figure_id, context in session.execute(
                    linked.where(link.parent_id.in_([row[0].item_id for row in rows])).order_by(
                        link.parent_id, experiment.item_id
                    )
                ):
                    related.setdefault(figure_id, []).append(context.to_dto())
            matches = []
            for row in rows:
                evidence, context, file, head, _count = row[:5]
                state = head.source_details or {}
                generation = state.get("generation")
                if (
                    not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or head.published_manifest_id is None
                ):
                    raise PublicationError("published registry state is invalid")
                matches.append(
                    QueryMatch(
                        item=evidence.to_dto(),
                        experiment=context.to_dto() if context is not None else None,
                        source=file.to_dto(),
                        manifest_id=head.published_manifest_id,
                        generation=generation,
                        related_experiments=tuple(related.get(evidence.item_id, ())),
                    )
                )
            return QuerySlice(
                matches=tuple(matches),
                total=int(rows[0][4]) if rows else 0,
                experiment_total=int(rows[0][5]) if rows and experiment_limit is not None else None,
            )

    def assert_current(self, matches: tuple[QueryMatch, ...]) -> None:
        expected = {match.item.origin_id: (match.manifest_id, match.generation) for match in matches}
        if not expected:
            return
        with self._read_only() as session:
            self._require_postgres(session)
            for row in session.scalars(
                select(PublicBenchmarkRecordRow).where(
                    PublicBenchmarkRecordRow.item_id.in_([registry_id(origin) for origin in expected])
                )
            ):
                state = row.source_details or {}
                approval = state.get("approval")
                if (
                    expected.pop(row.origin_id, None) != (row.published_manifest_id, state.get("generation"))
                    or state.get("enabled") is not True
                    or not isinstance(approval, dict)
                    or approval.get("verified_public") is not True
                ):
                    raise ConflictError("public benchmark publication or source access changed during the query; retry")
            if expected:
                raise ConflictError("public benchmark publication changed during the query; retry")
