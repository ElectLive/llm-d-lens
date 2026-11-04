"""Public benchmark facts and their immutable provenance."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import JsonValue
from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Mapped, mapped_column

from llm_d_lens.agentic.public_benchmark_record.contracts import KIND_FIELDS, REQUIRED_FIELDS, PublicBenchmarkRecord
from llm_d_lens.db.base import Base, DeclarativeDtoMixin, UTCDateTime

_NULLABLE_JSON = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class _ExactNumeric(TypeDecorator[Decimal]):
    impl = Numeric().with_variant(Text(), "sqlite")
    cache_ok = True

    def process_bind_param(self, value: Decimal | None, dialect: Dialect) -> Decimal | str | None:
        return str(value) if value is not None and dialect.name == "sqlite" else value

    def process_result_value(self, value: Decimal | str | None, dialect: Dialect) -> Decimal | None:
        return Decimal(value) if value is not None else None


def _shape_checks() -> list[CheckConstraint]:
    checks = []
    specific_fields = set().union(*KIND_FIELDS.values())
    for kind, allowed in KIND_FIELDS.items():
        predicates = [f"{field} IS NULL" for field in sorted(specific_fields - allowed)]
        predicates += [f"{field} IS NOT NULL" for field in sorted(REQUIRED_FIELDS[kind])]
        checks.append(CheckConstraint(f"kind <> '{kind}' OR ({' AND '.join(predicates)})", name=f"ck_rag_{kind}_shape"))
    return checks


class PublicBenchmarkRecordRow(Base, DeclarativeDtoMixin):
    __tablename__ = "public_benchmark_records"
    dto_type = PublicBenchmarkRecord
    column_map = {name: name for name in PublicBenchmarkRecord.model_fields}

    item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    origin_id: Mapped[str] = mapped_column(Text, nullable=False)
    logical_key: Mapped[str] = mapped_column(Text, nullable=False)
    source_revision: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    access_classification: Mapped[str] = mapped_column(String(32), nullable=False)
    publication_state: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("public_benchmark_records.item_id", ondelete="RESTRICT"))
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    format: Mapped[str] = mapped_column(Text, nullable=False)
    source_role: Mapped[str | None] = mapped_column(String(16))
    source_details: Mapped[dict[str, JsonValue] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    manifest_item_ids: Mapped[list[str] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    published_manifest_id: Mapped[str | None] = mapped_column(
        ForeignKey("public_benchmark_records.item_id", ondelete="RESTRICT")
    )
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("public_benchmark_records.item_id", ondelete="RESTRICT")
    )
    model: Mapped[str | None] = mapped_column(Text)
    hardware: Mapped[str | None] = mapped_column(Text)
    runtime: Mapped[str | None] = mapped_column(Text)
    workload: Mapped[dict[str, JsonValue] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    configuration: Mapped[dict[str, JsonValue] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime())
    timestamp_kind: Mapped[str | None] = mapped_column(String(16))
    source_pointer: Mapped[dict[str, JsonValue] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    metric: Mapped[str | None] = mapped_column(Text)
    statistic: Mapped[str | None] = mapped_column(Text)
    value: Mapped[Decimal | None] = mapped_column(_ExactNumeric())
    unit: Mapped[str | None] = mapped_column(Text)
    original_value: Mapped[Decimal | None] = mapped_column(_ExactNumeric())
    original_unit: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[Decimal | None] = mapped_column(_ExactNumeric())
    normalized_unit: Mapped[str | None] = mapped_column(Text)
    normalization_rule: Mapped[str | None] = mapped_column(Text)
    series: Mapped[dict[str, JsonValue] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    evidence_type: Mapped[str | None] = mapped_column(String(32))
    passage_text: Mapped[str | None] = mapped_column(Text)
    heading_path: Mapped[list[str] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text)
    coverage: Mapped[str | None] = mapped_column(String(16))
    missing_series: Mapped[list[str] | None] = mapped_column(_NULLABLE_JSON, nullable=True)
    referenced_item_id: Mapped[str | None] = mapped_column(
        ForeignKey("public_benchmark_records.item_id", ondelete="RESTRICT")
    )

    __table_args__ = (
        CheckConstraint("origin_id = 'prism:public-snapshot'", name="ck_public_benchmark_prism_origin"),
        UniqueConstraint("origin_id", "logical_key", "source_revision", name="uq_rag_item_version"),
        CheckConstraint(
            "kind IN ('source','experiment','measurement','passage','figure','figure_link')",
            name="ck_rag_kind",
        ),
        CheckConstraint("access_classification = 'verified_public'", name="ck_rag_public_only"),
        CheckConstraint("publication_state IN ('staged','published','retired','tombstone')", name="ck_rag_state"),
        CheckConstraint(
            "source_role IS NULL OR source_role IN ('registry','manifest','file')",
            name="ck_rag_source_role",
        ),
        CheckConstraint(
            "(source_role = 'manifest' AND manifest_item_ids IS NOT NULL) OR "
            "((source_role IS NULL OR source_role <> 'manifest') AND manifest_item_ids IS NULL)",
            name="ck_rag_manifest_membership",
        ),
        CheckConstraint(
            "published_manifest_id IS NULL OR (source_role IS NOT NULL AND source_role = 'registry')",
            name="ck_rag_registry_pointer",
        ),
        CheckConstraint(
            "(timestamp IS NULL AND timestamp_kind IS NULL) OR "
            "(timestamp IS NOT NULL AND timestamp_kind IS NOT NULL "
            "AND timestamp_kind IN ('experiment','submission','publication'))",
            name="ck_rag_timestamp",
        ),
        CheckConstraint(
            "evidence_type IS NULL OR evidence_type IN ('raw_json','published_table')",
            name="ck_rag_evidence_type",
        ),
        CheckConstraint(
            "coverage IS NULL OR coverage IN ('complete','partial','summary_only','unknown')",
            name="ck_rag_coverage",
        ),
        CheckConstraint(
            "kind <> 'measurement' OR (value = original_value AND "
            "((unit IS NULL AND original_unit IS NULL) OR "
            "(unit IS NOT NULL AND original_unit IS NOT NULL AND unit = original_unit)))",
            name="ck_rag_original_measurement",
        ),
        CheckConstraint(
            "(normalized_value IS NULL AND normalized_unit IS NULL AND normalization_rule IS NULL) OR "
            "(normalized_value IS NOT NULL AND normalized_unit IS NOT NULL AND normalization_rule IS NOT NULL)",
            name="ck_rag_normalization",
        ),
        CheckConstraint(
            "(value IS NULL OR value NOT IN ('NaN','Infinity','-Infinity')) AND "
            "(original_value IS NULL OR original_value NOT IN ('NaN','Infinity','-Infinity')) AND "
            "(normalized_value IS NULL OR normalized_value NOT IN ('NaN','Infinity','-Infinity'))",
            name="ck_rag_finite_numbers",
        ),
        CheckConstraint(
            "(parent_id IS NULL OR parent_id <> item_id) AND "
            "(experiment_id IS NULL OR experiment_id <> item_id) AND "
            "(referenced_item_id IS NULL OR referenced_item_id <> item_id) AND "
            "(published_manifest_id IS NULL OR published_manifest_id <> item_id)",
            name="ck_rag_no_self_link",
        ),
        CheckConstraint(
            "length(content_sha256) = 64 AND length(origin_id) > 0 AND length(logical_key) > 0 AND "
            "length(source_revision) > 0 AND length(source_path) > 0 AND length(format) > 0",
            name="ck_rag_identity",
        ),
        *_shape_checks(),
        Index("ix_rag_origin_revision_kind", "origin_id", "source_revision", "kind"),
        Index("ix_rag_parent", "parent_id"),
        Index("ix_rag_experiment", "experiment_id"),
        Index("ix_rag_model_hardware", "model", "hardware"),
        Index("ix_rag_metric_stat_unit_value", "metric", "statistic", "unit", "value"),
        Index("ix_rag_published_manifest", "published_manifest_id"),
        Index("ix_rag_referenced_item", "referenced_item_id"),
    )
