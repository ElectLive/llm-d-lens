"""Create structured Prism-only public benchmark records.

Revision ID: 8bf2a6c1d903
Revises: 42a3e9b47726
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "8bf2a6c1d903"
down_revision: str | Sequence[str] | None = "42a3e9b47726"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    nullable_json = sa.JSON(none_as_null=True).with_variant(postgresql.JSONB(none_as_null=True), "postgresql")
    op.create_table(
        "public_benchmark_records",
        sa.Column("item_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("origin_id", sa.Text(), nullable=False),
        sa.Column("logical_key", sa.Text(), nullable=False),
        sa.Column("source_revision", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("access_classification", sa.String(32), nullable=False),
        sa.Column("publication_state", sa.String(16), nullable=False),
        sa.Column("parent_id", sa.String(64), nullable=True),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("format", sa.Text(), nullable=False),
        sa.Column("source_role", sa.String(16), nullable=True),
        sa.Column("source_details", nullable_json, nullable=True),
        sa.Column("manifest_item_ids", nullable_json, nullable=True),
        sa.Column("published_manifest_id", sa.String(64), nullable=True),
        sa.Column("experiment_id", sa.String(64), nullable=True),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("hardware", sa.Text(), nullable=True),
        sa.Column("runtime", sa.Text(), nullable=True),
        sa.Column("workload", nullable_json, nullable=True),
        sa.Column("configuration", nullable_json, nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timestamp_kind", sa.String(16), nullable=True),
        sa.Column("source_pointer", nullable_json, nullable=True),
        sa.Column("metric", sa.Text(), nullable=True),
        sa.Column("statistic", sa.Text(), nullable=True),
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("unit", sa.Text(), nullable=True),
        sa.Column("original_value", sa.Numeric(), nullable=True),
        sa.Column("original_unit", sa.Text(), nullable=True),
        sa.Column("normalized_value", sa.Numeric(), nullable=True),
        sa.Column("normalized_unit", sa.Text(), nullable=True),
        sa.Column("normalization_rule", sa.Text(), nullable=True),
        sa.Column("series", nullable_json, nullable=True),
        sa.Column("evidence_type", sa.String(32), nullable=True),
        sa.Column("passage_text", sa.Text(), nullable=True),
        sa.Column("heading_path", nullable_json, nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("coverage", sa.String(16), nullable=True),
        sa.Column("missing_series", nullable_json, nullable=True),
        sa.Column("referenced_item_id", sa.String(64), nullable=True),
        sa.CheckConstraint("origin_id = 'prism:public-snapshot'", name="ck_public_benchmark_prism_origin"),
        sa.CheckConstraint(
            "kind IN ('source','experiment','measurement','passage','figure','figure_link')",
            name="ck_rag_kind",
        ),
        sa.CheckConstraint("access_classification = 'verified_public'", name="ck_rag_public_only"),
        sa.CheckConstraint("publication_state IN ('staged','published','retired','tombstone')", name="ck_rag_state"),
        sa.CheckConstraint(
            "source_role IS NULL OR source_role IN ('registry','manifest','file')", name="ck_rag_source_role"
        ),
        sa.CheckConstraint(
            "(source_role = 'manifest' AND manifest_item_ids IS NOT NULL) OR "
            "((source_role IS NULL OR source_role <> 'manifest') AND manifest_item_ids IS NULL)",
            name="ck_rag_manifest_membership",
        ),
        sa.CheckConstraint(
            "published_manifest_id IS NULL OR (source_role IS NOT NULL AND source_role = 'registry')",
            name="ck_rag_registry_pointer",
        ),
        sa.CheckConstraint(
            "(timestamp IS NULL AND timestamp_kind IS NULL) OR "
            "(timestamp IS NOT NULL AND timestamp_kind IS NOT NULL "
            "AND timestamp_kind IN ('experiment','submission','publication'))",
            name="ck_rag_timestamp",
        ),
        sa.CheckConstraint(
            "evidence_type IS NULL OR evidence_type IN ('raw_json','published_table')", name="ck_rag_evidence_type"
        ),
        sa.CheckConstraint(
            "coverage IS NULL OR coverage IN ('complete','partial','summary_only','unknown')", name="ck_rag_coverage"
        ),
        sa.CheckConstraint(
            "kind <> 'measurement' OR (value = original_value AND "
            "((unit IS NULL AND original_unit IS NULL) OR "
            "(unit IS NOT NULL AND original_unit IS NOT NULL AND unit = original_unit)))",
            name="ck_rag_original_measurement",
        ),
        sa.CheckConstraint(
            "(normalized_value IS NULL AND normalized_unit IS NULL AND normalization_rule IS NULL) OR "
            "(normalized_value IS NOT NULL AND normalized_unit IS NOT NULL AND normalization_rule IS NOT NULL)",
            name="ck_rag_normalization",
        ),
        sa.CheckConstraint(
            "(value IS NULL OR value NOT IN ('NaN','Infinity','-Infinity')) AND "
            "(original_value IS NULL OR original_value NOT IN ('NaN','Infinity','-Infinity')) AND "
            "(normalized_value IS NULL OR normalized_value NOT IN ('NaN','Infinity','-Infinity'))",
            name="ck_rag_finite_numbers",
        ),
        sa.CheckConstraint(
            "(parent_id IS NULL OR parent_id <> item_id) AND "
            "(experiment_id IS NULL OR experiment_id <> item_id) AND "
            "(referenced_item_id IS NULL OR referenced_item_id <> item_id) AND "
            "(published_manifest_id IS NULL OR published_manifest_id <> item_id)",
            name="ck_rag_no_self_link",
        ),
        sa.CheckConstraint(
            "length(content_sha256) = 64 AND length(origin_id) > 0 AND length(logical_key) > 0 AND "
            "length(source_revision) > 0 AND length(source_path) > 0 AND length(format) > 0",
            name="ck_rag_identity",
        ),
        sa.CheckConstraint(
            "kind <> 'source' OR (caption IS NULL AND configuration IS NULL AND "
            "coverage IS NULL AND evidence_type IS NULL AND "
            "experiment_id IS NULL AND hardware IS NULL AND heading_path IS NULL AND "
            "metric IS NULL AND missing_series IS NULL AND "
            "model IS NULL AND normalization_rule IS NULL AND normalized_unit IS NULL AND normalized_value IS NULL AND "
            "original_unit IS NULL AND original_value IS NULL AND passage_text IS NULL AND "
            "referenced_item_id IS NULL AND runtime IS NULL AND series IS NULL AND source_pointer IS NULL AND "
            "statistic IS NULL AND timestamp IS NULL AND timestamp_kind IS NULL AND unit IS NULL AND "
            "value IS NULL AND workload IS NULL AND source_details IS NOT NULL AND source_role IS NOT NULL)",
            name="ck_rag_source_shape",
        ),
        sa.CheckConstraint(
            "kind <> 'experiment' OR (caption IS NULL AND coverage IS NULL AND evidence_type IS NULL AND "
            "experiment_id IS NULL AND heading_path IS NULL AND manifest_item_ids IS NULL AND metric IS NULL AND "
            "missing_series IS NULL AND normalization_rule IS NULL AND normalized_unit IS NULL AND "
            "normalized_value IS NULL AND original_unit IS NULL AND original_value IS NULL AND "
            "passage_text IS NULL AND published_manifest_id IS NULL AND referenced_item_id IS NULL AND "
            "series IS NULL AND source_details IS NULL AND source_role IS NULL AND statistic IS NULL AND "
            "unit IS NULL AND value IS NULL AND source_pointer IS NOT NULL)",
            name="ck_rag_experiment_shape",
        ),
        sa.CheckConstraint(
            "kind <> 'measurement' OR (caption IS NULL AND configuration IS NULL AND "
            "coverage IS NULL AND hardware IS NULL AND "
            "heading_path IS NULL AND manifest_item_ids IS NULL AND missing_series IS NULL AND model IS NULL AND "
            "passage_text IS NULL AND published_manifest_id IS NULL AND "
            "referenced_item_id IS NULL AND runtime IS NULL AND "
            "source_details IS NULL AND source_role IS NULL AND timestamp IS NULL AND timestamp_kind IS NULL AND "
            "workload IS NULL AND evidence_type IS NOT NULL AND experiment_id IS NOT NULL AND metric IS NOT NULL AND "
            "original_value IS NOT NULL AND source_pointer IS NOT NULL AND value IS NOT NULL)",
            name="ck_rag_measurement_shape",
        ),
        sa.CheckConstraint(
            "kind <> 'passage' OR (caption IS NULL AND configuration IS NULL AND "
            "coverage IS NULL AND evidence_type IS NULL AND "
            "experiment_id IS NULL AND hardware IS NULL AND manifest_item_ids IS NULL AND metric IS NULL AND "
            "missing_series IS NULL AND model IS NULL AND normalization_rule IS NULL AND normalized_unit IS NULL AND "
            "normalized_value IS NULL AND original_unit IS NULL AND original_value IS NULL AND "
            "published_manifest_id IS NULL AND "
            "referenced_item_id IS NULL AND runtime IS NULL AND series IS NULL AND source_details IS NULL AND "
            "source_role IS NULL AND statistic IS NULL AND timestamp IS NULL AND "
            "timestamp_kind IS NULL AND unit IS NULL AND "
            "value IS NULL AND workload IS NULL AND heading_path IS NOT NULL AND passage_text IS NOT NULL AND "
            "source_pointer IS NOT NULL)",
            name="ck_rag_passage_shape",
        ),
        sa.CheckConstraint(
            "kind <> 'figure' OR (configuration IS NULL AND evidence_type IS NULL AND experiment_id IS NULL AND "
            "hardware IS NULL AND heading_path IS NULL AND manifest_item_ids IS NULL AND metric IS NULL AND "
            "model IS NULL AND normalization_rule IS NULL AND normalized_unit IS NULL AND "
            "normalized_value IS NULL AND original_unit IS NULL AND "
            "original_value IS NULL AND passage_text IS NULL AND published_manifest_id IS NULL AND "
            "referenced_item_id IS NULL AND runtime IS NULL AND series IS NULL AND source_details IS NULL AND "
            "source_role IS NULL AND statistic IS NULL AND timestamp IS NULL AND "
            "timestamp_kind IS NULL AND unit IS NULL AND "
            "value IS NULL AND workload IS NULL AND coverage IS NOT NULL AND source_pointer IS NOT NULL)",
            name="ck_rag_figure_shape",
        ),
        sa.CheckConstraint(
            "kind <> 'figure_link' OR (caption IS NULL AND configuration IS NULL AND coverage IS NULL AND "
            "evidence_type IS NULL AND experiment_id IS NULL AND hardware IS NULL AND heading_path IS NULL AND "
            "manifest_item_ids IS NULL AND metric IS NULL AND missing_series IS NULL AND model IS NULL AND "
            "normalization_rule IS NULL AND normalized_unit IS NULL AND "
            "normalized_value IS NULL AND original_unit IS NULL AND "
            "original_value IS NULL AND passage_text IS NULL AND published_manifest_id IS NULL AND runtime IS NULL AND "
            "series IS NULL AND source_details IS NULL AND source_role IS NULL AND "
            "statistic IS NULL AND timestamp IS NULL AND timestamp_kind IS NULL AND unit IS NULL AND "
            "value IS NULL AND workload IS NULL AND parent_id IS NOT NULL AND "
            "referenced_item_id IS NOT NULL AND source_pointer IS NOT NULL)",
            name="ck_rag_figure_link_shape",
        ),
        sa.ForeignKeyConstraint(["parent_id"], ["public_benchmark_records.item_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["experiment_id"], ["public_benchmark_records.item_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["published_manifest_id"], ["public_benchmark_records.item_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["referenced_item_id"], ["public_benchmark_records.item_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("item_id"),
        sa.UniqueConstraint("origin_id", "logical_key", "source_revision", name="uq_rag_item_version"),
    )
    op.create_index("ix_rag_origin_revision_kind", "public_benchmark_records", ["origin_id", "source_revision", "kind"])
    op.create_index("ix_rag_parent", "public_benchmark_records", ["parent_id"])
    op.create_index("ix_rag_experiment", "public_benchmark_records", ["experiment_id"])
    op.create_index("ix_rag_model_hardware", "public_benchmark_records", ["model", "hardware"])
    op.create_index(
        "ix_rag_metric_stat_unit_value", "public_benchmark_records", ["metric", "statistic", "unit", "value"]
    )
    op.create_index("ix_rag_published_manifest", "public_benchmark_records", ["published_manifest_id"])
    op.create_index("ix_rag_referenced_item", "public_benchmark_records", ["referenced_item_id"])


def downgrade() -> None:
    op.drop_table("public_benchmark_records")
