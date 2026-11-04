"""Frozen public benchmark DDL matches metadata and supports a reversible upgrade."""

import importlib

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.exc import IntegrityError

from llm_d_lens.agentic.public_benchmark_record.transfer_contracts import (
    PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION,
    BackupManifest,
)
from llm_d_lens.db.base import Base

migration = importlib.import_module(
    "llm_d_lens.db.migrations.versions.8bf2a6c1d903_structured_public_benchmark_records"
)


def assert_empty_benchmark_storage(engine):
    with engine.connect() as connection:
        assert connection.scalar(sa.text("SELECT count(*) FROM public_benchmark_records")) == 0
        assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == migration.revision
        assert set(sa.inspect(connection).get_table_names()) == set(Base.metadata.tables) | {"alembic_version"}
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
        # Alembic autogenerate does not compare CHECK expressions.
        model = Base.metadata.tables["public_benchmark_records"]
        expected = sa.Table(
            "expected_benchmark_checks",
            sa.MetaData(),
            *(sa.Column(column.name, column.type) for column in model.columns),
            *(
                sa.CheckConstraint(str(constraint.sqltext), name=constraint.name)
                for constraint in model.constraints
                if isinstance(constraint, sa.CheckConstraint)
            ),
            prefixes=["TEMPORARY"],
        )
        expected.create(connection)
        inspector = sa.inspect(connection)
        assert {
            check["name"]: check["sqltext"] for check in inspector.get_check_constraints("public_benchmark_records")
        } == {check["name"]: check["sqltext"] for check in inspector.get_check_constraints(expected.name)}


def test_revision_chain_and_backup_stamp(postgres_config):
    script = ScriptDirectory.from_config(postgres_config)
    assert script.get_revision(migration.revision).down_revision == "42a3e9b47726"
    assert migration.revision == PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION
    assert BackupManifest.model_fields["schema_revision"].default == migration.revision


def test_upgrade_matches_metadata(postgres_engine):
    assert_empty_benchmark_storage(postgres_engine)


def test_downgrade_and_reupgrade(postgres_engine, postgres_config):
    with postgres_engine.begin() as connection:
        connection.execute(
            sa.insert(Base.metadata.tables["public_benchmark_records"]).values(
                item_id="source-before-downgrade",
                kind="source",
                origin_id="prism:public-snapshot",
                logical_key="file",
                source_revision="revision",
                content_sha256="b" * 64,
                access_classification="verified_public",
                publication_state="staged",
                source_path="source.json",
                format="json",
                source_role="file",
                source_details={"parser_version": "test"},
            )
        )
        assert connection.scalar(sa.text("SELECT count(*) FROM public_benchmark_records")) == 1
    command.downgrade(postgres_config, migration.down_revision)
    with postgres_engine.connect() as connection:
        assert set(sa.inspect(connection).get_table_names()) == (
            set(Base.metadata.tables) - {"public_benchmark_records"}
        ) | {"alembic_version"}
        assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == migration.down_revision
    command.upgrade(postgres_config, migration.revision)
    assert_empty_benchmark_storage(postgres_engine)


def test_new_table_rejects_non_prism_origins(postgres_engine):
    with (
        pytest.raises(IntegrityError, match="ck_public_benchmark_prism_origin"),
        postgres_engine.begin() as connection,
    ):
        connection.execute(
            sa.text(
                "INSERT INTO public_benchmark_records (item_id, kind, origin_id, logical_key, "
                "source_revision, content_sha256, access_classification, publication_state, "
                "source_path, format, source_role, source_details) VALUES "
                "('not-prism', 'source', 'other-origin', 'file', 'revision', :hash, "
                "'verified_public', 'staged', 'source.json', 'json', 'file', "
                '\'{"parser_version":"test"}\'::jsonb)'
            ),
            {"hash": "b" * 64},
        )
