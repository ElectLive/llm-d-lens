from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from llm_d_lens.agentic.public_benchmark_record.publication_contracts import SnapshotManifest
from llm_d_lens.agentic.public_benchmark_record.transfer_contracts import PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION
from llm_d_lens.db.base import Base

_BACKEND = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def prism_initial_manifest() -> SnapshotManifest:
    path = _BACKEND / "tests/fixtures/public_benchmark_record/prism_initial_manifest.json"
    return SnapshotManifest.model_validate_json(path.read_bytes())


def _config(engine: Engine) -> Config:
    config = Config(str(_BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND / "src/llm_d_lens/db/migrations"))
    config.set_main_option("sqlalchemy.url", engine.url.render_as_string(hide_password=False).replace("%", "%%"))
    return config


@pytest.fixture(scope="session")
def postgres_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    pgserver = pytest.importorskip("pgserver", reason="PostgreSQL acceptance needs the embedded-db extra")
    server = pgserver.get_server(tmp_path_factory.mktemp("rag-postgres") / "data", cleanup_mode="stop")
    try:
        yield server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    finally:
        server.cleanup()


@pytest.fixture
def postgres_engine(postgres_url: str) -> Iterator[Engine]:
    schema = f"rag_test_{uuid4().hex}"
    admin = create_engine(postgres_url)
    with admin.begin() as connection:
        connection.execute(CreateSchema(schema))
    engine = create_engine(admin.url.update_query_dict({"options": f"-csearch_path={schema}"}))
    try:
        Base.metadata.create_all(
            engine, tables=[table for table in Base.metadata.sorted_tables if table.name != "public_benchmark_records"]
        )
        command.stamp(_config(engine), "42a3e9b47726")
        command.upgrade(_config(engine), PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        admin.dispose()


@pytest.fixture
def postgres_config(postgres_engine: Engine) -> Config:
    return _config(postgres_engine)
