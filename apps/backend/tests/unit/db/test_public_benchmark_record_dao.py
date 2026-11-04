from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from threading import Barrier

import pytest
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import CheckConstraint, Engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.public_benchmark_record.contracts import (
    ItemKind,
    PublicBenchmarkRecord,
    normalize_statistic,
    normalize_unit,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN
from llm_d_lens.db.dao.base import BaseDao
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao, PublicBenchmarkRecordDaoError
from llm_d_lens.db.engine import get_session_factory
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow
from llm_d_lens.db.models.public_benchmark_record import PublicBenchmarkRecordRow


def test_record_dao_directly_reuses_base_transaction_layer():
    assert PublicBenchmarkRecordDao.__bases__ == (BaseDao,)


@pytest.fixture(params=["sqlite", "postgresql"])
def dao(request: pytest.FixtureRequest) -> PublicBenchmarkRecordDao:
    if request.param == "postgresql":
        return PublicBenchmarkRecordDao(
            sessionmaker(bind=request.getfixturevalue("postgres_engine"), expire_on_commit=False)
        )
    return PublicBenchmarkRecordDao()


def _item(kind: ItemKind, item_id: str, **changes: object) -> PublicBenchmarkRecord:
    values: dict[str, object] = {
        "item_id": item_id,
        "kind": kind,
        "origin_id": PRISM_ORIGIN,
        "logical_key": item_id,
        "source_revision": "commit-a",
        "content_sha256": "a" * 64,
        "access_classification": "verified_public",
        "source_path": "guides/report.md",
        "format": "markdown",
    }
    specific: dict[str, dict[str, object]] = {
        "source": {"source_role": "file", "source_details": {"parser": "test-v1"}},
        "experiment": {"source_pointer": {"json_path": "$.records[0]"}},
        "measurement": {
            "experiment_id": "experiment",
            "metric": "throughput",
            "statistic": None,
            "value": Decimal("6131.06177725188"),
            "original_value": Decimal("6131.06177725188"),
            "unit": "output tokens/s",
            "original_unit": "output tokens/s",
            "source_pointer": {"json_path": "$.metrics.throughput"},
            "evidence_type": "raw_json",
        },
        "passage": {
            "passage_text": "A benchmark report captures aggregate workload statistics.",
            "heading_path": ["Benchmark report"],
            "source_pointer": {"heading": "Benchmark report", "start": 0},
        },
        "figure": {"coverage": "unknown", "source_pointer": {"path": "benchmark-results/plot.png"}},
        "figure_link": {"parent_id": "figure", "referenced_item_id": "experiment", "source_pointer": {"link": 0}},
    }
    values.update(specific[kind])
    values.update(changes)
    return PublicBenchmarkRecord.model_validate(values)


def _evidence(dao: PublicBenchmarkRecordDao) -> list[PublicBenchmarkRecord]:
    items = [
        _item("source", "file"),
        _item("experiment", "experiment", parent_id="file", model="Qwen3-0.6B", workload={"concurrency": None}),
        _item("measurement", "measurement", parent_id="experiment"),
        _item("passage", "passage", parent_id="file"),
        _item("figure", "figure", parent_id="file"),
        _item("figure_link", "figure-link"),
    ]
    for item in items:
        assert dao.save(item) == item
    return items


def test_all_kinds_round_trip_and_same_version_is_idempotent(dao: PublicBenchmarkRecordDao):
    items = _evidence(dao)
    for item in items:
        assert dao.get(item.item_id) == item
        assert dao.save(item) == item
    assert len(dao.list_version(PRISM_ORIGIN, "commit-a")) == 6
    assert dao.get("missing") is None
    experiment = dao.get("experiment")
    assert experiment is not None
    assert experiment.hardware is experiment.runtime is experiment.timestamp is None
    assert experiment.workload == {"concurrency": None}


def test_conflicts_cannot_overwrite_historical_versions(dao: PublicBenchmarkRecordDao):
    original = _item("passage", "passage")
    dao.save(original)
    for change in (
        {"content_sha256": "b" * 64},
        {"passage_text": "Changed text, even with the same declared hash"},
        {"item_id": "another-id"},
        {"logical_key": "another-key"},
    ):
        with pytest.raises(PublicBenchmarkRecordDaoError, match="immutable"):
            dao.save(PublicBenchmarkRecord.model_validate(original.model_dump() | change))
    newer = PublicBenchmarkRecord.model_validate(
        original.model_dump()
        | {
            "item_id": "passage-v2",
            "source_revision": "commit-b",
            "passage_text": "New pinned content.",
            "content_sha256": "b" * 64,
        }
    )
    dao.save(newer)
    assert dao.get("passage") == original
    assert dao.get("passage-v2") == newer


@pytest.mark.parametrize(
    "change",
    [
        {"experiment_id": "missing"},
        {"experiment_id": "figure"},
        {"experiment_id": "other-origin"},
        {"experiment_id": "other-revision"},
        {"parent_id": "figure"},
        {"parent_id": "another-experiment"},
    ],
)
def test_measurement_links_fail_closed(dao: PublicBenchmarkRecordDao, change: dict):
    dao.save(_item("experiment", "experiment"))
    dao.save(_item("experiment", "another-experiment"))
    dao.save(_item("figure", "figure"))
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("experiment", "other-origin", origin_id="public:other"))
    dao.save(_item("experiment", "other-revision", source_revision="commit-b"))
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("measurement", "invalid", **change))
    assert dao.get("invalid") is None


def test_figure_link_requires_figure_and_matching_experiment(dao: PublicBenchmarkRecordDao):
    _evidence(dao)
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("figure_link", "bad-parent", parent_id="passage"))
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("figure_link", "bad-target", referenced_item_id="file"))


def test_manifest_can_retain_old_pinned_items_without_publishing_them(dao: PublicBenchmarkRecordDao):
    items = _evidence(dao)
    manifest = _item(
        "source",
        "manifest",
        source_revision="commit-b",
        source_role="manifest",
        manifest_item_ids=[item.item_id for item in items],
    )
    assert dao.save(manifest) == manifest
    assert all(item.publication_state == "staged" for item in dao.list_version(PRISM_ORIGIN, "commit-a"))
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("passage", "foreign", origin_id="public:other"))
    for members in (["missing"], ["manifest"], ["foreign"]):
        with pytest.raises(PublicBenchmarkRecordDaoError):
            dao.save(_item("source", "bad-manifest", source_role="manifest", manifest_item_ids=members))
    registry = _item("source", "registry", source_role="registry", published_manifest_id="manifest")
    assert dao.save(registry) == registry
    with pytest.raises(PublicBenchmarkRecordDaoError):
        dao.save(_item("source", "bad-registry", source_role="registry", published_manifest_id="file"))


def test_original_units_and_submission_time_are_preserved(dao: PublicBenchmarkRecordDao):
    experiment = _item(
        "experiment",
        "experiment",
        timestamp=datetime(2026, 10, 13, tzinfo=UTC),
        timestamp_kind="submission",
    )
    assert dao.save(experiment) == experiment
    item = _item(
        "measurement",
        "weight-load",
        metric="weight_load_time",
        value=Decimal("0.93"),
        original_value=Decimal("0.93"),
        unit="s",
        original_unit="s",
        normalized_value=Decimal("930"),
        normalized_unit="ms",
        normalization_rule="unit-v1:s->ms",
        statistic=None,
        evidence_type="published_table",
    )
    assert dao.save(item) == item
    unknown = _item("measurement", "unknown", unit="undocumented-unit", original_unit="undocumented-unit")
    assert dao.save(unknown) == unknown
    assert unknown.normalized_unit is None


@pytest.mark.parametrize(
    ("kind", "changes"),
    [
        ("source", {"source_role": "manifest"}),
        ("source", {"source_role": "manifest", "manifest_item_ids": ["x", "x"]}),
        ("source", {"manifest_item_ids": []}),
        ("source", {"published_manifest_id": "x"}),
        ("experiment", {"value": "1"}),
        ("experiment", {"timestamp": "2026-10-13T00:00:00Z"}),
        ("experiment", {"timestamp": "2026-10-13T00:00:00", "timestamp_kind": "submission"}),
        ("measurement", {"evidence_type": "image_trend"}),
        ("measurement", {"source_pointer": None}),
        ("measurement", {"original_value": "1"}),
        ("measurement", {"value": "NaN", "original_value": "NaN"}),
        ("measurement", {"value": "Infinity", "original_value": "Infinity"}),
        ("measurement", {"normalized_value": "1"}),
        ("measurement", {"normalized_value": "1", "normalized_unit": "ms", "normalization_rule": "guessed"}),
        ("passage", {"passage_text": ""}),
        ("passage", {"surprise_field": True}),
        ("figure", {"coverage": "guessed"}),
        ("figure_link", {"referenced_item_id": None}),
        ("figure_link", {"parent_id": "item"}),
        ("passage", {"access_classification": "private"}),
        ("passage", {"content_sha256": "z" * 64}),
    ],
)
def test_invalid_contracts_are_rejected(kind: ItemKind, changes: dict):
    with pytest.raises(ValidationError):
        _item(kind, "item", **changes)


def test_unit_and_statistic_mappings_do_not_invent_information():
    assert normalize_unit(Decimal("0.93"), "s") == (Decimal("930"), "ms", "unit-v1:s->ms")
    assert normalize_unit(Decimal("1"), "us") == (Decimal("0.001"), "ms", "unit-v1:us->ms")
    assert normalize_unit(Decimal("1"), "unknown") is None
    assert normalize_statistic("avg") == "mean"
    assert normalize_statistic("median") == "p50"
    assert normalize_statistic(None) is None
    assert normalize_statistic("unknown") == "unknown"
    assert normalize_unit(Decimal("1"), "tokens/s") != normalize_unit(Decimal("1"), "output tokens/s")
    assert normalize_unit(Decimal("12345678901234567890123456789.12345"), "s") == (
        Decimal("12345678901234567890123456789123.45"),
        "ms",
        "unit-v1:s->ms",
    )
    with pytest.raises(ValueError, match="finite"):
        normalize_unit(Decimal("NaN"), "s")


def test_postgres_concurrent_same_version_import_is_idempotent(postgres_engine: Engine):
    dao = PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    barrier = Barrier(2)
    item = _item("passage", "concurrent")

    def save(_: int) -> PublicBenchmarkRecord:
        barrier.wait(timeout=10)
        return dao.save(item)

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(save, range(2))) == [item, item]
    assert dao.list_version(item.origin_id, item.source_revision) == [item]


def test_postgres_schema_numeric_filters_without_document_search(postgres_engine: Engine, postgres_config: Config):
    inspector = inspect(postgres_engine)
    assert {constraint["name"] for constraint in inspector.get_check_constraints("public_benchmark_records")} == {
        constraint.name
        for constraint in PublicBenchmarkRecordRow.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    columns = {column["name"]: str(column["type"]) for column in inspector.get_columns("public_benchmark_records")}
    assert columns["value"] == "NUMERIC"
    assert "search_vector" not in columns
    assert columns["workload"] == "JSONB"
    assert set(columns) == set(PublicBenchmarkRecordRow.__table__.columns.keys())
    dao = PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    _evidence(dao)
    with postgres_engine.connect() as connection:
        assert connection.scalar(
            select(PublicBenchmarkRecordRow.value).where(
                PublicBenchmarkRecordRow.metric == "throughput",
                PublicBenchmarkRecordRow.unit == "output tokens/s",
                PublicBenchmarkRecordRow.value > Decimal("6131.061777251879"),
                PublicBenchmarkRecordRow.value < Decimal("6131.061777251881"),
            )
        ) == Decimal("6131.06177725188")
        indexes = connection.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = current_schema() "
                "AND tablename = 'public_benchmark_records'"
            )
        ).all()
        assert not any("USING gin" in definition for name, definition in indexes)
        assert connection.scalar(select(func.count()).select_from(BenchmarkRecordRow)) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "unsupported"},
        {"access_classification": "private"},
        {"publication_state": "invalid"},
        {"value": Decimal("1")},
        {"timestamp": datetime(2026, 10, 13, tzinfo=UTC), "timestamp_kind": None},
        {"parent_id": "absent"},
        {"coverage": "partial"},
        {"source_role": "file"},
    ],
)
def test_postgres_constraints_reject_invalid_rows(postgres_engine: Engine, changes: dict):
    item = _item("experiment", "invalid")
    values = item.model_dump() | changes
    with pytest.raises(IntegrityError), postgres_engine.begin() as connection:
        connection.execute(PublicBenchmarkRecordRow.__table__.insert().values(**values))


@pytest.mark.parametrize(
    "changes",
    [
        {"value": Decimal("NaN"), "original_value": Decimal("NaN")},
        {"value": Decimal("Infinity"), "original_value": Decimal("Infinity")},
        {"value": Decimal("-Infinity"), "original_value": Decimal("-Infinity")},
        {"metric": None},
        {"original_unit": "requests/s"},
        {"normalized_value": Decimal("1")},
        {"evidence_type": "image_trend"},
    ],
)
def test_postgres_constraints_reject_invalid_measurements(postgres_engine: Engine, changes: dict):
    dao = PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    dao.save(_item("experiment", "experiment"))
    values = _item("measurement", "invalid").model_dump() | changes
    with pytest.raises(IntegrityError), postgres_engine.begin() as connection:
        connection.execute(PublicBenchmarkRecordRow.__table__.insert().values(**values))


def test_postgres_foreign_keys_restrict_deleting_cited_experiments(postgres_engine: Engine):
    dao = PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    _evidence(dao)
    with pytest.raises(IntegrityError), postgres_engine.begin() as connection:
        connection.execute(
            PublicBenchmarkRecordRow.__table__.delete().where(PublicBenchmarkRecordRow.item_id == "experiment")
        )
    assert dao.get("experiment") is not None
    assert dao.get("measurement") is not None


def test_sqlite_has_no_full_text_index():
    engine = get_session_factory()().get_bind()
    assert "search_vector" not in {column["name"] for column in inspect(engine).get_columns("public_benchmark_records")}
