"""SQL publication retains approval, immutable provenance and revocation without document services."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InventoryFile,
    ParsedFile,
    PinnedInventory,
    PublicationConflictError,
    PublicationError,
    SourceApproval,
    item_id,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, SourceArchive, allowed_paths
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao

PATH = "archives/llm-d-benchmarks.json"


@pytest.fixture
def service(postgres_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "evidence"))
    service = PublicBenchmarkRecordPublicationService(
        PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    )
    service.register_source(
        PRISM_ORIGIN,
        SourceApproval(
            verified_public=True,
            approved_by="test reviewer",
            approval_reference="fixture approval",
            license_reference="fixture license",
            parser_version="test-v1",
            allowed_paths=allowed_paths(PRISM_ORIGIN),
        ),
    )
    return service


def _stage(service, revision, value=42):
    digest = service.archive.put(f'{{"throughput":{value}}}'.encode())
    file = InventoryFile(path=PATH, sha256=digest, format="json")
    experiment_key = f"{PATH}#experiment"
    experiment = PublicBenchmarkRecord(
        item_id=item_id(PRISM_ORIGIN, experiment_key, revision),
        kind="experiment",
        origin_id=PRISM_ORIGIN,
        logical_key=experiment_key,
        source_revision=revision,
        content_sha256=digest,
        access_classification="verified_public",
        parent_id=item_id(PRISM_ORIGIN, f"file:{PATH}", revision),
        source_path=PATH,
        format="json",
        model="fixture-model",
        hardware="fixture-hardware",
        source_pointer={"json_pointer": "/"},
    )
    key = f"{PATH}#throughput"
    measurement = PublicBenchmarkRecord(
        item_id=item_id(PRISM_ORIGIN, key, revision),
        kind="measurement",
        origin_id=PRISM_ORIGIN,
        logical_key=key,
        source_revision=revision,
        content_sha256=digest,
        access_classification="verified_public",
        parent_id=experiment.item_id,
        experiment_id=experiment.item_id,
        source_path=PATH,
        format="json",
        metric="throughput",
        value=Decimal(value),
        original_value=Decimal(value),
        unit="tokens/s",
        original_unit="tokens/s",
        evidence_type="raw_json",
        source_pointer={"json_pointer": "/throughput"},
    )
    return service.stage(
        PinnedInventory(origin_id=PRISM_ORIGIN, revision=revision, complete=True, files=(file,)),
        (ParsedFile(path=PATH, items=(experiment, measurement)),),
        expected_counts={"source": 1, "experiment": 1, "measurement": 1},
    )


def test_default_publication_uses_sql_without_document_configuration():
    assert isinstance(PublicBenchmarkRecordPublicationService().dao, PublicBenchmarkRecordDao)


def test_stage_activate_replay_and_rollback_preserve_sql_history(service):
    first = _stage(service, "2026-10-13")
    assert service.dao.count_published(PRISM_ORIGIN) == 0
    result = service.activate(PRISM_ORIGIN, first.item_id)
    assert result.changed and result.generation == 1
    assert service.dao.count_published(PRISM_ORIGIN) == 3
    assert not service.activate(PRISM_ORIGIN, first.item_id).changed
    second = _stage(service, "2026-10-14", value=84)
    assert service.activate(PRISM_ORIGIN, second.item_id).generation == 2
    result = service.rollback(PRISM_ORIGIN, first.item_id, expected_generation=2)
    assert result.generation == 3
    assert service.dao.head(PRISM_ORIGIN).manifest.item_id == first.item_id
    assert service.dao.get(second.item_id) is not None
    with pytest.raises(PublicationConflictError):
        service.rollback(PRISM_ORIGIN, second.item_id, expected_generation=2)


def test_disabled_source_cannot_publish_or_read(service):
    manifest = _stage(service, "2026-10-13")
    service.activate(PRISM_ORIGIN, manifest.item_id)
    service.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=1)
    assert service.dao.count_published(PRISM_ORIGIN) == 0
    with pytest.raises(PublicationError, match="disabled"):
        _stage(service, "2026-10-14")


def test_missing_pinned_bytes_cannot_publish(service, tmp_path, monkeypatch):
    manifest = _stage(service, "2026-10-13")
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises((PublicationError, OSError)):
        service.activate(PRISM_ORIGIN, manifest.item_id)
    assert service.dao.head(PRISM_ORIGIN).manifest is None


def test_archive_concurrent_same_bytes_are_verified(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "evidence"))
    archive = SourceArchive()
    with ThreadPoolExecutor(max_workers=2) as executor:
        digests = list(executor.map(archive.put, [b"same pinned bytes"] * 2))
    assert digests[0] == digests[1]
    assert archive.read(digests[0]) == b"same pinned bytes"
