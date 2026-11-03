"""Checksummed, SQL-free public benchmark logical backups with immutable evidence bytes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationError
from llm_d_lens.agentic.public_benchmark_record.sources import SourceArchive
from llm_d_lens.agentic.public_benchmark_record.transfer_contracts import BackupManifest, RestoreResult
from llm_d_lens.db.dao.public_benchmark_record_transfer import PublicBenchmarkRecordTransferDao

MAX_BACKUP_BYTES = 1024 * 1024 * 1024


def _blob_ids(records: tuple[PublicBenchmarkRecord, ...]) -> set[str]:
    return {item.content_sha256 for item in records if item.source_role == "file"}


class PublicBenchmarkRecordBackupService:
    def __init__(self, dao: PublicBenchmarkRecordTransferDao, archive: SourceArchive | None = None) -> None:
        self.dao = dao
        self.archive = archive if archive is not None else SourceArchive()

    def backup(self, output: Path, *, max_bytes: int = MAX_BACKUP_BYTES) -> BackupManifest:
        records = self.dao.snapshot()
        payload = b"".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for item in records
        )
        blobs = {digest: self.archive.read(digest) for digest in sorted(_blob_ids(records))}
        metadata = BackupManifest(
            records_sha256=hashlib.sha256(payload).hexdigest(),
            row_count=len(records),
            counts=dict(Counter(item.kind for item in records)),
            blobs={digest: len(data) for digest, data in blobs.items()},
        )
        encoded = metadata.model_dump_json().encode()
        if max_bytes <= 0 or len(payload) + len(encoded) + sum(map(len, blobs.values())) > max_bytes:
            raise PublicationError("public benchmark backup exceeds the uncompressed size limit")
        output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=output.parent, prefix=".rag-backup-", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("manifest.json", encoded)
                bundle.writestr("records.jsonl", payload)
                for digest, data in blobs.items():
                    bundle.writestr(f"sources/{digest}", data)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(path, output)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            path.unlink(missing_ok=True)
        return metadata

    @staticmethod
    def _read(
        path: Path,
        max_bytes: int,
    ) -> tuple[BackupManifest, tuple[PublicBenchmarkRecord, ...], dict[str, bytes]]:
        if max_bytes <= 0:
            raise PublicationError("backup size limit must be positive")
        try:
            with zipfile.ZipFile(path) as bundle:
                entries = bundle.infolist()
                names = [entry.filename for entry in entries]
                if len(names) != len(set(names)):
                    raise PublicationError("backup has duplicate member names")
                if sum(entry.file_size for entry in entries) > max_bytes:
                    raise PublicationError("backup exceeds the uncompressed size limit")
                if any(
                    entry.is_dir()
                    or stat.S_ISLNK(entry.external_attr >> 16)
                    or entry.flag_bits & 1
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or not (
                        entry.filename in ("manifest.json", "records.jsonl")
                        or re.fullmatch(r"sources/[0-9a-f]{64}", entry.filename)
                    )
                    for entry in entries
                ):
                    raise PublicationError("backup has an unsupported or unsafe member")
                if "manifest.json" not in names or "records.jsonl" not in names:
                    raise PublicationError("backup is missing its manifest or records")
                if bundle.getinfo("manifest.json").file_size > 4 * 1024 * 1024:
                    raise PublicationError("backup manifest is too large")
                metadata = BackupManifest.model_validate_json(bundle.read("manifest.json"))
                if any(
                    not re.fullmatch(r"[0-9a-f]{64}", digest) or size < 0 for digest, size in metadata.blobs.items()
                ):
                    raise PublicationError("backup blob inventory is invalid")
                if set(names) != {
                    "manifest.json",
                    "records.jsonl",
                    *(f"sources/{digest}" for digest in metadata.blobs),
                }:
                    raise PublicationError("backup member inventory differs from its manifest")
                payload = bundle.read("records.jsonl")
                if hashlib.sha256(payload).hexdigest() != metadata.records_sha256:
                    raise PublicationError("backup record checksum mismatch")
                records = tuple(PublicBenchmarkRecord.model_validate_json(line) for line in payload.splitlines())
                if (
                    len(records) != metadata.row_count
                    or dict(Counter(item.kind for item in records)) != metadata.counts
                ):
                    raise PublicationError("backup record counts are invalid")
                PublicBenchmarkRecordTransferDao.validate_snapshot(records)
                if _blob_ids(records) != metadata.blobs.keys():
                    raise PublicationError("backup evidence inventory is incomplete")
                blobs = {}
                for digest, size in metadata.blobs.items():
                    data = bundle.read(f"sources/{digest}")
                    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                        raise PublicationError("backup source checksum mismatch")
                    blobs[digest] = data
                return metadata, records, blobs
        except zipfile.BadZipFile as error:
            raise PublicationError("invalid or corrupt public benchmark backup ZIP") from error

    @classmethod
    def verify(cls, path: Path, *, max_bytes: int = MAX_BACKUP_BYTES) -> BackupManifest:
        return cls._read(path, max_bytes)[0]

    def restore(self, path: Path, *, max_bytes: int = MAX_BACKUP_BYTES) -> RestoreResult:
        metadata, records, blobs = self._read(path, max_bytes)
        for digest, data in blobs.items():
            if self.archive.put(data) != digest:
                raise PublicationError("restored source archive identity differs")
        return self.dao.restore(records, metadata.backup_id)
