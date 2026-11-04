"""Versioned portable public benchmark backups; never an alternate application data store."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from llm_d_lens.agentic.public_benchmark_record.contracts import ItemKind
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationModel, content_digest

PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION = "8bf2a6c1d903"


class BackupManifest(PublicationModel):
    contract: Literal["benchmark-rag.backup.v1"] = "benchmark-rag.backup.v1"
    schema_revision: Literal["8bf2a6c1d903"] = PUBLIC_BENCHMARK_RECORD_SCHEMA_REVISION
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    counts: dict[ItemKind, int]
    blobs: dict[str, int]

    @property
    def backup_id(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class RestoreResult(PublicationModel):
    backup_id: str
    inserted: int
    updated_sources: int
    total_rows: int
