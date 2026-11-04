"""Internal source approval, inventory, and publication records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from llm_d_lens.agentic.public_benchmark_record.contracts import ItemKind, PublicBenchmarkRecord


class PublicationError(ValueError):
    """A source or release failed validation; the published pointer is unchanged."""


class PublicationConflictError(PublicationError):
    """The source advanced since staging; prepare against the current release."""


class PublicationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def content_digest(value: object) -> str:
    """Hash the full public payload, without deployment hashing's redaction."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def item_id(origin_id: str, logical_key: str, revision: str) -> str:
    return content_digest([origin_id, logical_key, revision])


def registry_id(origin_id: str) -> str:
    return item_id(origin_id, "@registry", "registry-v1")


class SourceApproval(PublicationModel):
    verified_public: Literal[True]
    approved_by: str = Field(min_length=1)
    approval_reference: str = Field(min_length=1)
    license_reference: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)


InputFormat = Literal["markdown", "json", "png", "zip", "jq"]


class InventoryFile(PublicationModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: InputFormat


class PinnedInventory(PublicationModel):
    origin_id: str
    revision: str = Field(min_length=1)
    complete: Literal[True]
    files: tuple[InventoryFile, ...]

    @model_validator(mode="after")
    def unique_paths(self) -> Self:
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("inventory paths must be unique")
        if self.files != tuple(sorted(self.files, key=lambda file: file.path)):
            raise ValueError("inventory paths must be in canonical order")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class SnapshotManifest(PublicationModel):
    contract: Literal["benchmark-rag.snapshot.v1"] = "benchmark-rag.snapshot.v1"
    inventory: PinnedInventory
    expected_experiments: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def require_files(self) -> Self:
        if not self.inventory.files:
            raise ValueError("snapshot manifest must pin at least one file")
        return self


class ParsedFile(PublicationModel):
    path: str
    items: tuple[PublicBenchmarkRecord, ...]
    errors: tuple[str, ...] = ()


class ManifestFile(PublicationModel):
    file: InventoryFile
    file_item_id: str
    member_ids: tuple[str, ...] = Field(min_length=1)


class ReleaseManifest(PublicationModel):
    contract: Literal["benchmark-rag.manifest.v1"] = "benchmark-rag.manifest.v1"
    inventory: PinnedInventory
    parser_version: str
    base_manifest_id: str | None
    base_generation: int = Field(ge=0)
    files: tuple[ManifestFile, ...]
    counts: dict[ItemKind, int]
    removed_item_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        if tuple(file.file for file in self.files) != self.inventory.files:
            raise ValueError("manifest must cover the complete inventory in canonical order")
        members = [member for file in self.files for member in file.member_ids]
        if len(set(members)) != len(members):
            raise ValueError("manifest members must be unique")
        if any(file.file_item_id not in file.member_ids for file in self.files):
            raise ValueError("each manifest file must include its source item")
        if set(self.removed_item_ids).intersection(members):
            raise ValueError("removed items cannot remain members")
        if any(count < 0 for count in self.counts.values()) or sum(self.counts.values()) != len(members):
            raise ValueError("manifest counts must describe exact membership")
        return self

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member for file in self.files for member in file.member_ids)


class PublicationEvent(PublicationModel):
    operation: Literal["activate", "rollback", "enable", "disable"]
    manifest_id: str | None
    previous_manifest_id: str | None
    generation: int
    at: datetime
    counts: dict[ItemKind, int]


class RestoreReceipt(PublicationModel):
    backup_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0)
    published_manifest_id: str | None
    enabled: bool
    history: tuple[PublicationEvent, ...]


class RegistryState(PublicationModel):
    contract: Literal["benchmark-rag.registry.v1"] = "benchmark-rag.registry.v1"
    approval: SourceApproval
    enabled: bool = True
    generation: int = Field(default=0, ge=0)
    history: tuple[PublicationEvent, ...] = ()
    restores: tuple[RestoreReceipt, ...] = ()

    def was_published(self, manifest_id: str) -> bool:
        return any(event.manifest_id == manifest_id for event in self.history) or any(
            event.manifest_id == manifest_id for receipt in self.restores for event in receipt.history
        )


class SourceHead(PublicationModel):
    registry: PublicBenchmarkRecord
    state: RegistryState
    manifest: PublicBenchmarkRecord | None


class PublicationResult(PublicationModel):
    origin_id: str
    manifest_id: str
    previous_manifest_id: str | None
    generation: int
    counts: dict[ItemKind, int]
    changed: bool
