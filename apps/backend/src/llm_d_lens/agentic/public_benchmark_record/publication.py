"""Approved Prism SQL publication with retained manifests and audit history."""

from __future__ import annotations

import io
import json
import logging
import zipfile
from collections import Counter

from pydantic import ValidationError

from llm_d_lens.agentic.public_benchmark_record.contracts import KIND_FIELDS, ItemKind, PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InventoryFile,
    ManifestFile,
    ParsedFile,
    PinnedInventory,
    PublicationError,
    PublicationResult,
    ReleaseManifest,
    SourceApproval,
    content_digest,
    item_id,
    registry_id,
)
from llm_d_lens.agentic.public_benchmark_record.sources import (
    SourceArchive,
    input_format,
    validate_revision,
)
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao, PublicBenchmarkRecordDaoError

logger = logging.getLogger(__name__)


class PublicBenchmarkRecordPublicationService:
    def __init__(
        self,
        dao: PublicBenchmarkRecordDao | None = None,
        archive: SourceArchive | None = None,
    ) -> None:
        self.dao = dao if dao is not None else PublicBenchmarkRecordDao()
        self.archive = archive if archive is not None else SourceArchive()

    def register_source(self, origin_id: str, approval: SourceApproval) -> PublicBenchmarkRecord:
        """Administrative approval only; never accept this identity from an LLM."""
        return self.dao.register(origin_id, SourceApproval.model_validate(approval.model_dump()))

    def _verify_file(self, file: InventoryFile) -> None:
        data = self.archive.read(file.sha256)
        try:
            if file.format == "json":
                json.loads(data)
            elif file.format == "jq":
                data.decode("utf-8")
            elif file.format == "zip" and not zipfile.is_zipfile(io.BytesIO(data)):
                raise PublicationError("source is not a ZIP")
            elif file.format not in ("json", "jq", "zip"):
                raise PublicationError("unsupported Prism source format")
        except (json.JSONDecodeError, UnicodeError) as error:
            raise PublicationError("source format validation failed") from error

    def _verify_manifest(self, manifest: ReleaseManifest) -> None:
        for file in manifest.inventory.files:
            self._verify_file(file)

    def stage(
        self,
        inventory: PinnedInventory,
        parsed_files: tuple[ParsedFile, ...],
        *,
        expected_counts: dict[ItemKind, int],
    ) -> PublicBenchmarkRecord:
        try:
            return self._stage(inventory, parsed_files, expected_counts)
        except (PublicationError, PublicBenchmarkRecordDaoError, ValidationError, OSError) as error:
            logger.warning("event=rag_stage_failed origin=%s reason=%s", inventory.origin_id, type(error).__name__)
            raise

    def _stage(
        self,
        inventory: PinnedInventory,
        parsed_files: tuple[ParsedFile, ...],
        expected_counts: dict[ItemKind, int],
    ) -> PublicBenchmarkRecord:
        inventory = PinnedInventory.model_validate(inventory.model_dump())
        validate_revision(inventory.origin_id, inventory.revision)
        if any(key not in KIND_FIELDS or type(value) is not int or value < 0 for key, value in expected_counts.items()):
            raise PublicationError("expected per-kind counts must be explicit nonnegative integers")
        head = self.dao.head(inventory.origin_id)
        if not head.state.enabled:
            raise PublicationError("source is disabled")
        previous = ReleaseManifest.model_validate(head.manifest.source_details) if head.manifest is not None else None
        previous_files = {file.file.path: file for file in previous.files} if previous is not None else {}
        supplied = {file.path: ParsedFile.model_validate(file.model_dump()) for file in parsed_files}
        if len(supplied) != len(parsed_files) or set(supplied) - {file.path for file in inventory.files}:
            raise PublicationError("parsed files must be unique inventory members")
        staged = []
        files = []
        for file in inventory.files:
            if input_format(inventory.origin_id, file.path, head.state.approval) != file.format:
                raise PublicationError("inventory includes a nonapproved path or format")
            self._verify_file(file)
            old = previous_files.get(file.path)
            parsed = supplied.get(file.path)
            if old is not None and old.file == file and parsed is None:
                files.append(old)
                continue
            if parsed is None or parsed.errors:
                raise PublicationError("every new or reparsed file needs a complete, error-free parser result")
            key = f"file:{file.path}"
            source = PublicBenchmarkRecord(
                item_id=item_id(inventory.origin_id, key, inventory.revision),
                logical_key=key,
                origin_id=inventory.origin_id,
                source_revision=inventory.revision,
                content_sha256=file.sha256,
                kind="source",
                source_role="file",
                parent_id=registry_id(inventory.origin_id),
                access_classification="verified_public",
                source_path=file.path,
                format=file.format,
                source_details={"parser_version": head.state.approval.parser_version},
            )
            staged.append(source)
            for item in parsed.items:
                if (
                    item.kind == "source"
                    or item.origin_id != inventory.origin_id
                    or item.source_revision != inventory.revision
                    or item.source_path != file.path
                    or item.content_sha256 != file.sha256
                    or item.format != file.format
                    or item.item_id != item_id(item.origin_id, item.logical_key, item.source_revision)
                    or not item.logical_key.startswith(f"{file.path}#")
                    or item.publication_state != "staged"
                ):
                    raise PublicationError("parser result does not match its exact pinned source identity")
                if item.kind == "measurement" and (
                    (item.evidence_type == "raw_json" and file.format != "json")
                    or (item.evidence_type == "published_table" and file.format != "markdown")
                ):
                    raise PublicationError("this source cannot support the claimed measurement evidence type")
                staged.append(item)
            files.append(
                ManifestFile(
                    file=file,
                    file_item_id=source.item_id,
                    member_ids=tuple(sorted((source.item_id, *(item.item_id for item in parsed.items)))),
                )
            )
        members = {item.item_id: item for item in staged}
        for file in files:
            for member_id in file.member_ids:
                if member_id not in members:
                    retained = self.dao.get(member_id)
                    if retained is None:
                        raise PublicationError("unchanged manifest member is missing")
                    members[member_id] = retained
        counts = dict(Counter(item.kind for item in members.values()))
        if counts != {key: value for key, value in expected_counts.items() if value}:
            raise PublicationError("parsed release does not match expected per-kind counts")
        if previous is not None and (
            previous.inventory == inventory and previous.files == tuple(files) and previous.counts == counts
        ):
            if head.manifest is None:
                raise PublicationError("source head lost its manifest")
            return head.manifest
        previous_members = set(head.manifest.manifest_item_ids or ()) if head.manifest is not None else set()
        metadata = ReleaseManifest(
            inventory=inventory,
            parser_version=head.state.approval.parser_version,
            base_manifest_id=head.registry.published_manifest_id,
            base_generation=head.state.generation,
            files=tuple(files),
            counts=counts,
            removed_item_ids=tuple(sorted(previous_members - members.keys())),
        )
        details = metadata.model_dump(mode="json")
        digest = content_digest(details)
        key = f"@manifest:{digest}"
        manifest = PublicBenchmarkRecord(
            item_id=item_id(inventory.origin_id, key, inventory.revision),
            kind="source",
            source_role="manifest",
            origin_id=inventory.origin_id,
            logical_key=key,
            source_revision=inventory.revision,
            content_sha256=digest,
            access_classification="verified_public",
            parent_id=registry_id(inventory.origin_id),
            source_path=f"@manifests/{digest}",
            format="json",
            source_details=details,
            manifest_item_ids=list(metadata.member_ids),
        )
        result = self.dao.stage(manifest, staged)
        logger.info(
            "event=rag_release_staged origin=%s revision=%s manifest=%s counts=%s",
            inventory.origin_id,
            inventory.revision,
            result.item_id,
            counts,
        )
        return result

    def activate(
        self,
        origin_id: str,
        manifest_id: str,
    ) -> PublicationResult:
        return self._activate(origin_id, manifest_id)

    def rollback(self, origin_id: str, manifest_id: str, *, expected_generation: int) -> PublicationResult:
        return self._activate(origin_id, manifest_id, rollback_generation=expected_generation)

    def _activate(
        self,
        origin_id: str,
        manifest_id: str,
        *,
        rollback_generation: int | None = None,
    ) -> PublicationResult:
        try:
            result = self.dao.activate(
                origin_id,
                manifest_id,
                self._verify_manifest,
                rollback_generation=rollback_generation,
            )
        except (PublicationError, ValidationError, OSError) as error:
            logger.warning("event=rag_activation_failed origin=%s reason=%s", origin_id, type(error).__name__)
            raise
        logger.info(
            "event=rag_release_activated origin=%s manifest=%s previous=%s generation=%s changed=%s counts=%s",
            origin_id,
            manifest_id,
            result.previous_manifest_id,
            result.generation,
            result.changed,
            result.counts,
        )
        return result
