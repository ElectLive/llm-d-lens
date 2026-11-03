"""Immutable public benchmark records and transactional publication lifecycle."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from sqlalchemy import Select, func, select, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from llm_d_lens.agentic.public_benchmark_record.contracts import ItemKind, PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    PublicationConflictError,
    PublicationError,
    PublicationEvent,
    PublicationResult,
    RegistryState,
    ReleaseManifest,
    SourceApproval,
    SourceHead,
    content_digest,
    item_id,
    registry_id,
)
from llm_d_lens.agentic.public_benchmark_record.sources import (
    allowed_paths,
    input_format,
    validate_approval,
    validate_revision,
)
from llm_d_lens.db.dao.base import BaseDao
from llm_d_lens.db.models.public_benchmark_record import PublicBenchmarkRecordRow


class PublicBenchmarkRecordDaoError(ValueError):
    """Invalid links or a conflicting immutable public benchmark record version."""


class PublicBenchmarkRecordDao(BaseDao):
    def save(self, item: PublicBenchmarkRecord) -> PublicBenchmarkRecord:
        item = PublicBenchmarkRecord.model_validate(item.model_dump())
        with self._transaction() as session:
            return self._save_in_session(session, item)

    @classmethod
    def _save_in_session(cls, session: Session, item: PublicBenchmarkRecord) -> PublicBenchmarkRecord:
        cls._validate_links(session, item)
        try:
            with session.begin_nested():
                row = PublicBenchmarkRecordRow.from_dto(item)
                session.add(row)
                session.flush()
        except IntegrityError as exc:
            existing = session.scalar(
                select(PublicBenchmarkRecordRow).where(
                    PublicBenchmarkRecordRow.origin_id == item.origin_id,
                    PublicBenchmarkRecordRow.logical_key == item.logical_key,
                    PublicBenchmarkRecordRow.source_revision == item.source_revision,
                )
            )
            if existing is None or existing.to_dto() != item:
                raise PublicBenchmarkRecordDaoError(
                    "conflicting or invalid immutable public benchmark record version"
                ) from exc
            return existing.to_dto()
        return row.to_dto()

    def get(self, item_id: str) -> PublicBenchmarkRecord | None:
        """Import/audit lookup, not a permission-filtered retrieval method."""
        with self._read_only() as session:
            row = session.get(PublicBenchmarkRecordRow, item_id)
            return row.to_dto() if row is not None else None

    def list_version(self, origin_id: str, source_revision: str) -> list[PublicBenchmarkRecord]:
        """Inspect staged or historical imports; never use for end-user search."""
        with self._read_only() as session:
            statement = (
                select(PublicBenchmarkRecordRow)
                .where(
                    PublicBenchmarkRecordRow.origin_id == origin_id,
                    PublicBenchmarkRecordRow.source_revision == source_revision,
                )
                .order_by(PublicBenchmarkRecordRow.item_id)
            )
            return [row.to_dto() for row in session.scalars(statement)]

    @classmethod
    def _validate_links(cls, session: Session, item: PublicBenchmarkRecord) -> None:
        def lookup(item_id: str) -> PublicBenchmarkRecord | None:
            row = session.get(PublicBenchmarkRecordRow, item_id)
            return row.to_dto() if row is not None else None

        cls._validate_targets(item, lookup)

    @staticmethod
    def _validate_targets(item: PublicBenchmarkRecord, lookup: Callable[[str], PublicBenchmarkRecord | None]) -> None:
        def target(
            item_id: str, kind: str, *, same_revision: bool = True, role: str | None = None
        ) -> PublicBenchmarkRecord:
            row = lookup(item_id)
            if (
                row is None
                or row.kind != kind
                or row.origin_id != item.origin_id
                or (same_revision and row.source_revision != item.source_revision)
                or (role is not None and row.source_role != role)
            ):
                raise PublicBenchmarkRecordDaoError(f"invalid {kind} link: {item_id}")
            return row

        if item.parent_id is not None:
            if item.kind == "figure_link":
                target(item.parent_id, "figure")
            elif item.kind == "measurement":
                target(item.parent_id, "experiment")
                if item.parent_id != item.experiment_id:
                    raise PublicBenchmarkRecordDaoError("measurement parent must be its experiment")
            elif item.kind == "source":
                target(item.parent_id, "source", same_revision=False, role="registry")
            else:
                target(item.parent_id, "source", role="file")
        if item.experiment_id is not None:
            target(item.experiment_id, "experiment")
        if item.referenced_item_id is not None:
            target(item.referenced_item_id, "experiment")
        if item.published_manifest_id is not None:
            target(item.published_manifest_id, "source", same_revision=False, role="manifest")
        for member_id in item.manifest_item_ids or []:
            row = lookup(member_id)
            if (
                row is None
                or row.origin_id != item.origin_id
                or row.item_id == item.item_id
                or (row.kind == "source" and row.source_role != "file")
            ):
                raise PublicBenchmarkRecordDaoError(f"invalid manifest member: {member_id}")

    @staticmethod
    def _require_postgres(session: Session) -> None:
        if session.get_bind().dialect.name != "postgresql":
            raise PublicationError("publication requires PostgreSQL row locking and manifest membership queries")

    def register(self, origin_id: str, approval: SourceApproval) -> PublicBenchmarkRecord:
        validate_approval(origin_id, approval)
        state = RegistryState(approval=approval)
        item = PublicBenchmarkRecord(
            item_id=registry_id(origin_id),
            kind="source",
            source_role="registry",
            origin_id=origin_id,
            logical_key="@registry",
            source_revision="registry-v1",
            content_sha256=content_digest(approval.model_dump(mode="json")),
            access_classification="verified_public",
            source_path="@registry",
            format="json",
            source_details=state.model_dump(mode="json"),
        )
        with self._transaction() as session:
            self._require_postgres(session)
            existing = session.get(PublicBenchmarkRecordRow, item.item_id)
            if existing is not None:
                existing, current = self._registry(session, origin_id)
                if current.approval != approval:
                    raise PublicationError("source approval changes require an explicit compatibility review")
                return existing.to_dto()
            return self._save_in_session(session, item)

    def head(self, origin_id: str) -> SourceHead:
        with self._read_only() as session:
            self._require_postgres(session)
            row, state = self._registry(session, origin_id)
            manifest = (
                session.get(PublicBenchmarkRecordRow, row.published_manifest_id) if row.published_manifest_id else None
            )
            if row.published_manifest_id is not None and manifest is None:
                raise PublicationError("published manifest is missing")
            return SourceHead(
                registry=row.to_dto(),
                state=state,
                manifest=manifest.to_dto() if manifest is not None else None,
            )

    @staticmethod
    def _registry(
        session: Session, origin_id: str, *, lock: bool = False
    ) -> tuple[PublicBenchmarkRecordRow, RegistryState]:
        allowed_paths(origin_id)
        statement = select(PublicBenchmarkRecordRow).where(PublicBenchmarkRecordRow.item_id == registry_id(origin_id))
        if lock:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None or row.kind != "source" or row.source_role != "registry" or row.origin_id != origin_id:
            raise PublicationError("source is not registered")
        state = RegistryState.model_validate(row.source_details)
        validate_approval(origin_id, state.approval)
        if (
            row.logical_key != "@registry"
            or row.source_revision != "registry-v1"
            or row.content_sha256 != content_digest(state.approval.model_dump(mode="json"))
        ):
            raise PublicationError("source approval identity or hash is invalid")
        return row, state

    @staticmethod
    def _manifest(item: PublicBenchmarkRecord, origin_id: str, approval: SourceApproval) -> ReleaseManifest:
        if (
            item.kind != "source"
            or item.source_role != "manifest"
            or item.origin_id != origin_id
            or item.parent_id != registry_id(origin_id)
        ):
            raise PublicationError("invalid source manifest")
        metadata = ReleaseManifest.model_validate(item.source_details)
        validate_revision(origin_id, metadata.inventory.revision)
        digest = content_digest(metadata.model_dump(mode="json"))
        if (
            metadata.inventory.origin_id != origin_id
            or metadata.inventory.revision != item.source_revision
            or metadata.parser_version != approval.parser_version
            or item.content_sha256 != digest
            or item.logical_key != f"@manifest:{digest}"
            or item.item_id != item_id(origin_id, item.logical_key, item.source_revision)
            or tuple(item.manifest_item_ids or ()) != metadata.member_ids
        ):
            raise PublicationError("manifest identity, hash, parser, or membership mismatch")
        return metadata

    @classmethod
    def _validate_members(
        cls,
        session: Session,
        manifest: PublicBenchmarkRecord,
        metadata: ReleaseManifest,
        approval: SourceApproval,
    ) -> None:
        rows = session.scalars(
            select(PublicBenchmarkRecordRow).where(PublicBenchmarkRecordRow.item_id.in_(metadata.member_ids))
        ).all()
        items = {row.item_id: row.to_dto() for row in rows}
        registry = session.get(PublicBenchmarkRecordRow, registry_id(manifest.origin_id))
        if registry is not None:
            items[registry.item_id] = registry.to_dto()
        members = tuple(item for key, item in items.items() if key in metadata.member_ids)
        if (
            members
            or session.scalar(
                select(PublicBenchmarkRecordRow.item_id)
                .where(
                    PublicBenchmarkRecordRow.origin_id == manifest.origin_id,
                    PublicBenchmarkRecordRow.kind.in_(("experiment", "measurement", "figure", "figure_link")),
                )
                .limit(1)
            )
            is None
        ):
            cls._require_structured_release(manifest.origin_id, members)
        cls._validate_member_map(manifest, metadata, approval, items)

    @classmethod
    def _validate_member_map(
        cls,
        manifest: PublicBenchmarkRecord,
        metadata: ReleaseManifest,
        approval: SourceApproval,
        items: dict[str, PublicBenchmarkRecord],
    ) -> None:
        members = {key: items[key] for key in metadata.member_ids if key in items}
        if len(members) != len(metadata.member_ids):
            raise PublicationError("manifest has missing members")
        if dict(Counter(item.kind for item in members.values())) != metadata.counts:
            raise PublicationError("manifest kind counts are invalid")
        logical_keys = [item.logical_key for item in members.values()]
        if len(set(logical_keys)) != len(logical_keys):
            raise PublicationError("a release cannot mix versions of one logical item")
        for file in metadata.files:
            if input_format(manifest.origin_id, file.file.path, approval) != file.file.format:
                raise PublicationError("manifest includes an unapproved path or format")
            source = members[file.file_item_id]
            if (
                source.kind != "source"
                or source.source_role != "file"
                or source.parent_id != registry_id(manifest.origin_id)
                or source.source_details != {"parser_version": metadata.parser_version}
            ):
                raise PublicationError("manifest file source is invalid")
            for member_id in file.member_ids:
                item = members[member_id]
                if (
                    item.origin_id != manifest.origin_id
                    or item.source_path != file.file.path
                    or item.content_sha256 != file.file.sha256
                    or item.format != file.file.format
                    or item.access_classification != "verified_public"
                    or item.publication_state == "tombstone"
                    or item.source_revision != source.source_revision
                    or item.item_id != item_id(item.origin_id, item.logical_key, item.source_revision)
                    or (item.kind == "source" and item.item_id != source.item_id)
                ):
                    raise PublicationError("member does not match its pinned file")
                dependencies = {item.parent_id, item.experiment_id, item.referenced_item_id} - {None}
                if item.kind == "source":
                    dependencies.discard(registry_id(manifest.origin_id))
                if not dependencies.issubset(members):
                    raise PublicationError("manifest is not closed over its source/experiment/figure links")
                if item.kind in ("experiment", "passage", "figure") and item.parent_id != source.item_id:
                    raise PublicationError("item must link to its exact pinned source file")
                if item.kind == "measurement" and item.parent_id != item.experiment_id:
                    raise PublicationError("measurement must link to its experiment")
                cls._validate_targets(item, items.get)

    @staticmethod
    def _require_structured_release(origin_id: str, items: Sequence[PublicBenchmarkRecord]) -> None:
        from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN

        if origin_id != PRISM_ORIGIN:
            raise PublicationError("public_benchmark_records accepts only Prism; documents belong in OpenSearch")

    def stage(self, manifest: PublicBenchmarkRecord, items: Sequence[PublicBenchmarkRecord]) -> PublicBenchmarkRecord:
        with self._transaction() as session:
            self._require_postgres(session)
            registry, state = self._registry(session, manifest.origin_id, lock=True)
            if not state.enabled:
                raise PublicationError("source is disabled")
            metadata = self._manifest(manifest, registry.origin_id, state.approval)
            if (
                metadata.base_generation != state.generation
                or metadata.base_manifest_id != registry.published_manifest_id
            ):
                raise PublicationConflictError("source advanced while preparing this release")
            if {item.item_id for item in items} - set(metadata.member_ids):
                raise PublicationError("staged items must belong to the exact manifest")
            pending = {item.item_id: PublicBenchmarkRecord.model_validate(item.model_dump()) for item in items}
            if len(pending) != len(items):
                raise PublicationError("duplicate staged item IDs")
            while pending:
                ready = [
                    item
                    for item in pending.values()
                    if not ({item.parent_id, item.experiment_id, item.referenced_item_id} - {None}).intersection(
                        pending
                    )
                ]
                if not ready:
                    raise PublicationError("cyclic staged item dependencies")
                for item in ready:
                    self._save_in_session(session, item)
                    del pending[item.item_id]
            self._validate_members(session, manifest, metadata, state.approval)
            previous = (
                session.get(PublicBenchmarkRecordRow, registry.published_manifest_id)
                if registry.published_manifest_id
                else None
            )
            old_members = set(previous.manifest_item_ids or ()) if previous is not None else set()
            if set(metadata.removed_item_ids) != old_members - set(metadata.member_ids):
                raise PublicationError("manifest removal set does not match the current release")
            return self._save_in_session(session, manifest)

    def activate(
        self,
        origin_id: str,
        manifest_id: str,
        verify_sources: Callable[[ReleaseManifest], None],
        *,
        rollback_generation: int | None = None,
    ) -> PublicationResult:
        with self._transaction() as session:
            self._require_postgres(session)
            registry, state = self._registry(session, origin_id, lock=True)
            if not state.enabled:
                raise PublicationError("source is disabled")
            row = session.get(PublicBenchmarkRecordRow, manifest_id)
            if row is None:
                raise PublicationError("manifest does not exist")
            manifest = row.to_dto()
            metadata = self._manifest(manifest, origin_id, state.approval)
            previous_id = registry.published_manifest_id
            if rollback_generation is not None:
                if rollback_generation != state.generation:
                    raise PublicationConflictError("source advanced before rollback")
                if not state.was_published(manifest_id):
                    raise PublicationError("rollback target was never published")
            elif previous_id != manifest_id and (
                metadata.base_generation != state.generation or metadata.base_manifest_id != previous_id
            ):
                raise PublicationConflictError("staged release is stale; rebuild against the current source head")
            self._validate_members(session, manifest, metadata, state.approval)
            verify_sources(metadata)
            if previous_id == manifest_id:
                return PublicationResult(
                    origin_id=origin_id,
                    manifest_id=manifest_id,
                    previous_manifest_id=previous_id,
                    generation=state.generation,
                    counts=metadata.counts,
                    changed=False,
                )
            if rollback_generation is None:
                self._tombstones(session, registry, manifest, metadata)
            event = PublicationEvent(
                operation="activate" if rollback_generation is None else "rollback",
                manifest_id=manifest_id,
                previous_manifest_id=previous_id,
                generation=state.generation + 1,
                at=datetime.now(UTC),
                counts=metadata.counts,
            )
            registry.published_manifest_id = manifest_id
            registry.source_details = state.model_copy(
                update={
                    "generation": event.generation,
                    "history": (*state.history, event),
                }
            ).model_dump(mode="json")
            session.flush()
            return PublicationResult(
                origin_id=origin_id,
                manifest_id=manifest_id,
                previous_manifest_id=previous_id,
                generation=event.generation,
                counts=metadata.counts,
                changed=True,
            )

    @classmethod
    def _tombstones(
        cls,
        session: Session,
        registry: PublicBenchmarkRecordRow,
        manifest: PublicBenchmarkRecord,
        metadata: ReleaseManifest,
    ) -> None:
        if not metadata.removed_item_ids:
            return
        removed = session.scalars(
            select(PublicBenchmarkRecordRow).where(PublicBenchmarkRecordRow.item_id.in_(metadata.removed_item_ids))
        ).all()
        for source in removed:
            if source.kind != "source" or source.source_role != "file":
                continue
            key = f"@tombstone:{manifest.item_id}:{source.logical_key}"
            tombstone = PublicBenchmarkRecord(
                item_id=item_id(registry.origin_id, key, manifest.source_revision),
                kind="source",
                source_role="file",
                logical_key=key,
                origin_id=registry.origin_id,
                source_revision=manifest.source_revision,
                content_sha256=source.content_sha256,
                access_classification="verified_public",
                publication_state="tombstone",
                parent_id=registry.item_id,
                source_path=source.source_path,
                format=source.format,
                source_details={
                    "replaced_in_manifest": manifest.item_id,
                    "source_item_id": source.item_id,
                    "removed_item_ids": sorted(row.item_id for row in removed if row.source_path == source.source_path),
                },
            )
            cls._save_in_session(session, tombstone)

    def set_enabled(self, origin_id: str, enabled: bool, *, expected_generation: int) -> None:
        with self._transaction() as session:
            self._require_postgres(session)
            registry, state = self._registry(session, origin_id, lock=True)
            if state.generation != expected_generation:
                raise PublicationConflictError("source advanced before access-policy change")
            if state.enabled == enabled:
                return
            event = PublicationEvent(
                operation="enable" if enabled else "disable",
                manifest_id=registry.published_manifest_id,
                previous_manifest_id=registry.published_manifest_id,
                generation=state.generation + 1,
                at=datetime.now(UTC),
                counts={},
            )
            registry.source_details = state.model_copy(
                update={
                    "enabled": enabled,
                    "generation": event.generation,
                    "history": (*state.history, event),
                }
            ).model_dump(mode="json")

    @staticmethod
    def _published_members(origin_id: str) -> Select[tuple[PublicBenchmarkRecordRow]]:
        allowed_paths(origin_id)
        registry = aliased(PublicBenchmarkRecordRow, name="rag_registry")
        manifest = aliased(PublicBenchmarkRecordRow, name="rag_manifest")
        members = func.jsonb_array_elements_text(manifest.manifest_item_ids).table_valued("value").lateral("members")
        return (
            select(PublicBenchmarkRecordRow)
            .select_from(registry)
            .join(manifest, registry.published_manifest_id == manifest.item_id)
            .join(members, true())
            .join(PublicBenchmarkRecordRow, PublicBenchmarkRecordRow.item_id == members.c.value)
            .where(
                registry.item_id == registry_id(origin_id),
                registry.kind == "source",
                registry.source_role == "registry",
                registry.origin_id == origin_id,
                registry.source_details["contract"].as_string() == "benchmark-rag.registry.v1",
                registry.source_details["enabled"].as_boolean().is_(True),
                registry.source_details["approval"]["verified_public"].as_boolean().is_(True),
                manifest.kind == "source",
                manifest.source_role == "manifest",
                manifest.origin_id == origin_id,
                manifest.source_details["contract"].as_string() == "benchmark-rag.manifest.v1",
                PublicBenchmarkRecordRow.origin_id == origin_id,
                PublicBenchmarkRecordRow.access_classification == "verified_public",
                PublicBenchmarkRecordRow.publication_state != "tombstone",
            )
        )

    def list_published(self, origin_id: str, *, kind: ItemKind | None = None) -> list[PublicBenchmarkRecord]:
        """Internal membership only; query services enforce caller authentication."""
        with self._read_only() as session:
            self._require_postgres(session)
            statement = self._published_members(origin_id)
            if kind is not None:
                statement = statement.where(PublicBenchmarkRecordRow.kind == kind)
            return [row.to_dto() for row in session.scalars(statement.order_by(PublicBenchmarkRecordRow.item_id))]

    def count_published(self, origin_id: str, *, kind: ItemKind | None = None) -> int:
        with self._read_only() as session:
            self._require_postgres(session)
            statement = self._published_members(origin_id)
            if kind is not None:
                statement = statement.where(PublicBenchmarkRecordRow.kind == kind)
            return session.execute(select(func.count()).select_from(statement.subquery())).scalar_one()
