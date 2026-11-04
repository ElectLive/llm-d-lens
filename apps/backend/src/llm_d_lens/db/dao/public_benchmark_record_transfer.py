"""public benchmark-only schema preflight, consistent export and atomic immutable merges."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    PublicationError,
    PublicationEvent,
    RegistryState,
    RestoreReceipt,
    content_digest,
    item_id,
    registry_id,
)
from llm_d_lens.agentic.public_benchmark_record.sources import input_format, validate_approval
from llm_d_lens.agentic.public_benchmark_record.transfer_contracts import RestoreResult
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao
from llm_d_lens.db.models.public_benchmark_record import PublicBenchmarkRecordRow


class PublicBenchmarkRecordTransferDao(PublicBenchmarkRecordDao):
    @staticmethod
    def _admin_lock(session: Session) -> None:
        session.execute(
            text(
                "SELECT pg_advisory_xact_lock(hashtextextended("
                "current_database() || '.' || current_schema() || '.rag-transfer', 0))"
            )
        )

    def prepare(self, *, initialize: bool = False) -> None:
        with self._transaction() as session:
            self._require_postgres(session)
            self._admin_lock(session)
            connection = session.connection()
            inspector = inspect(connection)
            if not inspector.has_table("public_benchmark_records"):
                if not initialize:
                    raise PublicationError(
                        "public_benchmark_records is missing; migrate Lens or initialize a standalone structured schema"
                    )
                if inspector.has_table("alembic_version"):
                    raise PublicationError(
                        "managed Lens schema requires its reviewed Alembic migration, not standalone initialization"
                    )
                PublicBenchmarkRecordRow.__table__.create(connection)
                inspector = inspect(connection)
            table = PublicBenchmarkRecordRow.__table__
            columns = {column["name"]: column for column in inspector.get_columns("public_benchmark_records")}
            if set(columns) != set(table.columns.keys()):
                raise PublicationError("public_benchmark_records columns are incompatible with this code version")
            for column in table.columns:
                actual = columns[column.name]
                if (
                    actual["type"].compile(dialect=connection.dialect)
                    != column.type.compile(dialect=connection.dialect)
                    or actual["nullable"] != column.nullable
                ):
                    raise PublicationError(f"incompatible public_benchmark_records column: {column.name}")
            names = {
                value["name"]
                for values in (
                    inspector.get_check_constraints("public_benchmark_records"),
                    inspector.get_unique_constraints("public_benchmark_records"),
                    inspector.get_foreign_keys("public_benchmark_records"),
                )
                for value in values
            }
            expected = {constraint.name for constraint in table.constraints if constraint.name is not None}
            if not expected.issubset(names):
                raise PublicationError("public_benchmark_records constraints are incompatible")
            if inspector.get_pk_constraint("public_benchmark_records")["constrained_columns"] != ["item_id"]:
                raise PublicationError("public_benchmark_records primary key is incompatible")
            if not {index.name for index in table.indexes}.issubset(
                index["name"] for index in inspector.get_indexes("public_benchmark_records")
            ):
                raise PublicationError("public_benchmark_records indexes are incompatible")

    @classmethod
    def validate_snapshot(cls, records: tuple[PublicBenchmarkRecord, ...]) -> dict[str, RegistryState]:
        items = {item.item_id: item for item in records}
        if not records or len(items) != len(records):
            raise PublicationError("public benchmark backup must contain unique records")
        versions = {(item.origin_id, item.logical_key, item.source_revision) for item in records}
        if len(versions) != len(records):
            raise PublicationError("public benchmark backup contains duplicate immutable versions")
        states: dict[str, RegistryState] = {}
        for item in records:
            if item.item_id != item_id(item.origin_id, item.logical_key, item.source_revision):
                raise PublicationError("public benchmark backup item identity is invalid")
            if item.source_role != "registry":
                continue
            state = RegistryState.model_validate(item.source_details)
            validate_approval(item.origin_id, state.approval)
            if (
                item.item_id != registry_id(item.origin_id)
                or item.parent_id is not None
                or item.content_sha256 != content_digest(state.approval.model_dump(mode="json"))
            ):
                raise PublicationError("public benchmark backup source approval identity is invalid")
            states[item.origin_id] = state
        for item in records:
            if item.origin_id not in states:
                raise PublicationError("public benchmark backup is missing its canonical source registry")
            cls._validate_targets(item, items.get)
            approval = states[item.origin_id].approval
            if item.source_role == "file" and (
                item.parent_id != registry_id(item.origin_id)
                or input_format(item.origin_id, item.source_path, approval) != item.format
            ):
                raise PublicationError("public benchmark backup source file is not approved")
            if item.source_role == "manifest":
                metadata = cls._manifest(item, item.origin_id, approval)
                cls._validate_member_map(item, metadata, approval, items)
                for reference in (*metadata.removed_item_ids, metadata.base_manifest_id):
                    if reference is not None:
                        target = items.get(reference)
                        if target is None or target.origin_id != item.origin_id:
                            raise PublicationError("public benchmark backup is missing retained release history")
                if metadata.base_manifest_id is not None and items[metadata.base_manifest_id].source_role != "manifest":
                    raise PublicationError("public benchmark backup base release is not a manifest")
        for origin, state in states.items():
            cls._require_structured_release(origin, tuple(item for item in records if item.origin_id == origin))
            if len({receipt.backup_id for receipt in state.restores}) != len(state.restores):
                raise PublicationError("public benchmark backup has duplicate restore receipts")
            for receipt in state.restores:
                if receipt.published_manifest_id is not None:
                    target = items.get(receipt.published_manifest_id)
                    if target is None or target.source_role != "manifest" or target.origin_id != origin:
                        raise PublicationError("public benchmark backup restore receipt points to a missing manifest")
            histories = [
                (state.generation, state.history),
                *((receipt.generation, receipt.history) for receipt in state.restores),
            ]
            for generation, history in histories:
                numbers = [event.generation for event in history]
                if numbers != sorted(set(numbers)) or any(number < 1 or number > generation for number in numbers):
                    raise PublicationError("public benchmark backup publication history generation is invalid")
                for event in history:
                    for reference in (event.manifest_id, event.previous_manifest_id):
                        if reference is not None:
                            target = items.get(reference)
                            if target is None or target.source_role != "manifest" or target.origin_id != origin:
                                raise PublicationError("public benchmark backup history points to a missing manifest")
        return states

    def snapshot(self) -> tuple[PublicBenchmarkRecord, ...]:
        with self._read_only() as session:
            self._require_postgres(session)
            # One SELECT captures registry pointers and every immutable row in one MVCC snapshot.
            records = tuple(
                row.to_dto()
                for row in session.scalars(select(PublicBenchmarkRecordRow).order_by(PublicBenchmarkRecordRow.item_id))
            )
            self.validate_snapshot(records)
            return records

    def restore(self, records: tuple[PublicBenchmarkRecord, ...], backup_id: str) -> RestoreResult:
        incoming_states = self.validate_snapshot(records)
        incoming = {item.item_id: item for item in records}
        with self._transaction() as session:
            self._require_postgres(session)
            self._admin_lock(session)
            registries: dict[str, PublicBenchmarkRecordRow] = {}
            states: dict[str, RegistryState] = {}
            inserted = 0
            for origin, incoming_state in sorted(incoming_states.items()):
                registry = session.get(PublicBenchmarkRecordRow, registry_id(origin))
                if registry is None:
                    item = incoming[registry_id(origin)].model_copy(
                        update={
                            "published_manifest_id": None,
                            "source_details": RegistryState(
                                approval=incoming_state.approval,
                                enabled=incoming_state.enabled,
                            ).model_dump(mode="json"),
                        }
                    )
                    self._save_in_session(session, item)
                    inserted += 1
                registry, state = self._registry(session, origin, lock=True)
                if state.approval.parser_version != incoming_state.approval.parser_version:
                    raise PublicationError("restore parser compatibility requires explicit review")
                registries[origin], states[origin] = registry, state
            existing_rows = session.scalars(select(PublicBenchmarkRecordRow)).all()
            existing = {row.item_id: row.to_dto() for row in existing_rows}
            pending: dict[str, PublicBenchmarkRecord] = {}
            for item in records:
                if item.source_role == "registry":
                    continue
                previous = existing.get(item.item_id)
                if previous is not None:
                    if previous != item:
                        raise PublicationError("restore conflicts with an existing immutable public benchmark item")
                else:
                    pending[item.item_id] = item
            inserted_rows: list[PublicBenchmarkRecordRow] = []
            while pending:
                ready = [
                    item
                    for item in pending.values()
                    if not (
                        {item.parent_id, item.experiment_id, item.referenced_item_id, *(item.manifest_item_ids or ())}
                        - {None}
                    ).intersection(pending)
                ]
                if not ready:
                    raise PublicationError("restore contains cyclic public benchmark dependencies")
                rows = [PublicBenchmarkRecordRow.from_dto(item) for item in ready]
                session.add_all(rows)
                session.flush()
                inserted_rows.extend(rows)
                inserted += len(rows)
                for item in ready:
                    del pending[item.item_id]
            updated = 0
            for origin, source_state in sorted(incoming_states.items()):
                registry, state = registries[origin], states[origin]
                if any(receipt.backup_id == backup_id for receipt in state.restores):
                    continue
                restored = {receipt.backup_id: receipt for receipt in state.restores}
                for receipt in source_state.restores:
                    if receipt.backup_id in restored and restored[receipt.backup_id] != receipt:
                        raise PublicationError("restore receipt history conflicts")
                    restored[receipt.backup_id] = receipt
                restored[backup_id] = RestoreReceipt(
                    backup_id=backup_id,
                    generation=source_state.generation,
                    published_manifest_id=incoming[registry_id(origin)].published_manifest_id,
                    enabled=source_state.enabled,
                    history=source_state.history,
                )
                target = incoming[registry_id(origin)].published_manifest_id or registry.published_manifest_id
                counts = (
                    self._manifest(
                        session.get(PublicBenchmarkRecordRow, target).to_dto(),
                        origin,
                        state.approval,
                    ).counts
                    if target is not None
                    else {}
                )
                event = PublicationEvent(
                    operation="activate",
                    manifest_id=target,
                    previous_manifest_id=registry.published_manifest_id,
                    generation=state.generation + 1,
                    at=datetime.now(UTC),
                    counts=counts,
                )
                registry.published_manifest_id = target
                registry.source_details = state.model_copy(
                    update={
                        "enabled": state.enabled and source_state.enabled,
                        "generation": event.generation,
                        "history": (*state.history, event),
                        "restores": tuple(restored.values()),
                    }
                ).model_dump(mode="json")
                updated += 1
            session.flush()
            combined = tuple(row.to_dto() for row in (*existing_rows, *inserted_rows))
            self.validate_snapshot(combined)
            return RestoreResult(
                backup_id=backup_id,
                inserted=inserted,
                updated_sources=updated,
                total_rows=len(combined),
            )


@contextmanager
def transfer_database(database_url: str, *, schema: str = "public") -> Iterator[PublicBenchmarkRecordTransferDao]:
    try:
        url = make_url(database_url)
    except (ArgumentError, ValueError) as error:
        raise PublicationError("invalid PostgreSQL connection URL") from error
    if url.drivername not in ("postgresql", "postgresql+psycopg"):
        raise PublicationError("public benchmark administration requires an explicit PostgreSQL connection")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema):
        raise PublicationError("public benchmark schema must be an unquoted lowercase PostgreSQL identifier")
    options = url.query.get("options", os.environ.get("PGOPTIONS", ""))
    if not isinstance(options, str):
        raise PublicationError("PostgreSQL connection options must have one value")
    try:
        engine = create_engine(
            url.set(drivername="postgresql+psycopg"),
            isolation_level="READ COMMITTED",
            connect_args={"options": f"{options} -csearch_path={schema}".strip()},
        )
    except (SQLAlchemyError, ValueError) as error:
        raise PublicationError("invalid PostgreSQL connection configuration") from error
    try:
        with engine.connect() as connection:
            if connection.scalar(text("SELECT current_schema()")) != schema:
                raise PublicationError("target schema does not exist or is not accessible")
        yield PublicBenchmarkRecordTransferDao(sessionmaker(bind=engine, expire_on_commit=False))
    except SQLAlchemyError as error:
        raise PublicationError(
            f"database operation failed ({type(error).__name__}); check target, schema and permissions"
        ) from error
    finally:
        engine.dispose()
