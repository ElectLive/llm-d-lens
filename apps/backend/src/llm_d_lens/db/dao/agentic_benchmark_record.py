"""DAO backing ``llm_d_lens.agentic.benchmark_store`` -- see design doc section 5.4.12.

Append-only: ``save`` is idempotent (a no-op if the id already exists),
matching ``JsonBenchmarkRecordStore.save``'s "write once" semantics -- there
is deliberately no update/delete method on this repository.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from llm_d_lens.agentic.benchmark_evidence import BenchmarkRecord
from llm_d_lens.agentic.planning_requirements import normalize_hardware
from llm_d_lens.db.dao.base import BaseDao
from llm_d_lens.db.models.agentic_benchmark_record import BenchmarkRecordRow


class BenchmarkRecordDaoError(Exception):
    """Raised when benchmark evidence cannot be durably persisted."""


class BenchmarkRecordDao(BaseDao):
    def get(self, benchmark_id: str) -> BenchmarkRecord | None:
        try:
            with self._read_only() as session:
                row = session.get(BenchmarkRecordRow, benchmark_id)
                return row.to_dto() if row else None
        except SQLAlchemyError as error:
            raise BenchmarkRecordDaoError("Unable to read benchmark evidence") from error

    def save_many(self, records: list[BenchmarkRecord]) -> list[str]:
        """Atomically append a batch, tolerating concurrent inserts of the same IDs."""
        inserted = []
        try:
            with self._transaction() as session:
                dialect = session.get_bind().dialect.name
                for record in records:
                    if dialect in ("postgresql", "sqlite"):
                        row = BenchmarkRecordRow.from_dto(record)
                        values = {
                            column.key: getattr(row, column.key) for column in BenchmarkRecordRow.__table__.columns
                        }
                        insert = postgres_insert if dialect == "postgresql" else sqlite_insert
                        statement = (
                            insert(BenchmarkRecordRow)
                            .values(**values)
                            .on_conflict_do_nothing(index_elements=["benchmark_id"])
                            .returning(BenchmarkRecordRow.benchmark_id)
                        )
                        if session.scalar(statement) is not None:
                            inserted.append(record.benchmark_id)
                        continue
                    if session.get(BenchmarkRecordRow, record.benchmark_id) is not None:
                        continue
                    try:
                        with session.begin_nested():
                            session.add(BenchmarkRecordRow.from_dto(record))
                            session.flush()
                    except IntegrityError:
                        if session.get(BenchmarkRecordRow, record.benchmark_id) is None:
                            raise
                    else:
                        inserted.append(record.benchmark_id)
        except SQLAlchemyError as error:
            raise BenchmarkRecordDaoError("Unable to persist benchmark evidence") from error
        return inserted

    def save(self, record: BenchmarkRecord) -> BenchmarkRecord:
        with self._transaction() as session:
            existing = session.get(BenchmarkRecordRow, record.benchmark_id)
            if existing is not None:
                return existing.to_dto()
            row = BenchmarkRecordRow.from_dto(record)
            session.add(row)
            session.flush()
            return row.to_dto()

    def list(self) -> list[BenchmarkRecord]:
        with self._read_only() as session:
            stmt = select(BenchmarkRecordRow).order_by(BenchmarkRecordRow.timestamp, BenchmarkRecordRow.benchmark_id)
            return [row.to_dto() for row in session.scalars(stmt)]

    def find_compatible(
        self,
        *,
        accelerator: str | None,
        backend: str | None,
        quantization: str | None,
        topology: str,
        minimum_vram_per_gpu_gib: float | None,
        observed_after: datetime | None = None,
    ) -> list[BenchmarkRecord]:
        """Return records eligible for deterministic retriever fine scoring.

        These predicates exactly match the retriever's hard compatibility rules.
        Continuous similarity inputs and outcome remain unfiltered so Agentic can
        compute weights, confidence, and negative evidence consistently.
        Missing query identities cannot establish compatibility and return no matches.
        """
        if any(value is None for value in (accelerator, backend, quantization, minimum_vram_per_gpu_gib)):
            return []
        accelerator_key = normalize_hardware(accelerator).device_model
        if accelerator_key is None:
            return []
        with self._read_only() as session:
            stmt = select(BenchmarkRecordRow).where(
                BenchmarkRecordRow.runtime_backend == backend,
                BenchmarkRecordRow.model_quantization == quantization,
                BenchmarkRecordRow.configuration_topology == topology,
                BenchmarkRecordRow.hardware_vram_per_gpu_gib >= minimum_vram_per_gpu_gib,
            )
            if observed_after is not None:
                stmt = stmt.where(BenchmarkRecordRow.timestamp >= observed_after)
            stmt = stmt.order_by(BenchmarkRecordRow.timestamp, BenchmarkRecordRow.benchmark_id)
            # Stored labels retain source provenance, including older normalized labels.
            records = (row.to_dto() for row in session.scalars(stmt))
            return [record for record in records if record.hardware.accelerator_key == accelerator_key]

    def find_seed_candidates(
        self,
        *,
        accelerator: str | None,
        backend: str | None,
        quantization: str | None,
        minimum_vram_per_gpu_gib: float | None,
        observed_after: datetime | None = None,
    ) -> list[BenchmarkRecord]:
        """Return hard-compatible history across topologies for seed discovery.

        The caller must still aggregate each configuration through
        ``BenchmarkEvidenceRetriever`` and validate every resulting proposal.
        Incomplete query identities return no seed candidates.
        """
        if any(value is None for value in (accelerator, backend, quantization, minimum_vram_per_gpu_gib)):
            return []
        accelerator_key = normalize_hardware(accelerator).device_model
        if accelerator_key is None:
            return []
        with self._read_only() as session:
            stmt = select(BenchmarkRecordRow).where(
                BenchmarkRecordRow.runtime_backend == backend,
                BenchmarkRecordRow.model_quantization == quantization,
                BenchmarkRecordRow.hardware_vram_per_gpu_gib >= minimum_vram_per_gpu_gib,
            )
            if observed_after is not None:
                stmt = stmt.where(BenchmarkRecordRow.timestamp >= observed_after)
            stmt = stmt.order_by(BenchmarkRecordRow.timestamp, BenchmarkRecordRow.benchmark_id)
            records = (row.to_dto() for row in session.scalars(stmt))
            return [record for record in records if record.hardware.accelerator_key == accelerator_key]
