"""Reviewed manual refresh jobs for pinned local Prism snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from llm_d_lens.agentic.public_benchmark_record.contracts import ItemKind
from llm_d_lens.agentic.public_benchmark_record.prism import PARSER_VERSION as PRISM_PARSER
from llm_d_lens.agentic.public_benchmark_record.prism import PrismSnapshotImporter
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    PinnedInventory,
    PublicationConflictError,
    PublicationError,
    PublicationModel,
    PublicationResult,
    ReleaseManifest,
    SourceApproval,
    SourceHead,
    content_digest,
)
from llm_d_lens.agentic.public_benchmark_record.sources import (
    PRISM_ORIGIN,
    input_format,
    validate_approval,
    validate_revision,
)

RefreshTrigger = Literal["manual"]


class RefreshPlan(PublicationModel):
    contract: Literal["benchmark-rag.refresh.v1"] = "benchmark-rag.refresh.v1"
    review_reference: str = Field(min_length=1)
    approval: SourceApproval
    inventory: PinnedInventory
    expected_counts: dict[ItemKind, StrictInt]
    base_generation: int = Field(ge=0)
    base_manifest_id: str | None
    triggers: tuple[RefreshTrigger, ...] = ("manual",)
    allow_empty: bool = False

    @model_validator(mode="after")
    def reviewed_scope(self) -> Self:
        origin = self.inventory.origin_id
        if origin != PRISM_ORIGIN:
            raise PublicationError("only Prism SQL benchmark refresh is available; document provider is not installed")
        validate_approval(origin, self.approval)
        validate_revision(origin, self.inventory.revision)
        if not self.triggers or len(set(self.triggers)) != len(self.triggers):
            raise ValueError("refresh triggers must be nonempty and unique")
        if any(value < 0 for value in self.expected_counts.values()):
            raise ValueError("reviewed counts must be nonnegative")
        if self.expected_counts.get("source", 0) != len(self.inventory.files):
            raise ValueError("source count must match the complete inventory")
        if not self.inventory.files and not self.allow_empty:
            raise ValueError("removing every file needs explicit allow_empty approval")
        for file in self.inventory.files:
            if input_format(origin, file.path, self.approval) != file.format:
                raise ValueError("refresh inventory includes an unapproved path or format")
        if self.approval.parser_version != PRISM_PARSER:
            raise ValueError("refresh plan requires the supported source parser")
        if self.triggers != ("manual",):
            raise ValueError("Prism refresh is manual")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class RefreshReport(PublicationModel):
    contract: Literal["benchmark-rag.refresh-report.v1"] = "benchmark-rag.refresh-report.v1"
    status: Literal["published", "unchanged"]
    trigger: RefreshTrigger
    plan_sha256: str
    review_reference: str
    revision: str
    inventory_sha256: str
    file_count: int
    started_at: datetime
    finished_at: datetime
    publication: PublicationResult


class PublicBenchmarkRecordRefreshService:
    def __init__(self, publication: PublicBenchmarkRecordPublicationService) -> None:
        self.publication = publication

    @staticmethod
    def _check_head(plan: RefreshPlan, head: SourceHead) -> None:
        if not head.state.enabled:
            raise PublicationError("source is disabled; refresh cannot re-enable it")
        if head.state.approval != plan.approval:
            raise PublicationError("refresh approval differs from the registered source")
        if (
            head.state.generation == plan.base_generation
            and head.registry.published_manifest_id == plan.base_manifest_id
        ):
            return
        # Only a direct successful replay may reuse the reviewed base. A rollback
        # or access-policy change must not be undone by an old reviewed plan.
        if head.manifest is not None and head.state.history:
            event = head.state.history[-1]
            manifest = ReleaseManifest.model_validate(head.manifest.source_details)
            if (
                head.state.generation == plan.base_generation + 1
                and event.operation == "activate"
                and event.previous_manifest_id == plan.base_manifest_id
                and event.manifest_id == head.manifest.item_id
                and manifest.base_generation == plan.base_generation
                and manifest.base_manifest_id == plan.base_manifest_id
                and manifest.inventory == plan.inventory
                and manifest.counts == {kind: count for kind, count in plan.expected_counts.items() if count}
            ):
                return
        raise PublicationConflictError("reviewed refresh base is stale; review a new plan against current status")

    def run(
        self,
        plan: RefreshPlan,
        source: Path | None = None,
        *,
        trigger: RefreshTrigger = "manual",
    ) -> RefreshReport:
        plan = RefreshPlan.model_validate(plan.model_dump())
        if trigger not in plan.triggers:
            raise PublicationError("trigger is not approved by this refresh plan")
        started = datetime.now(UTC)
        origin = plan.inventory.origin_id
        head = self.publication.dao.head(origin)
        self._check_head(plan, head)
        if source is None:
            raise PublicationError("Prism refresh requires an explicit local snapshot source")
        staged = PrismSnapshotImporter(self.publication).stage(
            source,
            plan.inventory.revision,
            {file.path: file.sha256 for file in plan.inventory.files},
            plan.approval,
            expected_experiments=plan.expected_counts.get("experiment", 0),
            allow_empty=plan.allow_empty,
        )
        metadata = ReleaseManifest.model_validate(staged.source_details)
        if metadata.inventory != plan.inventory or metadata.counts != {
            kind: count for kind, count in plan.expected_counts.items() if count
        }:
            raise PublicationError("extraction differs from the complete reviewed refresh plan")
        current = self.publication.dao.head(origin)
        if (
            current.state.generation != head.state.generation
            or current.registry.published_manifest_id != head.registry.published_manifest_id
            or current.state.approval != head.state.approval
            or not current.state.enabled
        ):
            raise PublicationConflictError("source changed while collecting refresh inputs; retry after review")
        if staged.item_id != current.registry.published_manifest_id and (
            metadata.base_generation != plan.base_generation or metadata.base_manifest_id != plan.base_manifest_id
        ):
            raise PublicationConflictError("staged refresh does not match the reviewed base")
        result = self.publication.activate(origin, staged.item_id)
        if result.generation != head.state.generation + int(result.changed):
            raise PublicationConflictError("source changed before refresh completion; review current status")
        return RefreshReport(
            status="published" if result.changed else "unchanged",
            trigger=trigger,
            plan_sha256=plan.digest,
            review_reference=plan.review_reference,
            revision=plan.inventory.revision,
            inventory_sha256=plan.inventory.digest,
            file_count=len(plan.inventory.files),
            started_at=started,
            finished_at=datetime.now(UTC),
            publication=result,
        )
