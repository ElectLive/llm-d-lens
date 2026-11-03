"""Caller-scoped public benchmark requests and deterministic, provenance-aware responses."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, validate_path

QUERY_PERMISSION = "benchmark-rag:query:read"
PUBLIC_ORIGINS = (PRISM_ORIGIN,)
QueryIntent = Literal["auto", "numeric", "comparison", "definition", "figure"]
EvidenceType = Literal["raw_json", "published_table", "prose_summary", "image_trend", "agentic_measurement"]


class QueryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=False)


class QueryFilters(QueryModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    origins: tuple[str, ...] = Field(default=PUBLIC_ORIGINS, min_length=1, max_length=len(PUBLIC_ORIGINS))
    source_revision: str | None = Field(default=None, min_length=1, max_length=64)
    source_path: str | None = Field(default=None, min_length=1, max_length=1024)
    record_id: str | None = Field(default=None, min_length=1, max_length=256)
    experiment_id: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    hardware: str | None = Field(default=None, min_length=1, max_length=256)
    runtime: str | None = Field(default=None, min_length=1, max_length=256)
    metric: str | None = Field(default=None, min_length=1, max_length=256)
    statistic: str | None = Field(default=None, min_length=1, max_length=64)
    unit: str | None = Field(default=None, min_length=1, max_length=64)
    workload: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)
    series: dict[str, JsonValue] = Field(default_factory=dict, max_length=16)
    value_min: Decimal | None = None
    value_max: Decimal | None = None
    normalized: bool = False

    @field_validator("origins")
    @classmethod
    def approved_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or set(value) - set(PUBLIC_ORIGINS):
            raise ValueError("origins must be unique approved public sources")
        return value

    @field_validator("source_path")
    @classmethod
    def canonical_path(cls, value: str | None) -> str | None:
        if value is not None:
            validate_path(value)
        return value

    @model_validator(mode="after")
    def range_units(self) -> Self:
        if (self.value_min is not None or self.value_max is not None) and self.unit is None:
            raise ValueError("numeric ranges require an explicit unit")
        if self.value_min is not None and self.value_max is not None and self.value_min > self.value_max:
            raise ValueError("value_min must not exceed value_max")
        return self


class QueryRequest(QueryModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2048)
    intent: QueryIntent = "auto"
    filters: QueryFilters = Field(default_factory=QueryFilters)
    limit: int = Field(default=10, ge=1, le=50)
    include_internal: bool = False


class Citation(QueryModel):
    origin_id: str
    source_revision: str
    source_path: str
    sha256: str
    artifact_uri: str
    source_pointer: dict[str, JsonValue]
    heading_path: tuple[str, ...] = ()
    published_manifest_id: str


class ExperimentContext(QueryModel):
    item_id: str
    origin_id: str
    model: str | None
    hardware: str | None
    runtime: str | None
    workload: dict[str, JsonValue] | None
    configuration: dict[str, JsonValue] | None
    source_pointer: dict[str, JsonValue]


class AgenticCitation(QueryModel):
    origin_id: Literal["agentic:evaluate"] = "agentic:evaluate"
    evidence_type: Literal["agentic_measurement"] = "agentic_measurement"
    benchmark_id: str
    evaluate_run_id: str
    cluster_id: str
    resource_uri: str
    record_sha256: str
    metric_field: str
    record_timestamp: datetime
    access_scope: Literal["owning_evaluation"] = "owning_evaluation"
    numeric_precision: Literal["stored_float"] = "stored_float"


class QueryEvidence(QueryModel):
    item_id: str
    origin: Literal["public", "agentic", "public_and_agentic"] = "public"
    evidence_type: EvidenceType
    citation: Citation | AgenticCitation
    corroborating_citations: tuple[AgenticCitation, ...] = ()
    identity_verification: Literal["canonical_evaluate_run_and_metric"] | None = None
    verified_evaluate_run_id: str | None = None
    experiment_time: datetime | None = None
    last_successful_sync_at: datetime | None = None
    experiment: ExperimentContext | None = None
    related_experiments: tuple[ExperimentContext, ...] = ()
    metric: str | None = None
    statistic: str | None = None
    value: Decimal | None = None
    unit: str | None = None
    original_value: Decimal | None = None
    original_unit: str | None = None
    normalized_value: Decimal | None = None
    normalized_unit: str | None = None
    normalization_rule: str | None = None
    series: dict[str, JsonValue] | None = None
    text: str | None = None
    coverage: Literal["complete", "partial", "summary_only", "unknown"] | None = None
    missing_series: tuple[str, ...] = ()


class Abstention(QueryModel):
    code: str
    message: str


class QueryResponse(QueryModel):
    contract: Literal["benchmark-rag.query.v1"] = "benchmark-rag.query.v1"
    question: str
    intent: Literal["numeric", "comparison", "definition", "figure"]
    filters: QueryFilters
    status: Literal["answered", "partial", "abstained"]
    answer: str
    evidence: tuple[QueryEvidence, ...]
    matched_items: int = Field(ge=0)
    matched_experiments: int | None = Field(default=None, ge=0)
    returned_experiments: int | None = Field(default=None, ge=0)
    experiment_limit: int | None = Field(default=None, ge=1, le=50)
    truncated: bool
    abstentions: tuple[Abstention, ...] = ()
    warnings: tuple[str, ...] = ()
    generation_mode: Literal["deterministic"] = "deterministic"
    internal_branch_enabled: bool = False
    returned_observations: int = Field(default=0, ge=0)
