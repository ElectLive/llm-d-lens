"""Versioned storage contracts; these are not public query responses."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, localcontext
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

if TYPE_CHECKING:
    from llm_d_lens.agentic.planner import PlanningFacts

ItemKind = Literal["source", "experiment", "measurement", "passage", "figure", "figure_link"]
PublicationState = Literal["staged", "published", "retired", "tombstone"]
SourceRole = Literal["registry", "manifest", "file"]
Coverage = Literal["complete", "partial", "summary_only", "unknown"]

WORKLOAD_ALIASES = {
    "mean_input_tokens": ("mean_input_tokens", "isl", "input_tokens"),
    "mean_output_tokens": ("mean_output_tokens", "osl", "output_tokens"),
}
LATENCY_STATISTICS = ("mean", "p50", "p90", "p95", "p99")
EXPERIMENT_RANKING_POLICY = "workload-first-lower-latency-v1"
WORKLOAD_NUMBER_PATTERN = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
DEFAULT_PUBLIC_RUNTIME = "vllm"
UNKNOWN_RUNTIME_LABELS = ("", "unknown")


def effective_runtime(runtime: str | None) -> str:
    """Apply the public corpus's vLLM default without rewriting source evidence."""
    if runtime is None or runtime.strip().casefold() in UNKNOWN_RUNTIME_LABELS:
        return DEFAULT_PUBLIC_RUNTIME
    return runtime


def workload_number(value: JsonValue) -> Decimal | None:
    """Compare numeric JSON scalars and legacy precise strings without rewriting evidence."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    return Decimal(text) if re.fullmatch(WORKLOAD_NUMBER_PATTERN, text) else None


def metric_aliases(metric: str) -> tuple[str, ...]:
    return ("latency", "e2e") if metric in ("latency", "e2e") else (metric,)


def latency_axes(targets: tuple[tuple[str, str | None], ...]) -> tuple[tuple[str, str | None], ...]:
    if targets:
        return tuple(
            dict.fromkeys(
                ("latency" if metric == "e2e" else metric, effective_statistic(statistic))
                for metric, statistic in targets
            )
        )
    return tuple((metric, statistic) for metric in ("ttft", "tpot", "latency") for statistic in LATENCY_STATISTICS)


def planning_latency_axes(facts: PlanningFacts) -> tuple[tuple[str, str | None], ...]:
    targets = (
        tuple(
            (target.field, target.statistic)
            for target in facts.requirements.performance_targets
            if target.field in {"ttft", "tpot", "e2e"}
        )
        if facts.requirements is not None
        else ()
    )
    if not targets:
        targets = tuple(
            (metric, "mean")
            for metric, value in (
                ("ttft", facts.ttft_slo_ms),
                ("tpot", facts.tpot_slo_ms),
                ("latency", facts.end_to_end_latency_slo_ms),
            )
            if value is not None
        )
    return latency_axes(targets)


def latency_value_ms(entry: dict[str, Any]) -> float | None:
    metric = "latency" if entry.get("metric") == "e2e" else entry.get("metric")
    if metric not in {"ttft", "tpot", "latency"}:
        return None
    value, unit = (
        (entry.get("normalized_value"), entry["normalized_unit"])
        if entry.get("normalized_unit") is not None
        else (entry.get("value"), entry.get("unit"))
    )
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(number) or number <= 0:
        return None
    if unit == ("ms/token" if metric == "tpot" else "ms"):
        return number
    if unit == ("s/token" if metric == "tpot" else "s"):
        converted = number * 1000
        return converted if isfinite(converted) else None
    return None


def experiment_ranking(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep workload priority and Pareto latency comparisons independent of SLO satisfaction."""
    result = []
    for profile in profiles:
        workload = profile["workload"]
        requested = workload["requested"]
        measured = [
            field for field in workload["fields"].values() if not field["basis"].startswith("published_report.")
        ]
        coverage = len(measured) / len(requested) if requested else None
        score = sum(field["similarity"] for field in measured) / len(requested) if measured else None
        fallback = score is None and not profile["latency_values_ms"]
        result.append(
            {
                "policy": EXPERIMENT_RANKING_POLICY,
                "queue": "fallback" if fallback else "ranked",
                "fallback_reason": "no_comparable_workload_or_valid_latency" if fallback else None,
                "workload_score": score,
                "workload_coverage": coverage,
                "latency_values_ms": profile["latency_values_ms"],
                "latency_coverage": len(profile["latency_values_ms"]),
                "dominated_by": 0,
            }
        )
    for current in result:
        values = current["latency_values_ms"]
        if not values:
            continue
        current["dominated_by"] = sum(
            other["workload_score"] == current["workload_score"]
            and other["workload_coverage"] == current["workload_coverage"]
            and other["latency_values_ms"].keys() == values.keys()
            and all(other["latency_values_ms"][axis] <= value for axis, value in values.items())
            and any(other["latency_values_ms"][axis] < value for axis, value in values.items())
            for other in result
        )
    return result


def experiment_rank_key(ranking: dict[str, Any]) -> tuple[Any, ...]:
    return (
        ranking.get("queue") == "fallback",
        ranking["workload_score"] is None,
        -(ranking["workload_score"] or 0),
        -(ranking["workload_coverage"] or 0),
        -ranking["latency_coverage"],
        ranking["dominated_by"],
    )


def model_identity_aliases(model: str) -> tuple[str, ...]:
    name = model.strip().lower()
    if not name:
        raise ValueError("model name must not be blank")
    return tuple(dict.fromkeys(alias for alias in (name, name.rsplit("/", 1)[-1], name.replace("/", "-")) if alias))


def model_identities_match(first: str | None, second: str | None) -> bool:
    if not first or not first.strip() or not second or not second.strip():
        return False
    return bool(set(model_identity_aliases(first)).intersection(model_identity_aliases(second)))


_COMMON_FIELDS = {
    "item_id",
    "kind",
    "origin_id",
    "logical_key",
    "source_revision",
    "content_sha256",
    "access_classification",
    "publication_state",
    "parent_id",
    "source_path",
    "source_url",
    "format",
}
KIND_FIELDS: dict[str, set[str]] = {
    "source": {"source_role", "source_details", "manifest_item_ids", "published_manifest_id"},
    "experiment": {
        "model",
        "hardware",
        "runtime",
        "workload",
        "configuration",
        "timestamp",
        "timestamp_kind",
        "source_pointer",
    },
    "measurement": {
        "experiment_id",
        "metric",
        "statistic",
        "value",
        "unit",
        "original_value",
        "original_unit",
        "normalized_value",
        "normalized_unit",
        "normalization_rule",
        "series",
        "source_pointer",
        "evidence_type",
    },
    "passage": {"passage_text", "heading_path", "source_pointer"},
    "figure": {"caption", "coverage", "missing_series", "source_pointer"},
    "figure_link": {"referenced_item_id", "source_pointer"},
}
REQUIRED_FIELDS: dict[str, set[str]] = {
    "source": {"source_role", "source_details"},
    "experiment": {"source_pointer"},
    "measurement": {"experiment_id", "metric", "value", "original_value", "source_pointer", "evidence_type"},
    "passage": {"passage_text", "heading_path", "source_pointer"},
    "figure": {"coverage", "source_pointer"},
    "figure_link": {"parent_id", "referenced_item_id", "source_pointer"},
}


class PublicBenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    item_id: str = Field(min_length=1, max_length=64)
    kind: ItemKind
    origin_id: str = Field(min_length=1)
    logical_key: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_classification: Literal["verified_public"]
    publication_state: PublicationState = "staged"
    parent_id: str | None = None
    source_path: str = Field(min_length=1)
    source_url: str | None = None
    format: str = Field(min_length=1)

    source_role: SourceRole | None = None
    source_details: dict[str, JsonValue] | None = None
    manifest_item_ids: list[str] | None = None
    published_manifest_id: str | None = None

    experiment_id: str | None = None
    model: str | None = None
    hardware: str | None = None
    runtime: str | None = None
    workload: dict[str, JsonValue] | None = None
    configuration: dict[str, JsonValue] | None = None
    timestamp: datetime | None = None
    timestamp_kind: Literal["experiment", "submission", "publication"] | None = None
    source_pointer: dict[str, JsonValue] | None = None

    metric: str | None = Field(default=None, min_length=1)
    statistic: str | None = Field(default=None, min_length=1)
    value: Decimal | None = None
    unit: str | None = Field(default=None, min_length=1)
    original_value: Decimal | None = None
    original_unit: str | None = Field(default=None, min_length=1)
    normalized_value: Decimal | None = None
    normalized_unit: str | None = None
    normalization_rule: str | None = None
    series: dict[str, JsonValue] | None = None
    evidence_type: Literal["raw_json", "published_table"] | None = None

    passage_text: str | None = Field(default=None, min_length=1)
    heading_path: list[str] | None = None
    caption: str | None = None
    coverage: Coverage | None = None
    missing_series: list[str] | None = None
    referenced_item_id: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        present = {name for name in type(self).model_fields if getattr(self, name) is not None}
        forbidden = present - _COMMON_FIELDS - KIND_FIELDS[self.kind]
        missing = REQUIRED_FIELDS[self.kind] - present
        if forbidden or missing:
            raise ValueError(f"{self.kind}: forbidden fields {sorted(forbidden)}; missing fields {sorted(missing)}")
        if self.timestamp is not None and self.timestamp.utcoffset() is None:
            raise ValueError("timestamp requires a timezone")
        if (self.timestamp is None) != (self.timestamp_kind is None):
            raise ValueError("timestamp and timestamp_kind must be supplied together")
        if self.item_id in (self.parent_id, self.experiment_id, self.referenced_item_id, self.published_manifest_id):
            raise ValueError("an item cannot reference itself")
        if self.source_role == "manifest":
            if self.manifest_item_ids is None or len(set(self.manifest_item_ids)) != len(self.manifest_item_ids):
                raise ValueError("a manifest requires unique versioned item IDs")
        elif self.manifest_item_ids is not None:
            raise ValueError("only a manifest can carry membership")
        if self.published_manifest_id is not None and self.source_role != "registry":
            raise ValueError("only a registry can point to a published manifest")
        if self.kind == "measurement":
            if self.value != self.original_value or self.unit != self.original_unit:
                raise ValueError("value/unit must preserve the original; use normalized fields for conversions")
            normalized = (self.normalized_value, self.normalized_unit, self.normalization_rule)
            if any(part is not None for part in normalized) and (
                self.value is None or self.unit is None or normalized != normalize_unit(self.value, self.unit)
            ):
                raise ValueError("normalization requires an explicit supported unit conversion")
        return self


# Denominators remain distinct: output tokens/s is not total tokens/s.
_UNIT_RULES: dict[str, tuple[Decimal, str]] = {
    "s": (Decimal("1000"), "ms"),
    "ms": (Decimal("1"), "ms"),
    "us": (Decimal("0.001"), "ms"),
    "s/token": (Decimal("1000"), "ms/token"),
    "ms/token": (Decimal("1"), "ms/token"),
    "output tokens/s": (Decimal("1"), "output tokens/s"),
    "tokens/s": (Decimal("1"), "tokens/s"),
    "requests/s": (Decimal("1"), "requests/s"),
}
_STATISTIC_ALIASES = {"avg": "mean", "median": "p50"}


def normalize_unit(value: Decimal, unit: str) -> tuple[Decimal, str, str] | None:
    if not value.is_finite():
        raise ValueError("measurement must be finite")
    rule = _UNIT_RULES.get(unit)
    if rule is None:
        return None
    factor, normalized_unit = rule
    with localcontext() as context:
        context.prec = max(context.prec, len(value.as_tuple().digits) + len(factor.as_tuple().digits))
        normalized = value * factor
    return normalized, normalized_unit, f"unit-v1:{unit}->{normalized_unit}"


def normalize_statistic(statistic: str | None) -> str | None:
    """Map only explicit aliases; missing and unknown statistics remain unchanged."""
    if statistic is not None and re.fullmatch(r"p[0-9]+p[0-9]+", statistic):
        return "p" + statistic[1:].replace("p", ".")
    return _STATISTIC_ALIASES.get(statistic, statistic) if statistic is not None else None


def statistic_aliases(statistic: str) -> tuple[str, ...]:
    canonical = normalize_statistic(statistic)
    assert canonical is not None
    aliases = [canonical, *(alias for alias, value in _STATISTIC_ALIASES.items() if value == canonical)]
    if re.fullmatch(r"p[0-9]+\.[0-9]+", canonical):
        aliases.append(canonical.replace(".", "p"))
    return tuple(aliases)


def effective_statistic(statistic: str | None) -> str:
    """Apply the query/planning default without rewriting the source measurement."""
    return "mean" if statistic is None else normalize_statistic(statistic) or statistic
