"""Versioned user requirements, without inferred measurements or device variants."""

from __future__ import annotations

import re
from dataclasses import replace
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from llm_d_lens.hardware.registry import all_profiles

if TYPE_CHECKING:
    from .planner import PlanningFacts


class RequirementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(RequirementModel):
    source: Literal["form", "text", "verified_discovery", "inference"]
    field: str
    raw: str | None = None
    start: int | None = None
    end: int | None = None


class NumericRequirement(RequirementModel):
    field: Literal["input_tokens", "output_tokens", "concurrency", "request_rate", "ttft", "tpot", "e2e"]
    statistic: Literal["unspecified", "mean", "p50", "p90", "p95", "p99"] = Field(
        default="unspecified", validate_default=True, description="Latency requirements default to mean."
    )
    operator: Literal["<", "<=", "=", ">=", ">"] = "="
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["tokens", "requests", "requests/s", "ms"]
    provenance: Provenance

    @field_validator("statistic")
    @classmethod
    def default_latency_statistic(
        cls, value: Literal["unspecified", "mean", "p50", "p90", "p95", "p99"], info: ValidationInfo
    ) -> Literal["unspecified", "mean", "p50", "p90", "p95", "p99"]:
        if value == "unspecified" and info.data.get("field") in {"ttft", "tpot", "e2e"}:
            return "mean"
        return value

    @model_validator(mode="after")
    def validate_unit(self) -> NumericRequirement:
        expected = {
            "input_tokens": "tokens",
            "output_tokens": "tokens",
            "concurrency": "requests",
            "request_rate": "requests/s",
            "ttft": "ms",
            "tpot": "ms",
            "e2e": "ms",
        }[self.field]
        if self.unit != expected:
            raise ValueError(f"{self.field}: expected {expected}")
        return self


class SoftPreference(RequirementModel):
    preference: Literal["latency", "throughput", "cost"]
    order: int = Field(ge=0)
    weight: float | None = Field(default=None, ge=0, le=1)
    provenance: Provenance


class HardwareIdentity(RequirementModel):
    profile_id: str | None = None
    device_model: str | None = None
    memory_gib: float | None = Field(default=None, gt=0)
    memory_unit: Literal["GB", "GiB"] | None = None
    form_factor: str | None = None
    device_count: int | None = Field(default=None, ge=1)
    interconnect: str | None = None
    raw: str | None = None


class ModelIdentity(RequirementModel):
    model_id: str | None = None
    revision: str | None = None
    architecture: Literal["dense", "moe"] | None = None
    quantization: str | None = None


class RuntimeIdentity(RequirementModel):
    backend: str | None = None
    backend_version: str | None = None
    accelerator_runtime: str | None = None
    driver_version: str | None = None
    container_digest: str | None = None


class RequirementIssue(RequirementModel):
    field: str
    message: str
    sources: tuple[Provenance, ...] = ()


class PlanningRequirements(RequirementModel):
    schema_version: Literal["1"] = "1"
    hardware: HardwareIdentity = Field(default_factory=HardwareIdentity)
    model: ModelIdentity = Field(default_factory=ModelIdentity)
    runtime: RuntimeIdentity = Field(default_factory=RuntimeIdentity)
    workload: tuple[NumericRequirement, ...] = ()
    hard_constraints: tuple[NumericRequirement, ...] = ()
    performance_targets: tuple[NumericRequirement, ...] = ()
    soft_preferences: tuple[SoftPreference, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    conflicts: tuple[RequirementIssue, ...] = ()
    errors: tuple[RequirementIssue, ...] = ()

    def require_valid(self) -> PlanningRequirements:
        if self.conflicts or self.errors:
            raise ValueError("; ".join(f"{issue.field}: {issue.message}" for issue in (*self.conflicts, *self.errors)))
        return self


def normalize_hardware(value: str | None, *, device_count: int | None = None) -> HardwareIdentity:
    """Split explicit count, memory and form factor; never infer missing variants."""
    if not value:
        return HardwareIdentity(device_count=device_count)
    raw = value.strip()
    text = raw.upper().replace("_", " ")
    if re.search(r"[,;+/]|(?:\sAND\s)|[\u3001\uff0c]", text):
        return HardwareIdentity(raw=raw, device_count=device_count)
    count = re.match(r"^\s*(\d+)\s*[X×*]\s*", text)
    if count:
        parsed_count = int(count.group(1))
        if device_count is not None and device_count != parsed_count:
            raise ValueError("hardware.device_count: conflicting explicit counts")
        device_count = parsed_count
        text = text[count.end() :]
    memory = re.search(r"(\d+(?:\.\d+)?)\s*(GIB|GB)\b", text)
    memory_gib = None
    memory_unit = None
    if memory:
        memory_unit = "GiB" if memory.group(2) == "GIB" else "GB"
        memory_gib = float(memory.group(1)) * (1 if memory_unit == "GiB" else 10**9 / 2**30)
        text = text[: memory.start()] + text[memory.end() :]
    factor = re.search(r"\b(PCIE|PCI-E|SXM\d*)\b", text)
    form_factor = factor.group(1).replace("PCI-E", "PCIE") if factor else None
    if factor:
        text = text[: factor.start()] + text[factor.end() :]
    profiles = [
        profile
        for profile in all_profiles()
        if any(re.search(pattern, raw, re.IGNORECASE) for pattern in profile.planning.aic_system_patterns)
    ]
    for profile in profiles:
        text = re.sub(rf"\b{re.escape(profile.vendor.upper())}\b", "", text)
    device_model = re.sub(r"[\s-]+", " ", text).strip() or None
    if device_model and device_model.lower() in {"unknown", "unavailable", "cluster", "none"}:
        device_model = None
    profile_ids = {profile.id for profile in profiles}
    return HardwareIdentity(
        profile_id=next(iter(profile_ids)) if len(profile_ids) == 1 else None,
        device_model=device_model,
        memory_gib=memory_gib,
        memory_unit=memory_unit,
        form_factor=form_factor,
        device_count=device_count,
        raw=raw,
    )


def enrich_planning_requirements(
    facts: PlanningFacts,
    *,
    model_name: str,
    hardware_model: str | None,
    profile_id: str | None = None,
    device_count: int | None = None,
    model_config: dict[str, Any] | None = None,
) -> PlanningFacts:
    """Fill verified identity gaps without replacing user requirements or inventing runtime facts."""
    from .planner import normalize_planning_facts

    normalized = normalize_planning_facts(facts, facts.operator_preference)
    requirements = normalized.requirements
    assert requirements is not None
    hardware = normalize_hardware(hardware_model, device_count=device_count)
    if profile_id and hardware.profile_id and profile_id != hardware.profile_id:
        raise ValueError("hardware.profile_id: conflicting discovery identity")
    hardware_values = hardware.model_dump(exclude={"raw"})
    hardware_values["profile_id"] = profile_id or hardware.profile_id
    if hardware_model and hardware.memory_gib is None and facts.vram_per_gpu_gib > 0:
        hardware_values.update(memory_gib=facts.vram_per_gpu_gib, memory_unit="GiB")
    config = model_config or {}
    architecture = config.get("architecture")
    if not isinstance(architecture, str) or architecture not in {"dense", "moe"}:
        architectures = config.get("architectures")
        architecture = (
            "moe"
            if (
                isinstance(architectures, list)
                and any(isinstance(name, str) and "moe" in name.lower() for name in architectures)
            )
            or any(
                isinstance(config.get(key), (int, float)) and not isinstance(config.get(key), bool) and config[key] > 0
                for key in ("num_local_experts", "num_experts", "n_routed_experts")
            )
            else None
        )
    quantization = config.get("quantization")
    if not isinstance(quantization, str):
        quantization_config = config.get("quantization_config")
        quantization = quantization_config.get("quant_method") if isinstance(quantization_config, dict) else None
    revision = config.get("_commit_hash") or config.get("revision")
    model_values = ModelIdentity(
        model_id=model_name.strip() or None,
        architecture=architecture,
        quantization=quantization if isinstance(quantization, str) else None,
        revision=revision if isinstance(revision, str) else None,
    ).model_dump()
    provenance = list(requirements.provenance)
    conflicts = list(requirements.conflicts)

    def merge(identity: RequirementModel, values: dict[str, Any], prefix: str) -> dict[str, Any]:
        merged = identity.model_dump()
        for field, value in values.items():
            if value is None:
                continue
            source = Provenance(source="verified_discovery", field=f"{prefix}.{field}", raw=str(value))
            old_value = merged.get(field)
            if old_value is not None and old_value != value:
                original = tuple(item for item in provenance if item.field == source.field)
                issue = RequirementIssue(
                    field=source.field,
                    message="Explicit requirement conflicts with verified discovery",
                    sources=(*original, source)
                    if original
                    else (
                        Provenance(source="form", field=source.field, raw=str(old_value)),
                        source,
                    ),
                )
                if issue not in conflicts:
                    conflicts.append(issue)
            else:
                merged[field] = value
            if source not in provenance:
                provenance.append(source)
        return merged

    merged_hardware = merge(requirements.hardware, hardware_values, "hardware")
    merged_hardware["raw"] = requirements.hardware.raw or hardware.raw
    enriched = requirements.model_copy(
        update={
            "hardware": HardwareIdentity.model_validate(merged_hardware),
            "model": ModelIdentity.model_validate(merge(requirements.model, model_values, "model")),
            "provenance": tuple(provenance),
            "conflicts": tuple(conflicts),
        }
    )
    enriched.require_valid()
    return replace(normalized, requirements=enriched)


_FIELDS = {
    "isl": ("input_tokens", "tokens"),
    "input tokens": ("input_tokens", "tokens"),
    "\u8f93\u5165\u957f\u5ea6": ("input_tokens", "tokens"),
    "osl": ("output_tokens", "tokens"),
    "output tokens": ("output_tokens", "tokens"),
    "\u8f93\u51fa\u957f\u5ea6": ("output_tokens", "tokens"),
    "concurrency": ("concurrency", "requests"),
    "\u5e76\u53d1": ("concurrency", "requests"),
    "\u5e76\u53d1\u6570": ("concurrency", "requests"),
    "\u5e76\u53d1\u7528\u6237\u6570": ("concurrency", "requests"),
    "qps": ("request_rate", "requests/s"),
    "request rate": ("request_rate", "requests/s"),
    "ttft": ("ttft", "ms"),
    "\u9996\u5b57\u5ef6\u8fdf": ("ttft", "ms"),
    "tpot": ("tpot", "ms"),
    "e2e": ("e2e", "ms"),
}
_STAT = r"p(?:50|90|95|99)|mean|average|\u5e73\u5747"
_OP = (
    r"less than or equal to|at most|no more than|less than|under|below|at least|more than|<=|>=|<|>|=|"
    r"\u2264|\u2265|\u4e0d\u8d85\u8fc7|\u4e0d\u9ad8\u4e8e|\u5c0f\u4e8e|\u81f3\u5c11|\u5927\u4e8e"
)
_OPERATORS = {
    "less than": "<",
    "under": "<",
    "below": "<",
    "\u5c0f\u4e8e": "<",
    "less than or equal to": "<=",
    "at most": "<=",
    "no more than": "<=",
    "\u4e0d\u8d85\u8fc7": "<=",
    "\u4e0d\u9ad8\u4e8e": "<=",
    "\u2264": "<=",
    "at least": ">=",
    "\u81f3\u5c11": ">=",
    "\u2265": ">=",
    "more than": ">",
    "\u5927\u4e8e": ">",
}
_PATTERN = re.compile(
    rf"(?<![a-z])(?:(?P<before>{_STAT})\s*)?"
    rf"(?P<field>{'|'.join(map(re.escape, sorted(_FIELDS, key=len, reverse=True)))})(?![a-z])"
    rf"\s*(?:(?P<after>{_STAT})\s*)?(?:of\s*|[:\uff1a]\s*)?"
    rf"(?P<op>{_OP})?\s*(?P<value>-?(?:\d+(?:\.\d+)?|\.\d+))"
    r"\s*(?P<unit>(?:(?:milliseconds?|seconds?|msec|ms|secs?|s)/tokens?|"
    r"milliseconds?|seconds?|msec|ms|secs?|s|tokens?|requests?/s|qps|requests?|users?|"
    r"\u6beb\u79d2|\u79d2|\u4e2a)(?![^\W\d_]|/))?",
    re.IGNORECASE,
)
_PREFERENCE_PATTERN = (
    r"(?:prefer|prioritize)\s+(?:latency|throughput|cost)|"
    r"low latency|latency first|\u4f4e\u5ef6\u8fdf|high throughput|\u9ad8\u541e\u5410|"
    r"low cost|\u4f4e\u6210\u672c"
)


def normalize_requirements(
    text: str | None = None,
    *,
    form: dict[str, object] | None = None,
    hardware: HardwareIdentity | None = None,
    model: ModelIdentity | None = None,
    runtime: RuntimeIdentity | None = None,
    provenance: tuple[Provenance, ...] = (),
) -> PlanningRequirements:
    """Parse common numeric statements, retaining errors and conflicting sources."""
    numeric: list[NumericRequirement] = []
    errors: list[RequirementIssue] = []
    form = form or {}
    form_fields = {
        "mean_input_tokens": ("input_tokens", "mean", "tokens"),
        "p95_input_tokens": ("input_tokens", "p95", "tokens"),
        "mean_output_tokens": ("output_tokens", "mean", "tokens"),
        "concurrency": ("concurrency", "unspecified", "requests"),
        "request_rate": ("request_rate", "unspecified", "requests/s"),
        "ttft_slo_ms": ("ttft", "unspecified", "ms"),
        "tpot_slo_ms": ("tpot", "unspecified", "ms"),
        "end_to_end_latency_slo_ms": ("e2e", "unspecified", "ms"),
    }
    text = text or ""
    for match in _PATTERN.finditer(text):
        field, canonical_unit = _FIELDS[match["field"].lower()]
        text_provenance = Provenance(
            source="text",
            field=field,
            raw=match.group(),
            start=match.start(),
            end=match.end(),
        )
        unit = (match["unit"] or "").lower()
        per_token = unit.endswith(("/token", "/tokens"))
        if per_token and field == "tpot":
            unit = unit.split("/", 1)[0]
        value = float(match["value"])
        suffix = text[match.end() :]
        # A trailing unknown unit must not silently become milliseconds/tokens.
        bad_unit = re.match(r"\s*([^\W\d_]+|/)", suffix) if not unit else None
        continuation = _PATTERN.match(suffix.lstrip()) or re.match(
            _PREFERENCE_PATTERN,
            suffix.lstrip(),
            re.IGNORECASE,
        )
        invalid_suffix = suffix.startswith("/") or (
            bad_unit
            and not continuation
            and bad_unit.group(1).lower()
            not in {
                "and",
                "or",
                "with",
                "under",
                "for",
                "requiring",
            }
        )
        valid_units = {
            "ms": {
                "",
                "ms",
                "msec",
                "millisecond",
                "milliseconds",
                "s",
                "sec",
                "secs",
                "second",
                "seconds",
                "\u79d2",
                "\u6beb\u79d2",
            },
            "tokens": {"", "token", "tokens"},
            "requests": {"", "requests", "request", "users", "user", "\u4e2a"},
            "requests/s": {"", "qps", "request/s", "requests/s"},
        }
        if unit not in valid_units[canonical_unit] or invalid_suffix or value <= 0 or not isfinite(value):
            errors.append(RequirementIssue(field=field, message="Invalid value or unit", sources=(text_provenance,)))
            continue
        if canonical_unit == "ms" and unit in {"s", "sec", "secs", "second", "seconds", "\u79d2"}:
            value *= 1000
        statistic = (match["before"] or match["after"] or "unspecified").lower()
        statistic = "mean" if statistic in {"average", "\u5e73\u5747"} else statistic
        operator = (match["op"] or ("<=" if canonical_unit == "ms" else "=")).lower()
        numeric.append(
            NumericRequirement(
                field=field,
                statistic=statistic,
                operator=_OPERATORS.get(operator, operator),
                value=value,
                unit=canonical_unit,
                provenance=text_provenance,
            )
        )
    concurrent_pattern = rf"(?:(?P<op>{_OP})\s+)?(?P<value>\d+)\s+concurrent\s+(?:users?|requests?)\b"
    for match in re.finditer(concurrent_pattern, text, re.IGNORECASE):
        if float(match["value"]) <= 0:
            errors.append(RequirementIssue(field="concurrency", message="Expected a positive value"))
            continue
        operator = (match["op"] or "=").lower()
        numeric.append(
            NumericRequirement(
                field="concurrency",
                value=float(match["value"]),
                unit="requests",
                operator=_OPERATORS.get(operator, operator),
                provenance=Provenance(
                    source="text",
                    field="concurrency",
                    raw=match.group(),
                    start=match.start(),
                    end=match.end(),
                ),
            )
        )
    form_numeric: list[NumericRequirement] = []
    for key, (field, statistic, unit) in form_fields.items():
        value = form.get(key)
        if value is not None:
            if unit == "ms":
                text_statistics = {item.statistic for item in numeric if item.field == field}
                if len(text_statistics) == 1:
                    statistic = next(iter(text_statistics))
            try:
                form_numeric.append(
                    NumericRequirement(
                        field=field,
                        statistic=statistic,
                        unit=unit,
                        value=value,
                        operator="<=" if unit == "ms" else "=",
                        provenance=Provenance(source="form", field=key, raw=str(value)),
                    )
                )
            except ValueError:
                errors.append(RequirementIssue(field=key, message="Expected a positive finite numeric value"))
    numeric = form_numeric + numeric
    conflicts: list[RequirementIssue] = []
    for index, left in enumerate(numeric):
        for right in numeric[index + 1 :]:
            if left.field != right.field:
                continue
            same_statistic = left.statistic == right.statistic or (
                {left.statistic, right.statistic} <= {"unspecified", "mean"} and left.unit != "ms"
            )
            if not same_statistic:
                continue
            # Distinct form/text targets require confirmation even if both are upper bounds.
            cross_source = left.provenance.source != right.provenance.source
            incompatible = left.value != right.value or (
                left.operator != right.operator and {left.operator, right.operator} != {"=", "<="}
            )
            disjoint = _disjoint(left, right)
            if (incompatible and (cross_source or left.operator == right.operator == "=")) or disjoint:
                conflicts.append(
                    RequirementIssue(
                        field=left.field,
                        message="Conflicting explicit requirements",
                        sources=(left.provenance, right.provenance),
                    )
                )
    preferences = []
    for match in re.finditer(_PREFERENCE_PATTERN, text, re.IGNORECASE):
        phrase = match.group().lower()
        preference = (
            "latency"
            if "latency" in phrase or phrase == "\u4f4e\u5ef6\u8fdf"
            else ("throughput" if "throughput" in phrase or phrase == "\u9ad8\u541e\u5410" else "cost")
        )
        preferences.append(
            SoftPreference(
                preference=preference,
                order=len(preferences),
                weight=_preference_weight(text[match.end() :]),
                provenance=Provenance(
                    source="text",
                    field="soft_preferences",
                    raw=match.group(),
                    start=match.start(),
                    end=match.end(),
                ),
            )
        )
    workload = tuple(item for item in numeric if item.unit != "ms")
    targets = tuple(item for item in numeric if item.unit == "ms")
    return PlanningRequirements(
        hardware=hardware or HardwareIdentity(),
        model=model or ModelIdentity(),
        runtime=runtime or RuntimeIdentity(),
        workload=workload,
        performance_targets=targets,
        hard_constraints=tuple(numeric),
        soft_preferences=tuple(preferences),
        provenance=(*provenance, *(item.provenance for item in numeric)),
        errors=tuple(errors),
        conflicts=tuple(conflicts),
    )


def _preference_weight(suffix: str) -> float | None:
    match = re.match(
        r"\s*(?:weight|[:=]|\u6743\u91cd)\s*[:=]?\s*(\d+(?:\.\d+)?%|0(?:\.\d+)?|1(?:\.0+)?)",
        suffix,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match[1]
    return float(value[:-1]) / 100 if value.endswith("%") else float(value)


def _disjoint(left: NumericRequirement, right: NumericRequirement) -> bool:
    if left.operator == "=":
        return not {
            "<": left.value < right.value,
            "<=": left.value <= right.value,
            "=": left.value == right.value,
            ">": left.value > right.value,
            ">=": left.value >= right.value,
        }[right.operator]
    if right.operator == "=":
        return _disjoint(right, left)
    if left.operator in {">", ">="}:
        left, right = right, left
    return (
        left.operator in {"<", "<="}
        and right.operator in {">", ">="}
        and (
            right.value > left.value or (right.value == left.value and (left.operator == "<" or right.operator == ">"))
        )
    )
