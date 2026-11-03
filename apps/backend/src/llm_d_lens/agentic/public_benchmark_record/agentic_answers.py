"""Caller-authorized Agentic envelopes and conservative source-identity merging."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from uuid import UUID

from llm_d_lens.agentic.public_benchmark_record.agentic_metrics import measured_values
from llm_d_lens.agentic.public_benchmark_record.contracts import model_identities_match
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    AgenticCitation,
    Citation,
    ExperimentContext,
    QueryEvidence,
    QueryFilters,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN
from llm_d_lens.db.dao.public_benchmark_record_query import QueryMatch

if TYPE_CHECKING:
    from llm_d_lens.db.dao.benchmark_experiment_query import AgenticMatch


def agentic_evidence(match: AgenticMatch, filters: QueryFilters) -> tuple[QueryEvidence, ...]:
    record = match.record
    checksum = hashlib.sha256(
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    context = ExperimentContext(
        item_id=record.benchmark_id,
        origin_id="agentic:evaluate",
        model=record.model.model_id,
        hardware=record.hardware.accelerator,
        runtime=record.runtime.backend,
        workload=record.workload.model_dump(mode="json"),
        configuration={
            **record.configuration.model_dump(mode="json"),
            "model_descriptor": record.model.model_dump(mode="json"),
            "hardware_descriptor": record.hardware.model_dump(mode="json"),
            "runtime_fingerprint": record.runtime.model_dump(mode="json"),
        },
        source_pointer={"benchmark_id": record.benchmark_id, "evaluate_run_id": match.evaluate_run_id},
    )
    return tuple(
        QueryEvidence(
            item_id=f"agentic:{record.benchmark_id}:{spec.field}",
            origin="agentic",
            evidence_type="agentic_measurement",
            verified_evaluate_run_id=record.benchmark_id,
            experiment_time=record.timestamp,
            citation=AgenticCitation(
                benchmark_id=record.benchmark_id,
                evaluate_run_id=match.evaluate_run_id,
                cluster_id=match.cluster_id,
                resource_uri=f"/api/v1/evaluate/runs/{match.evaluate_run_id}",
                record_sha256=checksum,
                metric_field=spec.field,
                record_timestamp=record.timestamp,
            ),
            experiment=context,
            metric=spec.metric,
            statistic=spec.statistic,
            value=value,
            unit=spec.unit,
            original_value=value,
            original_unit=spec.unit,
            normalized_value=value if spec.normalized else None,
            normalized_unit=spec.unit if spec.normalized else None,
            normalization_rule=f"agentic-stored-float-v1:{spec.unit}" if spec.normalized else None,
        )
        for spec, value in measured_values(record, filters)
    )


def pinned_evaluate_identity(match: QueryMatch, data: bytes) -> str | None:
    """A source-scoped name alone is not proof: check its retained JSON object."""
    experiment = match.experiment
    if match.item.origin_id != PRISM_ORIGIN or match.item.evidence_type != "raw_json" or experiment is None:
        return None
    pointer = experiment.source_pointer or {}
    identity = pointer.get("record_id")
    path = pointer.get("json_pointer")
    if not isinstance(identity, str) or not identity.startswith("evaluate:") or not isinstance(path, str):
        return None
    identifier = identity.removeprefix("evaluate:")
    try:
        if str(UUID(identifier)) != identifier:
            return None
    except ValueError:
        return None
    value = json.loads(data)
    if path:
        if not path.startswith("/"):
            return None
        for segment in path[1:].split("/"):
            key = segment.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdecimal() and int(key) < len(value):
                value = value[int(key)]
            else:
                return None
    return identity if isinstance(value, dict) and value.get("id") == identity else None


def merge_verified(
    public: list[QueryEvidence],
    internal: list[QueryEvidence],
    identities: dict[str, str],
) -> list[QueryEvidence]:
    remaining = list(internal)
    result = []
    for entry in public:
        identity = identities.get(entry.item_id)
        matched = []
        for candidate in remaining:
            citation = candidate.citation
            if (
                identity is not None
                and isinstance(entry.citation, Citation)
                and isinstance(citation, AgenticCitation)
                and citation.benchmark_id == identity
                and entry.metric == candidate.metric
                and entry.statistic == candidate.statistic
                and entry.unit == candidate.unit
                and entry.value == candidate.value
                and not entry.series
                and entry.experiment is not None
                and candidate.experiment is not None
                and model_identities_match(entry.experiment.model, candidate.experiment.model)
                and entry.experiment.hardware == candidate.experiment.hardware
                and entry.experiment.runtime == candidate.experiment.runtime
                and all(
                    candidate.experiment.workload is not None and candidate.experiment.workload.get(key) == value
                    for key, value in (entry.experiment.workload or {}).items()
                )
            ):
                matched.append(candidate)
        if len(matched) == 1:
            candidate = matched[0]
            remaining.remove(candidate)
            entry = entry.model_copy(
                update={
                    "origin": "public_and_agentic",
                    "corroborating_citations": (candidate.citation,),
                    "identity_verification": "canonical_evaluate_run_and_metric",
                }
            )
        result.append(entry)
    return result + remaining
