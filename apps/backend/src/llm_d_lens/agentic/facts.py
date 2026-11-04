"""Resolve and freeze planning evidence owned by existing Prism domains."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from dataclasses import asdict, replace
from math import isfinite
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from llm_d_lens.agentic.public_benchmark_record.access import require_query_access
from llm_d_lens.agentic.public_benchmark_record.contracts import (
    WORKLOAD_ALIASES,
    effective_statistic,
    experiment_rank_key,
    experiment_ranking,
    latency_axes,
    latency_value_ms,
    model_identity_aliases,
    planning_latency_axes,
)
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationError
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryFilters, QueryRequest
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService
from llm_d_lens.aic.models import DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS, AICRequest
from llm_d_lens.aic.service import AICError, check_support, search
from llm_d_lens.auth.contracts import Principal
from llm_d_lens.capacity import load_model_config
from llm_d_lens.cluster import registry
from llm_d_lens.cluster.errors import ClusterOverviewError
from llm_d_lens.cluster.service import build_overview
from llm_d_lens.core.exceptions import (
    DomainError,
    ForbiddenError,
    PasswordChangeRequiredError,
    UnauthenticatedError,
)
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao
from llm_d_lens.hardware.models import HardwareProfile
from llm_d_lens.hardware.registry import all_profiles

from .planner import AICCandidatePrediction, PlanningFacts, public_rag_workload, resolve_workload_signals
from .planning_requirements import NumericRequirement, enrich_planning_requirements

_GIB = 1024**3
logger = logging.getLogger(__name__)


class PlanningEvidence(BaseModel):
    id: str
    source: str
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class ResolvedPlanningFacts(BaseModel):
    facts: PlanningFacts
    cluster_snapshot: dict[str, Any]
    evidence: list[PlanningEvidence] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class PlanningHardwareSnapshot:
    """One discovery attempt shared only by a single create/refine operation."""

    def __init__(self, cluster_id: str) -> None:
        self.cluster_id = cluster_id
        self._task: asyncio.Task[dict[str, Any]] | None = None

    async def get(self, cluster_id: str) -> dict[str, Any]:
        if cluster_id != self.cluster_id:
            raise ValueError("Planning hardware snapshot belongs to a different cluster")
        if self._task is None:
            self._task = asyncio.create_task(build_overview(registry.require_cluster(cluster_id)))
        if self._task.cancelled():
            raise TimeoutError("Planning hardware discovery was cancelled; no snapshot is available")
        return deepcopy(await self._task)


def _hardware_model_key(model: str, profile: HardwareProfile) -> tuple[str, ...]:
    decoration = {
        token
        for label in (profile.vendor, *profile.accelerator_aliases)
        for token in re.findall(r"[a-z0-9]+", label.lower())
    }
    return tuple(token for token in re.findall(r"[a-z0-9]+", model.lower()) if token not in decoration)


def _public_rag_hardware(overview: dict[str, Any]) -> tuple[HardwareProfile, str] | None:
    accelerators = [
        item for item in (overview.get("hardware") or {}).get("accelerators", []) if item.get("gpuCount", 0) > 0
    ]
    if len(accelerators) != 1:
        return None
    accelerator = accelerators[0]
    profile = next((profile for profile in all_profiles() if profile.id == accelerator.get("id")), None)
    if profile is None:
        return None
    models = {model.strip() for model in accelerator.get("models") or [] if isinstance(model, str) and model.strip()}
    families = {name.lower() for name in profile.planning.model_family_names}
    models = {model for model in models if model.lower() not in families}
    keys = {_hardware_model_key(model, profile) for model in models}
    if len(keys) != 1 or not next(iter(keys)):
        return None
    return profile, sorted(models)[0]


def _reported_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value.replace(",", "") if isinstance(value, str) else value)
    except (ValueError, OverflowError):
        return None
    return number if isfinite(number) and number >= 0 else None


def _public_workload_match(item: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    experiment = item.get("experiment") or {}
    workload = experiment.get("workload") or {}
    configuration = experiment.get("configuration") or {}
    fields = {}
    for name, target in requested.items():
        for key in WORKLOAD_ALIASES.get(name, (name,)):
            value = _reported_number(workload.get(key))
            if value is not None:
                fields[name] = {"requested": target, "reported": value, "basis": f"experiment.workload.{key}"}
                break
    # These explicitly labelled prompt shapes provide nominal lengths, never measured means or P95s.
    report = configuration.get("published_report") or ""
    shape = re.search(
        r"([\d,]+)-token shared (?:system prompt|prefix)\s*\+\s*"
        r"([\d,]+)-token question\s*,\s*([\d,]+)-token output",
        report,
        re.IGNORECASE,
    )
    if shape:
        prefix, question, output = (int(value.replace(",", "")) for value in shape.groups())
        for name, value in (("mean_input_tokens", prefix + question), ("mean_output_tokens", output)):
            if name in requested and name not in fields:
                fields[name] = {
                    "requested": requested[name],
                    "reported": value,
                    "basis": "published_report.nominal_length_proxy_not_measured_mean",
                }
    for name, comparison in fields.items():
        target, value = comparison["requested"], comparison["reported"]
        comparison["difference"] = value - target
        comparison["relative_difference"] = (value - target) / target if target else None
        comparison["similarity"] = (
            1 - abs(value - target)
            if name == "shared_prefix_ratio"
            else min(value, target) / max(value, target)
            if value or target
            else 1.0
        )
    return {
        "requested": requested,
        "fields": fields,
        "unknown_fields": [name for name in requested if name not in fields],
        "similarity": sum(field["similarity"] for field in fields.values()) / len(fields) if fields else None,
        "coverage": len(fields) / len(requested) if requested else None,
        "reported_workload": workload or None,
        "nominal_prompt_shape": (
            {"shared_prefix_tokens": prefix, "question_tokens": question, "output_tokens": output} if shape else None
        ),
    }


def _public_latency_value(item: dict[str, Any]) -> float | None:
    return latency_value_ms(item)


def rank_public_rag_evidence(
    observations: list[dict[str, Any]], facts: PlanningFacts | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank model-alias/hardware matches without discarding unknown workload or latency."""
    workload = public_rag_workload(facts) if facts else {}
    targets = {
        metric: target
        for metric, target in (
            ("ttft", facts.ttft_slo_ms if facts else None),
            ("tpot", facts.tpot_slo_ms if facts else None),
            ("latency", facts.end_to_end_latency_slo_ms if facts else None),
        )
        if target is not None
    }
    typed_targets: dict[str, list[NumericRequirement]] = {}
    if facts and facts.requirements:
        for requirement in facts.requirements.performance_targets:
            metric = "latency" if requirement.field == "e2e" else requirement.field
            typed_targets.setdefault(metric, []).append(requirement)
            targets.setdefault(metric, requirement.value)
    groups: dict[str, list[dict[str, Any]]] = {}
    scopes = {}
    for item in observations:
        experiment = item.get("experiment") or {}
        citation = item.get("citation") or {}
        scope = json.dumps(
            [
                citation.get("origin_id"),
                citation.get("sha256"),
                experiment.get("item_id") or item["item_id"],
                experiment.get("configuration"),
                experiment.get("workload"),
                item.get("series"),
            ],
            sort_keys=True,
        )
        scopes[item["item_id"]] = scope
        groups.setdefault(scope, []).append(item)
    axes = planning_latency_axes(facts) if facts is not None else latency_axes(())
    profiles = []
    scope_keys = list(groups)
    for scope in scope_keys:
        peers = groups[scope]
        values = {}
        for metric, statistic in axes:
            numbers = [
                value
                for peer in peers
                if ("latency" if peer.get("metric") == "e2e" else peer.get("metric")) == metric
                and effective_statistic(peer.get("statistic")) == statistic
                and (value := _public_latency_value(peer)) is not None
            ]
            if numbers:
                # Conflicting copies must not cherry-pick the fastest reported value.
                values[f"{metric}:{statistic or 'unspecified'}"] = max(numbers)
        profiles.append({"workload": _public_workload_match(peers[0], workload), "latency_values_ms": values})
    rankings = dict(zip(scope_keys, experiment_ranking(profiles), strict=True))
    annotations = []
    for item in observations:
        experiment = item.get("experiment") or {}
        configuration = experiment.get("configuration") or {}
        latency_fields = {}
        for metric, target in targets.items():
            constraints = list(
                dict.fromkeys(
                    (effective_statistic(requirement.statistic), requirement.value, requirement.operator)
                    for requirement in typed_targets.get(metric, ())
                )
            ) or [(effective_statistic(None), target, "<=")]
            assessments = []
            for statistic, bound, operator in constraints:
                values = [
                    {
                        "evidence_id": f"public-rag:{peer['item_id']}",
                        "statistic": effective_statistic(peer.get("statistic")),
                        **(
                            {"statistic_original": None, "statistic_basis": "default_mean"}
                            if peer.get("statistic") is None
                            else {}
                        ),
                        "value_ms": value,
                    }
                    for peer in groups[scopes[item["item_id"]]]
                    if ("latency" if peer.get("metric") == "e2e" else peer.get("metric")) == metric
                    and effective_statistic(peer.get("statistic")) == statistic
                    and (value := _public_latency_value(peer)) is not None
                ]
                for value in values:
                    deviation = value["value_ms"] / bound - 1
                    value["relative_deviation"] = (
                        -deviation if operator in {">", ">="} else abs(deviation) if operator == "=" else deviation
                    )
                    value["target_satisfied"] = {
                        "<": value["value_ms"] < bound,
                        "<=": value["value_ms"] <= bound,
                        ">": value["value_ms"] > bound,
                        ">=": value["value_ms"] >= bound,
                        "=": value["value_ms"] == bound,
                    }[operator]
                assessments.append(
                    {"target_ms": bound, "operator": operator, "statistic": statistic, "observations": values}
                )
            latency_fields[metric] = {
                **assessments[-1],
                **({"targets": assessments} if len(assessments) > 1 else {}),
            }
        assessments = [assessment for field in latency_fields.values() for assessment in field.get("targets", [field])]
        deviations = [value["relative_deviation"] for assessment in assessments for value in assessment["observations"]]
        unknown_metrics = [
            metric
            for metric, field in latency_fields.items()
            if any(not assessment["observations"] for assessment in field.get("targets", [field]))
        ]
        status = (
            "not_requested"
            if not targets
            else "not_satisfied"
            if any(not value["target_satisfied"] for assessment in assessments for value in assessment["observations"])
            else "unknown"
            if unknown_metrics
            else "satisfied"
        )
        annotations.append(
            {
                "evidence_id": f"public-rag:{item['item_id']}",
                "workload": _public_workload_match(item, workload),
                "ranking": rankings[scopes[item["item_id"]]],
                "latency": {
                    "status": status,
                    "fields": latency_fields,
                    "unknown_metrics": unknown_metrics,
                    "relative_violation": max((max(0, value) for value in deviations), default=None),
                    "scope": "same published experiment and series; not current deployment SLO compliance",
                },
                "configuration_basis": {
                    "experiment_id": experiment.get("item_id"),
                    "runtime": experiment.get("runtime"),
                    "configuration": {key: value for key, value in configuration.items() if key != "published_report"},
                    "source_pointer": experiment.get("source_pointer"),
                },
            }
        )

    def rank(annotation: dict[str, Any]) -> tuple[Any, ...]:
        return (
            *experiment_rank_key(annotation["ranking"]),
            annotation["evidence_id"],
        )

    annotations.sort(key=rank)
    by_id = {f"public-rag:{item['item_id']}": item for item in observations}
    return [by_id[annotation["evidence_id"]] for annotation in annotations], annotations


async def resolve_retrieval_hardware(
    cluster_server_id: str,
    model: str,
    facts: PlanningFacts,
    principal: Principal | None,
    *,
    hardware_snapshot: PlanningHardwareSnapshot | None = None,
) -> tuple[PlanningFacts, dict[str, Any]]:
    """Share verified target identity before parallel SQL retrieval, without AIC or history reads."""
    try:
        require_query_access(principal)
        async with asyncio.timeout(15):
            overview = (
                await hardware_snapshot.get(cluster_server_id)
                if hardware_snapshot is not None
                else await build_overview(registry.require_cluster(cluster_server_id))
            )
        target = _public_rag_hardware(overview)
        if target is None:
            logger.warning("event=planning_retrieval_hardware reason=ambiguous_hardware")
            return facts, {"status": "unavailable", "reason": "ambiguous_hardware"}
        profile, hardware = target
        hardware_data = overview.get("hardware") or {}
        gpu_count = hardware_data.get("gpuCount")
        vram_bytes = hardware_data.get("vramBytes")
        verified_vram = (
            vram_bytes / gpu_count / _GIB
            if isinstance(gpu_count, (int, float))
            and gpu_count > 0
            and isinstance(vram_bytes, (int, float))
            and vram_bytes > 0
            else 0
        )
        enriched = enrich_planning_requirements(
            replace(facts, vram_per_gpu_gib=verified_vram),
            model_name=model,
            hardware_model=hardware,
            profile_id=profile.id,
            model_config=facts.model_config_dict,
        )
        return replace(enriched, vram_per_gpu_gib=facts.vram_per_gpu_gib), {
            "status": "available",
            "hardware": hardware,
        }
    except PasswordChangeRequiredError:
        raise
    except (DomainError, ClusterOverviewError, OSError, TimeoutError) as error:
        logger.warning("event=planning_retrieval_hardware reason=%s", type(error).__name__)
        return facts, {"status": "unavailable", "reason": type(error).__name__}


async def resolve_public_rag_evidence(
    model_name: str,
    principal: Principal | None,
    *,
    cluster_server_id: str,
    planning_facts: PlanningFacts | None = None,
    hardware_snapshot: PlanningHardwareSnapshot | None = None,
) -> PlanningEvidence:
    """Read only published public evidence under the actual caller's global grant."""
    query = QueryRequest(
        question="Published deployment benchmark evidence",
        intent="comparison",
        filters=QueryFilters(model=model_name),
        limit=50,
        include_internal=False,
    )
    target_hardware: dict[str, Any] = {"cluster_id": cluster_server_id}
    try:
        require_query_access(principal)
        async with asyncio.timeout(15):
            overview = (
                await hardware_snapshot.get(cluster_server_id)
                if hardware_snapshot is not None
                else await build_overview(registry.require_cluster(cluster_server_id))
            )
            target = _public_rag_hardware(overview)
            if target is None:
                message = (
                    "Published benchmark evidence was not queried: the target cluster does not identify one "
                    "unambiguous accelerator model. Continuing with existing planning evidence."
                )
                logger.warning("Agentic published benchmark hardware unavailable for cluster %s", cluster_server_id)
                return PlanningEvidence(
                    id="public-rag:query",
                    source="public-benchmark-rag",
                    status="unavailable",
                    details={"code": "rag_hardware_unresolved", "message": message, "target_hardware": target_hardware},
                )
            profile, hardware_model = target
            target_hardware.update(profile_id=profile.id, model=hardware_model)
            dao = PublicBenchmarkRecordQueryDao()
            dao.planning_facts = planning_facts
            options = await run_in_threadpool(dao.hardware_options, query.filters)
            aliases = [
                name
                for name in options
                if _hardware_model_key(name, profile) == _hardware_model_key(hardware_model, profile)
            ]
            # Keep exact published labels in the request and citation; never collapse device variants.
            hardware_filter = hardware_model if hardware_model in aliases else (sorted(aliases) or [hardware_model])[0]
            query = query.model_copy(update={"filters": query.filters.model_copy(update={"hardware": hardware_filter})})
            response = await run_in_threadpool(
                PublicBenchmarkRecordQueryService(dao=dao).query,
                query,
                principal,
                experiment_limit=50 if planning_facts is not None else None,
            )
    except (ForbiddenError, UnauthenticatedError) as error:
        status = "forbidden"
        message = (
            "Published benchmark evidence was not queried: authenticated global published benchmark "
            "read permission is required."
        )
        details = {"code": error.code}
    except (DomainError, PublicationError, SQLAlchemyError, OSError, TimeoutError) as error:
        status = "unavailable"
        message = (
            "Published benchmark retrieval failed; continuing with existing planning evidence and supplementary AIC."
        )
        details = {"code": error.code if isinstance(error, DomainError) else type(error).__name__}
        logger.warning("Agentic published benchmark retrieval unavailable: %s", details["code"])
    else:
        status = "available" if response.evidence else "no_matches"
        message = (
            f"Published benchmark SQL observations: retrieved {len(response.evidence)} "
            f"for {model_name} on {hardware_filter}; "
            "applicable evidence takes precedence over AIC estimates."
            if response.evidence
            else f"Published benchmark SQL observations: no matches for {model_name} on {hardware_filter}; "
            "no cross-hardware fallback. "
            "Continuing with existing planning evidence and supplementary AIC."
        )
        payload = response.model_dump(mode="json")
        ranked, matches = rank_public_rag_evidence(payload["evidence"], planning_facts)
        payload["evidence"] = ranked
        workload = public_rag_workload(planning_facts) if planning_facts else {}
        targets = {
            metric: target
            for metric, target in (
                ("ttft", planning_facts.ttft_slo_ms if planning_facts else None),
                ("tpot", planning_facts.tpot_slo_ms if planning_facts else None),
            )
            if target is not None
        }
        details = {
            "response": payload,
            "retrieval": {
                "mode": "approximate_workload" if workload else "latency_targets" if targets else "model_hardware",
                "model_identity_aliases": model_identity_aliases(model_name),
                "all_matches": False,
                "workload": workload,
                "latency_targets_ms": targets,
                "matched_observations": response.matched_items,
                "matched_experiments": response.matched_experiments,
                "returned_experiments": response.returned_experiments,
                "experiment_limit": response.experiment_limit,
                "returned_observations": len(ranked),
                "truncated": response.truncated,
                "ranking": (
                    "explicit workload similarity and coverage, then lower comparable experiment latency (Pareto); "
                    "neither comparable workload nor valid latency: fallback references, not low-latency evidence"
                ),
            },
            "matches": matches,
        }
    return PlanningEvidence(
        id="public-rag:query",
        source="public-benchmark-rag",
        status=status,
        details={
            "query": query.model_dump(mode="json"),
            "target_hardware": target_hardware,
            "message": message,
            **details,
        },
    )


def with_public_rag_evidence(resolved: ResolvedPlanningFacts, evidence: PlanningEvidence) -> ResolvedPlanningFacts:
    """Retain the query outcome and pinned citations across generator/fallback paths."""
    response = evidence.details.get("response", {})
    matches = {item["evidence_id"]: item for item in evidence.details.get("matches", [])}
    observations = tuple(
        {
            "evidence_id": f"public-rag:{item['item_id']}",
            **item,
            **(
                {"planning_match": matches[f"public-rag:{item['item_id']}"]}
                if f"public-rag:{item['item_id']}" in matches
                else {}
            ),
        }
        for item in response.get("evidence", [])
    )
    return resolved.model_copy(
        update={
            "facts": replace(
                resolved.facts,
                public_rag_evidence=observations,
            ),
            "evidence": [*resolved.evidence, evidence],
            "trace": [
                {"phase": "public_rag", "status": evidence.status, "message": evidence.details["message"]},
                *resolved.trace,
            ],
        }
    )


async def resolve_planning_facts(
    cluster_server_id: str,
    model_name: str,
    requested: PlanningFacts,
    *,
    include_supplementary_evidence: bool = True,
    hardware_snapshot: PlanningHardwareSnapshot | None = None,
) -> ResolvedPlanningFacts:
    """Use a live cluster overview for hard constraints and record other evidence."""
    requested = resolve_workload_signals(requested)
    cluster = registry.require_cluster(cluster_server_id)
    overview = (
        await hardware_snapshot.get(cluster_server_id)
        if hardware_snapshot is not None
        else await build_overview(cluster)
    )
    hardware = overview.get("hardware") or {}
    available_gpus = int(hardware.get("availableGpuCount") or 0)
    gpu_count = int(hardware.get("gpuCount") or 0)
    vram_bytes = int(hardware.get("vramBytes") or 0)
    if gpu_count < 1 or vram_bytes < 1:
        raise ValueError("cluster overview does not report usable accelerator VRAM")
    selected_gpu_count = _select_gpu_capacity(requested, available_gpus, gpu_count)
    vram_per_gpu_gib = vram_bytes / gpu_count / _GIB
    memory_bytes = int(hardware.get("memoryBytes") or 0)
    memory_usage = hardware.get("memoryUsagePercent")
    cpu_buffer_gib = requested.cpu_buffer_gib
    if memory_bytes and isinstance(memory_usage, (int, float)):
        cpu_buffer_gib = max(0.0, memory_bytes * (1 - memory_usage / 100) / _GIB)

    cluster_evidence = PlanningEvidence(
        id=f"cluster-overview:{cluster.id}",
        source="cluster-overview",
        status="available",
        details={
            "cluster_id": cluster.id,
            "available_gpu_count": available_gpus,
            "total_gpu_count": gpu_count,
            "selected_gpu_count": selected_gpu_count,
            "capacity_mode": requested.hardware_capacity_mode,
            "vram_per_gpu_gib": vram_per_gpu_gib,
            "cpu_buffer_gib": cpu_buffer_gib,
        },
    )
    evidence = [cluster_evidence]
    if include_supplementary_evidence:
        evidence.extend(await _supplementary_evidence(model_name, requested, selected_gpu_count, hardware))

    model_config_dict = requested.model_config_dict
    if model_config_dict is None and model_name:
        loaded = load_model_config(model_name)
        if loaded is not None:
            model_config_dict = loaded.raw if hasattr(loaded, "raw") else getattr(loaded, "to_dict", lambda: None)()
            if model_config_dict is not None and isinstance(model_config_dict, dict) and include_supplementary_evidence:
                evidence.append(
                    PlanningEvidence(
                        id=f"capacity-planner:{model_name}",
                        source="capacity-planner",
                        status="available",
                        details={
                            "model_name": model_name,
                            "architectures": model_config_dict.get("architectures", []),
                            "num_attention_heads": model_config_dict.get("num_attention_heads"),
                        },
                    )
                )

    aic_predictions = tuple(
        prediction
        for item in evidence
        if item.id == "aic:search" and item.status == "available"
        for prediction in _aic_predictions(item.details.get("configs", []))
    )
    facts = resolve_workload_signals(
        PlanningFacts(
            model_weight_gib=requested.model_weight_gib,
            vram_per_gpu_gib=vram_per_gpu_gib,
            free_gpu_count=selected_gpu_count,
            cpu_buffer_gib=cpu_buffer_gib,
            required_cpu_buffer_gib=requested.required_cpu_buffer_gib,
            context_length=requested.context_length,
            gpu_memory_utilization=requested.gpu_memory_utilization,
            shared_prefix_ratio=requested.shared_prefix_ratio,
            prefill_heavy=requested.prefill_heavy,
            use_case=requested.use_case,
            ttft_slo_ms=requested.ttft_slo_ms,
            tpot_slo_ms=requested.tpot_slo_ms,
            end_to_end_latency_slo_ms=requested.end_to_end_latency_slo_ms,
            operator_preference=requested.operator_preference,
            vllm_arguments=requested.vllm_arguments,
            evidence_ids=tuple(item.id for item in evidence),
            workload_profile=requested.workload_profile,
            aic_predictions=aic_predictions,
            hardware_capacity_mode=requested.hardware_capacity_mode,
            custom_gpu_count=requested.custom_gpu_count,
            model_name=model_name or requested.model_name,
            model_config_dict=model_config_dict,
            public_rag_evidence=requested.public_rag_evidence,
            configuration_cases=requested.configuration_cases,
            configuration_case_status=requested.configuration_case_status,
            requirements=requested.requirements,
        )
    )
    discovered = _public_rag_hardware(overview)
    facts = enrich_planning_requirements(
        facts,
        model_name=model_name,
        hardware_model=discovered[1] if discovered is not None else None,
        profile_id=discovered[0].id if discovered is not None else None,
        device_count=selected_gpu_count if discovered is not None else None,
        model_config=model_config_dict,
    )
    resolved = ResolvedPlanningFacts(
        facts=facts,
        cluster_snapshot=overview,
        evidence=evidence,
        trace=[
            {
                "phase": "workload",
                "status": "complete",
                "profile": asdict(facts.workload_profile) if facts.workload_profile else None,
                "ttft_slo_ms": facts.ttft_slo_ms,
                "tpot_slo_ms": facts.tpot_slo_ms,
                "context_length": facts.context_length,
            },
            {
                "phase": "facts",
                "status": "complete",
                "available_gpu_count": available_gpus,
                "selected_gpu_count": selected_gpu_count,
                "vram_per_gpu_gib": vram_per_gpu_gib,
                "cpu_buffer_gib": cpu_buffer_gib,
                "evidence": [{"id": item.id, "status": item.status} for item in evidence],
                "aic_prediction_count": len(aic_predictions),
            },
            {
                "phase": "aic",
                "status": "requested" if include_supplementary_evidence else "skipped",
                "request": {
                    "model_name": model_name,
                    "gpu_count": max(selected_gpu_count, 1),
                    "mean_input_tokens": int(facts.workload_profile.mean_input_tokens)
                    if facts.workload_profile and facts.workload_profile.mean_input_tokens
                    else DEFAULT_INPUT_TOKENS,
                    "mean_output_tokens": int(facts.workload_profile.mean_output_tokens)
                    if facts.workload_profile and facts.workload_profile.mean_output_tokens
                    else DEFAULT_OUTPUT_TOKENS,
                    "target_concurrency": facts.workload_profile.concurrency if facts.workload_profile else None,
                    "target_request_rate": facts.workload_profile.request_rate if facts.workload_profile else None,
                    "ttft_target_ms": facts.ttft_slo_ms,
                    "tpot_target_ms": facts.tpot_slo_ms,
                }
                if include_supplementary_evidence
                else None,
                "support_status": next((item.status for item in evidence if item.id == "aic:support"), "not_requested"),
                "search_status": next((item.status for item in evidence if item.id == "aic:search"), "not_requested"),
                "prediction_count": len(aic_predictions),
            },
        ],
    )
    logger.info(
        "Agentic facts cluster=%s selected_gpus=%d aic_predictions=%d evidence=%s",
        cluster_server_id,
        selected_gpu_count,
        len(aic_predictions),
        [(item.id, item.status) for item in evidence],
    )
    return resolved


def _select_gpu_capacity(requested: PlanningFacts, available_gpus: int, gpu_count: int) -> int:
    if requested.hardware_capacity_mode == "free":
        return available_gpus
    if requested.hardware_capacity_mode == "all":
        return gpu_count
    if requested.custom_gpu_count is None or requested.custom_gpu_count < 1:
        raise ValueError("custom GPU capacity requires a positive custom_gpu_count")
    if requested.custom_gpu_count > gpu_count:
        raise ValueError("custom GPU capacity cannot exceed total cluster GPUs")
    return requested.custom_gpu_count


async def _supplementary_evidence(
    model_name: str,
    facts: PlanningFacts,
    gpu_count: int,
    hardware: dict[str, Any],
) -> list[PlanningEvidence]:
    evidence: list[PlanningEvidence] = []
    try:
        profiles = {profile.id: profile for profile in all_profiles()}
        systems = []
        for accelerator in hardware.get("accelerators", []):
            profile = profiles.get(accelerator.get("id"))
            for model in accelerator.get("models") or [""]:
                matches = {
                    row["system"]
                    for row in (profile.planning.aic_system_models if profile else ())
                    if row.get("pattern") and re.search(row["pattern"], model, re.IGNORECASE)
                }
                systems.append(next(iter(matches)) if len(matches) == 1 else None)
        if not systems or None in systems or len(set(systems)) != 1:
            raise ValueError("Cluster hardware does not identify a single supported AIC system")
        request = AICRequest(
            aic_system_name=systems[0],
            model_name=model_name,
            gpu_count=max(gpu_count, 1),
            mean_input_tokens=int(facts.workload_profile.mean_input_tokens)
            if facts.workload_profile and facts.workload_profile.mean_input_tokens
            else DEFAULT_INPUT_TOKENS,
            mean_output_tokens=int(facts.workload_profile.mean_output_tokens)
            if facts.workload_profile and facts.workload_profile.mean_output_tokens
            else DEFAULT_OUTPUT_TOKENS,
            target_concurrency=facts.workload_profile.concurrency if facts.workload_profile else None,
            target_request_rate=facts.workload_profile.request_rate if facts.workload_profile else None,
            ttft_target_ms=facts.ttft_slo_ms,
            tpot_target_ms=facts.tpot_slo_ms,
            max_candidates=3,
        )
        support = await check_support(request)
        evidence.append(
            PlanningEvidence(
                id="aic:support",
                source="aic",
                status="available",
                details=support.model_dump(),
            )
        )
        if support.supported:
            prediction = await search(request)
            evidence.append(
                PlanningEvidence(
                    id="aic:search",
                    source="aic",
                    status="available",
                    details=prediction.model_dump(),
                )
            )
        elif facts.workload_profile and (
            facts.workload_profile.concurrency is not None or facts.workload_profile.request_rate is not None
        ):
            raise ValueError(
                "AIC does not support this model and hardware. Choose a supported source or clear the load target."
            )
    except (AICError, ValueError) as error:
        if facts.workload_profile and (
            facts.workload_profile.concurrency is not None or facts.workload_profile.request_rate is not None
        ):
            raise ValueError(f"AIC load-target prediction failed: {error}") from error
        evidence.append(PlanningEvidence(id="aic:support", source="aic", status="unavailable"))
    evidence.extend(
        PlanningEvidence(
            id=identity,
            source=source,
            status="unavailable",
            details={"reason": "unscoped_history_disabled", "replacement": "configuration-cases:query"},
        )
        for identity, source in (("deploy-history", "deploy-history"), ("simulation-history", "simulation"))
    )
    return evidence


def _aic_predictions(configs: list[dict[str, Any]]) -> list[AICCandidatePrediction]:
    """Normalize only AIC configurations that can match Prism's bounded topologies."""
    predictions: list[AICCandidatePrediction] = []
    for config in configs:
        try:
            mode = config.get("mode")
            if mode == "agg":
                predictions.append(
                    AICCandidatePrediction(
                        mode="agg",
                        tensor_parallel_size=int(config["tp"]),
                        replicas=int(config["replicas"]),
                        ttft_ms=_positive_float(config.get("ttft_ms")),
                        tpot_ms=_positive_float(config.get("tpot_ms")),
                        throughput_tokens_per_sec=_positive_float(config.get("throughput_tokens_per_sec")),
                    )
                )
            elif mode == "disagg":
                predictions.append(
                    AICCandidatePrediction(
                        mode="disagg",
                        tensor_parallel_size=int(config["decode_tp"]),
                        replicas=int(config["decode_replicas"]),
                        prefill_tensor_parallel_size=int(config["prefill_tp"]),
                        prefill_replicas=int(config["prefill_replicas"]),
                        ttft_ms=_positive_float(config.get("ttft_ms")),
                        tpot_ms=_positive_float(config.get("tpot_ms")),
                        throughput_tokens_per_sec=_positive_float(config.get("throughput_tokens_per_sec")),
                    )
                )
        except (KeyError, TypeError, ValueError):
            continue
    return predictions


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None
