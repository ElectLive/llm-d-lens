"""Compatibility API for candidate search, backed by the local AIC service."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from itertools import product
from typing import Annotated, Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from llm_d_lens.aic import service as aic_service
from llm_d_lens.aic.models import AICEstimateRequest, AICExperimentRequest, AICRequest
from llm_d_lens.auth.context import require_route_permission

router = APIRouter(include_in_schema=False)
logger = logging.getLogger(__name__)
SEARCH_SOURCES = {"baseline", "pd", "epd", "tiered-cache", "precise-prefix-cache"}
SOURCE_LABELS = {
    "aic": "AIC Prediction",
    "baseline": "Optimized Baseline",
    "pd": "P/D Disaggregation",
    "epd": "E/P/D Disaggregation",
    "tiered-cache": "Tiered Prefix Cache",
    "precise-prefix-cache": "Precise Prefix Cache Routing",
    "manual": "Manual Configuration",
}
DEPLOYMENT_NOTE = "The current deployment backend does not support E/P/D topology."
NO_FEASIBLE = "No feasible configurations found for the given parameters."


class CandidateError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.public_message = message


def record(value) -> dict:
    return value if isinstance(value, dict) else {}


def finite_number(value, *, missing=False) -> float | None:
    if missing:
        return None
    try:
        number = float(0 if value is None or value == "" else value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def positive_integer(value, fallback: int, maximum: int = 128) -> int:
    number = finite_number(value)
    return min(int(number), maximum) if number is not None and number > 0 and number.is_integer() else fallback


def number_field(source: dict, name: str) -> float | None:
    return finite_number(source.get(name), missing=name not in source)


def predict_payload(workload: dict, config: dict) -> dict:
    return {
        "model_name": str(workload.get("model") or ""),
        "gpu_count": positive_integer(config.get("totalGpus"), 8, 64),
        "mean_input_tokens": positive_integer(workload.get("isl"), 4000, 1_000_000),
        "mean_output_tokens": positive_integer(workload.get("osl"), 1000, 1_000_000),
        "target_concurrency": workload.get("concurrency"),
        "target_request_rate": workload.get("request_rate"),
        "ttft_target_ms": number_field(workload, "ttftMs"),
        "tpot_target_ms": number_field(workload, "tpotMs"),
        "aic_system_name": str(config.get("aicSystemName") or "").strip(),
        "aic_backend_name": str(config.get("aicBackendName") or "vllm").strip().lower(),
        "aic_database_mode": str(config.get("aicDatabaseMode") or "SILICON").strip().upper(),
    }


async def aic_call(request: Request, operation: str, payload: dict) -> dict:
    scope = dict(request.scope, path=f"/api/v1/aic/{operation}", method="POST")
    await require_route_permission(Request(scope))
    model, service = {
        "support": (AICRequest, aic_service.check_support),
        "search": (AICRequest, aic_service.search),
        "experiments": (AICExperimentRequest, aic_service.experiments),
        "estimate": (AICEstimateRequest, aic_service.estimate),
    }[operation]
    try:
        parsed = model.model_validate(payload)
        timeout = float(os.environ.get("AIC_REQUEST_TIMEOUT_MS", "240000")) / 1000
        async with asyncio.timeout(timeout):
            result = await service(parsed)
            return result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    except ValidationError as error:
        detail = "; ".join(f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors())
        raise CandidateError(detail, 422) from error
    except aic_service.AICError as error:
        if str(error) == NO_FEASIBLE:
            raise CandidateError(NO_FEASIBLE) from error
        logger.exception("AIC candidate search failed")
        raise CandidateError("AIC candidate search failed") from error
    except TimeoutError as error:
        raise CandidateError("AIC candidate search timed out", 504) from error


def candidate_base(source: str, config_id: str, index: int) -> dict:
    return {
        "id": f"{source}-{config_id}-{index}",
        "source": source,
        "sourceLabel": SOURCE_LABELS.get(source, source),
        "family": "Predictive" if source == "aic" else "Search-based",
        "confidence": None,
        "backend": "AIConfigurator",
    }


def normalize_candidate(raw: dict, index: int, label: str | None = None) -> dict:
    mode = str(raw.get("mode") or (raw.get("serving_mode") if label else None) or "disagg")
    aggregated = mode == "agg"
    prefill_tp = 0 if aggregated else positive_integer(raw.get("prefill_tp"), 1)
    prefill_replicas = 0 if aggregated else positive_integer(raw.get("prefill_replicas"), 1)
    decode_tp = positive_integer(raw.get("tp" if aggregated else "decode_tp"), 1)
    pipeline_parallel = positive_integer(raw.get("pp"), 1) if aggregated else 1
    reported = number_field(raw, "num_total_gpus")
    inferred = (
        max(1, math.floor(reported / (decode_tp * pipeline_parallel))) if reported and aggregated and label else 1
    )
    decode_replicas = positive_integer(raw.get("replicas" if aggregated else "decode_replicas"), inferred)
    computed = prefill_tp * prefill_replicas + decode_tp * decode_replicas
    config_id = (
        f"agg-tp{decode_tp}-rep{decode_replicas}"
        if aggregated
        else f"{mode}-p{prefill_tp}x{prefill_replicas}-d{decode_tp}x{decode_replicas}"
    )
    raw_name = str(raw.get("name") or raw.get("experiment_name") or f"manual-{index + 1}")
    candidate = {
        **candidate_base(
            "manual" if label else "aic", re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name) if label else config_id, index
        ),
        "name": raw_name if label else f"AIC #{index + 1} \u00b7 {config_id}",
        "topologyMode": mode,
        "prefillTp": prefill_tp,
        "prefillReplicas": prefill_replicas,
        "decodeTp": decode_tp,
        "decodeReplicas": decode_replicas,
        "encodeTp": positive_integer(raw.get("encode_tp"), 1) if mode == "epd" else None,
        "encodeReplicas": positive_integer(raw.get("encode_replicas"), 1) if mode == "epd" else None,
        "totalGpus": positive_integer(raw.get("num_total_gpus"), computed),
        "predicted": {
            "ttftMs": number_field(raw, "ttft_ms"),
            "tpotMs": number_field(raw, "tpot_ms"),
            "throughputTps": number_field(raw, "throughput_tokens_per_sec"),
        },
        "deployable": mode != "epd",
        "deploymentNote": DEPLOYMENT_NOTE if mode == "epd" else None,
    }
    if label:
        candidate.update(
            sourceLabel=label,
            family="Existing / User",
            pipelineParallel=pipeline_parallel,
            isChosenExperiment=raw.get("is_chosen") is True,
        )
    else:
        candidate["aicRank"] = index + 1
    return candidate


def grid_candidates(source: str, budget: int, limit: int) -> list[dict]:
    configs = []
    if source in {"baseline", "tiered-cache"}:
        dimensions = product([0], [0], [1, 2, 4, 8], range(1, budget + 1), [0], [0])
    elif source == "epd":
        dimensions = product([1, 2], [1, 2], [1, 2], [1, 2, 3], [1, 2], [1, 2])
    else:
        dimensions = product([1, 2, 4], range(1, budget + 1), [1, 2, 4], range(1, budget + 1), [0], [0])
    for prefill_tp, prefill_replicas, decode_tp, decode_replicas, encode_tp, encode_replicas in dimensions:
        used = prefill_tp * prefill_replicas + decode_tp * decode_replicas + encode_tp * encode_replicas
        if used > budget:
            continue
        config_id = f"pd-p{prefill_tp}x{prefill_replicas}-d{decode_tp}x{decode_replicas}"
        if source == "epd":
            config_id = (
                f"epd-e{encode_tp}x{encode_replicas}-p{prefill_tp}x{prefill_replicas}-d{decode_tp}x{decode_replicas}"
            )
        elif source in {"baseline", "tiered-cache"}:
            config_id = f"baseline-tp{decode_tp}-rep{decode_replicas}"
        configs.append(
            {
                "config_id": config_id,
                "totalGpus": used,
                "prefillTp": prefill_tp,
                "prefillReplicas": prefill_replicas,
                "decodeTp": decode_tp,
                "decodeReplicas": decode_replicas,
                "encodeTp": encode_tp or None,
                "encodeReplicas": encode_replicas or None,
            }
        )
    configs.sort(key=lambda candidate: (-candidate["totalGpus"], candidate["config_id"]))
    candidates = []
    for index, config in enumerate(configs[:limit]):
        config_id = config.pop("config_id")
        if source == "tiered-cache":
            config_id = config_id.replace("baseline-", "tpc-", 1)
        candidates.append(
            {
                **candidate_base(source, config_id, index),
                **config,
                "name": config_id,
                "topologyMode": "agg" if source in {"baseline", "tiered-cache"} else source,
                "predicted": None,
                "deployable": source != "epd",
                "deploymentNote": DEPLOYMENT_NOTE if source == "epd" else None,
            }
        )
    return candidates


async def search_candidates(request: Request, body: dict) -> dict:
    workload, config, manual = (record(body.get(key)) for key in ("workload", "searchConfig", "manualConfig"))
    sources = body.get("sourceIds")
    sources = list(dict.fromkeys(str(value) for value in sources)) if isinstance(sources, list) else []
    sources = [source for source in sources if source in SEARCH_SOURCES | {"aic", "manual"}]
    if not sources:
        return {"candidates": []}
    budget = positive_integer(config.get("totalGpus"), 8, 64)
    limit = positive_integer(config.get("maxCandidatesPerMode"), 10, 50)
    payload = predict_payload(workload, config)
    candidates = []
    if "aic" in sources:
        if not payload["aic_system_name"]:
            raise CandidateError("AIC system is required for Predictive search")
        support = await aic_call(request, "support", payload)
        if support.get("supported") is False:
            raise CandidateError(
                f"{workload.get('model') or 'Model'} is not supported on "
                f"{payload['aic_system_name']} with {payload['aic_backend_name']}"
            )
        prediction = await aic_call(request, "search", payload)
        candidates.extend(
            normalize_candidate(raw, index) for index, raw in enumerate(prediction.get("configs", [])[:limit])
        )
    if "manual" in sources:
        if manual.get("inputMode") == "yaml":
            text = str(manual.get("yamlText") or "").strip()
            if not text:
                raise CandidateError("Upload an AIConfigurator experiment YAML file first")
            result = await aic_call(request, "experiments", {"yaml_text": text, "top_n": limit})
            candidates.extend(
                normalize_candidate(raw, index, "YAML Experiment")
                for index, raw in enumerate(result.get("experiments", []))
            )
        else:
            if not payload["aic_system_name"]:
                raise CandidateError("AIC system is required for Manual Estimate")
            mode = str(manual.get("servingMode") or "disagg")
            manual_payload = dict(payload, scenario="inference_scheduling" if mode == "agg" else "pd_disaggregation")
            for key, field in (
                ("tp", "tp"),
                ("pp", "pp"),
                ("replicas", "replicas"),
                ("prefill_tp", "prefillTp"),
                ("prefill_replicas", "prefillReplicas"),
                ("decode_tp", "decodeTp"),
                ("decode_replicas", "decodeReplicas"),
            ):
                manual_payload[key] = positive_integer(manual.get(field), 1)
            manual_payload["gpu_count"] = (
                manual_payload["tp"] * manual_payload["pp"] * manual_payload["replicas"]
                if mode == "agg"
                else manual_payload["prefill_tp"] * manual_payload["prefill_replicas"]
                + manual_payload["decode_tp"] * manual_payload["decode_replicas"]
            )
            support = await aic_call(request, "support", manual_payload)
            if support.get("agg_supported" if mode == "agg" else "disagg_supported") is False:
                label = "Aggregated" if mode == "agg" else "P/D disaggregated"
                raise CandidateError(f"{label} serving is not supported for this model and hardware")
            estimate = await aic_call(request, "estimate", manual_payload)
            candidates.append(
                normalize_candidate(
                    dict(estimate, name=str(manual.get("name") or "manual-estimate")),
                    0,
                    "Manual Estimate",
                )
            )
    for source in sources:
        if source in SEARCH_SOURCES:
            candidates.extend(grid_candidates(source, budget, limit))
    return {"candidates": candidates, "backend": "local-aic"}


def complete_candidate(candidate: dict, workload: dict) -> bool:
    targets = [key for key in ("ttftMs", "tpotMs") if workload.get(key) is not None]
    predicted = candidate["predicted"]
    return (
        candidate["deployable"]
        and all((predicted[key] or 0) > 0 for key in targets)
        and (bool(targets) or (predicted["throughputTps"] or 0) > 0)
    )


def violation(candidate: dict, workload: dict) -> float:
    return max(
        (candidate["predicted"][key] / workload[key] if workload.get(key) is not None else 0)
        for key in ("ttftMs", "tpotMs")
    )


async def relaxed_candidates(request: Request, body: dict) -> dict:
    workload, config = record(body.get("workload")), record(body.get("searchConfig"))
    budget = config.get("totalGpus")
    if (
        not str(workload.get("model") or "").strip()
        or isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or int(budget) != budget
        or not 1 <= budget <= 64
    ):
        raise CandidateError("Model and explicit totalGpus (1-64) are required")
    targets = [workload.get("ttftMs"), workload.get("tpotMs")]
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0)
        for value in targets
    ):
        raise CandidateError("Latency targets must be positive finite numbers")
    payload = predict_payload(workload, config)
    support = await aic_call(request, "support", payload)
    if support.get("supported") is False:
        return {"candidates": [], "attempts": [], "anchor": None, "sloSatisfied": False, "reason": "unsupported"}
    attempts, candidates = [], []
    for factor in [1, 1.25, 1.5, 1.75, 2] if any(value is not None for value in targets) else [1]:
        search_payload = dict(payload)
        for key in ("ttft_target_ms", "tpot_target_ms"):
            if search_payload[key] is not None:
                search_payload[key] *= factor
        try:
            result = await aic_call(request, "search", search_payload)
        except CandidateError as error:
            if error.status != 400 or str(error) != NO_FEASIBLE:
                raise
            result = {"configs": []}
        configs = result.get("configs", [])
        normalized = []
        for raw in configs:
            keys = (
                ("tp", "replicas")
                if raw.get("mode") == "agg"
                else (
                    "prefill_tp",
                    "prefill_replicas",
                    "decode_tp",
                    "decode_replicas",
                )
            )
            if raw.get("mode") not in {"agg", "disagg"} or any((number_field(raw, key) or 0) <= 0 for key in keys):
                continue
            candidate = normalize_candidate(raw, len(normalized))
            normalized.append(candidate)
        normalized = [
            candidate
            for candidate in normalized
            if complete_candidate(candidate, workload)
            and candidate["prefillTp"] * candidate["prefillReplicas"]
            + candidate["decodeTp"] * candidate["decodeReplicas"]
            <= budget
        ]
        normalized = normalized[: positive_integer(config.get("maxCandidatesPerMode"), 10, 50)]
        candidates.extend(normalized)
        attempts.append(
            {
                "factor": factor,
                "ttftMs": search_payload["ttft_target_ms"],
                "tpotMs": search_payload["tpot_target_ms"],
                "resultCount": len(configs),
                "completeCount": len(normalized),
            }
        )
        if normalized:
            break
    candidates.sort(key=lambda candidate: (violation(candidate, workload), candidate["totalGpus"]))
    anchor = candidates[0] if candidates else None
    return {
        "candidates": candidates,
        "attempts": attempts,
        "anchor": anchor,
        "sloSatisfied": anchor is not None and violation(anchor, workload) <= 1,
        "reason": None if anchor else "no_complete_anchor",
        "backend": "local-aic",
    }


async def support_candidate(request: Request, body: dict) -> dict:
    payload = predict_payload(record(body.get("workload")), record(body.get("searchConfig")))
    if not payload["model_name"]:
        raise CandidateError("Model is required for the AIC support check")
    if not payload["aic_system_name"]:
        raise CandidateError("AIC system is required for the support check")
    return await aic_call(request, "support", payload)


async def candidate_endpoint(request: Request, body: Annotated[Any, Body()] = None):
    operation = {
        "/api/candidate-search": search_candidates,
        "/api/candidate-search/relaxed": relaxed_candidates,
        "/api/candidate-support": support_candidate,
    }[request.url.path]
    try:
        return await operation(request, record(body))
    except CandidateError as error:
        return JSONResponse({"error": error.public_message}, status_code=error.status)


for route_path in ("/api/candidate-search", "/api/candidate-search/relaxed", "/api/candidate-support"):
    router.add_api_route(route_path, candidate_endpoint, methods=["POST"])
