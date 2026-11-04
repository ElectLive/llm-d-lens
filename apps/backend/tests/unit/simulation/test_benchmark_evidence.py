"""Simulation evidence must retain its own test-time facts, not planning defaults."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_d_lens.agentic import benchmark_context
from llm_d_lens.agentic.benchmark_evidence import BenchmarkEvidenceRetriever
from llm_d_lens.db.dao.agentic_benchmark_record import BenchmarkRecordDao, BenchmarkRecordDaoError
from llm_d_lens.db.dao.simulation_task import SimulationTaskDao
from llm_d_lens.infrastructure.artifacts.manifests import configuration_checksum
from llm_d_lens.simulation import benchmark_evidence as evidence
from llm_d_lens.simulation import service
from llm_d_lens.simulation.models import SimulationTask


@pytest.fixture
def task(tmp_path):
    content = {"model": {"name": "org/model"}, "decode": {"tensorParallelSize": 1, "replicaCount": 1}}
    task = SimulationTask.model_validate(
        {
            "id": "abcdef12",
            "name": "Measured replay",
            "description": "",
            "scenario": "chat",
            "status": "completed",
            "endpoint_mode": "deployment",
            "endpoint_deployment_execution_id": "execution-1",
            "endpoint_cluster_id": "cluster1",
            "endpoint_namespace": "serving",
            "endpoint_url": "http://service.local",
            "model_name": "org/model",
            "task_dir": str(tmp_path / "abcdef12"),
            "simulation": {
                "backend": "aiperf",
                "backend_options": {},
                "duration_seconds": 2.0,
                "stream": True,
                "grace_period_seconds": 30.0,
            },
            "prompt": {
                "type": "trace",
                "dataset": {"name": "mooncake-arxiv", "scenario": "chat", "tokenizer": None},
                "trace": {
                    "path": str(tmp_path / "trace.jsonl"),
                    "format": "mooncake_trace",
                    "fixed_schedule": True,
                    "timeout_seconds": 30.0,
                    "synthesis_speedup_ratio": 1.0,
                },
            },
            "progress_percent": 100.0,
            "progress_message": "completed",
            "logs": [],
            "created_at": "2026-10-20T12:00:00Z",
            "started_at": "2026-10-20T12:00:30Z",
            "execution_started_at": "2026-10-20T12:01:00Z",
            "completed_at": "2026-10-20T12:02:00Z",
            "result": {
                "run_id": "abcdef12",
                "backend": "aiperf",
                "backend_version": "0.12.0",
                "summary": {
                    "total_requests": 2.0,
                    "successful_requests": 2.0,
                    "failed_requests": 0.0,
                    "duration_seconds": 2.0,
                    "throughput_rps": 1.0,
                    "throughput_tps": 20.0,
                },
                "artifacts": [],
                "backend_metrics": {},
                "warnings": [],
                "per_request": [
                    {
                        "request_id": "a",
                        "successful": True,
                        "started_at_seconds": 0.0,
                        "completed_at_seconds": 1.0,
                        "input_tokens": 100.0,
                        "output_tokens": 20.0,
                        "ttft_ms": 10.0,
                        "tpot_ms": 2.0,
                        "latency_ms": 1000.0,
                    },
                    {
                        "request_id": "b",
                        "successful": True,
                        "started_at_seconds": 0.0,
                        "completed_at_seconds": 2.0,
                        "input_tokens": 200.0,
                        "output_tokens": 20.0,
                        "ttft_ms": 90.0,
                        "tpot_ms": 4.0,
                        "latency_ms": 2000.0,
                    },
                ],
            },
        }
    )
    task.benchmark_evidence = {
        "schema_version": 1,
        "source": "simulation",
        "id": task.id,
        **evidence._task_configuration(task),
        "deployment_configuration": {
            "type": "baseline",
            "format": "helm",
            "provider_ref": "baseline-vllm",
            "content": content,
            "checksum": configuration_checksum(content),
        },
        "benchmark_context": {
            "schema_version": 1,
            "captured_at": "2026-10-20T12:00:45Z",
            "verified_at": "2026-10-20T12:02:30Z",
            "configuration_checksum": configuration_checksum(content),
            "model": {
                "model_id": "org/model",
                "family": "actual-family",
                "architecture": "moe",
                "parameter_count_b": 8.0,
                "quantization": "none",
                "tokenizer_id": "org/model",
            },
            "hardware": {"accelerator": "Actual XPU", "vram_per_gpu_gib": 64.0, "gpu_count": 1},
            "runtime": {"backend": "vllm", "backend_version": "1.2.3", "accelerator_runtime": "actual-runtime"},
        },
    }
    return task


def test_saved_metadata_is_used_without_live_or_planning_lookups(task, monkeypatch):
    monkeypatch.setattr(evidence, "get_execution_context", lambda _: pytest.fail("Historical API query"))
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    (record,) = records
    assert record.benchmark_id == "simulation:abcdef12"
    assert record.timestamp.isoformat() == "2026-10-20T12:02:00+00:00"
    assert record.model.architecture == "moe"
    assert record.hardware.accelerator == "Actual XPU"
    assert record.hardware.vram_per_gpu_gib == 64
    assert record.runtime.backend_version == "1.2.3"
    assert record.workload.mean_input_tokens == 150
    assert record.workload.p95_input_tokens == 200
    assert record.workload.concurrency == 1.5
    assert record.workload.shared_prefix_ratio is None
    assert record.workload.prefill_fraction == pytest.approx(150 / 170)
    assert record.metrics.ttft_p95_ms == 90
    assert record.metrics.error_rate == 0


def test_unknown_runtime_version_does_not_block_simulation_archival(task):
    task.benchmark_evidence["benchmark_context"]["runtime"]["backend_version"] = "unknown"
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    assert records[0].runtime.backend_version is None
    assert records[0].provenance["attribution"] == "serving-snapshot"


@pytest.mark.parametrize("backend", ["aiperf", "trace-replayer"])
@pytest.mark.parametrize("failed_interval", [(-1.5, -1.0), (2.5, 3.5)])
def test_duration_scope_allows_failures_outside_successful_window(task, backend, failed_interval):
    task.simulation = task.simulation.model_copy(update={"backend": backend})
    task.result.backend = backend
    task.benchmark_evidence["simulation"] = task.simulation.model_dump(mode="json")
    task.result.per_request.append(
        task.result.per_request[0].model_copy(
            update={
                "request_id": "failed",
                "successful": False,
                "started_at_seconds": failed_interval[0],
                "completed_at_seconds": failed_interval[1],
                "status_code": 503,
            }
        )
    )
    duration = 2.0 if backend == "aiperf" else 3.5
    task.result.summary.update(total_requests=3, failed_requests=1, duration_seconds=duration)
    original_summary = deepcopy(task.result.summary)
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    record = records[0]
    assert record.workload.concurrency == pytest.approx((3 + failed_interval[1] - failed_interval[0]) / 3.5)
    assert record.workload.request_rate == 1
    assert record.metrics.throughput_tokens_per_s == 20
    assert record.metrics.error_rate == pytest.approx(1 / 3)
    assert record.provenance["measurement_windows"] == {
        "reported_duration_seconds": duration,
        "reported_duration_scope": "successful-requests" if backend == "aiperf" else "all-requests",
        "all_requests_duration_seconds": 3.5,
        "successful_requests_duration_seconds": 2.0,
    }
    assert task.result.summary == original_summary


@pytest.mark.parametrize("backend", ["aiperf", "trace-replayer"])
@pytest.mark.parametrize("request_window,eligible", [(3.0, True), (3.01, False)])
def test_reported_duration_checks_matching_scope_at_tolerance_boundary(task, backend, request_window, eligible):
    task.simulation = task.simulation.model_copy(update={"backend": backend})
    task.result.backend = backend
    task.benchmark_evidence["simulation"] = task.simulation.model_dump(mode="json")
    task.result.per_request[1] = task.result.per_request[1].model_copy(update={"completed_at_seconds": request_window})
    records, issues = evidence.normalize_simulation_task(task)
    assert bool(records) == eligible
    assert bool(issues) != eligible
    if not eligible:
        assert "reported measurement duration" in issues[0]["reason"]


@pytest.mark.parametrize("backend", ["aiperf", "trace-replayer"])
def test_trace_replayer_duration_includes_failed_requests(task, backend):
    task.simulation = task.simulation.model_copy(update={"backend": backend})
    task.result.backend = backend
    task.benchmark_evidence["simulation"] = task.simulation.model_dump(mode="json")
    task.result.per_request.append(
        task.result.per_request[0].model_copy(
            update={
                "request_id": "late-failure",
                "successful": False,
                "started_at_seconds": 3.0,
                "completed_at_seconds": 4.0,
            }
        )
    )
    task.result.summary.update(total_requests=3, failed_requests=1)
    records, issues = evidence.normalize_simulation_task(task)
    if backend == "aiperf":
        assert records
        assert not issues
    else:
        assert not records
        assert "all-requests" in issues[0]["reason"]


def test_failed_request_window_cannot_exceed_execution_duration(task):
    task.result.per_request.append(
        task.result.per_request[0].model_copy(
            update={
                "request_id": "outlier",
                "successful": False,
                "started_at_seconds": -100.0,
                "completed_at_seconds": -99.0,
            }
        )
    )
    task.result.summary.update(total_requests=3, failed_requests=1)
    records, issues = evidence.normalize_simulation_task(task)
    assert not records
    assert "execution duration" in issues[0]["reason"]


@pytest.mark.parametrize(
    "updates",
    [
        {"started_at_seconds": float("nan")},
        {"completed_at_seconds": float("inf")},
        {"started_at_seconds": 2.0, "completed_at_seconds": 1.0},
    ],
)
def test_invalid_request_times_remain_ineligible(task, updates):
    task.result.per_request[0] = task.result.per_request[0].model_copy(update=updates)
    records, issues = evidence.normalize_simulation_task(task)
    assert not records
    assert issues[0]["reason"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: setattr(t, "benchmark_evidence", None),
        lambda t: t.benchmark_evidence["benchmark_context"].pop("verified_at"),
        lambda t: t.benchmark_evidence.update(benchmark_context_error="API unavailable"),
        lambda t: t.benchmark_evidence["benchmark_context"].update(captured_at="2026-10-20T12:01:30Z"),
        lambda t: t.benchmark_evidence.update(task_id="12345678"),
        lambda t: setattr(t, "endpoint_cluster_id", "other"),
        lambda t: setattr(t, "endpoint_url", "http://other.local"),
        lambda t: setattr(t, "status", "cancelled"),
        lambda t: setattr(t, "artifacts_incomplete", True),
        lambda t: t.result.summary.update(total_requests=99),
        lambda t: t.result.summary.update(failed_requests=True),
        lambda t: t.result.summary.update(duration_seconds=0),
        lambda t: t.result.summary.update(duration_seconds=float("nan")),
        lambda t: t.result.summary.update(throughput_rps=float("nan")),
        lambda t: t.result.per_request.pop(),
        lambda t: t.result.per_request.__setitem__(1, t.result.per_request[0]),
        lambda t: setattr(t, "execution_started_at", None),
    ],
)
def test_invalid_missing_or_changed_context_is_never_inferred(task, mutate):
    mutate(task)
    records, issues = evidence.normalize_simulation_task(task)
    assert not records
    assert issues[0]["reason"]


def test_missing_latency_is_not_fabricated(task):
    task.result.per_request = [
        row.model_copy(update={"ttft_ms": None, "tpot_ms": None, "latency_ms": None}) for row in task.result.per_request
    ]
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    assert records[0].metrics.ttft_p95_ms is None
    assert records[0].metrics.throughput_tokens_per_s == 20


def test_append_only_ingestion_and_preview_preserve_unknown_prefix(task):
    assert evidence.ingest_simulation_task(task, apply=False)["status"] == "eligible"
    assert not BenchmarkRecordDao().list()
    assert evidence.ingest_simulation_task(task)["inserted_ids"] == ["simulation:abcdef12"]
    task.result.summary["throughput_tps"] = 999
    assert evidence.ingest_simulation_task(task)["inserted_ids"] == []
    stored = BenchmarkRecordDao().get("simulation:abcdef12")
    assert stored.metrics.throughput_tokens_per_s == 20
    assert stored.workload.shared_prefix_ratio is None


def test_unknown_prefix_is_not_a_perfect_match_even_when_both_unknown(task):
    (record,), _ = evidence.normalize_simulation_task(task)
    profile = record.workload
    assert BenchmarkEvidenceRetriever.workload_similarity(profile, profile) == pytest.approx(6 / 7)
    known = profile.model_copy(update={"shared_prefix_ratio": 0.0})
    assert BenchmarkEvidenceRetriever.workload_similarity(known, known) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("published", [False, True])
async def test_metadata_capture_uses_execution_source_and_queries_missing_facts(task, monkeypatch, published):
    snapshot = deepcopy(task.benchmark_evidence)
    if published:
        task.model_service_group_id = "msg-1"
        task.model_name = "published-model"
        monkeypatch.setattr(
            evidence,
            "_routing_snapshot",
            AsyncMock(
                return_value={
                    "model_ref": "org/model",
                    "pod_selector": "app=serving",
                }
            ),
        )
    configuration = SimpleNamespace(
        content=snapshot["deployment_configuration"]["content"],
        model_dump=lambda **_: deepcopy(snapshot["deployment_configuration"]),
    )
    context = SimpleNamespace(cluster_id="cluster1", namespace="serving", run_id="run-1", case_id="case-1")
    run = SimpleNamespace(
        cases=[SimpleNamespace(id="case-1", execution_id="execution-1", source_configuration_ordinal=0)],
        source_configurations=[configuration],
    )
    monkeypatch.setattr(evidence, "get_execution_context", lambda _: context)
    monkeypatch.setattr(evidence.deployment_run_manager.store, "get_run", lambda _: run)

    async def collect(source):
        assert source["deployment_configuration"] == snapshot["deployment_configuration"]
        source["benchmark_context"] = deepcopy(snapshot["benchmark_context"])

    query = AsyncMock(side_effect=collect)
    monkeypatch.setattr(evidence, "capture_benchmark_context", query)
    await evidence.capture_simulation_context(task)
    query.assert_awaited_once()
    assert task.benchmark_evidence["benchmark_context"]["hardware"]["accelerator"] == "Actual XPU"
    configuration.content["decode"]["replicaCount"] = 8
    assert task.benchmark_evidence["deployment_configuration"]["content"]["decode"]["replicaCount"] == 1


@pytest.mark.asyncio
async def test_external_or_multi_target_context_never_guesses_metadata(task, monkeypatch):
    query = AsyncMock()
    monkeypatch.setattr(evidence, "capture_benchmark_context", query)
    task.endpoint_mode = "external"
    await evidence.capture_simulation_context(task)
    assert "unverified" in task.benchmark_evidence["benchmark_context_error"]
    query.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_save_keeps_source_on_dao_error_and_persists_diagnostics(task, monkeypatch, tmp_path):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    service.reset_service_state()

    def unavailable(_self, _records):
        raise BenchmarkRecordDaoError("offline")

    monkeypatch.setattr(BenchmarkRecordDao, "save_many", unavailable)
    await service.save_task(task)
    loaded = SimulationTaskDao().get(task.id)
    assert loaded.status == "completed"
    assert loaded.result.summary["throughput_tps"] == 20
    assert loaded.benchmark_evidence["ingestion"]["status"] == "error"
    service.reset_service_state()


@pytest.mark.asyncio
async def test_running_snapshot_roundtrips_without_a_result(task, monkeypatch, tmp_path):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    service.reset_service_state()
    task.status = "running"
    task.result = None
    await service.save_task(task)
    loaded = SimulationTaskDao().get(task.id)
    assert loaded.result is None
    assert loaded.benchmark_evidence == task.benchmark_evidence
    service.reset_service_state()


@pytest.mark.parametrize("backend", ["aiperf", "trace-replayer"])
def test_real_saved_backend_artifact_shapes_are_normalized(task, backend):
    rows = task.result.per_request
    task.result.per_request = []
    task.simulation = task.simulation.model_copy(update={"backend": backend})
    task.result.backend = backend
    task.benchmark_evidence["simulation"] = task.simulation.model_dump(mode="json")
    directory = Path(task.task_dir) / "artifacts" / backend
    directory.mkdir(parents=True)
    if backend == "aiperf":
        filename = "profile_export.jsonl"
        raw = [
            {
                "request_id": row.request_id,
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": 0,
                    "request_end_ns": int(row.completed_at_seconds * 1e9),
                },
                "metrics": {
                    key: {"value": value}
                    for key, value in {
                        "input_sequence_length": row.input_tokens,
                        "output_sequence_length": row.output_tokens,
                        "request_latency": row.latency_ms,
                        "time_to_first_token": row.ttft_ms,
                        "inter_token_latency": row.tpot_ms,
                    }.items()
                },
            }
            for row in rows
        ]
    else:
        filename = "requests.jsonl"
        raw = [
            {
                "request_id": row.request_id,
                "s_time": 0,
                "e_time": row.completed_at_seconds * 1000,
                "status": 200,
                "input_length": row.input_tokens,
                "output_length": row.output_tokens,
                "first_token_time": row.ttft_ms,
                "avg_time_between_tokens": row.tpot_ms,
            }
            for row in rows
        ]
    (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in raw))
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    assert records[0].metrics.ttft_p95_ms == 90
    assert records[0].metrics.tpot_p95_ms == 4
    assert records[0].workload.mean_input_tokens == 150
    if backend == "trace-replayer":
        for row in raw:
            row.pop("avg_time_between_tokens")
            row["total_time"] = 1000
        (directory / filename).write_text("".join(json.dumps(row) + "\n" for row in raw))
        records, issues = evidence.normalize_simulation_task(task)
        assert not issues
        assert records[0].metrics.tpot_p95_ms is None


@pytest.mark.asyncio
async def test_supplementary_evidence_never_reads_unscoped_simulation_history(task, monkeypatch):
    from llm_d_lens.agentic import facts
    from llm_d_lens.agentic.planner import PlanningFacts
    from llm_d_lens.aic.service import AICError

    async def offline(_):
        raise AICError("offline")

    monkeypatch.setattr(facts, "check_support", offline)
    monkeypatch.setattr(SimulationTaskDao, "list_ids", lambda *_: pytest.fail("Unscoped simulation history query"))
    for snapshot in (task.benchmark_evidence, None):
        task.benchmark_evidence = snapshot
        resolved = await facts._supplementary_evidence("org/model", PlanningFacts(8, 1, 99), 99, {})
        history = next(item for item in resolved if item.id == "simulation-history")
        assert history.status == "unavailable"
        assert history.details == {
            "reason": "unscoped_history_disabled",
            "replacement": "configuration-cases:query",
        }


def test_failed_evidence_requires_actual_measurements_and_reason(task):
    task.status = "failed"
    assert not evidence.normalize_simulation_task(task)[0]
    task.error_message = "Measured client failure"
    records, issues = evidence.normalize_simulation_task(task)
    assert not issues
    assert records[0].outcome == "failed"
    assert records[0].failure_reason == "Measured client failure"


def test_cli_preview_never_writes_and_apply_saves_report(task, monkeypatch):
    import sys

    SimulationTaskDao().save(task)
    # DAO-only fixtures need the same persisted request sidecar as the service.
    directory = Path(task.task_dir)
    directory.mkdir(parents=True)
    (directory / "per_request.json").write_text(
        json.dumps([row.model_dump(mode="json") for row in task.result.per_request])
    )
    monkeypatch.setattr(sys, "argv", ["benchmark_evidence", "--task-id", task.id])
    evidence.main()
    assert not BenchmarkRecordDao().list()
    assert "ingestion" not in SimulationTaskDao().get(task.id).benchmark_evidence
    monkeypatch.setattr(sys, "argv", ["benchmark_evidence", "--task-id", task.id, "--apply"])
    evidence.main()
    assert BenchmarkRecordDao().get("simulation:abcdef12")
    assert SimulationTaskDao().get(task.id).benchmark_evidence["ingestion"]["status"] == "stored"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
async def test_lifecycle_persists_snapshot_before_traffic_and_verifies_before_ingestion(
    task, monkeypatch, tmp_path, terminal_status
):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    service.reset_service_state()
    task.status = "queued"
    result = task.result
    task.result = None
    snapshot = deepcopy(task.benchmark_evidence)
    timestamps = iter(
        (
            task.started_at,
            task.execution_started_at,
            task.completed_at,
            snapshot["benchmark_context"].pop("verified_at"),
        )
    )
    monkeypatch.setattr(service, "_now", timestamps.__next__)
    task.benchmark_evidence = None
    task.execution_started_at = None
    order = []

    async def capture(current):
        order.append("capture")
        current.benchmark_evidence = deepcopy(snapshot)

    async def verify(current):
        order.append("verify")
        assert current.completed_at
        assert not service.save_jobs.get(current.id)
        assert BenchmarkRecordDao().get(f"simulation:{current.id}") is None
        saved_snapshot = current.benchmark_evidence
        await asyncio.sleep(0)
        assert current.benchmark_evidence is saved_snapshot
        saved_snapshot["benchmark_context"]["verified_at"] = service._now()

    ingest = evidence.ingest_simulation_task

    def ingest_after_verification(current):
        order.append("ingest")
        assert current.benchmark_evidence["benchmark_context"]["verified_at"]
        return ingest(current)

    async def run(current, context):
        order.append("traffic")
        assert SimulationTaskDao().get(task.id).benchmark_evidence == snapshot
        context.on_ready()
        context.progress(99, "Finalizing simulation results")
        if terminal_status != "completed":
            current.result = result
            if terminal_status == "cancelled":
                raise service.SimulationCancelledError("Cancelled after measurement")
            raise service.SimulationError("Failed after measurement")
        return result

    monkeypatch.setattr(evidence, "capture_simulation_context", capture)
    monkeypatch.setattr(evidence, "verify_simulation_context", verify)
    monkeypatch.setattr(evidence, "ingest_simulation_task", ingest_after_verification)
    monkeypatch.setattr(
        service,
        "_resolve_task_endpoint_url",
        AsyncMock(
            return_value=service.ResolvedTaskEndpoint(
                url=task.endpoint_url,
                cluster_id=task.endpoint_cluster_id,
                namespace=task.endpoint_namespace,
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "get_backend",
        lambda _: SimpleNamespace(
            descriptor=lambda: SimpleNamespace(version="0.12.0"),
            run=run,
        ),
    )
    await service._run_task(task)
    assert order == ["capture", "traffic", "verify", "ingest"]
    saved = SimulationTaskDao().get(task.id)
    assert saved.status == terminal_status
    assert saved.benchmark_evidence["benchmark_context"]["verified_at"]
    record = BenchmarkRecordDao().get("simulation:abcdef12")
    if terminal_status == "cancelled":
        assert record is None
        assert saved.benchmark_evidence["ingestion"]["status"] == "skipped"
    else:
        assert record is not None
        assert record.outcome == ("succeeded" if terminal_status == "completed" else "failed")
        assert saved.benchmark_evidence["ingestion"]["status"] == "stored"
    service.reset_service_state()


@pytest.mark.asyncio
async def test_model_service_route_and_pool_are_read_and_multi_target_traffic_is_rejected(task, monkeypatch):
    from llm_d_lens.model_service.gateway_providers import INFERENCE_POOL_API_GROUP

    task.model_service_group_id = "msg-1"
    task.model_name = "published-model"
    group = SimpleNamespace(
        id="msg-1", name="published-model", status="active", model_ref="org/model", served_name="org/model"
    )
    member = SimpleNamespace(
        id="member-1",
        cluster_id="cluster1",
        status="active",
        weight=1,
        pool_name="serving/pool-1",
        target_namespace="serving",
        execution_id="execution-1",
    )
    members = [member]
    monkeypatch.setattr(
        "llm_d_lens.model_service.service.default_service",
        lambda: SimpleNamespace(
            groups=SimpleNamespace(get=lambda _: group),
            members=SimpleNamespace(list_by_group=lambda _: members),
        ),
    )
    monkeypatch.setattr(
        "llm_d_lens.cluster.registry.get_cluster",
        lambda _: SimpleNamespace(
            id="cluster1",
            gateway_namespace="gateway",
        ),
    )
    route = {
        "metadata": {"uid": "route-uid", "generation": 1},
        "status": {
            "parents": [
                {
                    "conditions": [
                        {"type": name, "status": "True", "observedGeneration": 1}
                        for name in ("Accepted", "ResolvedRefs")
                    ]
                }
            ]
        },
        "spec": {
            "rules": [
                {
                    "backendRefs": [
                        {
                            "group": INFERENCE_POOL_API_GROUP,
                            "kind": "InferencePool",
                            "namespace": "serving",
                            "name": "pool-1",
                        }
                    ]
                }
            ]
        },
    }
    pool = {"metadata": {"uid": "pool-uid", "generation": 1}, "spec": {"selector": {"matchLabels": {"app": "serving"}}}}
    calls = []

    async def query(argv, **_):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps(route if "httproute" in argv else pool))

    monkeypatch.setattr(evidence, "scoped_runner", lambda _: SimpleNamespace(run=query))
    snapshot = await evidence._routing_snapshot(task)
    assert snapshot["pod_selector"] == "app=serving"
    assert len(calls) == 2
    route["spec"]["rules"][0]["backendRefs"].append(
        {
            "group": INFERENCE_POOL_API_GROUP,
            "kind": "InferencePool",
            "namespace": "other",
            "name": "pool-2",
        }
    )
    with pytest.raises(ValueError, match="exactly"):
        await evidence._routing_snapshot(task)
    members.append(
        SimpleNamespace(
            cluster_id="cluster1",
            status="unhealthy",
            weight=1,
            execution_id="other-execution",
        )
    )
    with pytest.raises(ValueError, match="single deployment"):
        await evidence._routing_snapshot(task)


@pytest.mark.asyncio
async def test_route_change_invalidates_context_instead_of_relabeling_history(task, monkeypatch):
    task.benchmark_evidence["routing"] = {"route_uid": "old"}
    monkeypatch.setattr(evidence, "_routing_snapshot", AsyncMock(return_value={"route_uid": "new"}))
    verify = AsyncMock()
    monkeypatch.setattr(evidence, "verify_benchmark_context", verify)
    await evidence.verify_simulation_context(task)
    assert "routing changed" in task.benchmark_evidence["benchmark_context_error"]
    assert "verified_at" not in task.benchmark_evidence["benchmark_context"]
    verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [["/bin/bash", "-c"], ["/opt/serve-model"]])
@pytest.mark.parametrize("probe_exit", [0, 1])
@pytest.mark.parametrize("model_service", [False, True])
async def test_validated_identity_does_not_require_parsing_serving_command(
    task, monkeypatch, command, probe_exit, model_service
):
    saved = deepcopy(task.benchmark_evidence)
    if model_service:
        task.model_service_group_id = "msg-1"
        task.model_name = "published-model"
    routing = {
        "group_id": "msg-1",
        "member_id": "member-1",
        "execution_id": "execution-1",
        "model_ref": "org/model",
        "served_name": "published-model",
        "pod_selector": "app=serving",
    }
    monkeypatch.setattr(evidence, "_routing_snapshot", AsyncMock(return_value=routing if model_service else None))
    configuration = SimpleNamespace(
        content=saved["deployment_configuration"]["content"],
        model_dump=lambda **_: deepcopy(saved["deployment_configuration"]),
    )
    context = SimpleNamespace(cluster_id="cluster1", namespace="serving", run_id="run-1", case_id="case-1")
    run = SimpleNamespace(
        cases=[SimpleNamespace(id="case-1", execution_id="execution-1", source_configuration_ordinal=0)],
        source_configurations=[configuration],
    )
    monkeypatch.setattr(evidence, "get_execution_context", lambda _: context)
    monkeypatch.setattr(evidence.deployment_run_manager.store, "get_run", lambda _: run)
    container = {
        "name": "modelserver",
        "command": command,
        "args": ["exec vllm serve org/model --tensor-parallel-size=1"] if command[0] == "/bin/bash" else [],
    }
    pod = {
        "metadata": {"name": "model-0", "uid": "pod-uid"},
        "status": {
            "containerStatuses": [
                {"name": "modelserver", "ready": True, "imageID": "sha256:actual-image", "restartCount": 0}
            ]
        },
    }
    probe = {
        "model": {"model_id": "org/model"},
        "hardware": saved["benchmark_context"]["hardware"],
        "runtime": saved["benchmark_context"]["runtime"],
        "warnings": ["model:ValueError"],
    }
    execute = AsyncMock(return_value=SimpleNamespace(returncode=probe_exit, stdout=json.dumps(probe)))
    monkeypatch.setattr(
        benchmark_context, "_model_pods", AsyncMock(return_value=(SimpleNamespace(run=execute), [(pod, container)]))
    )
    monkeypatch.setattr(benchmark_context, "enrich_model_metadata", AsyncMock())
    monkeypatch.setattr(
        benchmark_context,
        "_argument",
        lambda *_: pytest.fail("Validated Simulation identity must not parse serving argv"),
    )

    await evidence.capture_simulation_context(task)
    snapshot = task.benchmark_evidence["benchmark_context"]
    assert snapshot["model"]["model_id"] == "org/model"
    expected_identity = {
        "source": "deployment-configuration",
        "model_id": "org/model",
        "execution_id": "execution-1",
    }
    if model_service:
        expected_identity.update(source="model-service-routing", group_id="msg-1", member_id="member-1")
    assert snapshot["model_identity"] == expected_identity
    assert snapshot["serving_configuration"][0]["model_source"] is None
    assert snapshot["serving_configuration"][0]["arguments"]["--tensor-parallel-size"] is None
    assert snapshot["warnings"]
    execute.assert_awaited_once()
    assert execute.call_args.args[0][7:9] == ["--", "python3"]
    assert execute.call_args.args[0][-4:] == ["", "org/model", "", ""]

    await evidence.verify_simulation_context(task)
    assert snapshot["verified_at"]
    snapshot["captured_at"] = saved["benchmark_context"]["captured_at"]
    snapshot["verified_at"] = saved["benchmark_context"]["verified_at"]
    report = evidence.ingest_simulation_task(task)
    assert report["status"] == "stored"
    record = BenchmarkRecordDao().get("simulation:abcdef12")
    assert record.model.model_id == "org/model"
    assert record.model.quantization is None
    assert record.model.revision is None
    assert record.provenance["serving_snapshot"]["model_identity"] == snapshot["model_identity"]
    assert record.hardware.accelerator == (None if probe_exit else "Actual XPU")
    assert record.runtime.backend == (None if probe_exit else "vllm")
    assert record.configuration.decode_tp == 1
    assert not BenchmarkRecordDao().find_compatible(
        accelerator="Actual XPU",
        backend="vllm",
        quantization="none",
        topology=record.configuration.topology,
        minimum_vram_per_gpu_gib=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conflict", "reason"),
    [
        ("model_ref", "identity does not match"),
        ("group_id", "identity does not match"),
        ("execution_id", "identity does not match"),
        ("allocation", "saved deployment allocation"),
        ("replicas", "saved deployment topology"),
        ("per-replica-allocation", "saved deployment topology"),
    ],
)
async def test_validated_identity_does_not_allow_attribution_or_allocation_conflicts(
    task, monkeypatch, conflict, reason
):
    source = deepcopy(task.benchmark_evidence)
    source.pop("benchmark_context")
    source["model_service_group_id"] = "msg-1"
    source["routing"] = {
        "group_id": "msg-1",
        "member_id": "member-1",
        "execution_id": "execution-1",
        "model_ref": "org/model",
    }
    if conflict in source["routing"]:
        source["routing"][conflict] = "different"
    counts = [1]
    if conflict == "allocation":
        counts = [2]
    elif conflict == "replicas":
        counts = [None, None]
    elif conflict == "per-replica-allocation":
        counts = [1, 3]
        content = source["deployment_configuration"]["content"]
        content["decode"].update(tensorParallelSize=2, replicaCount=2)
        source["deployment_configuration"]["checksum"] = configuration_checksum(content)
    targets = [
        (
            {
                "metadata": {"name": f"model-{index}", "uid": f"pod-{index}"},
                "status": {
                    "containerStatuses": [
                        {"name": "modelserver", "ready": True, "imageID": "sha256:actual-image", "restartCount": 0}
                    ]
                },
            },
            {"name": "modelserver"},
        )
        for index in range(len(counts))
    ]
    responses = [
        SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"model": {"model_id": "org/model"}, "hardware": {"gpu_count": count}, "runtime": {}, "warnings": []}
            ),
        )
        for count in counts
    ]
    monkeypatch.setattr(
        benchmark_context,
        "_model_pods",
        AsyncMock(return_value=(SimpleNamespace(run=AsyncMock(side_effect=responses)), targets)),
    )
    monkeypatch.setattr(benchmark_context, "enrich_model_metadata", AsyncMock())
    await benchmark_context.capture_benchmark_context(source)
    assert "benchmark_context" not in source
    assert reason in source["benchmark_context_error"]


@pytest.mark.asyncio
async def test_queued_model_service_binding_survives_restart_without_claiming_a_snapshot(task, monkeypatch, tmp_path):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    service.reset_service_state()
    task.model_service_group_id = "msg-1"
    task.benchmark_evidence = None
    task.status = "queued"
    task.result = None
    await service.save_task(task)
    loaded = SimulationTaskDao().get(task.id)
    assert loaded.model_service_group_id == "msg-1"
    assert loaded.benchmark_evidence == {"model_service_group_id": "msg-1"}
    assert not evidence.normalize_simulation_task(loaded)[0]
    service.reset_service_state()


@pytest.mark.asyncio
async def test_rerun_does_not_inherit_a_verified_snapshot_or_ingestion(task, monkeypatch, tmp_path):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path))
    service.reset_service_state()
    evidence.ingest_simulation_task(task)
    monkeypatch.setattr(service, "load_task", AsyncMock(return_value=task))
    monkeypatch.setattr(service, "save_task", AsyncMock())
    monkeypatch.setattr(service, "_schedule_task", lambda _: None)
    monkeypatch.setattr(service, "_task_id_exists", lambda _: False)
    monkeypatch.setattr(
        service,
        "_resolve_task_endpoint_url",
        AsyncMock(
            return_value=service.ResolvedTaskEndpoint(
                url=task.endpoint_url,
                cluster_id=task.endpoint_cluster_id,
                namespace=task.endpoint_namespace,
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "get_backend",
        lambda _: SimpleNamespace(
            descriptor=lambda: SimpleNamespace(capabilities={}),
            validate=AsyncMock(),
        ),
    )
    rerun = await service.rerun_task(task.id)
    assert rerun.id != task.id
    assert rerun.benchmark_evidence is None
    assert rerun.result is None
    assert rerun.execution_started_at is None
    assert task.benchmark_evidence["ingestion"]["status"] == "stored"
    service.reset_service_state()
