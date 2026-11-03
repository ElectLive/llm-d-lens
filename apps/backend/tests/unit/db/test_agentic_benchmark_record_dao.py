from datetime import UTC, datetime

import pytest

from llm_d_lens.agentic.benchmark_evidence import (
    BenchmarkRecord,
    CandidateConfiguration,
    HardwareDescriptor,
    ModelDescriptor,
    RuntimeFingerprint,
    WorkloadProfile,
)
from llm_d_lens.db.dao.agentic_benchmark_record import BenchmarkRecordDao
from llm_d_lens.db.models.agentic_benchmark_record import _evaluate_run_id


def test_stage_evidence_soft_reference_retains_only_the_source_run_id():
    run_id = "00000000-0000-4000-8000-000000000001"
    assert _evaluate_run_id(f"evaluate:{run_id}:matrix:0:stage:1") == run_id
    assert _evaluate_run_id(f"evaluate:{run_id}:stage:0") == run_id
    assert _evaluate_run_id(f"evaluate:{run_id}") == run_id
    assert _evaluate_run_id("simulation:abcd1234") is None


def _record(benchmark_id: str, *, outcome: str = "succeeded", **overrides) -> BenchmarkRecord:
    values = {
        "timestamp": datetime(2026, 9, 20, tzinfo=UTC),
        "outcome": outcome,
        "model": ModelDescriptor(
            model_id="Qwen/Qwen3-8B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=8,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-8B",
        ),
        "hardware": HardwareDescriptor(accelerator="Intel Gaudi 3", vram_per_gpu_gib=96, gpu_count=4),
        "runtime": RuntimeFingerprint(backend="vllm", backend_version="0.10.0", accelerator_runtime="synapse"),
        "workload": WorkloadProfile(
            mean_input_tokens=4096,
            p95_input_tokens=8192,
            mean_output_tokens=512,
            concurrency=16,
            request_rate=4,
            shared_prefix_ratio=0.5,
            prefill_fraction=0.5,
        ),
        "configuration": CandidateConfiguration(topology="baseline-vllm", decode_tp=2, decode_replicas=2),
    }
    values.update(overrides)
    return BenchmarkRecord(benchmark_id=benchmark_id, **values)


def test_find_compatible_preserves_matching_success_and_failure_evidence():
    dao = BenchmarkRecordDao()
    matching_failure = _record("matching-failure", outcome="failed", failure_reason="OOM")
    matching_success = _record("matching-success")
    incompatible_accelerator = _record(
        "wrong-accelerator",
        hardware=HardwareDescriptor(accelerator="NVIDIA H100", vram_per_gpu_gib=96, gpu_count=4),
    )
    insufficient_vram = _record(
        "insufficient-vram",
        hardware=HardwareDescriptor(accelerator="Intel Gaudi 3", vram_per_gpu_gib=80, gpu_count=4),
    )
    old_record = _record("old", timestamp=datetime(2026, 6, 1, tzinfo=UTC))
    wrong_topology = _record(
        "wrong-topology",
        configuration=CandidateConfiguration(topology="pd-disaggregation", decode_tp=2, decode_replicas=2),
    )
    for record in (
        matching_failure,
        matching_success,
        incompatible_accelerator,
        insufficient_vram,
        old_record,
        wrong_topology,
    ):
        dao.save(record)

    records = dao.find_compatible(
        accelerator="Intel Gaudi 3",
        backend="vllm",
        quantization="bf16",
        topology="baseline-vllm",
        minimum_vram_per_gpu_gib=96,
        observed_after=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert [record.benchmark_id for record in records] == ["matching-failure", "matching-success"]
    assert [record.outcome for record in records] == ["failed", "succeeded"]


def test_find_compatible_does_not_filter_continuous_similarity_inputs():
    dao = BenchmarkRecordDao()
    different_model_and_workload = _record(
        "continuous-difference",
        model=ModelDescriptor(
            model_id="Qwen/Qwen3-32B",
            family="Qwen",
            architecture="dense",
            parameter_count_b=32,
            quantization="bf16",
            tokenizer_id="Qwen/Qwen3-32B",
        ),
        workload=WorkloadProfile(
            mean_input_tokens=8192,
            p95_input_tokens=16384,
            mean_output_tokens=1024,
            concurrency=32,
            request_rate=8,
            shared_prefix_ratio=0.1,
            prefill_fraction=0.8,
        ),
    )
    dao.save(different_model_and_workload)

    records = dao.find_compatible(
        accelerator="Intel Gaudi 3",
        backend="vllm",
        quantization="bf16",
        topology="baseline-vllm",
        minimum_vram_per_gpu_gib=96,
    )

    assert records == [different_model_and_workload]


def test_find_seed_candidates_returns_hard_compatible_records_across_topologies():
    dao = BenchmarkRecordDao()
    baseline = _record("baseline")
    disaggregated = _record(
        "disaggregated",
        configuration=CandidateConfiguration(
            topology="pd-disaggregation",
            decode_tp=2,
            decode_replicas=2,
            prefill_tp=2,
            prefill_replicas=1,
        ),
    )
    incompatible = _record(
        "wrong-backend",
        runtime=RuntimeFingerprint(backend="tgi", backend_version="0.10.0", accelerator_runtime="synapse"),
    )
    for record in (baseline, disaggregated, incompatible):
        dao.save(record)

    records = dao.find_seed_candidates(
        accelerator="Intel Gaudi 3",
        backend="vllm",
        quantization="bf16",
        minimum_vram_per_gpu_gib=96,
    )

    assert [record.benchmark_id for record in records] == ["baseline", "disaggregated"]


@pytest.mark.parametrize("saved_label", ["Intel Gaudi 3", "INTEL GAUDI 3", "4x Intel Gaudi 3 96GiB"])
@pytest.mark.parametrize("query_label", ["Intel Gaudi 3", "INTEL GAUDI 3", "4x Intel Gaudi 3 96GiB"])
def test_hardware_aliases_match_raw_and_previously_normalized_records(saved_label, query_label):
    dao = BenchmarkRecordDao()
    record = _record(
        "hardware-alias",
        hardware=HardwareDescriptor(accelerator=saved_label, vram_per_gpu_gib=96, gpu_count=4),
    )
    dao.save(record)
    filters = {
        "accelerator": query_label,
        "backend": "vllm",
        "quantization": "bf16",
        "minimum_vram_per_gpu_gib": 96,
    }
    assert dao.find_compatible(**filters, topology="baseline-vllm") == [record]
    assert dao.find_seed_candidates(**filters) == [record]
    assert dao.get(record.benchmark_id).hardware.accelerator == saved_label


@pytest.mark.parametrize("accelerator", ["unknown", "H100 + A100"])
def test_unknown_or_mixed_hardware_cannot_establish_dao_compatibility(accelerator):
    dao = BenchmarkRecordDao()
    dao.save(_record("known"))
    filters = {
        "accelerator": accelerator,
        "backend": "vllm",
        "quantization": "bf16",
        "minimum_vram_per_gpu_gib": 96,
    }
    assert dao.find_compatible(**filters, topology="baseline-vllm") == []
    assert dao.find_seed_candidates(**filters) == []
