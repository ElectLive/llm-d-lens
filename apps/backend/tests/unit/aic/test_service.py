"""Tests for the AISimulate estimate adapter."""

from types import SimpleNamespace

import pytest
import yaml

from llm_d_lens.aic import service


def estimate_request(**kwargs):
    return service.AICEstimateRequest(aic_system_name="test-system", **kwargs)


@pytest.mark.parametrize("target", ["target_concurrency", "target_request_rate"])
def test_search_routes_load_target_to_recommend_and_scales_replicas(monkeypatch, target):
    import pandas as pd
    from aisimulate.legacy_cli import api

    captured = {}

    def recommend(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            chosen_exp="agg",
            best_configs={
                "agg": pd.DataFrame(
                    [{"tp": 2, "num_total_gpus": 2, "replicas_needed": 3, "tokens/s": 100, "ttft": 100, "tpot": 10}]
                )
            },
        )

    monkeypatch.setattr(api, "cli_recommend", recommend)
    monkeypatch.setattr(service, "_cli", lambda: (lambda **kwargs: pytest.fail("default called"), None, None))
    result = service.search_sync(
        service.AICRequest(model_name="Qwen/Qwen3-8B", aic_system_name="test-system", gpu_count=8, **{target: 10})
    )
    assert captured[target] == 10
    assert captured["isl"] == 4000
    assert captured["osl"] == 1000
    assert "total_gpus" not in captured
    assert result.configs[0]["replicas"] == 3
    assert result.configs[0]["num_total_gpus"] == 6
    assert result.configs[0]["throughput_tokens_per_sec"] == 300


def test_search_rejects_joint_load_targets():
    with pytest.raises(ValueError, match="choose one and clear the other"):
        service.AICRequest(
            model_name="Qwen/Qwen3-8B",
            aic_system_name="test-system",
            target_concurrency=10,
            target_request_rate=2,
        )


def test_search_defaults_match_native_search_and_recommend():
    from inspect import signature

    from aisimulate.legacy_cli.api import cli_default, cli_recommend

    request = service.AICRequest(model_name="Qwen/Qwen3-8B", aic_system_name="test-system")
    for operation in (cli_default, cli_recommend):
        assert request.mean_input_tokens == signature(operation).parameters["isl"].default
        assert request.mean_output_tokens == signature(operation).parameters["osl"].default


def test_search_without_load_target_keeps_default_search(monkeypatch):
    captured = {}

    def default(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(best_configs={}, chosen_exp=None)

    monkeypatch.setattr(service, "_cli", lambda: (default, None, None))
    service.search_sync(service.AICRequest(model_name="model", aic_system_name="system", gpu_count=4))
    assert captured["total_gpus"] == 4
    assert "target_concurrency" not in captured
    assert "target_request_rate" not in captured


def test_search_scales_both_pd_worker_pools():
    result = service._candidate(
        "disagg",
        {
            "(p)tp": 2,
            "(p)workers": 1,
            "(d)tp": 1,
            "(d)workers": 2,
            "num_total_gpus": 4,
            "replicas_needed": 3,
            "tokens/s": 100,
        },
    )
    assert result["prefill_replicas"] == 3
    assert result["decode_replicas"] == 6
    assert result["num_total_gpus"] == 12
    assert result["throughput_tokens_per_sec"] == 300


def test_search_rejects_closest_recommendation_that_exceeds_slo(monkeypatch):
    import pandas as pd
    from aisimulate.legacy_cli import api

    monkeypatch.setattr(service, "_cli", lambda: (None, None, None))
    monkeypatch.setattr(
        api,
        "cli_recommend",
        lambda **kwargs: SimpleNamespace(
            chosen_exp="agg",
            best_configs={"agg": pd.DataFrame([{"tp": 1, "num_total_gpus": 1, "ttft": 3000, "tpot": 50}])},
        ),
    )
    with pytest.raises(service.AICError, match="no configuration meeting the TTFT/TPOT"):
        service.search_sync(
            service.AICRequest(
                model_name="model",
                aic_system_name="system",
                gpu_count=4,
                target_concurrency=10,
            )
        )


def test_search_rejects_recommendation_above_capacity(monkeypatch):
    import pandas as pd
    from aisimulate.legacy_cli import api

    monkeypatch.setattr(service, "_cli", lambda: (None, None, None))
    monkeypatch.setattr(
        api,
        "cli_recommend",
        lambda **kwargs: SimpleNamespace(
            chosen_exp="agg",
            best_configs={"agg": pd.DataFrame([{"tp": 2, "num_total_gpus": 2, "replicas_needed": 3}])},
        ),
    )
    with pytest.raises(service.AICError, match="Minimum recommended GPUs: 6"):
        service.search_sync(
            service.AICRequest(
                model_name="Qwen/Qwen3-8B",
                aic_system_name="test-system",
                gpu_count=4,
                target_request_rate=10,
            )
        )


def test_estimate_rejects_a_different_aggregated_topology(monkeypatch):
    monkeypatch.setattr(
        service,
        "experiments_sync",
        lambda request: SimpleNamespace(
            experiments=[
                {
                    "mode": "agg",
                    "tp": 2,
                    "replicas": 4,
                    "ttft_ms": 120,
                    "tpot_ms": 20,
                }
            ]
        ),
    )

    with pytest.raises(service.AICError, match="topology"):
        service.estimate_sync(
            estimate_request(
                scenario="inference_scheduling",
                model_name="Qwen/Qwen3-8B",
                gpu_count=4,
                tp=2,
                replicas=2,
                ttft_target_ms=200,
            )
        )


def test_estimate_passes_exact_aggregated_topology_and_workload(monkeypatch):
    captured = {}

    def experiments(request):
        captured.update(yaml.safe_load(request.yaml_text)["manual"])
        return SimpleNamespace(
            experiments=[
                {
                    "mode": "agg",
                    "tp": 2,
                    "replicas": 2,
                    "ttft_ms": 120,
                    "tpot_ms": 20,
                }
            ]
        )

    monkeypatch.setattr(service, "experiments_sync", experiments)
    result = service.estimate_sync(
        estimate_request(
            scenario="inference_scheduling",
            model_name="Qwen/Qwen3-8B",
            gpu_count=4,
            tp=2,
            replicas=2,
            mean_input_tokens=4096,
            mean_output_tokens=256,
            ttft_target_ms=200,
        )
    )

    assert result["ttft_ms"] == 120
    assert captured["total_gpus"] == 4
    assert captured["agg_tp_candidates"] == [2]
    assert captured["agg_num_gpu_candidates"] == [2]
    assert captured["isl"] == 4096


def test_estimate_rejects_inconsistent_gpu_count(monkeypatch):
    monkeypatch.setattr(service, "experiments_sync", lambda request: pytest.fail("SDK must not be called"))

    with pytest.raises(service.AICError, match="requires 4 GPUs"):
        service.estimate_sync(
            estimate_request(
                scenario="inference_scheduling",
                model_name="Qwen/Qwen3-8B",
                gpu_count=8,
                tp=2,
                replicas=2,
            )
        )


def test_estimate_rejects_unverifiable_pipeline_parallelism(monkeypatch):
    monkeypatch.setattr(service, "experiments_sync", lambda request: pytest.fail("SDK must not be called"))
    with pytest.raises(service.AICError, match="PP greater than 1"):
        service.estimate_sync(
            estimate_request(
                scenario="inference_scheduling",
                model_name="Qwen/Qwen3-8B",
                gpu_count=4,
                tp=2,
                pp=2,
            )
        )


def test_estimate_rejects_a_different_disaggregated_worker_allocation(monkeypatch):
    monkeypatch.setattr(
        service,
        "experiments_sync",
        lambda request: SimpleNamespace(
            experiments=[
                {
                    "mode": "disagg",
                    "prefill_tp": 2,
                    "prefill_replicas": 1,
                    "decode_tp": 1,
                    "decode_replicas": 2,
                    "ttft_ms": 120,
                    "tpot_ms": 20,
                }
            ]
        ),
    )

    with pytest.raises(service.AICError, match="topology"):
        service.estimate_sync(
            estimate_request(
                scenario="pd_disaggregation",
                model_name="Qwen/Qwen3-8B",
                gpu_count=3,
                prefill_tp=2,
                prefill_replicas=1,
                decode_tp=1,
                decode_replicas=1,
                ttft_target_ms=200,
            )
        )


def test_estimate_accepts_matching_disaggregated_topology(monkeypatch):
    captured = {}

    def experiments(request):
        captured.update(yaml.safe_load(request.yaml_text)["manual"])
        return SimpleNamespace(
            experiments=[
                {
                    "mode": "disagg",
                    "prefill_tp": 2,
                    "prefill_replicas": 1,
                    "decode_tp": 1,
                    "decode_replicas": 2,
                    "num_total_gpus": 4,
                    "ttft_ms": 180,
                    "tpot_ms": 25,
                }
            ]
        )

    monkeypatch.setattr(service, "experiments_sync", experiments)
    result = service.estimate_sync(
        estimate_request(
            scenario="pd_disaggregation",
            model_name="Qwen/Qwen3-8B",
            gpu_count=4,
            prefill_tp=2,
            prefill_replicas=1,
            decode_tp=1,
            decode_replicas=2,
            ttft_target_ms=200,
        )
    )

    assert result["tpot_ms"] == 25
    assert captured["prefill_tp_candidates"] == [2]
    assert captured["decode_tp_candidates"] == [1]
    assert captured["total_gpus"] == 4


def test_estimate_without_targets_uses_direct_aggregated_prediction(monkeypatch):
    from aisimulate.legacy_cli import api

    captured = {}

    def direct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ttft=45,
            tpot=7,
            raw={
                "tp": 1,
                "pp": 1,
                "num_total_gpus": 1,
                "tokens/s": 150,
            },
        )

    monkeypatch.setattr(api, "cli_estimate", direct)
    result = service.estimate_sync(
        estimate_request(
            scenario="inference_scheduling",
            model_name="Qwen/Qwen3-0.6B",
            gpu_count=1,
            tp=1,
            replicas=1,
            mean_input_tokens=1024,
            mean_output_tokens=256,
        )
    )

    assert result["ttft_ms"] == 45
    assert result["throughput_tokens_per_sec"] == 150
    assert captured["isl"] == 1024
    assert captured["osl"] == 256
    assert captured["batch_size"] == 1
    assert "ttft" not in captured and "tpot" not in captured


def test_estimate_without_targets_rejects_mismatched_direct_topology(monkeypatch):
    from aisimulate.legacy_cli import api

    monkeypatch.setattr(
        api,
        "cli_estimate",
        lambda **kwargs: SimpleNamespace(
            ttft=45,
            tpot=7,
            raw={"(p)tp": 1, "(p)workers": 1, "(d)tp": 1, "(d)workers": 2, "num_total_gpus": 3},
        ),
    )
    with pytest.raises(service.AICError, match="topology"):
        service.estimate_sync(
            estimate_request(
                scenario="pd_disaggregation",
                model_name="Qwen/Qwen3-0.6B",
                gpu_count=2,
                prefill_tp=1,
                prefill_replicas=1,
                decode_tp=1,
                decode_replicas=1,
            )
        )


def test_estimate_without_targets_accepts_matching_direct_disaggregation(monkeypatch):
    from aisimulate.legacy_cli import api

    captured = {}

    def direct(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            ttft=44,
            tpot=7,
            raw={
                "(p)tp": 1,
                "(p)workers": 2,
                "(d)tp": 1,
                "(d)workers": 3,
                "num_total_gpus": 5,
                "tokens/s": 300,
            },
        )

    monkeypatch.setattr(api, "cli_estimate", direct)
    result = service.estimate_sync(
        estimate_request(
            scenario="pd_disaggregation",
            model_name="Qwen/Qwen3-0.6B",
            gpu_count=5,
            prefill_tp=1,
            prefill_replicas=2,
            decode_tp=1,
            decode_replicas=3,
            mean_input_tokens=1024,
            mean_output_tokens=256,
        )
    )

    assert result["throughput_tokens_per_sec"] == 300
    assert result["prefill_replicas"] == 2
    assert result["decode_replicas"] == 3
    assert captured["prefill_num_workers"] == 2
    assert captured["decode_num_workers"] == 3
    assert captured["isl"] == 1024 and captured["osl"] == 256
    assert "ttft" not in captured and "tpot" not in captured


def test_estimate_without_targets_does_not_infer_multi_replica_throughput(monkeypatch):
    from aisimulate.legacy_cli import api

    monkeypatch.setattr(
        api,
        "cli_estimate",
        lambda **kwargs: SimpleNamespace(
            ttft=45,
            tpot=7,
            raw={"tp": 1, "pp": 1, "num_total_gpus": 1, "tokens/s": 150},
        ),
    )
    result = service.estimate_sync(
        estimate_request(
            scenario="inference_scheduling",
            model_name="Qwen/Qwen3-0.6B",
            gpu_count=2,
            tp=1,
            replicas=2,
        )
    )

    assert result["replicas"] == 2
    assert result["throughput_tokens_per_sec"] is None
