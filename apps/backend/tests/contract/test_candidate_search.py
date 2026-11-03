from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from llm_d_lens.aic.models import AICExperimentResponse, AICSearchResponse, AICSupportResponse
from llm_d_lens.api.main import app
from llm_d_lens.simulation import candidate_search as candidates


@pytest.mark.parametrize("supported", [True, False])
def test_predictive_search_handles_aic_response_models(monkeypatch, supported):
    monkeypatch.setattr(
        candidates.aic_service,
        "check_support",
        AsyncMock(
            return_value=AICSupportResponse(supported=supported, agg_supported=supported, disagg_supported=supported)
        ),
    )
    search = AsyncMock(
        return_value=AICSearchResponse(configs=[{"mode": "agg", "tp": 2, "replicas": 1, "ttft_ms": 100, "tpot_ms": 10}])
    )
    monkeypatch.setattr(candidates.aic_service, "search", search)
    response = TestClient(app).post(
        "/api/candidate-search",
        json={
            "sourceIds": ["aic"],
            "workload": {"model": "example/model"},
            "searchConfig": {"totalGpus": 3, "aicSystemName": "test-system"},
        },
    )
    if not supported:
        assert response.status_code == 400
        assert "not supported" in response.json()["error"]
        search.assert_not_awaited()
        return
    assert response.status_code == 200
    candidate = response.json()["candidates"][0]
    assert candidate["totalGpus"] == 2
    assert candidate["predicted"] == {"ttftMs": 100, "tpotMs": 10, "throughputTps": None}
    search.assert_awaited_once()


@pytest.mark.parametrize("mode", ["agg", "disagg"])
def test_manual_estimate_preserves_dictionary_response(monkeypatch, mode):
    monkeypatch.setattr(
        candidates.aic_service,
        "check_support",
        AsyncMock(return_value=AICSupportResponse(supported=True, agg_supported=True, disagg_supported=True)),
    )
    estimate = AsyncMock(return_value={"mode": mode, "tp": 1, "ttft_ms": 100, "tpot_ms": 10})
    monkeypatch.setattr(candidates.aic_service, "estimate", estimate)
    response = TestClient(app).post(
        "/api/candidate-search",
        json={
            "sourceIds": ["manual"],
            "workload": {"model": "example/model"},
            "searchConfig": {"aicSystemName": "test-system"},
            "manualConfig": {"servingMode": mode},
        },
    )
    assert response.status_code == 200
    candidate = response.json()["candidates"][0]
    assert candidate["sourceLabel"] == "Manual Estimate"
    assert candidate["topologyMode"] == mode
    assert candidate["predicted"]["ttftMs"] == 100
    estimate.assert_awaited_once()


def test_grid_candidates_preserve_order_budget_and_names():
    response = TestClient(app).post(
        "/api/candidate-search",
        json={
            "sourceIds": ["baseline", "baseline", "tiered-cache", "pd", "epd", "unknown"],
            "searchConfig": {"totalGpus": 4, "maxCandidatesPerMode": 2},
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["backend"] == "local-aic"
    assert len(result["candidates"]) == 8
    assert result["candidates"][0]["id"] == "baseline-baseline-tp1-rep4-0"
    assert result["candidates"][2]["id"] == "tiered-cache-tpc-tp1-rep4-0"
    assert all(candidate["totalGpus"] <= 4 for candidate in result["candidates"])
    assert all(not candidate["deployable"] for candidate in result["candidates"][-2:])


@pytest.mark.parametrize("budget", [0, 65, "4", True, 1.5])
def test_relaxed_requires_explicit_valid_budget(budget):
    response = TestClient(app).post(
        "/api/candidate-search/relaxed",
        json={
            "workload": {"model": "example/model"},
            "searchConfig": {"totalGpus": budget},
        },
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Model and explicit totalGpus (1-64) are required"}


def test_relaxed_retries_no_feasible_and_rejects_over_budget(monkeypatch):
    search = AsyncMock(
        side_effect=[
            candidates.aic_service.AICError(candidates.NO_FEASIBLE),
            AICSearchResponse(
                configs=[
                    {"mode": "agg", "tp": 8, "replicas": 1, "ttft_ms": 100, "tpot_ms": 10},
                    {"mode": "agg", "tp": 2, "replicas": 2, "ttft_ms": 120, "tpot_ms": 10},
                ]
            ),
        ]
    )
    monkeypatch.setattr(
        candidates.aic_service,
        "check_support",
        AsyncMock(return_value=AICSupportResponse(supported=True, agg_supported=True, disagg_supported=True)),
    )
    monkeypatch.setattr(candidates.aic_service, "search", search)
    response = TestClient(app).post(
        "/api/candidate-search/relaxed",
        json={
            "workload": {"model": "example/model", "ttftMs": 100},
            "searchConfig": {"totalGpus": 4, "aicSystemName": "test-system"},
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert [attempt["factor"] for attempt in result["attempts"]] == [1, 1.25]
    assert result["anchor"]["totalGpus"] == 4
    assert result["anchor"]["aicRank"] == 2
    assert result["sloSatisfied"] is False
    assert search.call_args_list[1].args[0].ttft_target_ms == 125


def test_manual_yaml_preserves_chosen_and_replica_inference(monkeypatch):
    monkeypatch.setattr(
        candidates.aic_service,
        "experiments",
        AsyncMock(
            return_value=AICExperimentResponse(
                experiments=[
                    {"mode": "agg", "tp": 2, "pp": 2, "num_total_gpus": 8, "is_chosen": True, "name": "chosen config"},
                ]
            )
        ),
    )
    response = TestClient(app).post(
        "/api/candidate-search",
        json={
            "sourceIds": ["manual"],
            "manualConfig": {"inputMode": "yaml", "yamlText": "experiments: []"},
        },
    )
    assert response.status_code == 200
    candidate = response.json()["candidates"][0]
    assert candidate["decodeReplicas"] == 2
    assert candidate["pipelineParallel"] == 2
    assert candidate["isChosenExperiment"] is True
    assert candidate["id"] == "manual-chosen-config-0"


def test_unknown_sources_are_ignored():
    assert TestClient(app).post("/api/candidate-search", json={"sourceIds": ["unknown"]}).json() == {"candidates": []}


@pytest.mark.parametrize("field", ["concurrency", "request_rate"])
def test_support_preserves_load_target_and_upstream_token_defaults(monkeypatch, field):
    result = AICSupportResponse(supported=True, agg_supported=True, disagg_supported=True)
    support = AsyncMock(return_value=result)
    monkeypatch.setattr(candidates.aic_service, "check_support", support)
    response = TestClient(app).post(
        "/api/candidate-support",
        json={
            "workload": {"model": "example/model", field: 10},
            "searchConfig": {"aicSystemName": "test-system"},
        },
    )
    assert response.status_code == 200
    assert response.json() == result.model_dump(mode="json")
    payload = support.call_args.args[0]
    assert payload.mean_input_tokens == 4000
    assert payload.mean_output_tokens == 1000
    assert getattr(payload, f"target_{field}") == 10


def test_aic_error_details_are_logged_not_exposed(monkeypatch, caplog):
    detail = "internal traceback /private/aic/config credential-value"
    monkeypatch.setattr(
        candidates.aic_service, "check_support", AsyncMock(side_effect=candidates.aic_service.AICError(detail))
    )
    response = TestClient(app).post(
        "/api/candidate-support",
        json={"workload": {"model": "example/model"}, "searchConfig": {"aicSystemName": "test-system"}},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "AIC candidate search failed"}
    assert detail not in response.text
    assert detail in caplog.text


def test_support_rejects_joint_load_targets_before_calling_aic(monkeypatch):
    support = AsyncMock()
    monkeypatch.setattr(candidates.aic_service, "check_support", support)
    response = TestClient(app).post(
        "/api/candidate-support",
        json={
            "workload": {"model": "example/model", "concurrency": 10, "request_rate": 2},
            "searchConfig": {"aicSystemName": "test-system"},
        },
    )
    assert response.status_code == 422
    assert "only one load target" in response.json()["error"]
    support.assert_not_awaited()
