from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import Engine, event
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.planner import PlanningFacts, WorkloadProfile
from llm_d_lens.agentic.planning_requirements import PlanningRequirements, RuntimeIdentity
from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord, normalize_unit
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InventoryFile,
    ParsedFile,
    PinnedInventory,
    PublicationError,
    SourceApproval,
    item_id,
)
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION, QueryFilters, QueryRequest
from llm_d_lens.agentic.public_benchmark_record.service import (
    EvidenceUnavailableError,
    PublicBenchmarkRecordQueryService,
    plan_query,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, allowed_paths
from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
from llm_d_lens.core.exceptions import ConflictError, DomainValidationError, ForbiddenError, UnauthenticatedError
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao

_PATH = "archives/llm-d-benchmarks.json"
_REVISION = "2026-10-13"
LLM_D_ORIGIN = "https://github.com/llm-d/llm-d"
BENCHMARK_ORIGIN = "https://github.com/llm-d/llm-d-benchmark.git"
_RAW = (
    b'{"runs":[{"id":"run-a","throughput":6131.06177725188,"latency":1.234567890123456789},'
    b'{"id":"run-b","throughput":99,"latency":0.5}]}'
)


def reader() -> Principal:
    return Principal(
        user_id="rag-reader",
        username="reader",
        bindings=(RoleBinding(role="custom", scope_type=ScopeType.GLOBAL),),
        role_permissions={"custom": frozenset({QUERY_PERMISSION})},
    )


def approval(origin: str) -> SourceApproval:
    return SourceApproval(
        verified_public=True,
        approved_by="isolated query fixture",
        approval_reference="fixture only",
        license_reference="not production approval",
        parser_version="query-fixture-v1",
        allowed_paths=allowed_paths(origin),
    )


def publish_numeric(
    publication: PublicBenchmarkRecordPublicationService,
    revision: str = _REVISION,
    *,
    activate: bool = True,
    measurement_copies: int = 1,
    model: str = "Qwen3-0.6B",
):
    digest = publication.archive.put(_RAW)
    file = InventoryFile(path=_PATH, sha256=digest, format="json")
    inventory = PinnedInventory(origin_id=PRISM_ORIGIN, revision=revision, complete=True, files=(file,))
    items = []
    for index, (identity, hardware, throughput, latency, unit) in enumerate(
        (
            ("run-a", "H100", Decimal("6131.06177725188"), Decimal("1.234567890123456789"), "output tokens/s"),
            ("run-b", "H200", Decimal("99"), Decimal("0.5"), "tokens/s"),
        )
    ):
        key = f"{_PATH}#experiment:{identity}"
        experiment = PublicBenchmarkRecord(
            item_id=item_id(PRISM_ORIGIN, key, revision),
            logical_key=key,
            kind="experiment",
            origin_id=PRISM_ORIGIN,
            source_revision=revision,
            content_sha256=digest,
            access_classification="verified_public",
            source_path=_PATH,
            format="json",
            parent_id=item_id(PRISM_ORIGIN, f"file:{_PATH}", revision),
            model=model,
            hardware=hardware,
            runtime="reported-runtime",
            workload={"concurrency": 4 if index == 0 else None},
            configuration={"scenario": identity},
            source_pointer={"record_id": identity, "json_pointer": f"/runs/{index}"},
        )
        items.append(experiment)
        for metric, value, measurement_unit, copy in (
            (metric, value, measurement_unit, copy)
            for copy in range(measurement_copies)
            for metric, value, measurement_unit in (("throughput", throughput, unit), ("latency", latency, "s"))
        ):
            key = f"{_PATH}#measurement:{identity}:{metric}"
            if measurement_copies > 1:
                key += f":copy:{copy}"
            normalized = normalize_unit(value, measurement_unit)
            items.append(
                PublicBenchmarkRecord(
                    item_id=item_id(PRISM_ORIGIN, key, revision),
                    logical_key=key,
                    kind="measurement",
                    origin_id=PRISM_ORIGIN,
                    source_revision=revision,
                    content_sha256=digest,
                    access_classification="verified_public",
                    source_path=_PATH,
                    format="json",
                    parent_id=experiment.item_id,
                    experiment_id=experiment.item_id,
                    metric=metric,
                    statistic="mean",
                    value=value,
                    original_value=value,
                    unit=measurement_unit,
                    original_unit=measurement_unit,
                    normalized_value=normalized[0] if normalized else None,
                    normalized_unit=normalized[1] if normalized else None,
                    normalization_rule=normalized[2] if normalized else None,
                    evidence_type="raw_json",
                    series={"scenario": identity},
                    source_pointer={"record_id": identity, "json_pointer": f"/runs/{index}/{metric}"},
                )
            )
    publication.register_source(PRISM_ORIGIN, approval(PRISM_ORIGIN))
    manifest = publication.stage(
        inventory,
        (ParsedFile(path=_PATH, items=tuple(items)),),
        expected_counts={"source": 1, "experiment": 2, "measurement": 4 * measurement_copies},
    )
    return publication.activate(PRISM_ORIGIN, manifest.item_id) if activate else manifest


@pytest.fixture
def query_service(postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "query-evidence"))
    dao = PublicBenchmarkRecordQueryDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    service = PublicBenchmarkRecordQueryService(dao=dao)
    publication = PublicBenchmarkRecordPublicationService(dao=dao, archive=service.archive)
    publish_numeric(publication)
    assert dao.count_published(PRISM_ORIGIN) > 0
    return service


def numeric(**fields) -> QueryRequest:
    return QueryRequest(
        question="Reported throughput",
        intent="numeric",
        filters=QueryFilters(record_id="run-a", metric="throughput", **fields),
    )


def test_migration_environment_preserves_existing_loggers(postgres_config: Config, monkeypatch: pytest.MonkeyPatch):
    logger = logging.getLogger("llm_d_lens.db.migration_logging_regression")
    monkeypatch.setattr(logger, "disabled", False)

    command.current(postgres_config)

    assert not logger.disabled


def test_hardware_options_are_model_specific_and_published_only(query_service):
    assert query_service.dao.hardware_options(QueryFilters(model="qwen3-0.6b")) == ("H100", "H200")
    assert query_service.dao.hardware_options(QueryFilters(model="unreported-model")) == ()
    publish_numeric(
        PublicBenchmarkRecordPublicationService(dao=query_service.dao), revision="2026-10-14", activate=False
    )
    assert query_service.dao.hardware_options(QueryFilters(model="Qwen3-0.6B")) == ("H100", "H200")
    with pytest.raises(ValidationError, match="approved public"):
        QueryFilters(model="Qwen3-0.6B", origins=(LLM_D_ORIGIN,))


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "qwen/qwen3-0.6b", "Qwen3-0.6B", "another-org/Qwen3-0.6B"])
def test_generated_model_aliases_preserve_hardware_and_source_scope(query_service, model):
    filters = QueryFilters(model=model, hardware="H100", origins=(PRISM_ORIGIN,))
    result = query_service.query(
        QueryRequest(question="Published deployment benchmark evidence", intent="comparison", filters=filters),
        reader(),
    )
    assert result.matched_items == 2
    assert len(result.evidence) == 2
    assert result.filters.model == model
    assert all(item.experiment.model == "Qwen3-0.6B" for item in result.evidence)
    assert all(item.experiment.hardware == "H100" for item in result.evidence)
    assert all(item.citation.origin_id == PRISM_ORIGIN for item in result.evidence)
    assert query_service.dao.hardware_options(filters) == ("H100", "H200")
    assert (
        query_service.dao.search(
            filters.model_copy(update={"hardware": "H100-SXM"}),
            intent="comparison",
            text_query="",
            limit=50,
        ).total
        == 0
    )
    with pytest.raises(PublicationError, match="allowlist"):
        query_service.dao.hardware_options(filters.model_copy(update={"origins": (LLM_D_ORIGIN,)}))


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-0.6B-FP8", "Qwen/Qwen3-8B", "Qwen3-0.6B-v2", "Qwen3-%"],
)
def test_model_aliases_do_not_strip_variant_suffixes_or_use_wildcards(query_service, model):
    filters = QueryFilters(model=model, hardware="H100")
    assert query_service.dao.hardware_options(filters) == ()
    assert query_service.dao.search(filters, intent="comparison", text_query="", limit=50).total == 0


def test_published_full_and_short_names_support_reverse_alias_queries_without_copying_rows(query_service):
    publication = PublicBenchmarkRecordPublicationService(dao=query_service.dao, archive=query_service.archive)
    for index, stored in enumerate(("Publisher/Falcon-7B", "Falcon-7B", "Publisher-Falcon-7B")):
        publish_numeric(publication, revision=f"2026-11-{index + 1:02d}", model=stored)
        requests = ("Publisher/Falcon-7B",)
        if "/" in stored or stored == "Falcon-7B":
            requests += ("falcon-7b",)
        if "/" in stored or stored == "Publisher-Falcon-7B":
            requests += ("Publisher-Falcon-7B",)
        for requested in requests:
            filters = QueryFilters(model=requested, hardware="H100")
            result = query_service.query(
                QueryRequest(question="Published benchmark evidence", intent="comparison", filters=filters), reader()
            )
            assert result.matched_items == len(result.evidence) == 2
            assert all(item.experiment.model == stored for item in result.evidence)
            assert query_service.dao.hardware_options(filters) == ("H100", "H200")
        assert query_service.dao.count_published(PRISM_ORIGIN, kind="experiment") == 2
        assert query_service.dao.count_published(PRISM_ORIGIN, kind="measurement") == 4
        if "/" not in stored:
            missing_boundary = "Publisher-Falcon-7B" if stored == "Falcon-7B" else "Falcon-7B"
            assert (
                query_service.dao.search(
                    QueryFilters(model=missing_boundary), intent="comparison", text_query="", limit=50
                ).total
                == 0
            )


def test_question_and_filter_accept_generated_model_aliases():
    plan = plan_query(
        QueryRequest(
            question="Reported throughput for Qwen3-0.6B on H100",
            filters=QueryFilters(model="Qwen/Qwen3-0.6B"),
        )
    )
    assert plan.filters.model == "Qwen/Qwen3-0.6B"
    with pytest.raises(DomainValidationError, match="model conflicts"):
        plan_query(
            QueryRequest(
                question="Reported throughput for Qwen3-0.6B on H100",
                filters=QueryFilters(model="Qwen/Qwen3-0.6B-FP8"),
            )
        )


def test_multiple_spellings_of_one_model_do_not_require_multiple_model_scope():
    plan = plan_query(
        QueryRequest(
            question="Reported throughput: Publisher/Falcon-7B (Publisher-Falcon-7B) on H100",
            filters=QueryFilters(model="Falcon-7B"),
        )
    )
    assert plan.precondition is None
    assert plan.filters.model == "Falcon-7B"


def test_public_planning_ranks_relevant_workload_before_fifty_row_limit(query_service):
    dao = query_service.dao
    revision = "2026-10-15"
    keys = [f"{_PATH}#planning:{index}" for index in range(53)]
    relevant = max(range(53), key=lambda index: item_id(PRISM_ORIGIN, keys[index] + ":measurement", revision))
    wrong_runtime = next(index for index in range(53) if index != relevant)
    records = [{"id": f"planning-{index}", "throughput": 42} for index in range(53)]
    digest = query_service.archive.put(json.dumps(records).encode())
    items = []
    for index, key in enumerate(keys):
        common = {
            "origin_id": PRISM_ORIGIN,
            "source_revision": revision,
            "source_path": _PATH,
            "content_sha256": digest,
            "access_classification": "verified_public",
            "format": "json",
        }
        experiment_id = item_id(PRISM_ORIGIN, key, revision)
        items.append(
            PublicBenchmarkRecord(
                **common,
                item_id=experiment_id,
                logical_key=key,
                kind="experiment",
                parent_id=item_id(PRISM_ORIGIN, f"file:{_PATH}", revision),
                model="Qwen3-0.6B",
                hardware="H100",
                runtime="incompatible-runtime" if index == wrong_runtime else "vllm",
                workload={"mean_input_tokens": 1000 if index in (relevant, wrong_runtime) else 10},
                source_pointer={"record_id": f"planning-{index}", "json_pointer": f"/{index}"},
            )
        )
        measurement_key = key + ":measurement"
        items.append(
            PublicBenchmarkRecord(
                **common,
                item_id=item_id(PRISM_ORIGIN, measurement_key, revision),
                logical_key=measurement_key,
                kind="measurement",
                parent_id=experiment_id,
                experiment_id=experiment_id,
                evidence_type="raw_json",
                metric="throughput",
                value=Decimal("42"),
                original_value=Decimal("42"),
                unit="tokens/s",
                original_unit="tokens/s",
                source_pointer={"json_pointer": f"/{index}/throughput"},
            )
        )
    publication = PublicBenchmarkRecordPublicationService(dao=dao, archive=query_service.archive)
    manifest = publication.stage(
        PinnedInventory(
            origin_id=PRISM_ORIGIN,
            revision=revision,
            complete=True,
            files=(InventoryFile(path=_PATH, sha256=digest, format="json"),),
        ),
        (ParsedFile(path=_PATH, items=tuple(items)),),
        expected_counts={"source": 1, "experiment": 53, "measurement": 53},
    )
    publication.activate(PRISM_ORIGIN, manifest.item_id)
    filters = QueryFilters(model="Qwen3-0.6B", hardware="H100", metric="throughput")
    relevant_id = item_id(PRISM_ORIGIN, keys[relevant] + ":measurement", revision)
    baseline = dao.search(filters, intent="numeric", text_query="", limit=50)
    assert baseline.total == 53
    assert relevant_id not in {match.item.item_id for match in baseline.matches}

    dao.planning_facts = PlanningFacts(
        model_weight_gib=1,
        vram_per_gpu_gib=80,
        free_gpu_count=8,
        workload_profile=WorkloadProfile(mean_input_tokens=1000),
        requirements=PlanningRequirements(runtime=RuntimeIdentity(backend="vllm")),
    )
    result = dao.search(filters, intent="numeric", text_query="", limit=50)
    assert result.total == 52 and len(result.matches) == 50
    assert result.matches[0].item.item_id == relevant_id
    assert all(match.experiment.runtime == "vllm" for match in result.matches)


def publish_planning_experiments(service, runs):
    revision = "2026-10-16"
    raw = json.dumps(runs).encode()
    digest = service.archive.put(raw)
    items = []
    for index, run in enumerate(runs):
        key = f"{_PATH}#ranked:{run['id']}"
        experiment_id = item_id(PRISM_ORIGIN, key, revision)
        common = {
            "origin_id": PRISM_ORIGIN,
            "source_revision": revision,
            "source_path": _PATH,
            "content_sha256": digest,
            "access_classification": "verified_public",
            "format": "json",
        }
        items.append(
            PublicBenchmarkRecord(
                **common,
                item_id=experiment_id,
                logical_key=key,
                kind="experiment",
                parent_id=item_id(PRISM_ORIGIN, f"file:{_PATH}", revision),
                model=run.get("model", "Qwen3-0.6B"),
                hardware="H100",
                runtime=run.get("runtime", "vllm"),
                workload=run["workload"],
                configuration={"tp": 1},
                source_pointer={"record_id": run["id"], "json_pointer": f"/{index}"},
            )
        )
        for offset, measurement in enumerate(run["measurements"]):
            measurement_key = key + f":measurement:{offset}"
            value = Decimal(str(measurement["value"]))
            unit = measurement["unit"]
            normalized = normalize_unit(value, unit)
            items.append(
                PublicBenchmarkRecord(
                    **common,
                    item_id=item_id(PRISM_ORIGIN, measurement_key, revision),
                    logical_key=measurement_key,
                    kind="measurement",
                    parent_id=experiment_id,
                    experiment_id=experiment_id,
                    evidence_type="raw_json",
                    metric=measurement["metric"],
                    statistic=measurement.get("statistic"),
                    value=value,
                    original_value=value,
                    unit=unit,
                    original_unit=unit,
                    normalized_value=normalized[0] if normalized else None,
                    normalized_unit=normalized[1] if normalized else None,
                    normalization_rule=normalized[2] if normalized else None,
                    series=measurement.get("series"),
                    source_pointer={"json_pointer": f"/{index}/measurements/{offset}/value"},
                )
            )
    publication = PublicBenchmarkRecordPublicationService(dao=service.dao, archive=service.archive)
    manifest = publication.stage(
        PinnedInventory(
            origin_id=PRISM_ORIGIN,
            revision=revision,
            complete=True,
            files=(InventoryFile(path=_PATH, sha256=digest, format="json"),),
        ),
        (ParsedFile(path=_PATH, items=tuple(items)),),
        expected_counts={
            "source": 1,
            "experiment": len(runs),
            "measurement": sum(len(run["measurements"]) for run in runs),
        },
    )
    publication.activate(PRISM_ORIGIN, manifest.item_id)


@pytest.mark.parametrize("runtime", [None, "", " ", "unknown", "UNKNOWN", "vllm", "sglang"])
def test_unknown_public_runtime_defaults_to_vllm_without_rewriting_evidence(query_service, runtime):
    from llm_d_lens.agentic.public_benchmark_record.case_retrieval import build_case_bundle

    publish_planning_experiments(
        query_service,
        [
            {
                "id": "legacy",
                "runtime": runtime,
                "workload": {"mean_input_tokens": 2000},
                "measurements": [{"metric": "latency", "value": 20, "unit": "ms"}],
            },
            {
                "id": "other-runtime",
                "runtime": "sglang",
                "workload": {"mean_input_tokens": 2000},
                "measurements": [{"metric": "latency", "value": 10, "unit": "ms"}],
            },
        ],
    )
    filters = QueryFilters(model="Qwen/Qwen3-0.6B", hardware="H100", runtime="vllm")
    request = QueryRequest(question="Reported latency", intent="comparison", filters=filters)
    expected_count = 0 if runtime == "sglang" else 1
    response = query_service.query(request, reader())
    assert response.matched_items == expected_count
    assert len(response.evidence) == expected_count

    other = query_service.query(
        request.model_copy(update={"filters": filters.model_copy(update={"runtime": "sglang"})}), reader()
    )
    assert other.matched_items == (2 if runtime == "sglang" else 1)
    assert all(entry.experiment.runtime == "sglang" for entry in other.evidence)

    facts = PlanningFacts(
        model_weight_gib=1,
        vram_per_gpu_gib=80,
        free_gpu_count=8,
        workload_profile=WorkloadProfile(mean_input_tokens=2000),
        requirements=PlanningRequirements(runtime=RuntimeIdentity(backend="vllm")),
    )
    query_service.dao.planning_facts = facts
    planning_request = request.model_copy(update={"filters": filters.model_copy(update={"runtime": None})})
    planned = query_service.query(planning_request, reader(), experiment_limit=50)
    assert planned.matched_items == expected_count
    assert len(planned.evidence) == expected_count
    if not expected_count:
        return
    entry = planned.evidence[0]
    assert entry.experiment.runtime == "vllm"
    assert entry.experiment.source_pointer["record_id"] == "legacy"
    assert query_service.dao.get(entry.experiment.item_id).runtime == runtime

    observation = {"evidence_id": f"public-rag:{entry.item_id}", **entry.model_dump(mode="json")}
    bundle, _ = build_case_bundle(
        (observation,), facts, model=filters.model, hardware=filters.hardware, branch_status={}
    )
    assert len(bundle.cases) == 1
    assert bundle.cases[0].runtime == "vllm"
    assert "runtime" not in bundle.cases[0].unknown_fields
    assert "runtime.backend" not in bundle.cases[0].unknown_fields
    assert "runtime.backend_version" in bundle.cases[0].unknown_fields


@pytest.mark.parametrize("with_workload", [False, True])
@pytest.mark.parametrize("with_target", [False, True])
def test_planning_ranks_complete_experiments_before_limit(query_service, with_workload, with_target):
    from llm_d_lens.agentic.planner import normalize_planning_facts

    runs = [
        {
            "id": f"run-{index}",
            "workload": {"input_tokens": "1000" if index == 54 else "10000"},
            "measurements": [
                {"metric": "ttft", "statistic": "mean", "value": 500 - index, "unit": "ms"},
                {"metric": "tpot", "statistic": "mean", "value": 500 - index, "unit": "ms/token"},
                {"metric": "throughput", "value": 42, "unit": "output tokens/s"},
            ],
        }
        for index in range(55)
    ]
    runs[0]["measurements"][0]["value"] = 1
    runs[0]["measurements"][1]["value"] = 1
    publish_planning_experiments(query_service, runs)
    facts = normalize_planning_facts(
        PlanningFacts(1, 80, 8, workload_profile=WorkloadProfile(mean_input_tokens=1000) if with_workload else None),
        "TTFT mean <= 500 ms; TPOT mean <= 500 ms" if with_target else "",
    )
    query_service.dao.planning_facts = facts
    request = QueryRequest(
        question="Published deployment benchmark evidence",
        intent="comparison",
        filters=QueryFilters(model="Qwen/Qwen3-0.6B", hardware="H100"),
        limit=1,
    )
    result = query_service.query(request, reader(), experiment_limit=1)
    assert result.matched_items == 165
    assert result.matched_experiments == 55
    assert result.returned_experiments == 1
    assert result.experiment_limit == 1
    assert len(result.evidence) == 3
    expected = "run-54" if with_workload else "run-0"
    assert {entry.experiment.source_pointer["record_id"] for entry in result.evidence} == {expected}
    assert {entry.metric for entry in result.evidence} == {"ttft", "tpot", "throughput"}
    assert result.truncated
    complete = query_service.query(request, reader(), experiment_limit=50)
    assert len(complete.evidence) == 150
    assert complete.returned_experiments == 50
    ordinary = query_service.query(request, reader())
    assert len(ordinary.evidence) == 1 and ordinary.experiment_limit is None


def test_planning_latency_keeps_series_and_statistics_separate(query_service):
    from llm_d_lens.agentic.planner import normalize_planning_facts

    publish_planning_experiments(
        query_service,
        [
            {
                "id": "split",
                "workload": {},
                "measurements": [
                    {"metric": "ttft", "statistic": "mean", "value": 1, "unit": "ms", "series": {"concurrency": 1}},
                    {
                        "metric": "tpot",
                        "statistic": "mean",
                        "value": 1,
                        "unit": "ms/token",
                        "series": {"concurrency": 2},
                    },
                    {"metric": "ttft", "statistic": "p99", "value": 0.1, "unit": "ms", "series": {"concurrency": 2}},
                ],
            },
            {
                "id": "complete",
                "workload": {},
                "measurements": [
                    {"metric": "ttft", "statistic": "mean", "value": 10, "unit": "ms"},
                    {"metric": "tpot", "statistic": "mean", "value": 10, "unit": "ms/token"},
                ],
            },
            {
                "id": "invalid",
                "workload": {},
                "measurements": [
                    {"metric": "ttft", "statistic": "mean", "value": 0, "unit": "ms"},
                    {"metric": "tpot", "statistic": None, "value": 0.1, "unit": "ms/token"},
                ],
            },
        ],
    )
    query_service.dao.planning_facts = normalize_planning_facts(
        PlanningFacts(1, 80, 8), "TTFT mean <= 100 ms; TPOT mean <= 100 ms"
    )
    result = query_service.query(
        QueryRequest(
            question="Published evidence",
            intent="comparison",
            filters=QueryFilters(model="Qwen3-0.6B", hardware="H100"),
        ),
        reader(),
        experiment_limit=1,
    )
    assert result.matched_experiments == 4
    assert len(result.evidence) == 2
    assert {entry.experiment.source_pointer["record_id"] for entry in result.evidence} == {"complete"}


def test_null_statistic_is_mean_for_sql_ranking_and_filtering_without_changing_storage(query_service):
    from llm_d_lens.agentic.planner import normalize_planning_facts

    publish_planning_experiments(
        query_service,
        [
            {
                "id": identity,
                "workload": {},
                "measurements": [{"metric": "ttft", "statistic": statistic, "value": value, "unit": "ms"}],
            }
            for identity, statistic, value in (
                ("fast-default", None, 1),
                ("slow-mean", "mean", 20),
                ("p99", "p99", 0.1),
            )
        ],
    )
    query_service.dao.planning_facts = normalize_planning_facts(PlanningFacts(1, 80, 8), "TTFT mean <= 100 ms")
    request = QueryRequest(
        question="Published evidence",
        intent="comparison",
        filters=QueryFilters(model="Qwen3-0.6B", hardware="H100", metric="ttft", statistic="mean"),
    )
    result = query_service.query(request, reader(), experiment_limit=1)
    assert result.matched_experiments == 2
    assert result.evidence[0].experiment.source_pointer["record_id"] == "fast-default"
    assert result.evidence[0].statistic is None
    query_service.dao.planning_facts = None
    ordinary = query_service.query(request, reader())
    assert "mean (default; source statistic unreported)" in ordinary.answer
    assert query_service.dao.search(request.filters, intent="comparison", text_query="", limit=10).total == 2


def test_planning_e2e_alias_ranks_lower_latency_without_a_target(query_service):
    publish_planning_experiments(
        query_service,
        [
            {
                "id": identity,
                "workload": {},
                "measurements": [{"metric": "e2e", "statistic": "mean", "value": value, "unit": "ms"}],
            }
            for identity, value in (("slow", 100), ("fast", 5))
        ],
    )
    query_service.dao.planning_facts = PlanningFacts(1, 80, 8)
    result = query_service.query(
        QueryRequest(
            question="Published evidence",
            intent="comparison",
            filters=QueryFilters(model="Qwen3-0.6B", hardware="H100"),
        ),
        reader(),
        experiment_limit=1,
    )
    assert result.evidence[0].experiment.source_pointer["record_id"] == "fast"


@pytest.mark.parametrize("with_workload", [False, True])
def test_planning_fallback_does_not_claim_low_latency_and_follows_ranked_experiments(query_service, with_workload):
    from llm_d_lens.agentic.facts import rank_public_rag_evidence

    publish_planning_experiments(
        query_service,
        [
            {
                "id": "fallback",
                "workload": {},
                "measurements": [{"metric": "throughput", "value": 42, "unit": "tokens/s"}],
            },
            {
                "id": "ranked",
                "workload": {},
                "measurements": [{"metric": "ttft", "statistic": "mean", "value": 5, "unit": "ms"}],
            },
        ],
    )
    facts = PlanningFacts(1, 80, 8, workload_profile=WorkloadProfile(mean_input_tokens=1000) if with_workload else None)
    query_service.dao.planning_facts = facts
    request = QueryRequest(
        question="Published evidence",
        intent="comparison",
        filters=QueryFilters(model="Qwen3-0.6B", hardware="H100"),
    )
    result = query_service.query(request, reader(), experiment_limit=50)
    assert [entry.experiment.source_pointer["record_id"] for entry in result.evidence] == ["ranked", "fallback"]
    _, matches = rank_public_rag_evidence(result.model_dump(mode="json")["evidence"], facts)
    assert [match["ranking"]["queue"] for match in matches] == ["ranked", "fallback"]
    assert "lower-latency ranked experiments" not in result.answer
    assert "fallback references, not low-latency evidence" in result.answer
    selected = query_service.query(request, reader(), experiment_limit=1)
    assert selected.evidence[0].experiment.source_pointer["record_id"] == "ranked"


def test_planning_complete_history_finishes_inside_retrieval_deadline(query_service, postgres_engine):
    from time import perf_counter

    publish_planning_experiments(
        query_service,
        [
            {
                "id": f"run-{index}",
                "workload": {},
                "measurements": [
                    {
                        "metric": metric,
                        "statistic": statistic,
                        "value": index + 1,
                        "unit": "ms/token" if metric == "tpot" else "ms",
                    }
                    for metric in ("ttft", "tpot", "latency")
                    for statistic in ("mean", "p99")
                ],
            }
            for index in range(84)
        ]
        + [
            {
                "id": f"other-model-{index}",
                "model": "unrelated-model",
                "workload": {},
                "measurements": [
                    {"metric": "ttft", "statistic": "mean", "value": index + 1, "unit": "ms"} for _ in range(40)
                ],
            }
            for index in range(220)
        ],
    )
    query_service.dao.planning_facts = PlanningFacts(1, 80, 8)

    def bounded_query(_connection, cursor, statement, _parameters, _context, _executemany):
        if "planning_selected AS" in statement:
            cursor.execute("SET LOCAL statement_timeout = '5s'")

    event.listen(postgres_engine, "before_cursor_execute", bounded_query)
    started = perf_counter()
    try:
        result = query_service.query(
            QueryRequest(
                question="Published evidence",
                intent="comparison",
                filters=QueryFilters(model="Qwen3-0.6B", hardware="H100"),
            ),
            reader(),
            experiment_limit=50,
        )
    finally:
        event.remove(postgres_engine, "before_cursor_execute", bounded_query)
    assert perf_counter() - started < 15
    assert result.matched_items == 504
    assert result.matched_experiments == 84
    assert result.returned_experiments == 50
    assert result.returned_observations == 300


@pytest.mark.parametrize(
    "principal",
    [
        None,
        Principal(user_id="system", username="system", authz_version=0),
        Principal(user_id="unprivileged", username="user"),
        replace(
            reader(), bindings=(RoleBinding(role="custom", scope_type=ScopeType.CLUSTER, scope_cluster_id="cluster-a"),)
        ),
    ],
)
def test_service_authorization_precedes_every_read(principal):
    class NeverRead:
        def search(self, *args, **kwargs):
            pytest.fail("unauthorized request reached storage")

    service = PublicBenchmarkRecordQueryService(dao=NeverRead())
    with pytest.raises((UnauthenticatedError, ForbiddenError)):
        service.query(numeric(), principal)


@pytest.mark.parametrize("include_internal", [False, True])
def test_missing_model_abstains_before_either_branch_reads(include_internal):
    class NeverRead:
        def search(self, *args, **kwargs):
            pytest.fail("ambiguous model reached retrieval")

    service = PublicBenchmarkRecordQueryService(dao=NeverRead(), agentic_dao=NeverRead())
    result = service.query(
        QueryRequest(question="Show throughput for model on H100", include_internal=include_internal), reader()
    )
    assert result.status == "abstained"
    assert result.matched_items == 0
    assert not result.evidence
    assert [item.code for item in result.abstentions] == ["model_scope_required"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x", "principal": {"user_id": "admin"}},
        {"question": "x", "filters": {"origins": ["private:secret"]}},
        {"question": "x", "filters": {"value_min": "1"}},
        {"question": "x", "filters": {"unit": "s", "value_min": "2", "value_max": "1"}},
        {"question": "x", "filters": {"source_path": "../escape"}},
        {"question": "x", "limit": 51},
        {"question": " "},
    ],
)
def test_query_contract_rejects_invalid_or_unscoped_payloads(payload):
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(payload)


def test_numeric_values_units_and_conditions_are_exact(query_service, postgres_engine):
    statements = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(postgres_engine, "before_cursor_execute", capture)
    try:
        result = query_service.query(numeric(), reader())
    finally:
        event.remove(postgres_engine, "before_cursor_execute", capture)
    assert result.status == "answered"
    assert result.matched_items == 1
    evidence = result.evidence[0]
    assert evidence.value == Decimal("6131.06177725188")
    assert evidence.unit == "output tokens/s"
    assert evidence.experiment.model == "Qwen3-0.6B"
    assert evidence.experiment.workload == {"concurrency": 4}
    assert evidence.citation.source_pointer["json_pointer"] == "/runs/0/throughput"
    assert evidence.citation.sha256 == hashlib.sha256(_RAW).hexdigest()
    assert "6131.06177725188" in result.answer
    assert result.internal_branch_enabled is False
    assert not any("agentic_benchmark_records" in statement for statement in statements)
    assert not any(statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for statement in statements)


def test_required_unknown_condition_and_denominators_do_not_match(query_service):
    assert query_service.query(numeric(hardware="unreported accelerator"), reader()).matched_items == 0
    assert query_service.query(numeric(unit="tokens/s"), reader()).matched_items == 0
    assert query_service.query(numeric(workload={"concurrency": 8}), reader()).matched_items == 0
    assert query_service.query(numeric(series={"scenario": "run-b"}), reader()).matched_items == 0
    assert query_service.query(numeric(series={"scenario": "run-a"}), reader()).matched_items == 1
    result = query_service.query(numeric(source_revision="2026-10-20"), reader())
    assert result.status == "abstained" and not result.evidence


def test_complete_public_retrieval_preserves_all_observations_and_exact_identity(query_service):
    publication = PublicBenchmarkRecordPublicationService(dao=query_service.dao, archive=query_service.archive)
    publish_numeric(publication, revision="2026-10-15", measurement_copies=40)
    request = QueryRequest(
        question="Published deployment benchmark evidence",
        intent="comparison",
        limit=1,
        filters=QueryFilters(model="Qwen3-0.6B", hardware="H100"),
    )
    result = query_service.query(request, reader(), all_matches=True)
    assert result.returned_observations == result.matched_items == len(result.evidence) == 80
    assert result.truncated is False
    assert {item.experiment.hardware for item in result.evidence} == {"H100"}
    assert {item.experiment.model for item in result.evidence} == {"Qwen3-0.6B"}
    assert all(item.citation.source_revision == "2026-10-15" for item in result.evidence)
    assert "all 80" in result.answer
    bounded = query_service.query(request, reader())
    assert len(bounded.evidence) == 1
    assert bounded.truncated
    assert bounded.matched_items == 80


@pytest.mark.parametrize("filters", [QueryFilters(), QueryFilters(model="Qwen3-0.6B"), QueryFilters(hardware="H100")])
def test_complete_public_retrieval_requires_exact_model_and_hardware(filters):
    request = QueryRequest(question="Published deployment benchmark evidence", intent="comparison", filters=filters)
    with pytest.raises(DomainValidationError, match="exact model and hardware"):
        PublicBenchmarkRecordQueryService(dao=object()).query(request, reader(), all_matches=True)


@pytest.mark.parametrize(
    "question",
    [
        "What was the p99 latency for Qwen3-0.6B on H100?",
        "What was the throughput for Llama-3.1-70B on H100?",
    ],
)
def test_question_conditions_cannot_return_a_different_statistic_or_model(query_service, question):
    result = query_service.query(QueryRequest(question=question), reader())
    assert result.status == "abstained"
    assert result.matched_items == 0
    assert not result.evidence
    assert [item.code for item in result.abstentions] == ["no_exact_match"]


def test_question_mean_statistic_still_returns_matching_measurement(query_service):
    result = query_service.query(QueryRequest(question="What was the mean latency for Qwen3-0.6B on H100?"), reader())
    assert result.status == "answered"
    assert result.filters.statistic == "mean"
    assert result.evidence[0].statistic == "mean"
    assert result.evidence[0].value == Decimal("1.234567890123456789")


def test_numeric_normalization_range_and_statistic_alias(query_service):
    result = query_service.query(
        QueryRequest(
            question="Latency under the explicit conditions",
            intent="numeric",
            filters=QueryFilters(
                record_id="run-a",
                metric="latency",
                statistic="avg",
                normalized=True,
                unit="ms",
                value_min=Decimal("1200"),
                value_max=Decimal("1300"),
            ),
        ),
        reader(),
    )
    assert result.matched_items == 1
    assert result.evidence[0].value == Decimal("1.234567890123456789")
    assert result.evidence[0].unit == "s"
    assert result.evidence[0].normalized_value == Decimal("1234.567890123456789000")


def test_document_origins_are_not_sql_query_sources():
    with pytest.raises(ValidationError, match="approved public"):
        QueryFilters(origins=(BENCHMARK_ORIGIN,), source_path="docs/benchmark_report.md", metric="throughput")


def test_repository_definition_requires_an_installed_document_provider(query_service):
    with pytest.raises(EvidenceUnavailableError, match="document provider, which is not installed"):
        query_service.query(
            QueryRequest(
                question="What is TTFT?",
                intent="definition",
                filters=QueryFilters(
                    source_path="docs/benchmark_report.md",
                ),
            ),
            reader(),
        )
    with pytest.raises(PublicationError, match="allowlist"):
        query_service.dao.count_published(BENCHMARK_ORIGIN)


def test_repository_figure_evidence_is_not_stored_in_prism_sql(query_service):
    result = query_service.query(
        QueryRequest(
            question="Show explicitly linked H100 figure scenarios",
            intent="figure",
            filters=QueryFilters(hardware="H100"),
        ),
        reader(),
    )
    assert result.intent == "figure" and result.status == "abstained"
    assert result.matched_items == 0 and not result.evidence


def test_counts_are_filtered_before_limit_and_source_revocation(query_service):
    result = query_service.query(
        QueryRequest(
            question="List matching measurements",
            intent="numeric",
            limit=1,
            filters=QueryFilters(model="Qwen3-0.6B", metric="throughput"),
        ),
        reader(),
    )
    assert result.matched_items == 2 and len(result.evidence) == 1 and result.truncated is True
    head = query_service.dao.head(PRISM_ORIGIN)
    query_service.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=head.state.generation)
    hidden = query_service.query(numeric(), reader())
    assert hidden.matched_items == 0 and not hidden.evidence


def test_revocation_during_evidence_read_discards_entire_response(query_service):
    original = query_service.archive.read

    def revoke(digest):
        data = original(digest)
        head = query_service.dao.head(PRISM_ORIGIN)
        query_service.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=head.state.generation)
        return data

    query_service.archive.read = revoke
    with pytest.raises(ConflictError):
        query_service.query(numeric(), reader())


def test_missing_or_corrupt_bytes_are_errors_not_empty_success(query_service):
    root = query_service.archive._root(hashlib.sha256(_RAW).hexdigest())
    (root / "source").write_bytes(b"corrupt")
    with pytest.raises(EvidenceUnavailableError):
        query_service.query(numeric(), reader())


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_object",
        "non_object",
        "invalid_json",
        "invalid_utf8",
        "wrong_owner",
        "invalid_owner",
        "missing_files",
        "invalid_files",
        "invalid_entry",
        "missing_uri",
        "missing_path",
        "unregistered_source",
        "mismatched_path",
    ],
)
def test_corrupt_evidence_manifest_is_unavailable_and_recovers(query_service, corruption):
    request = numeric()
    expected = query_service.query(request, reader()).model_dump(mode="json")
    digest = hashlib.sha256(_RAW).hexdigest()
    manifest_path = query_service.archive._root(digest) / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    broken = {
        "empty_object": {},
        "non_object": [],
        "wrong_owner": {**manifest, "owner_id": "different-owner"},
        "invalid_owner": {**manifest, "owner_type": None},
        "missing_files": {key: value for key, value in manifest.items() if key != "files"},
        "invalid_files": {**manifest, "files": None},
        "invalid_entry": {**manifest, "files": [None]},
        "missing_uri": {**manifest, "files": [{"path": "source"}]},
        "missing_path": {**manifest, "files": [{"uri": manifest["files"][0]["uri"]}]},
        "unregistered_source": {**manifest, "files": []},
        "mismatched_path": {**manifest, "files": [{**manifest["files"][0], "path": "different"}]},
    }
    payload = (
        b"\xff"
        if corruption == "invalid_utf8"
        else b"{"
        if corruption == "invalid_json"
        else json.dumps(broken[corruption]).encode()
    )
    manifest_path.write_bytes(payload)
    with pytest.raises(PublicationError, match="manifest"):
        query_service.archive.read(digest)
    with pytest.raises(EvidenceUnavailableError, match="missing or corrupt"):
        query_service.query(request, reader())

    manifest_path.write_bytes(original)
    assert query_service.query(request, reader()).model_dump(mode="json") == expected


def test_manifest_membership_not_per_row_published_flag(query_service):
    original = query_service.dao.list_published(PRISM_ORIGIN, kind="measurement")[0]
    key = f"{_PATH}#unmanifested"
    forged = original.model_copy(
        update={
            "item_id": item_id(PRISM_ORIGIN, key, _REVISION),
            "logical_key": key,
            "publication_state": "published",
        }
    )
    query_service.dao.save(forged)
    result = query_service.query(numeric(), reader())
    assert result.matched_items == 1
    assert forged.item_id not in {item.item_id for item in result.evidence}


def test_old_release_and_generation_behavior(query_service):
    publication = PublicBenchmarkRecordPublicationService(dao=query_service.dao, archive=query_service.archive)
    old = query_service.dao.head(PRISM_ORIGIN)
    publish_numeric(publication, revision="2026-10-20", activate=False)
    assert query_service.query(numeric(source_revision="2026-10-20"), reader()).matched_items == 0
    assert query_service.query(numeric(source_revision=_REVISION), reader()).matched_items == 1
    publish_numeric(publication, revision="2026-10-20")
    assert query_service.query(numeric(source_revision=_REVISION), reader()).matched_items == 0
    head = query_service.dao.head(PRISM_ORIGIN)
    publication.rollback(PRISM_ORIGIN, old.registry.published_manifest_id, expected_generation=head.state.generation)
    assert query_service.query(numeric(source_revision=_REVISION), reader()).matched_items == 1


def test_question_plans_are_auditable_and_not_inferred_from_unrelated_permissions():
    request = QueryRequest(
        question="What was the recorded throughput for Prism ID 02c8f038-1cad-4ee8-a72e-724affa923f4?"
    )
    plan = plan_query(request)
    assert plan.filters.origins == (PRISM_ORIGIN,)
    assert plan.filters.record_id == "02c8f038-1cad-4ee8-a72e-724affa923f4"
    assert plan.filters.metric == "throughput"
    assert plan_query(QueryRequest(question="Throughput for an unspecified model")).precondition is not None
    assert plan_query(QueryRequest(question="Show a diagram")).intent == "figure"
    assert plan_query(QueryRequest(question="Compare H200 weight loading times")).intent == "comparison"
    assert plan_query(QueryRequest(question="Show internal records")).precondition.code == "internal_branch_disabled"


def test_database_evidence_contract_roundtrips_exact_decimal(query_service):
    from llm_d_lens.agentic.public_benchmark_record.query_contracts import QueryResponse

    response = query_service.query(numeric(), reader())
    assert QueryResponse.model_validate_json(response.model_dump_json()) == response
    assert response.model_dump(mode="json")["evidence"][0]["value"] == "6131.06177725188"


def test_real_prism_snapshot_answer_acceptance(postgres_engine, tmp_path, monkeypatch, prism_initial_manifest):
    from llm_d_lens.agentic.public_benchmark_record.prism import PARSER_VERSION as PRISM_PARSER_VERSION
    from llm_d_lens.agentic.public_benchmark_record.prism import PrismSnapshotImporter

    root = Path(__file__).resolve().parents[5]
    snapshot = root / "private/prism-public-2026-10-13"
    if not snapshot.exists():
        pytest.skip("real Prism acceptance needs the separately provisioned pinned local snapshot")
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "real-query-evidence"))
    dao = PublicBenchmarkRecordQueryDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    service = PublicBenchmarkRecordQueryService(dao=dao)
    publication = PublicBenchmarkRecordPublicationService(dao=dao, archive=service.archive)
    PrismSnapshotImporter(publication).import_snapshot(
        snapshot,
        prism_initial_manifest.inventory.revision,
        {file.path: file.sha256 for file in prism_initial_manifest.inventory.files},
        approval(PRISM_ORIGIN).model_copy(update={"parser_version": PRISM_PARSER_VERSION}),
        expected_experiments=prism_initial_manifest.expected_experiments,
    )
    prism = service.query(
        QueryRequest(question="What was the recorded throughput for Prism ID 02c8f038-1cad-4ee8-a72e-724affa923f4?"),
        reader(),
    )
    assert prism.status == "answered" and prism.matched_items == 1
    assert prism.evidence[0].value == Decimal("6131.06177725188")
    assert prism.evidence[0].unit == "output tokens/s"
    assert prism.evidence[0].experiment.model == "Qwen3-0.6B"
    assert dao.count_published(PRISM_ORIGIN, kind="experiment") == 249
    assert dao.count_published(PRISM_ORIGIN, kind="measurement") == 9060
    for origin in (LLM_D_ORIGIN, BENCHMARK_ORIGIN):
        with pytest.raises(PublicationError, match="allowlist"):
            dao.count_published(origin)
