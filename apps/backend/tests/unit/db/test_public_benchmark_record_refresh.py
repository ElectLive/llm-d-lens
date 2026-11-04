"""Reviewed Prism-only SQL refresh and source approval boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.public_benchmark_record.admin import _parser
from llm_d_lens.agentic.public_benchmark_record.prism import PARSER_VERSION as PRISM_PARSER
from llm_d_lens.agentic.public_benchmark_record.prism import PrismSnapshotImporter
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InventoryFile,
    PinnedInventory,
    PublicationConflictError,
    PublicationError,
    ReleaseManifest,
    SourceApproval,
    content_digest,
)
from llm_d_lens.agentic.public_benchmark_record.query_contracts import QUERY_PERMISSION, QueryFilters, QueryRequest
from llm_d_lens.agentic.public_benchmark_record.refresh import PublicBenchmarkRecordRefreshService, RefreshPlan
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService
from llm_d_lens.agentic.public_benchmark_record.sources import (
    PRISM_ORIGIN,
    allowed_paths,
    input_format,
    validate_approval,
)
from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
from llm_d_lens.db.dao.public_benchmark_record import PublicBenchmarkRecordDao
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao


@pytest.fixture
def publication(postgres_engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "evidence"))
    return PublicBenchmarkRecordPublicationService(
        PublicBenchmarkRecordDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    )


def _approval() -> SourceApproval:
    return SourceApproval(
        verified_public=True,
        approved_by="isolated refresh test",
        approval_reference="test only, not production approval",
        license_reference="fixture",
        parser_version=PRISM_PARSER,
        allowed_paths=allowed_paths(PRISM_ORIGIN),
    )


def _snapshot(tmp_path: Path, publication: PublicBenchmarkRecordPublicationService, revision: str, value: int = 11):
    root = tmp_path / revision
    root.mkdir()
    record = {
        "id": "historical",
        "model_name": "fixture-model",
        "timestamp": "20250923-201033",
        "metadata": {"hardware": "reported-hardware"},
        "workload": {},
        "source_info": {"file_identifier": "original.json"},
        "metrics": {"throughput": value},
    }
    documents = {
        "archives/archived_drive_data.json": json.dumps([record]),
        "archives/llm-d-benchmarks.json": "[]",
        "historical-merged-deduplicated.json": json.dumps([record]),
        "results/public-index.json": '{"items": [], "nextPageToken": null}',
        "results/promoted-index.json": '{"items": [], "nextPageToken": null}',
        "normalize-and-merge.jq": "# fixture",
    }
    files = []
    for path, text in sorted(documents.items()):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        files.append(
            InventoryFile(
                path=path,
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                format="jq" if path.endswith(".jq") else "json",
            )
        )
    approval = _approval()
    publication.register_source(PRISM_ORIGIN, approval)
    head = publication.dao.head(PRISM_ORIGIN)
    plan = RefreshPlan(
        review_reference="reviewed complete offline snapshot",
        approval=approval,
        inventory=PinnedInventory(origin_id=PRISM_ORIGIN, revision=revision, complete=True, files=tuple(files)),
        expected_counts={"source": 6, "experiment": 1, "measurement": 1},
        base_generation=head.state.generation,
        base_manifest_id=head.registry.published_manifest_id,
    )
    return root, plan


def test_refresh_cli_defers_missing_snapshot_error_to_reviewed_runner():
    args = _parser().parse_args(["refresh", "--data-dir", "/srv/lens-data", "--plan", "/secure/approved-plan.json"])
    assert args.source is None


def test_refresh_cli_accepts_an_explicit_local_prism_snapshot():
    args = _parser().parse_args(
        [
            "refresh",
            "--data-dir",
            "/srv/lens-data",
            "--plan",
            "/secure/approved-plan.json",
            "--source",
            "/srv/prism-snapshot",
        ]
    )
    assert args.source == Path("/srv/prism-snapshot")


def test_rebuild_cli_requires_an_explicit_snapshot_manifest():
    arguments = [
        "rebuild",
        "--data-dir",
        "/srv/lens-data",
        "--snapshot",
        "/srv/prism-snapshot",
        "--approval",
        "/secure/approval.json",
    ]
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(arguments)
    assert error.value.code == 2
    args = _parser().parse_args([*arguments, "--manifest", "/secure/snapshot-manifest.json"])
    assert args.manifest == Path("/secure/snapshot-manifest.json")


@pytest.mark.parametrize(
    "unsupported", [["--trigger", "commit"], ["--trigger", "periodic"], ["--event-revision", "a" * 40]]
)
def test_refresh_cli_rejects_uninstalled_event_workflows(unsupported):
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(
            ["refresh", "--data-dir", "/srv/lens-data", "--plan", "/secure/approved-plan.json", *unsupported]
        )
    assert error.value.code == 2


def test_transfer_cli_rejects_obsolete_backend_selection():
    with pytest.raises(SystemExit) as error:
        _parser().parse_args(["verify", "--bundle", "/secure/benchmark.zip", "--structured"])
    assert error.value.code == 2


def test_prism_refresh_still_requires_local_snapshot(publication, tmp_path):
    _, plan = _snapshot(tmp_path, publication, "2026-10-13")
    with pytest.raises(PublicationError, match="explicit local snapshot"):
        PublicBenchmarkRecordRefreshService(publication).run(plan)
    assert publication.dao.head(PRISM_ORIGIN).manifest is None


def test_prism_refresh_replay_is_idempotent(publication, tmp_path):
    root, plan = _snapshot(tmp_path, publication, "2026-10-13")
    runner = PublicBenchmarkRecordRefreshService(publication)
    first = runner.run(plan, root)
    repeated = runner.run(plan, root)
    assert first.status == "published" and repeated.status == "unchanged"
    assert repeated.publication.generation == first.publication.generation


def test_approved_empty_refresh_removes_publication_and_retains_history(publication, tmp_path):
    root, plan = _snapshot(tmp_path, publication, "2026-10-13")
    runner = PublicBenchmarkRecordRefreshService(publication)
    first = runner.run(plan, root)
    old_items = publication.dao.list_published(PRISM_ORIGIN)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = RefreshPlan(
        review_reference="explicitly reviewed removal of all published files",
        approval=plan.approval,
        inventory=PinnedInventory(origin_id=PRISM_ORIGIN, revision="2026-10-20", complete=True, files=()),
        expected_counts={},
        base_generation=first.publication.generation,
        base_manifest_id=first.publication.manifest_id,
        allow_empty=True,
    )
    before = publication.dao.head(PRISM_ORIGIN)
    with pytest.raises(ValidationError, match="explicit allow_empty approval"):
        runner.run(empty.model_copy(update={"allow_empty": False}), empty_root)
    assert publication.dao.head(PRISM_ORIGIN) == before
    result = runner.run(empty, empty_root)
    assert result.status == "published" and result.publication.generation == 2
    assert result.file_count == 0 and result.publication.counts == {}
    assert publication.dao.count_published(PRISM_ORIGIN) == 0
    head = publication.dao.head(PRISM_ORIGIN)
    manifest = ReleaseManifest.model_validate(head.manifest.source_details)
    assert set(manifest.removed_item_ids) == {item.item_id for item in old_items}
    for item in old_items:
        assert publication.dao.get(item.item_id) == item
        assert publication.archive.read(item.content_sha256)
    replay = runner.run(empty, empty_root)
    assert replay.status == "unchanged" and replay.publication.generation == 2
    publication.rollback(PRISM_ORIGIN, first.publication.manifest_id, expected_generation=2)
    assert publication.dao.count_published(PRISM_ORIGIN) == len(old_items)
    with pytest.raises(PublicationConflictError):
        runner.run(empty, empty_root)


@pytest.mark.parametrize("fault", ["approval", "disabled", "stale", "unreviewed", "count", "missing", "race"])
def test_empty_refresh_preserves_approval_inventory_and_generation_guards(publication, tmp_path, monkeypatch, fault):
    root, plan = _snapshot(tmp_path, publication, "2026-10-13")
    runner = PublicBenchmarkRecordRefreshService(publication)
    first = runner.run(plan, root)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    empty = RefreshPlan(
        review_reference="reviewed removal",
        approval=plan.approval,
        inventory=PinnedInventory(origin_id=PRISM_ORIGIN, revision="2026-10-20", complete=True, files=()),
        expected_counts={},
        base_generation=first.publication.generation,
        base_manifest_id=first.publication.manifest_id,
        allow_empty=True,
    )
    if fault == "approval":
        empty = empty.model_copy(
            update={"approval": plan.approval.model_copy(update={"approval_reference": "different review"})}
        )
    elif fault == "disabled":
        publication.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=1)
    elif fault == "stale":
        publication.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=1)
        publication.dao.set_enabled(PRISM_ORIGIN, True, expected_generation=2)
    elif fault == "unreviewed":
        empty_root = root
    elif fault == "count":
        empty = empty.model_copy(update={"expected_counts": {"measurement": 1}})
    elif fault == "missing":
        empty_root = tmp_path / "missing"
    else:
        stage = publication.stage

        def disable_after_staging(*args, **kwargs):
            manifest = stage(*args, **kwargs)
            publication.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=1)
            return manifest

        monkeypatch.setattr(publication, "stage", disable_after_staging)
    with pytest.raises(PublicationError):
        runner.run(empty, empty_root)
    head = publication.dao.head(PRISM_ORIGIN)
    assert head.registry.published_manifest_id == first.publication.manifest_id
    assert head.state.generation == (3 if fault == "stale" else 2 if fault in ("disabled", "race") else 1)


def test_empty_import_still_requires_explicit_approval(publication, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(PublicationError, match="reviewed hash inventory"):
        PrismSnapshotImporter(publication).stage(root, "2026-10-20", {}, _approval(), expected_experiments=0)


def test_prism_new_snapshot_exact_answer_old_citation_and_revocation(publication, tmp_path):
    runner = PublicBenchmarkRecordRefreshService(publication)
    root, plan = _snapshot(tmp_path, publication, "2026-10-13")
    first = runner.run(plan, root)
    old = publication.dao.list_published(PRISM_ORIGIN, kind="measurement")[0]
    newer, update = _snapshot(tmp_path, publication, "2026-10-20", value=23)
    result = runner.run(update, newer)
    assert result.publication.generation == first.publication.generation + 1
    current = publication.dao.list_published(PRISM_ORIGIN, kind="measurement")
    assert len(current) == 1 and current[0].original_value == 23
    assert publication.dao.get(old.item_id) == old
    assert b'"throughput": 11' in publication.archive.read(old.content_sha256)
    query = PublicBenchmarkRecordQueryService(
        PublicBenchmarkRecordQueryDao(publication.dao._get_session_factory()), publication.archive
    )
    reader = Principal(
        user_id="reader",
        username="reader",
        bindings=(RoleBinding(role="reader", scope_type=ScopeType.GLOBAL),),
        role_permissions={"reader": frozenset({QUERY_PERMISSION})},
    )
    request = QueryRequest(
        question="Reported throughput",
        intent="numeric",
        filters=QueryFilters(origins=(PRISM_ORIGIN,), metric="throughput", record_id="historical"),
    )
    answer = query.query(request, reader)
    assert len(answer.evidence) == 1 and answer.evidence[0].value == 23
    assert answer.evidence[0].citation.source_revision == "2026-10-20"
    publication.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=result.publication.generation)
    assert not query.query(request, reader).evidence
    with pytest.raises(PublicationError, match="disabled"):
        runner.run(update, newer)


@pytest.mark.parametrize("fault", ["pagination", "missing", "hash", "count", "measurement-count", "parser"])
def test_prism_refresh_failure_preserves_head(publication, tmp_path, fault):
    runner = PublicBenchmarkRecordRefreshService(publication)
    original, first = _snapshot(tmp_path, publication, "2026-10-13")
    runner.run(first, original)
    root, plan = _snapshot(tmp_path, publication, "2026-10-20")
    if fault == "count":
        plan = RefreshPlan.model_validate({**plan.model_dump(), "expected_counts": {"source": 6, "experiment": 2}})
    elif fault == "measurement-count":
        plan = RefreshPlan.model_validate(
            {**plan.model_dump(), "expected_counts": {"source": 6, "experiment": 1, "measurement": 2}}
        )
    elif fault == "missing":
        (root / "results/public-index.json").unlink()
    elif fault == "hash":
        (root / "normalize-and-merge.jq").write_text("# unreviewed")
    else:
        path = "results/public-index.json" if fault == "pagination" else "archives/archived_drive_data.json"
        target = root / path
        target.write_text('{"items": [], "nextPageToken": "more"}' if fault == "pagination" else "{")
        files = tuple(
            file.model_copy(update={"sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            if file.path == path
            else file
            for file in plan.inventory.files
        )
        plan = RefreshPlan.model_validate(
            {**plan.model_dump(), "inventory": {**plan.inventory.model_dump(), "files": files}}
        )
    before = publication.dao.head(PRISM_ORIGIN)
    with pytest.raises(PublicationError):
        runner.run(plan, root)
    assert publication.dao.head(PRISM_ORIGIN) == before


def test_reviewed_snapshot_cannot_undo_rollback_or_policy_aba(publication, tmp_path):
    root, first = _snapshot(tmp_path, publication, "2026-10-13")
    runner = PublicBenchmarkRecordRefreshService(publication)
    runner.run(first, root)
    newer, update = _snapshot(tmp_path, publication, "2026-10-20")
    head = publication.dao.head(PRISM_ORIGIN)
    publication.dao.set_enabled(PRISM_ORIGIN, False, expected_generation=head.state.generation)
    publication.dao.set_enabled(PRISM_ORIGIN, True, expected_generation=head.state.generation + 1)
    with pytest.raises(PublicationConflictError):
        runner.run(update, newer)


def test_invalid_plans_do_not_expand_scope(publication, tmp_path):
    root, plan = _snapshot(tmp_path, publication, "2026-10-13")
    for update in (
        {"expected_counts": {"source": 2}},
        {"expected_counts": {"source": 6, "measurement": True}},
        {"triggers": ()},
        {"triggers": ("periodic",)},
        {"inventory": {**plan.inventory.model_dump(), "complete": False}},
        {"inventory": {**plan.inventory.model_dump(), "origin_id": "https://unapproved.invalid/repo"}},
        {"inventory": {**plan.inventory.model_dump(), "files": ()}, "expected_counts": {}},
    ):
        with pytest.raises((ValidationError, PublicationError)):
            RefreshPlan.model_validate({**plan.model_dump(), **update})
    with pytest.raises(PublicationError, match="not approved"):
        PublicBenchmarkRecordRefreshService(publication).run(plan, root, trigger="periodic")


@pytest.mark.parametrize(
    "origin",
    [
        "https://llm-d.ai",
        "https://llm-d.ai.evil.invalid",
        "https://medium.com",
        "https://github.com",
        "https://cloud.google.com",
        "https://github.com/llm-d/llm-d",
        "https://github.com/llm-d/llm-d-benchmark.git",
    ],
)
def test_sql_source_scope_rejects_all_document_and_repository_origins(origin):
    with pytest.raises(PublicationError):
        allowed_paths(origin)
    with pytest.raises(PublicationError):
        input_format(origin, "blog/article.md")
    with pytest.raises(PublicationError):
        validate_approval(origin, _approval())
    with pytest.raises(PublicationError):
        input_format(origin, "archives/llm-d-benchmarks.json", _approval())


def test_prism_approval_cannot_expand_scope_or_duplicate_paths():
    path = "archives/archived_drive_data.json"
    approval = _approval().model_copy(update={"allowed_paths": (path,)})
    validate_approval(PRISM_ORIGIN, approval)
    assert input_format(PRISM_ORIGIN, path, approval) == "json"
    assert input_format(PRISM_ORIGIN, "archives/llm-d-benchmarks.json", approval) is None
    for paths in (("**/*.json",), ("private/article.md",), (path, path)):
        with pytest.raises(PublicationError):
            validate_approval(PRISM_ORIGIN, approval.model_copy(update={"allowed_paths": paths}))


def test_web_document_refresh_cannot_repopulate_sql(publication, tmp_path):
    origin = "https://llm-d.ai"
    path = "blog/kvcache-wins-you-can-see.md"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("# llm-d\nPublic document context.\n")
    file = InventoryFile(path=path, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), format="markdown")
    approval = _approval()
    with pytest.raises(ValidationError, match="document provider is not installed"):
        RefreshPlan(
            review_reference="reviewed document-only snapshot",
            approval=approval,
            inventory=PinnedInventory(
                origin_id=origin, revision=content_digest({path: file.sha256}), complete=True, files=(file,)
            ),
            expected_counts={"source": 1, "passage": 1},
            base_generation=0,
            base_manifest_id=None,
        )
