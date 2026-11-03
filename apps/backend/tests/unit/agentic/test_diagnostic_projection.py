"""Tests for replayable diagnostic evidence projections."""

import hashlib
import json

import pytest

from llm_d_lens.agentic.diagnostic_projection import (
    DiagnosticArtifactError,
    project_diagnostic_excerpt,
    project_planning_snapshot,
    projection_index_document,
    verify_artifact_evidence,
)
from llm_d_lens.agentic.evidence_search_repository import DiagnosticEvidenceChunk
from llm_d_lens.agentic.opensearch_evidence_search import EvidenceSearchError
from llm_d_lens.agentic.opensearch_evidence_search import verify_artifact_evidence as verify_search_artifact
from llm_d_lens.infrastructure.artifacts.store import register_artifacts


@pytest.mark.parametrize(
    ("verify", "error"),
    [(verify_artifact_evidence, DiagnosticArtifactError), (verify_search_artifact, EvidenceSearchError)],
)
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("content", "checksum"),
        ("manifest", "manifest checksum"),
        ("missing_manifest", "cannot be resolved"),
        ("invalid_manifest", "cannot be resolved"),
        ("retention", "cannot be resolved"),
        ("missing_references", "missing immutable references"),
        ("unapproved_root", "not approved"),
        ("missing_file", "cannot be resolved"),
    ],
)
def test_verify_artifact_evidence_rejects_untrusted_artifacts(tmp_path, verify, error, mutation, message):
    root = tmp_path / "evaluation"
    root.mkdir()
    path = root / "diagnostic.log"
    path.write_text("OOM", encoding="utf-8")
    manifest = register_artifacts(root, owner_type="evaluation", owner_id="run-1", retention_class="diagnostic")
    entry = manifest["files"][0]
    chunk = DiagnosticEvidenceChunk(
        evidence_id="evidence-1",
        source_type="deployment_log",
        excerpt="OOM",
        content_hash="summary-hash",
        artifact_uri=entry["uri"],
        file_sha256=entry["sha256"],
        manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        tenant_id="tenant-a",
        acl=("operator",),
        score=1,
        ranking_explanation="test",
    )
    assert verify(root, chunk, approved_roots=(tmp_path,)) == path
    approved_roots = (tmp_path,)
    if mutation == "content":
        path.write_text("replaced", encoding="utf-8")
    elif mutation == "manifest":
        (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    elif mutation == "missing_manifest":
        (root / "manifest.json").unlink()
    elif mutation == "invalid_manifest":
        (root / "manifest.json").write_text("{", encoding="utf-8")
    elif mutation == "retention":
        (root / "manifest.json").write_text(json.dumps({**manifest, "retention_class": "temporary"}))
    elif mutation == "missing_references":
        chunk = chunk.model_copy(update={"manifest_sha256": None})
    elif mutation == "unapproved_root":
        approved_roots = (tmp_path / "elsewhere",)
    elif mutation == "missing_file":
        path.unlink()
    with pytest.raises(error, match=message):
        verify(root, chunk, approved_roots=approved_roots)


def test_projects_logs_traces_and_reference_sources_as_bounded_excerpts():
    for source_type in ("deployment_log", "profiling_trace", "github", "guide", "paper"):
        projection = project_diagnostic_excerpt(
            evidence_id=f"{source_type}:1",
            source_type=source_type,
            title="Reference",
            excerpt="out of memory " * 1_000,
            tenant_id="tenant-a",
            acl=("operator",),
        )

        assert len(projection.excerpt) == 8_000
        assert projection.content_hash == hashlib.sha256(projection.excerpt.encode("utf-8")).hexdigest()


def test_projects_planning_snapshot_without_prompt_or_tool_trace(tmp_path):
    snapshot = {
        "schema_version": 1,
        "snapshot_id": "snapshot-1",
        "created_at": "2026-09-21T00:00:00+00:00",
        "event": "created",
        "run_id": "run-1",
        "selected_candidate": {
            "id": "candidate-1",
            "provider_ref": "baseline-vllm",
            "replicas": 1,
            "tensor_parallel_size": 1,
        },
        "decision": {"action": "select_candidate", "candidate_id": "candidate-1", "rationale": "secret reasoning"},
        "candidate_catalog": [{"id": "candidate-1"}],
        "planner": {"name": "deterministic"},
        "generator": {"name": "ai-mcp", "tool_trace": [{"arguments": "sensitive"}]},
        "request": {"planner_prompt": "must not be indexed"},
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    projection = project_planning_snapshot(path, tenant_id="tenant-a", acl=("operator",))
    document = projection_index_document(projection)

    assert projection.evidence_id == "planning-snapshot:snapshot-1"
    assert projection.content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    assert projection.plan_id == "run-1"
    assert "secret reasoning" not in projection.excerpt
    assert "sensitive" not in projection.excerpt
    assert "must not be indexed" not in projection.excerpt
    assert document["source_type"] == "planning_snapshot"
