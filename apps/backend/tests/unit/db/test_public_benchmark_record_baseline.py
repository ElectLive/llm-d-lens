from __future__ import annotations

import hashlib
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from llm_d_lens.agentic.public_benchmark_record.contracts import PublicBenchmarkRecord
from llm_d_lens.agentic.public_benchmark_record.evaluation import EvaluationCase, EvaluationSuite, run_evaluation
from llm_d_lens.agentic.public_benchmark_record.prism import PARSER_VERSION as PRISM_PARSER_VERSION
from llm_d_lens.agentic.public_benchmark_record.prism import PrismSnapshotImporter
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    PublicationError,
    ReleaseManifest,
    SnapshotManifest,
    SourceApproval,
    item_id,
    registry_id,
)
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    QUERY_PERMISSION,
    Citation,
    ExperimentContext,
    QueryEvidence,
    QueryFilters,
    QueryResponse,
)
from llm_d_lens.agentic.public_benchmark_record.service import PublicBenchmarkRecordQueryService, _verify_measurement
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, allowed_paths
from llm_d_lens.auth.contracts import Principal, RoleBinding, ScopeType
from llm_d_lens.core.exceptions import DomainError
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao, QueryMatch
from llm_d_lens.infrastructure.artifacts.store import artifact_uri

_ROOT = Path(__file__).resolve().parents[5]
_SUITE = Path(__file__).resolve().parents[2] / "fixtures/public_benchmark_record/public_baseline_v1.json"
_ORIGINS = (PRISM_ORIGIN,)


def _reader(*, authorized: bool = True) -> Principal:
    return Principal(
        user_id="public-baseline-reader",
        username="public-baseline-reader",
        bindings=(RoleBinding(role="baseline", scope_type=ScopeType.GLOBAL),),
        role_permissions={"baseline": frozenset({QUERY_PERMISSION}) if authorized else frozenset()},
    )


def _approval(origin: str, parser: str) -> SourceApproval:
    return SourceApproval(
        verified_public=True,
        approved_by="isolated fixed public baseline",
        approval_reference="local test only; not production approval",
        license_reference="existing reviewed public inputs",
        parser_version=parser,
        allowed_paths=allowed_paths(origin),
    )


@dataclass
class _Corpus:
    dao: PublicBenchmarkRecordQueryDao
    service: PublicBenchmarkRecordQueryService
    publication: PublicBenchmarkRecordPublicationService


@pytest.fixture
def public_baseline_corpus(
    postgres_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prism_initial_manifest: SnapshotManifest,
) -> _Corpus:
    snapshot = _ROOT / "private/prism-public-2026-10-13"
    if not snapshot.exists():
        pytest.skip("Prism structured baseline needs the separately provisioned pinned local snapshot")
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "public-baseline-evidence"))
    dao = PublicBenchmarkRecordQueryDao(sessionmaker(bind=postgres_engine, expire_on_commit=False))
    service = PublicBenchmarkRecordQueryService(dao=dao)
    publication = PublicBenchmarkRecordPublicationService(dao=dao, archive=service.archive)
    PrismSnapshotImporter(publication).import_snapshot(
        snapshot,
        prism_initial_manifest.inventory.revision,
        {file.path: file.sha256 for file in prism_initial_manifest.inventory.files},
        _approval(PRISM_ORIGIN, PRISM_PARSER_VERSION),
        expected_experiments=prism_initial_manifest.expected_experiments,
    )
    assert dao.count_published(PRISM_ORIGIN) > 0
    return _Corpus(dao=dao, service=service, publication=publication)


class _PublicCitationVerifier:
    """Resolve emitted identities against published DAO rows and retained source bytes."""

    def __init__(self, corpus: _Corpus, origins: tuple[str, ...] = _ORIGINS) -> None:
        self.corpus = corpus
        self.members = {
            origin: {item.item_id: item for item in corpus.dao.list_published(origin)} for origin in origins
        }

    def _bytes(self, item: PublicBenchmarkRecord) -> bytes:
        members = self.members[item.origin_id]
        source = members[item_id(item.origin_id, f"file:{item.source_path}", item.source_revision)]
        assert source.kind == "source" and source.source_role == "file"
        assert (source.origin_id, source.source_revision, source.source_path, source.content_sha256) == (
            item.origin_id,
            item.source_revision,
            item.source_path,
            item.content_sha256,
        )
        assert item.parent_id in members
        data = self.corpus.service.archive.read(source.content_sha256)
        assert hashlib.sha256(data).hexdigest() == source.content_sha256
        return data

    def _passage(self, item: PublicBenchmarkRecord) -> None:
        text = self._bytes(item).decode("utf-8")
        pointer = item.source_pointer or {}
        start, end = pointer["start_char"], pointer["end_char"]
        assert type(start) is int and type(end) is int and 0 <= start < end <= len(text)
        assert text[start:end] == item.passage_text

    def _context(self, context: ExperimentContext, origin: str) -> PublicBenchmarkRecord:
        item = self.members[origin][context.item_id]
        assert item.kind == "experiment" and item.origin_id == origin
        trusted_context = {
            name: getattr(item, name)
            for name in ("item_id", "origin_id", "model", "hardware", "runtime", "workload", "configuration")
        } | {"source_pointer": item.source_pointer or {}}
        assert context.model_dump() == trusted_context
        data = self._bytes(item)
        pointer = item.source_pointer or {}
        if "line" in pointer:
            lines = data.decode("utf-8").splitlines()
            assert 1 <= pointer["line"] <= len(lines)
            assert "|" in lines[pointer["line"] - 1]
        return item

    def _figure(self, item: PublicBenchmarkRecord, evidence: QueryEvidence, case: EvaluationCase) -> None:
        assert self._bytes(item).startswith(b"\x89PNG\r\n\x1a\n")
        assert evidence.value is None and evidence.experiment is None
        assert evidence.coverage == item.coverage
        assert evidence.missing_series == tuple(item.missing_series or ())
        pointer = item.source_pointer or {}
        assert pointer["numeric_extraction"] == "none"
        references = pointer["references"]
        assert isinstance(references, list) and references
        for reference in references:
            assert isinstance(reference, dict)
            passage = self.members[item.origin_id][reference["passage_id"]]
            assert passage.kind == "passage"
            assert passage.source_path == reference["report_path"]
            assert passage.content_sha256 == reference["report_sha256"]
            assert passage.heading_path == reference["heading_path"]
            self._passage(passage)
            line = self._bytes(passage).decode("utf-8").splitlines()[reference["line"] - 1]
            assert reference["target"] in line and reference["caption"] in line
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(passage.source_path), reference["target"]))
            assert resolved == item.source_path == reference["resolved_path"]
        links = [
            member
            for member in self.members[item.origin_id].values()
            if member.kind == "figure_link" and member.parent_id == item.item_id
        ]
        expected = set()
        for link in links:
            experiment = self.members[item.origin_id][link.referenced_item_id]
            if (
                case.request.filters.hardware is not None
                and (experiment.hardware or "").casefold() != case.request.filters.hardware.casefold()
            ):
                continue
            expected.add(experiment.item_id)
            self._bytes(link)
            assert (link.source_pointer or {})["report_sha256"] == experiment.content_sha256
            assert (link.source_pointer or {})["report_path"] == experiment.source_path
            assert (link.source_pointer or {})["table_index"] == (experiment.source_pointer or {})["table_index"]
        assert {context.item_id for context in evidence.related_experiments} == expected
        for context in evidence.related_experiments:
            self._context(context, item.origin_id)

    def __call__(self, case: EvaluationCase, evidence: QueryEvidence) -> tuple[str, ...]:
        try:
            assert evidence.origin == "public" and not evidence.corroborating_citations
            citation = evidence.citation
            assert isinstance(citation, Citation)
            item = self.corpus.dao.get(evidence.item_id)
            assert item is not None and item == self.members[citation.origin_id][evidence.item_id]
            head = self.corpus.dao.head(item.origin_id)
            assert head.state.enabled and head.manifest is not None
            assert item.item_id in (head.manifest.manifest_item_ids or ())
            trusted_citation = {
                "origin_id": item.origin_id,
                "source_revision": item.source_revision,
                "source_path": item.source_path,
                "sha256": item.content_sha256,
                "artifact_uri": artifact_uri("benchmark-rag", item.content_sha256, "source"),
                "source_pointer": item.source_pointer or {},
                "heading_path": tuple(item.heading_path or ()),
                "published_manifest_id": head.registry.published_manifest_id,
            }
            assert citation.model_dump() == trusted_citation
            data = self._bytes(item)
            assert evidence.text == ((item.passage_text or item.caption or "")[:2500] or None)
            for name in (
                "metric",
                "statistic",
                "value",
                "unit",
                "normalized_value",
                "normalized_unit",
                "normalization_rule",
                "series",
            ):
                assert getattr(evidence, name) == getattr(item, name)
            if item.kind == "measurement":
                assert evidence.evidence_type == item.evidence_type
                assert evidence.experiment is not None and not evidence.related_experiments
                experiment = self._context(evidence.experiment, item.origin_id)
                assert experiment.item_id == item.experiment_id
                source = self.members[item.origin_id][
                    item_id(item.origin_id, f"file:{item.source_path}", item.source_revision)
                ]
                _verify_measurement(
                    QueryMatch(item, experiment, source, citation.published_manifest_id, head.state.generation),
                    data,
                )
                cited_item = item.model_copy(update={"source_pointer": citation.source_pointer})
                _verify_measurement(
                    QueryMatch(cited_item, experiment, source, citation.published_manifest_id, head.state.generation),
                    data,
                )
            elif item.kind == "passage":
                assert evidence.evidence_type == "prose_summary" and evidence.value is None
                assert evidence.experiment is None and not evidence.related_experiments
                self._passage(item)
            else:
                assert item.kind == "figure" and evidence.evidence_type == "image_trend"
                self._figure(item, evidence, case)
        except (AssertionError, KeyError, IndexError, TypeError, ValueError, OSError, DomainError):
            return ("public_citation_not_verified_against_source",)
        return ()


def test_public_baseline_is_fixed_and_covers_all_public_categories(prism_initial_manifest: SnapshotManifest) -> None:
    suite = EvaluationSuite.model_validate_json(_SUITE.read_bytes())
    assert suite.name == "public-baseline-v1"
    assert len(suite.cases) == 10
    assert {case.category for case in suite.cases} == {
        "numeric",
        "false_join",
        "comparison",
        "access",
        "provenance",
        "overlap",
    }
    assert set(suite.source_pins) == set(_ORIGINS)
    assert suite.source_pins[PRISM_ORIGIN].revision == prism_initial_manifest.inventory.revision
    assert all(not case.request.include_internal for case in suite.cases)


def test_real_prism_structured_baseline(
    public_baseline_corpus: _Corpus, prism_initial_manifest: SnapshotManifest
) -> None:
    corpus = public_baseline_corpus
    suite = EvaluationSuite.model_validate_json(_SUITE.read_bytes())
    for origin, pin in suite.source_pins.items():
        head = corpus.dao.head(origin)
        assert head.state.enabled and head.manifest is not None
        assert head.registry.published_manifest_id == head.manifest.item_id
        manifest = ReleaseManifest.model_validate(head.manifest.source_details)
        assert manifest.inventory.origin_id == origin
        assert manifest.inventory.revision == head.manifest.source_revision == pin.revision
        if pin.inventory_sha256 is not None:
            assert manifest.inventory.digest == pin.inventory_sha256
        if origin == PRISM_ORIGIN:
            assert manifest.inventory == prism_initial_manifest.inventory
    verify = _PublicCitationVerifier(corpus, origins=(PRISM_ORIGIN,))
    emitted: list[tuple[EvaluationCase, QueryEvidence]] = []

    def execute(case: EvaluationCase) -> QueryResponse:
        if case.actor in ("inactive_source", "revoked_source"):
            origin = PRISM_ORIGIN
            head = corpus.dao.head(origin)
            corpus.dao.set_enabled(origin, False, expected_generation=head.state.generation)
            try:
                return corpus.service.query(case.request, _reader())
            finally:
                corpus.dao.set_enabled(origin, True, expected_generation=corpus.dao.head(origin).state.generation)
        if case.actor == "reject_unapproved_source":
            origin = "unapproved:fixed-baseline-negative-control"
            counts = {PRISM_ORIGIN: corpus.dao.count_published(PRISM_ORIGIN)}
            with pytest.raises(PublicationError, match="allowlist"):
                corpus.publication.register_source(origin, _approval(PRISM_ORIGIN, PRISM_PARSER_VERSION))
            with pytest.raises(ValidationError, match="approved public"):
                QueryFilters(origins=(origin,))
            assert corpus.dao.get(registry_id(origin)) is None
            assert {PRISM_ORIGIN: corpus.dao.count_published(PRISM_ORIGIN)} == counts
        principal = None if case.actor == "anonymous" else _reader(authorized=case.actor != "forbidden")
        return corpus.service.query(case.request, principal)

    def verify_evidence(case: EvaluationCase, evidence: QueryEvidence) -> tuple[str, ...]:
        emitted.append((case, evidence))
        return verify(case, evidence)

    report = run_evaluation(suite, execute=execute, verify_evidence=verify_evidence)
    if directory := os.environ.get("LENS_RAG_BASELINE_REPORT_DIR"):
        destination = Path(directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        pending = destination / "prism-structured-baseline.json.pending"
        pending.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        pending.replace(destination / "prism-structured-baseline.json")
    assert report.passed, report.model_dump_json(indent=2)
    assert report.total_cases == report.passed_cases == len(suite.cases)
    assert report.verified_citations == len(emitted) > 0
    assert {evidence.citation.origin_id for _, evidence in emitted} == {PRISM_ORIGIN}
    assert all(not verify(case, evidence) for case, evidence in emitted)

    case, evidence = next(pair for pair in emitted if pair[1].evidence_type == "raw_json")
    assert isinstance(evidence.citation, Citation)
    false_citation = evidence.citation.model_copy(update={"sha256": "0" * 64})
    assert verify(case, evidence.model_copy(update={"citation": false_citation}))
    false_pointer = evidence.citation.model_copy(update={"source_pointer": {"fabricated": True}})
    assert verify(case, evidence.model_copy(update={"citation": false_pointer}))
    assert verify(case, evidence.model_copy(update={"text": "fabricated source excerpt"}))
