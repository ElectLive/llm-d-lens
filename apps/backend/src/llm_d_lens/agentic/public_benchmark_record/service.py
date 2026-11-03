"""Deterministic public and opt-in authorized Agentic evidence; text is never executed."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal

from llm_d_lens.agentic.public_benchmark_record.access import require_query_access
from llm_d_lens.agentic.public_benchmark_record.agentic_answers import (
    agentic_evidence,
    merge_verified,
    pinned_evaluate_identity,
)
from llm_d_lens.agentic.public_benchmark_record.contracts import (
    PublicBenchmarkRecord,
    effective_runtime,
    model_identities_match,
    normalize_statistic,
)
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import PublicationError
from llm_d_lens.agentic.public_benchmark_record.query_contracts import (
    Abstention,
    AgenticCitation,
    Citation,
    ExperimentContext,
    QueryEvidence,
    QueryFilters,
    QueryRequest,
    QueryResponse,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, SourceArchive
from llm_d_lens.auth.contracts import Principal
from llm_d_lens.core.exceptions import DomainError, DomainValidationError
from llm_d_lens.db.dao.public_benchmark_record_query import PublicBenchmarkRecordQueryDao, QueryMatch
from llm_d_lens.infrastructure.artifacts.store import artifact_uri

if TYPE_CHECKING:
    from llm_d_lens.db.dao.benchmark_experiment_query import BenchmarkExperimentQueryDao

logger = logging.getLogger(__name__)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_EVALUATE_ID = re.compile(
    r"\bevaluate:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?::stage:(?:0|[1-9][0-9]*)|:matrix:(?:0|[1-9][0-9]*):stage:(?:0|[1-9][0-9]*))?(?![\w:.-])",
    re.IGNORECASE,
)
_HARDWARE_NAME = re.compile(
    r"\b(?:[HAB]\d{2,4}|MI\d{3}X?|(?:Intel\s+)?Gaudi\s*\d+|TPU(?:\s+v\d+[a-z]?)?)\b",
    re.IGNORECASE,
)
_STATISTIC = re.compile(
    r"\b(?:p\d{1,3}(?:\.\d+)?|\d{1,3}(?:\.\d+)?(?:st|nd|rd|th)?\s+percentile"
    r"|mean|average|avg|median|minimum|maximum|min|max)\b",
    re.IGNORECASE,
)
_MODEL_IDENTIFIER = re.compile(
    r"(?<![\w./-])(?:[a-zA-Z][\w.-]*/[\w.-]+|(?=[\w.-]*\d)[a-zA-Z]\w*(?:[.-]\w+)+)(?![\w./-])"
)
_METRIC_TERMS = (
    (("weight-load", "weight load", "weight loading"), "weight_load"),
    (("throughput", "token throughput"), "throughput"),
    (("ttft", "time to first token"), "ttft"),
    (("tpot", "time per output token"), "tpot"),
    (("latency", "request latency"), "latency"),
)
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "what",
        "which",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "does",
        "do",
        "can",
        "could",
        "should",
        "would",
        "for",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "and",
        "or",
        "it",
        "its",
        "this",
        "that",
        "from",
        "as",
        "about",
        "give",
        "tell",
        "me",
        "please",
        "intended",
        "capture",
        "recorded",
        "reported",
        "exact",
        "list",
        "unreported",
    ]
)


class EvidenceUnavailableError(DomainError):
    code = "rag_evidence_unavailable"
    status_code = 503


@dataclass(frozen=True)
class QueryPlan:
    intent: Literal["numeric", "comparison", "definition", "figure"]
    filters: QueryFilters
    text_query: str
    precondition: Abstention | None = None


def _query_statistic(value: str) -> str:
    value = value.casefold()
    if "percentile" in value:
        return "p" + re.split(r"(?:st|nd|rd|th)?\s+", value, maxsplit=1)[0]
    value = {"average": "avg", "minimum": "min", "maximum": "max"}.get(value, value)
    return normalize_statistic(value) or value


def _question_hardware(question: str) -> tuple[str | None, Literal["ambiguous", "unrecognized"] | None]:
    clauses = tuple(
        re.finditer(
            r"\b(?:on\s+|(?:hardware|accelerator|device|gpu)(?:\s*[:=]\s*|\s+))(.+?)"
            r"(?=\s+(?:for|with|at|under|using|in|concurrency|runtime)\b|[?;!]|$)",
            question,
            re.IGNORECASE,
        )
    )
    names = tuple(_HARDWARE_NAME.finditer(question))
    if clauses:
        hardware = clauses[0][1].strip().rstrip(".").strip('"`')
        if (
            len(clauses) > 1
            or re.search(r"\b(?:and|or|versus|vs)\b|[,;/+]", hardware, re.IGNORECASE)
            or any(not clauses[0].start(1) <= name.start() < clauses[0].end(1) for name in names)
        ):
            return hardware, "ambiguous"
        return hardware, None if _HARDWARE_NAME.search(hardware) else "unrecognized"
    values = tuple(dict.fromkeys(match[0].casefold() for match in names))
    if not values and re.search(r"\b(?:on|hardware|accelerator|device|gpu)\s*[:=]?\s*[?.!]*$", question, re.IGNORECASE):
        return None, "ambiguous"
    return (names[0][0] if names else None), "ambiguous" if len(values) > 1 else None


def _question_models(question: str, hardware: str | None) -> tuple[tuple[str, ...], bool]:
    named = tuple(
        re.finditer(
            r'(?<![\w./-])model(?![\w./-])(?:(?:\s*[:=]\s*|\s+)(?:"([^"]*)"|`([^`]*)`|([\w./-]+)))?',
            question,
            re.IGNORECASE,
        )
    )
    models = [
        match[0]
        for match in _MODEL_IDENTIFIER.finditer(question)
        if not _UUID.fullmatch(match[0]) and not any(label.start() <= match.start() < label.end() for label in named)
    ]
    ambiguous = False
    # Bare or multiword names need a model label or a "for ... on ..." subject;
    # structured identifiers do not depend on a hardcoded model-family catalog.
    for label in named:
        value = next((group for group in label.groups() if group is not None), "").strip()
        if not value or value.casefold() in _STOP_WORDS or value.casefold() in {"under", "using", "model"}:
            ambiguous = True
        else:
            models.append(value.rstrip("."))
    for match in re.finditer(r"\bfor\s+(.+?)(?=\s+(?:on|with|at|under|in|using)\b|[?;!]|$)", question, re.IGNORECASE):
        subject = match[1].strip().rstrip(".")
        if hardware is not None and subject.casefold() == hardware.casefold():
            continue
        if (
            re.match(r"(?:prism|record|run|benchmark|evaluate)\b", subject, re.IGNORECASE)
            or _UUID.fullmatch(subject)
            or re.fullmatch(
                r"(?:an?|the)\s+(?:(?:unregistered|registered|baseline|benchmark|reported|recorded|internal|public)"
                r"\s+)*(?:record|run)",
                subject,
                re.IGNORECASE,
            )
        ):
            continue
        if re.match(r"model(?:\s|[:=]|$)", subject, re.IGNORECASE):
            continue
        if re.fullmatch(r"[\w./-]+", subject) and subject.casefold() not in _STOP_WORDS:
            models.append(subject)
        else:
            ambiguous = True
    unique: list[str] = []
    for model in models:
        if hardware is not None and model.casefold() == hardware.casefold():
            continue
        if not any(model_identities_match(model, existing) for existing in unique):
            unique.append(model)
    return tuple(unique), ambiguous


def _metric_definition(question: str, filters: QueryFilters) -> bool:
    if any(
        (
            filters.record_id,
            filters.experiment_id,
            filters.model,
            filters.hardware,
            filters.runtime,
            filters.metric,
            filters.statistic,
            filters.unit,
            filters.workload,
            filters.series,
        )
    ):
        return False
    if re.search(r"\b(?:define|definitions?|meaning)\b|\bwhat (?:does|do)\b.*\b(?:mean|stand for)\b", question):
        return True
    subject = re.sub(r"^(?:what (?:is|are)|explain|describe)\s+(?:(?:the|a|an)\s+)?", "", question)
    subject = re.sub(r"\s+(?:metric|term)$", "", subject.rstrip("?!. "))
    return subject != question and any(subject in words for words, _ in _METRIC_TERMS)


def plan_query(request: QueryRequest) -> QueryPlan:
    question = request.question.casefold()
    ids = tuple(dict.fromkeys(value.lower() for value in _UUID.findall(question)))
    evaluate_ids = tuple(dict.fromkeys(match[0].lower() for match in _EVALUATE_ID.finditer(question)))
    metric = next(
        (
            metric
            for words, metric in _METRIC_TERMS
            if any(re.search(r"\b" + re.escape(word) + r"\b", question) for word in words)
        ),
        None,
    )
    intent = request.intent
    if intent == "auto":
        if metric is not None and not ids and _metric_definition(question, request.filters):
            intent = "definition"
        elif re.search(r"\bpng\b", question) or any(
            word in question for word in ("plot", "figure", "chart", ".png", "curve", "diagram", "plotted series")
        ):
            intent = "figure"
        elif any(
            word in question for word in (" versus ", " vs ", "compare", "comparison", "compared with", "comparison of")
        ):
            intent = "comparison"
        elif ids or request.filters.metric or metric is not None:
            intent = "numeric"
        else:
            intent = "definition"
    updates: dict[str, object] = {}
    internal_identity = request.include_internal and any(
        phrase in question for phrase in ("evaluate:", "internal benchmark", "agentic", "my evaluation")
    )
    if len(ids) == 1:
        record_id = (
            evaluate_ids[0]
            if internal_identity and len(evaluate_ids) == 1
            else (f"evaluate:{ids[0]}" if internal_identity else ids[0])
        )
        permitted_ids = {ids[0], record_id}
        if request.include_internal:
            permitted_ids.add(f"evaluate:{ids[0]}")
        if request.filters.record_id is not None and request.filters.record_id.casefold() not in permitted_ids:
            raise DomainValidationError("the question's record ID conflicts with the explicit filter")
        if request.filters.record_id is None:
            updates["record_id"] = record_id
        if not request.include_internal and "origins" not in request.filters.model_fields_set:
            updates["origins"] = (PRISM_ORIGIN,)
    condition_precondition = None
    if request.filters.statistic is not None:
        updates["statistic"] = _query_statistic(request.filters.statistic)
    if intent in ("numeric", "comparison"):
        statistics = tuple(dict.fromkeys(_query_statistic(match[0]) for match in _STATISTIC.finditer(question)))
        if len(statistics) > 1:
            condition_precondition = Abstention(
                code="statistic_scope_required",
                message="Use separate exact queries for different statistics; no statistic was discarded.",
            )
        elif statistics:
            if request.filters.statistic is not None and updates["statistic"] != statistics[0]:
                raise DomainValidationError("the question's statistic conflicts with the explicit filter")
            updates["statistic"] = statistics[0]
        if request.filters.metric is None and metric is not None:
            updates["metric"] = metric
    if intent in ("numeric", "comparison"):
        hardware, hardware_scope = _question_hardware(request.question)
        if hardware is not None:
            if request.filters.hardware is not None and request.filters.hardware.casefold() != hardware.casefold():
                raise DomainValidationError("the question's hardware conflicts with the explicit filter")
            if request.filters.hardware is None:
                updates["hardware"] = hardware
        if hardware_scope == "ambiguous" or (hardware_scope == "unrecognized" and request.filters.hardware is None):
            condition_precondition = condition_precondition or Abstention(
                code="hardware_scope_required",
                message=(
                    "Supply one exact hardware filter and an unambiguous question; "
                    "no unrecognized or compound hardware condition was discarded."
                ),
            )
        models, ambiguous_model = _question_models(request.question, hardware)
        if len(models) > 1 or ambiguous_model:
            condition_precondition = condition_precondition or Abstention(
                code="model_scope_required",
                message=(
                    "Use one exact model identifier per query, or an explicit model filter with an unambiguous "
                    "question; no model condition was discarded."
                ),
            )
        elif models:
            model = models[0]
            if request.filters.model is not None and not model_identities_match(request.filters.model, model):
                raise DomainValidationError("the question's model conflicts with the explicit filter")
            if request.filters.model is None:
                updates["model"] = model
        concurrency = re.search(r"(?:concurrency|concurrent)\s*[:=]?\s*(\d+)", question)
        if concurrency is not None:
            workload = dict(request.filters.workload)
            value = int(concurrency[1])
            if "concurrency" in workload and workload["concurrency"] != value:
                raise DomainValidationError("the question's concurrency conflicts with its explicit workload filter")
            workload["concurrency"] = value
            updates["workload"] = workload
    filters = QueryFilters.model_validate({**request.filters.model_dump(), **updates})
    tokens = tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[^\W_]+(?:-[^\W_]+)*", question, flags=re.UNICODE)
            if token not in _STOP_WORDS and len(token) > 1
        )
    )[:20]
    text_query = " OR ".join(f'"{token}"' for token in tokens)
    precondition = None
    if len(ids) > 1 or len(evaluate_ids) > 1:
        precondition = Abstention(
            code="comparison_requires_explicit_scope",
            message=(
                "Use separate exact record queries; this interface does not infer a comparison between multiple IDs."
            ),
        )
    elif "evaluate:" in question and len(tuple(_EVALUATE_ID.finditer(question))) != question.count("evaluate:"):
        precondition = Abstention(
            code="record_scope_required",
            message="Supply a complete canonical evaluation, stage, or matrix-stage record identifier.",
        )
    elif not request.include_internal and any(
        phrase in question
        for phrase in (
            "evaluate:",
            "internal benchmark",
            "agentic benchmark record",
            "my evaluation",
            "internal records",
        )
    ):
        precondition = Abstention(
            code="internal_branch_disabled",
            message="Internal Agentic record retrieval is disabled until its separate authorization gate passes.",
        )
    elif any(
        phrase in question
        for phrase in ("same experiment", "distinct experiment", "identical experiment", "semantic deduplication")
    ):
        precondition = Abstention(
            code="unverified_experiment_identity",
            message="Similar model/hardware names do not prove run identity or an independent-experiment count.",
        )
    elif condition_precondition is not None:
        precondition = condition_precondition
    elif intent in ("numeric", "comparison") and not any(
        (
            filters.record_id,
            filters.experiment_id,
            filters.source_path,
            filters.model,
            filters.hardware,
            filters.runtime,
            filters.workload,
            filters.series,
        )
    ):
        precondition = Abstention(
            code="experiment_conditions_required",
            message="Supply an exact record, source path, model or experimental-condition filter; no run was guessed.",
        )
    return QueryPlan(intent=intent, filters=filters, text_query=text_query, precondition=precondition)


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal, str)):
        raise EvidenceUnavailableError("pinned measurement does not resolve to a numeric source value")
    try:
        number = Decimal(str(value).replace(",", "").replace("\u2212", "-"))
    except InvalidOperation as error:
        raise EvidenceUnavailableError("pinned measurement token is invalid") from error
    if not number.is_finite():
        raise EvidenceUnavailableError("pinned measurement is not finite")
    return number


def _verify_measurement(match: QueryMatch, data: bytes) -> None:
    item = match.item
    pointer = item.source_pointer or {}
    if item.evidence_type == "raw_json":
        path = pointer.get("json_pointer")
        if not isinstance(path, str) or not path.startswith("/"):
            raise EvidenceUnavailableError("measurement has no exact original JSON pointer")
        try:
            value: object = json.loads(data, parse_float=Decimal)
            for segment in path[1:].split("/"):
                key = segment.replace("~1", "/").replace("~0", "~")
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdecimal() and int(key) < len(value):
                    value = value[int(key)]
                else:
                    raise EvidenceUnavailableError("measurement JSON pointer no longer resolves")
        except (json.JSONDecodeError, UnicodeError) as error:
            raise EvidenceUnavailableError("pinned measurement JSON is invalid") from error
        if isinstance(value, str) or _decimal(value) != item.original_value:
            raise EvidenceUnavailableError("measurement disagrees with its pinned JSON number")
    elif item.evidence_type == "published_table":
        line_number = pointer.get("line")
        column = pointer.get("column")
        cell = pointer.get("cell")
        subvalue = pointer.get("subvalue")
        lines = data.decode("utf-8").splitlines()
        if (
            type(line_number) is not int
            or not 1 <= line_number <= len(lines)
            or type(column) is not int
            or column < 0
            or not isinstance(cell, str)
            or not isinstance(subvalue, dict)
        ):
            raise EvidenceUnavailableError("measurement has no exact original table cell")
        line = lines[line_number - 1]
        cells = re.split(r"(?<!\\)\|", line)
        if line.lstrip().startswith("|"):
            cells = cells[1:]
        if line.rstrip().endswith("|"):
            cells = cells[:-1]
        start, end = subvalue.get("start"), subvalue.get("end")
        if (
            column >= len(cells)
            or cells[column] != cell
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(cell)
            or _decimal(cell[start:end]) != item.original_value
        ):
            raise EvidenceUnavailableError("measurement disagrees with its pinned table token")
    else:
        raise EvidenceUnavailableError("unsupported exact numeric evidence type")


def _context(item: PublicBenchmarkRecord) -> ExperimentContext:
    return ExperimentContext(
        item_id=item.item_id,
        origin_id=item.origin_id,
        model=item.model,
        hardware=item.hardware,
        runtime=effective_runtime(item.runtime),
        workload=item.workload,
        configuration=item.configuration,
        source_pointer=item.source_pointer or {},
    )


def _verify_passage_text(item: PublicBenchmarkRecord, data: bytes) -> None:
    pointer = item.source_pointer or {}
    start, end = pointer.get("start_char"), pointer.get("end_char")
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise EvidenceUnavailableError("passage original text is not UTF-8") from error
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(text)
        or text[start:end] != item.passage_text
    ):
        raise EvidenceUnavailableError("passage disagrees with its pinned original offsets")


class PublicBenchmarkRecordQueryService:
    def __init__(
        self,
        dao: PublicBenchmarkRecordQueryDao | None = None,
        archive: SourceArchive | None = None,
        agentic_dao: BenchmarkExperimentQueryDao | None = None,
    ) -> None:
        self.dao = dao if dao is not None else PublicBenchmarkRecordQueryDao()
        self.archive = archive if archive is not None else SourceArchive()
        self.agentic_dao = agentic_dao

    def query(
        self,
        request: QueryRequest,
        principal: Principal | None,
        *,
        all_matches: bool = False,
        experiment_limit: int | None = None,
    ) -> QueryResponse:
        require_query_access(principal)
        assert principal is not None
        request = QueryRequest.model_validate(request.model_dump(exclude_unset=True))
        plan = plan_query(request)
        if experiment_limit is not None and (
            all_matches
            or request.include_internal
            or plan.intent not in ("numeric", "comparison")
            or not plan.filters.model
            or not plan.filters.hardware
            or self.dao.planning_facts is None
            or not 1 <= experiment_limit <= 50
        ):
            raise DomainValidationError(
                "Planning experiment retrieval requires exact model/hardware and planning facts"
            )
        if all_matches and (
            request.include_internal
            or plan.intent not in ("numeric", "comparison")
            or not plan.filters.model
            or not plan.filters.hardware
        ):
            raise DomainValidationError("Complete public retrieval requires exact model and hardware scope")
        if plan.precondition is not None:
            return QueryResponse(
                question=request.question,
                intent=plan.intent,
                filters=plan.filters,
                status="abstained",
                answer=plan.precondition.message,
                evidence=(),
                matched_items=0,
                truncated=False,
                abstentions=(plan.precondition,),
                internal_branch_enabled=request.include_internal,
            )
        if plan.intent == "definition":
            raise EvidenceUnavailableError("Document definitions require a document provider, which is not installed")
        internal_result = None
        internal_dao = self.agentic_dao
        if request.include_internal and plan.intent in ("numeric", "comparison"):
            if internal_dao is None:
                from llm_d_lens.db.dao.benchmark_experiment_query import BenchmarkExperimentQueryDao

                internal_dao = BenchmarkExperimentQueryDao(self.dao._get_session_factory())
            internal_result = internal_dao.search(plan.filters, principal, limit=50)
        if experiment_limit is not None:
            result = self.dao.search(
                plan.filters,
                intent=plan.intent,
                text_query=plan.text_query,
                limit=None,
                experiment_limit=experiment_limit,
            )
        else:
            result = self.dao.search(
                plan.filters,
                intent=plan.intent,
                text_query=plan.text_query,
                limit=None if all_matches else 50 if request.include_internal else request.limit,
            )
        evidence: list[QueryEvidence] = []
        identities: dict[str, str] = {}
        cache: dict[str, bytes] = {}
        for match in result.matches:
            item = match.item
            if item.content_sha256 not in cache:
                try:
                    cache[item.content_sha256] = self.archive.read(item.content_sha256)
                except (OSError, PublicationError, json.JSONDecodeError) as error:
                    logger.warning("event=rag_query_evidence_unavailable reason=%s", type(error).__name__)
                    raise EvidenceUnavailableError("pinned evidence is missing or corrupt") from error
            data = cache[item.content_sha256]
            pointer = item.source_pointer or {}
            if item.kind == "measurement":
                _verify_measurement(match, data)
                identity = pinned_evaluate_identity(match, data)
                if identity is not None:
                    identities[item.item_id] = identity
            elif item.kind == "passage" and "start_char" in pointer:
                _verify_passage_text(item, data)
            if item.kind == "measurement":
                if item.evidence_type not in ("raw_json", "published_table"):
                    raise EvidenceUnavailableError("measurement evidence type is invalid")
                evidence_type = item.evidence_type
            else:
                evidence_type = "image_trend" if item.kind == "figure" else "prose_summary"
            evidence.append(
                QueryEvidence(
                    item_id=item.item_id,
                    evidence_type=evidence_type,
                    verified_evaluate_run_id=identities.get(item.item_id),
                    experiment_time=(
                        match.experiment.timestamp
                        if match.experiment is not None and match.experiment.timestamp_kind == "experiment"
                        else None
                    ),
                    citation=Citation(
                        origin_id=item.origin_id,
                        source_revision=item.source_revision,
                        source_path=item.source_path,
                        sha256=item.content_sha256,
                        artifact_uri=artifact_uri("benchmark-rag", item.content_sha256, "source"),
                        source_pointer=pointer,
                        heading_path=tuple(item.heading_path or ()),
                        published_manifest_id=match.manifest_id,
                    ),
                    experiment=_context(match.experiment) if match.experiment is not None else None,
                    related_experiments=tuple(_context(context) for context in match.related_experiments),
                    metric=item.metric,
                    statistic=item.statistic,
                    value=item.value,
                    unit=item.unit,
                    original_value=item.original_value if item.original_value is not None else item.value,
                    original_unit=item.original_unit or item.unit,
                    normalized_value=item.normalized_value,
                    normalized_unit=item.normalized_unit,
                    normalization_rule=item.normalization_rule,
                    series=item.series,
                    text=(item.passage_text or item.caption or "")[:2500] or None,
                    coverage=item.coverage,
                    missing_series=tuple(item.missing_series or ()),
                )
            )
        internal_evidence = []
        if internal_result is not None:
            for match in internal_result.matches:
                internal_evidence.extend(agentic_evidence(match, plan.filters))
            evidence = merge_verified(evidence, internal_evidence, identities)
        candidate_total = result.total + (internal_result.total if internal_result is not None else 0)
        window_truncated = (
            result.total > len(result.matches)
            or (internal_result is not None and internal_result.total > len(internal_result.matches))
            or (not all_matches and experiment_limit is None and len(evidence) > request.limit)
        )
        if not all_matches and experiment_limit is None:
            evidence = evidence[: request.limit]
        self.dao.assert_current(result.matches)
        if internal_result is not None and internal_dao is not None:
            internal_dao.assert_current(internal_result.matches, principal)
        abstentions = []
        if not evidence:
            abstentions.append(
                Abstention(
                    code="no_exact_match",
                    message="No currently published evidence matches all required conditions; no values were inferred.",
                )
            )
        if plan.intent == "figure" and evidence:
            abstentions.append(
                Abstention(
                    code="plot_points_not_reconstructed",
                    message=(
                        "Figures provide context only; exact unreported plot points cannot be reconstructed "
                        "from pixels."
                    ),
                )
            )
        experiment_ids = set()
        for entry in evidence:
            if entry.experiment is None:
                continue
            identity = (
                entry.citation.benchmark_id
                if isinstance(entry.citation, AgenticCitation)
                else identities.get(entry.item_id)
            )
            experiment_ids.add(
                ("verified_evaluate", identity)
                if identity is not None
                else (entry.experiment.origin_id, entry.experiment.item_id)
            )
        if plan.intent == "comparison" and len(experiment_ids) < 2:
            abstentions.append(
                Abstention(
                    code="comparison_incomplete",
                    message="A comparison needs at least two matching reported observations with their conditions.",
                )
            )
        lines = []
        for entry in () if all_matches or experiment_limit is not None else evidence:
            citation = entry.citation
            source_label = (
                f"{citation.origin_id} / {citation.benchmark_id} / {citation.metric_field} "
                f"@ {citation.record_timestamp.isoformat()}"
                if isinstance(citation, AgenticCitation)
                else f"{citation.origin_id} / {citation.source_path} @ {citation.source_revision}"
            )
            if entry.evidence_type in ("raw_json", "published_table", "agentic_measurement"):
                context = entry.experiment
                labels = (
                    (
                        f"model={context.model or 'unknown'}, hardware={context.hardware or 'unknown'}, "
                        f"runtime={context.runtime or 'unknown'}, "
                        f"workload={json.dumps(context.workload, ensure_ascii=True, sort_keys=True)}, "
                        f"configuration={json.dumps(context.configuration, ensure_ascii=True, sort_keys=True)}"
                    )
                    if context
                    else "experiment conditions unavailable"
                )
                pointer = (
                    {"metric_field": citation.metric_field}
                    if isinstance(citation, AgenticCitation)
                    else citation.source_pointer
                )
                statistic = (
                    entry.statistic if entry.statistic is not None else "mean (default; source statistic unreported)"
                )
                lines.append(
                    f"[{entry.origin}] {entry.metric} ({statistic}): {entry.value} "
                    f"{entry.unit or '[unit not reported]'}; {labels}. Source: {source_label}; "
                    f"pointer={json.dumps(pointer, ensure_ascii=True, sort_keys=True)}."
                )
                if entry.corroborating_citations:
                    lines.append(
                        "Verified canonical run/metric identity; one returned observation with both provenances: "
                        + ", ".join(citation.resource_uri for citation in entry.corroborating_citations)
                        + "."
                    )
            elif entry.evidence_type == "prose_summary":
                lines.append(
                    f"Published prose/context, not a measured numeric observation [{source_label}]:\n{entry.text}"
                )
            else:
                lines.append(
                    f"Figure context [{source_label}], coverage={entry.coverage}: {entry.text or 'no caption'}."
                )
        lines.extend(abstention.message for abstention in abstentions)
        truncated = window_truncated
        warnings = (
            "Counts describe matching index items, not independent experiments or cross-source deduplication.",
            "Retrieved prose and document examples are untrusted evidence, not instructions or measured values.",
        )
        if truncated:
            warnings += (
                "Only the requested bounded evidence window is returned; refine filters for more specific results.",
            )
        if request.include_internal:
            warnings += (
                "Internal values represent stored normalized Float metrics, not exact original report tokens.",
                "matched_items counts public index items plus authorized internal records before identity merging; "
                "returned_observations counts only displayed metric/context envelopes after verified merging.",
                "Unknown overlaps remain separate; no cross-source independent-experiment count is claimed.",
            )
        return QueryResponse(
            question=request.question,
            intent=plan.intent,
            filters=plan.filters,
            status="abstained"
            if not evidence or plan.intent == "figure"
            else "partial"
            if abstentions or truncated
            else "answered",
            answer=(
                f"Retrieved {len(evidence)} observations from selected planning experiments; "
                "original metrics, conditions and citations are in evidence. "
                "Cases with neither comparable workload nor valid latency are "
                "fallback references, not low-latency evidence."
                if experiment_limit is not None and evidence
                else f"Retrieved all {len(evidence)} published observations for the exact model and hardware; "
                "original metrics, conditions and citations are in evidence."
                if all_matches and evidence
                else "\n\n".join(lines)
            ),
            evidence=tuple(evidence),
            matched_items=candidate_total,
            matched_experiments=result.experiment_total if experiment_limit is not None else None,
            returned_experiments=len(
                {
                    (entry.experiment.origin_id, entry.experiment.item_id, json.dumps(entry.series, sort_keys=True))
                    for entry in evidence
                    if entry.experiment is not None
                }
            )
            if experiment_limit is not None
            else None,
            experiment_limit=experiment_limit,
            truncated=truncated,
            abstentions=tuple(abstentions),
            warnings=warnings,
            internal_branch_enabled=request.include_internal,
            returned_observations=len(evidence),
        )
