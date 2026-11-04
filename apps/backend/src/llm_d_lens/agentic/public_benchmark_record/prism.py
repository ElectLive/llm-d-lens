"""Offline Prism archive adapter and explicitly approved snapshot publication."""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path

import yaml
from pydantic import JsonValue
from yaml.nodes import ScalarNode

from llm_d_lens.agentic.public_benchmark_record.contracts import (
    ItemKind,
    PublicBenchmarkRecord,
    normalize_statistic,
    normalize_unit,
)
from llm_d_lens.agentic.public_benchmark_record.publication import PublicBenchmarkRecordPublicationService
from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    ParsedFile,
    PinnedInventory,
    PublicationError,
    PublicationResult,
    ReleaseManifest,
    SourceApproval,
    item_id,
)
from llm_d_lens.agentic.public_benchmark_record.sources import PRISM_ORIGIN, collect_snapshot

PARSER_VERSION = "prism-archive-v1"
MERGED_PATH = "historical-merged-deduplicated.json"
ARCHIVE_PATHS = ("archives/archived_drive_data.json", "archives/llm-d-benchmarks.json")
_STAGE = re.compile(r"benchmark_report_v0\.2,_stage_(\d+)_lifecycle_metrics\.json\.yaml$")
_ALIASES = {
    "throughput": ("output_tput",),
    "latency": ("e2e_latency",),
    "ttft": ("ttft_ms",),
    "tpot": ("time_per_output_token", "tpot_ms"),
    "itl": ("itl_ms",),
}
_ARCHIVE_UNITS = {
    "throughput": "output tokens/s",
    "input_tput": "input tokens/s",
    "request_rate": "requests/s",
    "latency": "ms",
    "ttft": "ms",
    "tpot": "ms/token",
    "itl": "ms/token",
    "error_count": "count",
    "requests_total": "count",
}
_RAW_METRICS = {
    "latency": {
        "request_latency": "latency",
        "time_to_first_token": "ttft",
        "time_per_output_token": "tpot",
        "inter_token_latency": "itl",
        "normalized_time_per_output_token": "normalized_tpot",
    },
    "throughput": {
        "output_token_rate": "throughput",
        "total_token_rate": "total_throughput",
        "request_rate": "request_rate",
    },
    "requests": {
        "failures": "error_count",
        "total": "requests_total",
        "input_length": "input_length",
        "output_length": "output_length",
    },
}


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PublicationError("expected a JSON object")
    result: dict[str, object] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise PublicationError("JSON object keys must be strings")
        result[key] = child
    return result


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise PublicationError("expected a JSON array")
    return list(value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicationError("expected a nonempty source identity")
    return value


def _number(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise PublicationError("measurement must be a JSON number, not a string or boolean")
    result = Decimal(value)
    if not result.is_finite():
        raise PublicationError("measurement must be finite")
    return result


def _json(value: object) -> JsonValue:
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, list):
        return [_json(child) for child in value]
    return {key: _json(child) for key, child in _object(value).items()}


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError("duplicate JSON object key")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise PublicationError(f"nonfinite JSON constant: {value}")


def _load(data: bytes) -> object:
    try:
        return json.loads(data, parse_float=Decimal, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError("invalid Prism JSON") from error


def _pointer(path: str, *parts: str | int) -> str:
    suffix = "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)
    return f"{path}/{suffix}" if suffix else path


def _statistics(values: dict[str, object]) -> list[tuple[str, Decimal, tuple[str, ...]]]:
    groups: dict[str, list[str]] = {}
    numbers: dict[str, Decimal] = {}
    for name, value in values.items():
        if name == "units" or value is None:
            continue
        if name and name not in ("min", "max", "mean", "avg", "median") and not re.fullmatch(r"p\d+(?:[p.]\d+)?", name):
            raise PublicationError("unsupported metric statistic requires compatibility review")
        statistic = normalize_statistic(name) or ""
        number = _number(value)
        if statistic in numbers and numbers[statistic] != number:
            raise PublicationError("conflicting statistic aliases")
        groups.setdefault(statistic, []).append(name)
        numbers[statistic] = number
    return [
        (statistic, numbers[statistic], tuple(sorted(names, key=lambda name: (name != statistic, name))))
        for statistic, names in groups.items()
    ]


class _ReportLoader(yaml.SafeLoader):
    pass


def _yaml_decimal(loader: _ReportLoader, node: ScalarNode) -> Decimal:
    try:
        value = Decimal(loader.construct_scalar(node).replace("_", ""))
    except InvalidOperation as error:
        raise PublicationError("invalid original YAML report number") from error
    if not value.is_finite():
        raise PublicationError("nonfinite original YAML report number")
    return value


_ReportLoader.add_constructor("tag:yaml.org,2002:float", _yaml_decimal)


def _yaml_report(data: bytes) -> dict[str, object]:
    loader = _ReportLoader(data)
    try:
        return _object(loader.get_single_data())
    finally:
        loader.dispose()


class PrismSnapshotImporter:
    """Internal administration only; never selects the application's database."""

    def __init__(self, service: PublicBenchmarkRecordPublicationService) -> None:
        self.service = service

    def import_snapshot(
        self,
        root: Path,
        revision: str,
        expected_files: dict[str, str],
        approval: SourceApproval,
        *,
        expected_experiments: int,
    ) -> PublicationResult:
        manifest = self.stage(root, revision, expected_files, approval, expected_experiments=expected_experiments)
        return self.service.activate(PRISM_ORIGIN, manifest.item_id)

    def stage(
        self,
        root: Path,
        revision: str,
        expected_files: dict[str, str],
        approval: SourceApproval,
        *,
        expected_experiments: int,
        allow_empty: bool = False,
    ) -> PublicBenchmarkRecord:
        if approval.parser_version != PARSER_VERSION:
            raise PublicationError("Prism parser version requires compatibility review")
        if type(expected_experiments) is not int or expected_experiments < 0:
            raise PublicationError("expected experiment count must be a nonnegative integer")
        inventory = collect_snapshot(
            root, revision, expected_files, approval, self.service.archive, allow_empty=allow_empty
        )
        parser = _SnapshotParser(inventory, self.service)
        parsed = parser.parse()
        counts: dict[ItemKind, int] = dict(Counter(item.kind for file in parsed for item in file.items))
        if counts.get("experiment", 0) != expected_experiments:
            raise PublicationError("Prism experiment count differs from reviewed snapshot")
        counts["source"] = len(inventory.files)
        self.service.register_source(PRISM_ORIGIN, approval)
        head = self.service.dao.head(PRISM_ORIGIN)
        previous = ReleaseManifest.model_validate(head.manifest.source_details) if head.manifest else None
        previous_files = {file.file.path: file.file for file in previous.files} if previous else {}
        changed_paths = {path for path, file in parser.files.items() if previous_files.get(path) != file}
        changed = []
        for file in parsed:
            # Unchanged payload bytes may still acquire new metadata or ZIP provenance.
            dependencies = {file.path} | {
                value
                for item in file.items
                for key, value in (item.source_pointer or {}).items()
                if key in ("merged_path", "original_zip_path") and isinstance(value, str)
            }
            if dependencies & changed_paths:
                changed.append(file)
        return self.service.stage(inventory, tuple(changed), expected_counts=counts)


class _SnapshotParser:
    def __init__(self, inventory: PinnedInventory, service: PublicBenchmarkRecordPublicationService) -> None:
        self.inventory = inventory
        self.files = {file.path: file for file in inventory.files}
        self.data = {file.path: service.archive.read(file.sha256) for file in inventory.files}
        self.items: dict[str, list[PublicBenchmarkRecord]] = {file.path: [] for file in inventory.files}
        self.rows: dict[str, tuple[int, dict[str, object]]] = {}
        self.seen: set[str] = set()

    def _read(self, path: str) -> object:
        if path not in self.data:
            raise PublicationError(f"required Prism snapshot input is missing: {path}")
        return _load(self.data[path])

    def _common(self, path: str, key: str) -> dict[str, object]:
        file = self.files[path]
        return {
            "item_id": item_id(PRISM_ORIGIN, key, self.inventory.revision),
            "logical_key": key,
            "origin_id": PRISM_ORIGIN,
            "source_revision": self.inventory.revision,
            "content_sha256": file.sha256,
            "access_classification": "verified_public",
            "source_path": path,
            "format": file.format,
        }

    def _provenance(self, record_id: str, pointer: str) -> dict[str, JsonValue]:
        index, _ = self.rows[record_id]
        return {
            "record_id": record_id,
            "snapshot_date": self.inventory.revision,
            "json_pointer": pointer,
            "merged_path": MERGED_PATH,
            "merged_sha256": self.files[MERGED_PATH].sha256,
            "merged_json_pointer": f"/{index}",
        }

    def _record(self, record_id: str) -> dict[str, object]:
        if record_id in self.seen or record_id not in self.rows:
            raise PublicationError("duplicate or unmerged Prism experiment identity")
        self.seen.add(record_id)
        return self.rows[record_id][1]

    def _experiment(
        self,
        path: str,
        record_id: str,
        row: dict[str, object],
        provenance: dict[str, JsonValue],
        *,
        submitted_at: str | None = None,
    ) -> PublicBenchmarkRecord:
        metadata = _object(row.get("metadata", {}))
        timestamp = None
        if submitted_at is not None:
            try:
                timestamp = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise PublicationError("invalid submission timestamp") from error
            if timestamp.utcoffset() is None:
                raise PublicationError("submission timestamp must have a timezone")
        key = f"{path}#experiment:{record_id}"
        configuration = {
            name: _json(value) for name, value in row.items() if name not in ("metrics", "workload", "_diagnostics")
        }
        item = PublicBenchmarkRecord.model_validate(
            self._common(path, key)
            | {
                "kind": "experiment",
                "parent_id": item_id(PRISM_ORIGIN, f"file:{path}", self.inventory.revision),
                "model": row.get("model_name", metadata.get("model_name")),
                "hardware": row.get("hardware", metadata.get("hardware")),
                "runtime": metadata.get("backend"),
                "workload": _json(_object(row.get("workload", {}))),
                "configuration": configuration,
                "timestamp": timestamp,
                "timestamp_kind": "submission" if timestamp else None,
                "source_pointer": provenance | {"timestamp_original": _json(row.get("timestamp"))},
            }
        )
        self.items[path].append(item)
        return item

    def _measurement(
        self,
        experiment: PublicBenchmarkRecord,
        metric: str,
        value: object,
        unit: str | None,
        statistic: str | None,
        pointer: str,
        *,
        extra: dict[str, JsonValue] | None = None,
    ) -> None:
        numeric = _number(value)
        normalized = normalize_unit(numeric, unit) if unit is not None else None
        key = f"{experiment.logical_key}#metric:{metric}:{statistic or 'unreported'}"
        provenance = dict(experiment.source_pointer or {})
        provenance["json_pointer"] = pointer
        provenance.update(extra or {})
        self.items[experiment.source_path].append(
            PublicBenchmarkRecord.model_validate(
                self._common(experiment.source_path, key)
                | {
                    "kind": "measurement",
                    "parent_id": experiment.item_id,
                    "experiment_id": experiment.item_id,
                    "metric": metric,
                    "value": numeric,
                    "original_value": numeric,
                    "unit": unit,
                    "original_unit": unit,
                    "statistic": normalize_statistic(statistic),
                    "normalized_value": normalized[0] if normalized else None,
                    "normalized_unit": normalized[1] if normalized else None,
                    "normalization_rule": normalized[2] if normalized else None,
                    "evidence_type": "raw_json",
                    "source_pointer": provenance,
                }
            )
        )

    def _archive(self, path: str) -> None:
        for index, value in enumerate(_array(self._read(path))):
            row = _object(value)
            record_id = _text(row.get("id"))
            if self._record(record_id) != row:
                raise PublicationError("merged historical record differs from its original archive")
            provenance = self._provenance(record_id, f"/{index}")
            provenance["original_source"] = _json(row.get("source_info"))
            experiment = self._experiment(path, record_id, row, provenance)
            metrics = _object(row.get("metrics"))
            consumed: set[str] = set()
            for metric, unit in _ARCHIVE_UNITS.items():
                names = (metric, *_ALIASES.get(metric, ()))
                present = [(name, metrics[name]) for name in names if metrics.get(name) is not None]
                if not present:
                    continue
                name, metric_value = present[0]
                values = _object(metric_value) if isinstance(metric_value, dict) else {"": metric_value}
                statistics = _statistics(values)
                canonical = next((number for statistic, number, _ in statistics if statistic in ("", "mean")), None)
                for _, alias_value in present[1:]:
                    if _number(alias_value) != _number(canonical):
                        raise PublicationError(f"conflicting archive aliases for {metric}")
                consumed.update(name for name, _ in present)
                for statistic, number, labels in statistics:
                    original = labels[0]
                    pointer = _pointer(f"/{index}/metrics", name, *([original] if original else []))
                    self._measurement(
                        experiment,
                        metric,
                        number,
                        unit,
                        statistic or None,
                        pointer,
                        extra={
                            "alias_json_pointers": [_pointer(f"/{index}/metrics", alias) for alias, _ in present[1:]],
                            "unit_basis": "Prism normalized archive convention",
                            "statistic_original": original or None,
                            "statistic_alias_json_pointers": [
                                _pointer(f"/{index}/metrics", name, alias) for alias in labels[1:]
                            ],
                        },
                    )
            if {name for name, value in metrics.items() if value is not None} - consumed:
                raise PublicationError("unsupported archive metric requires parser compatibility review")

    def _indexes(self) -> set[str]:
        runs: set[str] = set()
        for state in ("public", "promoted"):
            index = _object(self._read(f"results/{state}-index.json"))
            if "nextPageToken" not in index or index["nextPageToken"] is not None:
                raise PublicationError("Results Store pagination is incomplete")
            for value in _array(index.get("items")):
                record = _object(value)
                run_id = _text(record.get("runId"))
                if record.get("state") != state or run_id in runs:
                    raise PublicationError("Results Store index has an invalid state or duplicate run")
                runs.add(run_id)
        expected = {f"results/json/{run_id}.json" for run_id in runs}
        if {path for path in self.files if path.startswith("results/json/")} != expected:
            raise PublicationError("Results Store payload inventory differs from complete indexes")
        if {path for path in self.files if path.startswith("results/zip/")} != {
            f"results/zip/{run_id}.zip" for run_id in runs
        }:
            raise PublicationError("Results Store ZIP inventory differs from complete indexes")
        return runs

    def _bundle(self, run_id: str) -> None:
        path = f"results/json/{run_id}.json"
        bundle = _object(self._read(path))
        if bundle.get("runId") != run_id:
            raise PublicationError("Results Store run identity differs from its pinned path")
        entries = [_object(value) for value in _array(bundle.get("entries"))]
        zip_path = f"results/zip/{run_id}.zip"
        members: dict[int, str] = {}
        try:
            with zipfile.ZipFile(BytesIO(self.data[zip_path])) as archive:
                if archive.testzip() is not None:
                    raise PublicationError("corrupt original Results Store ZIP")
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise PublicationError("duplicate original ZIP member names")
                exported = {name: _yaml_report(archive.read(name)) for name in names if _STAGE.search(name)}
                for index, entry in enumerate(entries):
                    matches = [name for name, report in exported.items() if report == entry.get("raw_report")]
                    if len(matches) != 1:
                        raise PublicationError("original report does not uniquely match retained ZIP content")
                    members[index] = matches[0]
                if len(set(members.values())) != len(entries) or set(members.values()) != set(exported):
                    raise PublicationError("original ZIP reports differ from Results Store entries")
        except (zipfile.BadZipFile, yaml.YAMLError) as error:
            raise PublicationError("invalid original Results Store ZIP") from error
        filenames: set[str] = set()
        for index, entry in enumerate(entries):
            filename = _text(entry.get("filename"))
            if filename in filenames:
                raise PublicationError("duplicate Results Store report identity")
            filenames.add(filename)
            report = _object(entry.get("raw_report"))
            if report.get("version") != "0.2":
                raise PublicationError("unsupported Results Store report version")
            match = _STAGE.search(filename)
            pointer = f"/entries/{index}/raw_report"
            if match is None:
                if not filename.endswith("/report_v0.2.json"):
                    raise PublicationError("unsupported Results Store report type")
                key = f"{path}#summary:{index}"
                self.items[path].append(
                    PublicBenchmarkRecord.model_validate(
                        self._common(path, key)
                        | {
                            "kind": "passage",
                            "parent_id": item_id(PRISM_ORIGIN, f"file:{path}", self.inventory.revision),
                            "passage_text": f"Prism run {run_id}: {filename}. Run-level summary, context only; "
                            "not an additional experiment or independent measurement.",
                            "heading_path": ["Prism", run_id, "Run-level summary"],
                            "source_pointer": {
                                "snapshot_date": self.inventory.revision,
                                "json_pointer": pointer,
                                "original_report": filename,
                                "original_zip_path": zip_path,
                                "original_zip_sha256": self.files[zip_path].sha256,
                                "original_zip_member": members[index],
                                "context_only": True,
                            },
                        }
                    )
                )
                continue
            load = _object(_object(_object(report.get("scenario")).get("load")).get("standardized"))
            if type(load.get("stage")) is not int or load["stage"] != int(match.group(1)):
                raise PublicationError("stage number differs from its original report identity")
            record_id = f"prism:{run_id}:stage:{load['stage']}"
            row = self._record(record_id)
            info = _object(row.get("source_info"))
            if (
                row.get("run_id") != run_id
                or row.get("source") != "prism-results-store"
                or info.get("file_identifier") != filename
                or info.get("stage_run_id") != entry.get("run_id")
                or info.get("timestamp_kind") != "submission"
                or row.get("timestamp") != bundle.get("submitted_at")
                or row.get("model_name") != bundle.get("model_name")
            ):
                raise PublicationError("normalized stage identity differs from its original report")
            workload = {
                "input_tokens": _object(load.get("input_seq_len")).get("value"),
                "output_tokens": _object(load.get("output_seq_len")).get("value"),
                "target_qps": load.get("rate_qps"),
                "concurrency": load.get("concurrency"),
            }
            if row.get("workload") != workload:
                raise PublicationError("normalized workload differs from its original stage")
            provenance = self._provenance(record_id, pointer)
            provenance.update(
                {
                    "original_report": filename,
                    "original_zip_path": zip_path,
                    "original_zip_sha256": self.files[zip_path].sha256,
                    "original_zip_member": members[index],
                    "stage_run_id": _json(entry.get("run_id")),
                }
            )
            experiment = self._experiment(
                path, record_id, row, provenance, submitted_at=_text(bundle.get("submitted_at"))
            )
            aggregate = _object(_object(_object(report.get("results")).get("request_performance")).get("aggregate"))
            if set(aggregate) - set(_RAW_METRICS):
                raise PublicationError("unsupported performance group requires parser compatibility review")
            for group, names in _RAW_METRICS.items():
                values = _object(aggregate.get(group, {}))
                if set(values) - set(names):
                    raise PublicationError("unsupported raw metric requires parser compatibility review")
                for name, value in values.items():
                    if value is None:
                        continue
                    metric_pointer = _pointer(pointer, "results", "request_performance", "aggregate", group, name)
                    distribution = _object(value) if isinstance(value, dict) else {"": value}
                    unit = _text(distribution["units"]) if distribution.get("units") is not None else None
                    if group == "requests" and not isinstance(value, dict):
                        unit = "count"
                    for statistic, number, labels in _statistics(distribution):
                        original = labels[0]
                        self._measurement(
                            experiment,
                            names[name],
                            number,
                            unit,
                            statistic or None,
                            _pointer(metric_pointer, original) if original else metric_pointer,
                            extra={
                                "statistic_original": original or None,
                                "statistic_alias_json_pointers": [
                                    _pointer(metric_pointer, alias) for alias in labels[1:]
                                ],
                                "unit_json_pointer": _pointer(metric_pointer, "units")
                                if distribution.get("units") is not None
                                else None,
                            },
                        )

    def parse(self) -> tuple[ParsedFile, ...]:
        if not self.inventory.files:
            return ()
        merged = _array(self._read(MERGED_PATH))
        for index, value in enumerate(merged):
            row = _object(value)
            record_id = _text(row.get("id"))
            if record_id in self.rows:
                raise PublicationError("merged archive IDs must be unique")
            self.rows[record_id] = (index, row)
        for path in ARCHIVE_PATHS:
            self._archive(path)
        for run_id in sorted(self._indexes()):
            self._bundle(run_id)
        if self.seen != self.rows.keys():
            raise PublicationError("merged archive includes unverified or missing source records")
        if "normalize-and-merge.jq" not in self.data:
            raise PublicationError("snapshot normalization rule is missing")
        return tuple(ParsedFile(path=path, items=tuple(items)) for path, items in sorted(self.items.items()))
