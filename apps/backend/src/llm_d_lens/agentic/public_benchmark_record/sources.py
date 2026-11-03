"""Pinned Prism evidence collection from approved, reviewed offline snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from llm_d_lens.agentic.public_benchmark_record.publication_contracts import (
    InputFormat,
    InventoryFile,
    PinnedInventory,
    PublicationError,
    SourceApproval,
)
from llm_d_lens.infrastructure.artifacts.store import artifact_uri, register_artifacts, resolve_artifact
from llm_d_lens.infrastructure.storage.paths import storage_path

PRISM_ORIGIN = "prism:public-snapshot"
_PATHS = (
    "archives/archived_drive_data.json",
    "archives/llm-d-benchmarks.json",
    "results/promoted-index.json",
    "results/public-index.json",
    "results/json/*.json",
    "results/zip/*.zip",
    "normalize-and-merge.jq",
    "historical-merged-deduplicated.json",
)
_FORMATS: dict[str, InputFormat] = {
    ".json": "json",
    ".zip": "zip",
    ".jq": "jq",
}


def allowed_paths(origin_id: str) -> tuple[str, ...]:
    if origin_id != PRISM_ORIGIN:
        raise PublicationError("source is not in the v1 allowlist")
    return _PATHS


def validate_approval(origin_id: str, approval: SourceApproval) -> None:
    permitted = allowed_paths(origin_id)
    for pattern in approval.allowed_paths:
        if pattern in permitted:
            continue
        raise PublicationError("approval cannot expand the v1 source/path allowlist")
    if len(set(approval.allowed_paths)) != len(approval.allowed_paths):
        raise PublicationError("approved paths must be unique")


def validate_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or path != parsed.as_posix()
        or ".." in parsed.parts
        or "\\" in path
        or "\x00" in path
        or "\ufffd" in path
        or not parsed.parts
    ):
        raise PublicationError("source path must be canonical, relative, and without traversal")


def validate_revision(origin_id: str, revision: str) -> None:
    allowed_paths(origin_id)
    try:
        valid = date.fromisoformat(revision).isoformat() == revision
    except ValueError:
        valid = False
    if not valid:
        raise PublicationError("Prism source revision must be a pinned ISO snapshot date")


def _matches(path: str, pattern: str) -> bool:
    expression = re.escape(pattern).replace(r"\*\*/", "(?:[^/]+/)*").replace(r"\*", "[^/]*")
    return re.fullmatch(expression, path) is not None


def input_format(origin_id: str, path: str, approval: SourceApproval | None = None) -> InputFormat | None:
    validate_path(path)
    patterns = allowed_paths(origin_id)
    if approval is not None:
        validate_approval(origin_id, approval)
        patterns = approval.allowed_paths
    if any(_matches(path, pattern) for pattern in patterns):
        return _FORMATS[PurePosixPath(path).suffix]
    return None


class SourceArchive:
    """Content-addressed evidence files using the shared Lens artifact layout."""

    @staticmethod
    def _root(digest: str) -> Path:
        if not re.fullmatch("[0-9a-f]{64}", digest):
            raise PublicationError("invalid source digest")
        return storage_path("data", "artifacts", "benchmark-rag", digest)

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        root = self._root(digest)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = root / "source"
        if target.is_symlink():
            raise PublicationError("source archive symlinks are not supported")
        with NamedTemporaryFile(dir=root, prefix=".source-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise PublicationError("immutable source archive content conflicts") from None
            finally:
                temporary.unlink(missing_ok=True)
        register_artifacts(
            root,
            owner_type="benchmark-rag",
            owner_id=digest,
            retention_class="evidence",
            source_version={"sha256": digest},
            files={"source": {"kind": "source"}},
        )
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.read(digest)
        return digest

    def read(self, digest: str) -> bytes:
        root = self._root(digest)
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise PublicationError("pinned source manifest is not valid UTF-8 JSON") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("owner_type") != "benchmark-rag"
            or manifest.get("owner_id") != digest
            or not isinstance(manifest.get("files"), list)
            or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("uri"), str)
                or not isinstance(entry.get("path"), str)
                for entry in manifest["files"]
            )
        ):
            raise PublicationError("pinned source manifest has invalid ownership or file entries")
        uri = artifact_uri("benchmark-rag", digest, "source")
        try:
            target = resolve_artifact(root, manifest, uri)
        except ValueError as error:
            raise PublicationError("pinned source manifest cannot resolve its source") from error
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise PublicationError("pinned source bytes failed hash verification")
        return data


def collect_snapshot(
    root: Path,
    revision: str,
    expected_files: dict[str, str],
    approval: SourceApproval,
    archive: SourceArchive,
    *,
    origin_id: str = PRISM_ORIGIN,
    allow_empty: bool = False,
) -> PinnedInventory:
    validate_approval(origin_id, approval)
    validate_revision(origin_id, revision)
    if (
        (not expected_files and not allow_empty)
        or not root.is_dir()
        or any(path.is_symlink() for path in (root, *root.parents))
    ):
        raise PublicationError("snapshot needs a reviewed hash inventory and a non-symlink root")
    actual = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PublicationError("snapshot symlinks are not supported")
        if path.is_file() and input_format(origin_id, relative, approval) is not None:
            actual[relative] = path
    if set(actual) != set(expected_files):
        raise PublicationError("snapshot inventory is incomplete or contains unreviewed inputs")
    files = []
    for relative, digest in sorted(expected_files.items()):
        kind = input_format(origin_id, relative, approval)
        if kind is None:
            raise PublicationError("snapshot contains an unapproved file")
        if hashlib.sha256(actual[relative].read_bytes()).hexdigest() != digest:
            raise PublicationError("snapshot hash does not match the reviewed inventory")
        if archive.put(actual[relative].read_bytes()) != digest:
            raise PublicationError("snapshot changed during collection")
        files.append(InventoryFile(path=relative, sha256=digest, format=kind))
    return PinnedInventory(origin_id=origin_id, revision=revision, complete=True, files=tuple(files))
