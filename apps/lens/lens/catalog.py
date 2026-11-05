"""The stacks a source holds, read from the checkout; no stack or model is compiled in.

A stack is what a team deploys into one namespace:

* an llm-d guide: a directory with a ``guide.yaml`` (``guides/optimized-baseline``,
  ``guides/models/<model>``). Its ``deploy`` steps say what it applies: the router chart
  (``helm install``) and the model server overlays (``kubectl apply -k``), picked by the
  guide's variables (``--var ACCELERATOR_TYPE=...``);
* a kustomize directory (``kustomization.yaml``), such as one model server overlay of a guide
  (``guides/<guide>/modelserver/gpu/vllm/base``), applied as ``kubectl apply -k`` would;
* any other directory or YAML file named explicitly, whose manifests are read as they are.

``llmd stacks`` lists the guides; a kustomize directory or a manifest file is named directly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import yamlio
from .errors import NotFoundError, UsageError
from .process import Runner
from .sources import Source

KUSTOMIZATION_FILES = ("kustomization.yaml", "kustomization.yml", "Kustomization")
GUIDE_FILE = "guide.yaml"
#: Instance names become object names (``lens-instance-<name>``), so they stay short.
MAX_NAME = 40


@dataclass(frozen=True)
class Stack:
    id: str  # the path in the source, e.g. guides/optimized-baseline
    kind: str  # guide | kustomize | manifests
    source: Source

    @property
    def path(self) -> Path:
        return self.source.root / self.id

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "source": self.source.describe()}


def kustomization_file(directory: Path) -> Path | None:
    for name in KUSTOMIZATION_FILES:
        if (directory / name).is_file():
            return directory / name
    return None


def inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; refuse absolute paths and traversal outside the source."""
    if not relative or Path(relative).is_absolute():
        raise UsageError(f"expected a path relative to the source: {relative!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise UsageError(f"path escapes the source checkout: {relative}")
    return resolved


def local_includes(directory: Path) -> list[Path]:
    """The local directories a kustomize unit includes as ``resources`` or ``components``."""
    kustomization = kustomization_file(directory)
    if kustomization is None:
        return []
    try:
        data = yamlio.load_file(kustomization) or {}
    except (OSError, UsageError):
        return []
    if not isinstance(data, dict):
        return []
    entries = [*(data.get("resources") or []), *(data.get("components") or [])]
    return [
        (directory / entry).resolve()
        for entry in entries
        if isinstance(entry, str) and "://" not in entry and (directory / entry).is_dir()
    ]


def uses_helm(directory: Path, seen: set[Path] | None = None) -> bool:
    """Whether a kustomize unit renders a Helm chart (``helmCharts``) itself or in a unit it includes;
    ``kustomize build`` then needs ``--enable-helm``."""
    seen = set() if seen is None else seen
    directory = directory.resolve()
    kustomization = kustomization_file(directory)
    if directory in seen or kustomization is None:
        return False
    seen.add(directory)
    try:
        data = yamlio.load_file(kustomization) or {}
    except (OSError, UsageError):
        return False
    if isinstance(data, dict) and data.get("helmCharts"):
        return True
    return any(uses_helm(include, seen) for include in local_includes(directory))


def manifest_documents(path: Path) -> list[dict[str, Any]] | None:
    """The Kubernetes objects in a YAML file, or None when it is not a manifest file (a kustomization,
    a Helm values file, a guide, anything without apiVersion and kind)."""
    if path.name in (*KUSTOMIZATION_FILES, GUIDE_FILE) or path.name.endswith((".values.yaml", ".values.yml")):
        return None
    if path.name in ("values.yaml", "values.yml", "Chart.yaml"):
        return None
    try:
        documents = yamlio.load_documents(path.read_text(encoding="utf-8"), str(path))
    except (UsageError, UnicodeDecodeError, OSError):
        return None
    objects = [doc for doc in documents if doc.get("apiVersion") and doc.get("kind")]
    if not objects or {doc["kind"] for doc in objects} & {"Kustomization", "Component"}:
        return None
    return objects


def manifest_files(directory: Path) -> list[Path]:
    """Manifest files at or below ``directory``, leaving out nested kustomize units and Helm charts."""
    if directory.is_file():
        return [directory] if manifest_documents(directory) is not None else []
    found = []
    for path in sorted(directory.rglob("*")):
        if path.suffix not in (".yaml", ".yml") or not path.is_file():
            continue
        parents = [parent for parent in path.parents if parent.is_relative_to(directory)]
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if any(parent != directory and kustomization_file(parent) for parent in parents):
            continue
        if any((parent / "Chart.yaml").is_file() for parent in parents):
            continue
        if manifest_documents(path) is not None:
            found.append(path)
    return found


def _tracked_files(source: Source, runner: Runner) -> list[str] | None:
    """The checkout's files (tracked and untracked, not ignored) when it is a Git work tree."""
    if not source.revision:
        return None
    listed = runner(
        ["git", "-C", str(source.root), "ls-files", "--cached", "--others", "--exclude-standard"], timeout=60
    )
    return listed.stdout.splitlines() if listed.ok else None


def _guides(source: Source, runner: Runner) -> list[str]:
    files = _tracked_files(source, runner)
    if files is None:
        files = [path.relative_to(source.root).as_posix() for path in source.root.rglob(GUIDE_FILE)]
    found = set()
    for line in files:
        path = Path(line)
        hidden = any(part.startswith(".") for part in path.parts[:-1])
        if path.name == GUIDE_FILE and not hidden and path.parent.as_posix() != "." and (source.root / path).is_file():
            found.add(path.parent.as_posix())
    return sorted(found)


def list_stacks(source: Source, runner: Runner, prefix: str = "") -> list[Stack]:
    """Every guide of the source (a directory with a ``guide.yaml``), below ``prefix`` when given."""
    prefix = prefix.strip("/")
    return [
        Stack(id=guide, kind="guide", source=source)
        for guide in _guides(source, runner)
        if not prefix or guide == prefix or guide.startswith(prefix + "/")
    ]


def find_stack(source: Source, reference: str) -> Stack:
    """A stack by its path in the source: a guide, a kustomize unit, a directory of manifests, or one
    file. A guide's name alone (``optimized-baseline``) finds ``guides/<name>``."""
    reference = reference.strip().strip("/")
    if not reference:
        raise UsageError("no stack selected", hint="Run `llmd stacks` to list the guides of the source.")
    path = inside(source.root, reference)
    if not path.exists() and "/" not in reference:
        for candidate in (f"guides/{reference}", f"guides/models/{reference}"):
            if (source.root / candidate / GUIDE_FILE).is_file():
                return find_stack(source, candidate)
    identifier = path.relative_to(source.root.resolve()).as_posix()
    if path.is_dir() and (path / GUIDE_FILE).is_file():
        return Stack(id=identifier, kind="guide", source=source)
    if path.is_dir() and kustomization_file(path):
        return Stack(id=identifier, kind="kustomize", source=source)
    if path.exists() and manifest_files(path):
        return Stack(id=identifier, kind="manifests", source=source)
    if path.exists():
        raise UsageError(f"{reference} holds no Kubernetes manifests and no {GUIDE_FILE}")
    raise NotFoundError(
        f"stack {reference!r} not found in source {source.name}",
        hint="Run `llmd stacks` to list the guides of the source.",
    )


def default_name(stack: Stack) -> str:
    """An instance name from the stack's path: a guide's directory name, else the last two segments."""
    segments = stack.id.split("/")
    tail = segments[-1] if stack.kind == "guide" else "-".join(segments[-2:])
    name = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-") or "stack"
    if len(name) <= MAX_NAME:
        return name
    digest = hashlib.sha256(stack.id.encode()).hexdigest()[:8]
    return f"{name[: MAX_NAME - 9].rstrip('-')}-{digest}"
