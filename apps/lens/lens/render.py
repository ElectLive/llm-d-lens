"""Render a stack to Kubernetes objects on this machine. Rendering never contacts the cluster.

* An llm-d guide renders what its ``deploy`` steps apply (see :mod:`lens.guides`): the router
  chart with ``helm template`` (same chart, version, values and release name as the guide's
  ``helm install``), and each model server overlay with ``kustomize build``.
* A kustomize stack renders with ``kustomize build`` (``kubectl kustomize`` when kustomize is
  not installed), with ``--enable-helm`` when it includes a Helm chart.
* A manifest stack is its files, read as they are.

Either way the result is what the guide's own commands, or ``kubectl apply -k|-f``, would send.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import guides, yamlio
from .catalog import Stack, manifest_files, uses_helm
from .errors import CommandError
from .model import Finding
from .process import Runner, available, display


@dataclass(frozen=True)
class Tools:
    kube: str = "kubectl"
    kustomize: str = "kustomize"
    helm: str = "helm"

    def kustomize_command(self) -> list[str]:
        if self.kustomize and available(self.kustomize):
            return [self.kustomize, "build"]
        return [self.kube, "kustomize"]


@dataclass
class Rendered:
    documents: list[dict[str, Any]]
    #: The command (or the files read) that produced the documents, for the plan's evidence.
    command: str
    #: For a guide: how each of its steps was followed, and the variables they saw.
    steps: list[dict[str, str]] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def _short(message: str, limit: int = 1200) -> str:
    """Tool errors can be pages long; keep the lines that explain them."""
    if len(message) <= limit:
        return message
    errors = [line for line in message.splitlines() if line.lower().startswith("error")]
    text = "\n".join(errors) if errors else message
    return text if len(text) <= limit else "..." + text[-limit:]


def _kustomize(directory: Path, label: str, runner: Runner, tools: Tools) -> tuple[list[dict[str, Any]], str]:
    command = tools.kustomize_command()
    options = ["--enable-helm"] if uses_helm(directory) else []
    result = runner([*command, str(directory), *options], timeout=300, cwd=str(directory))
    if not result.ok:
        raise CommandError(f"kustomize render of {label} failed: {_short(result.message)}")
    return yamlio.load_documents(result.stdout, label), display([Path(command[0]).name, *command[1:], label, *options])


def _manifests(path: Path, label: str) -> tuple[list[dict[str, Any]], str]:
    files = manifest_files(path)
    documents = []
    for item in files:
        documents.extend(yamlio.load_documents(item.read_text(encoding="utf-8"), str(item)))
    return documents, f"read {len(files)} manifest file(s) under {label}"


def _helm(
    unit: guides.Unit, namespace: str, root: Path, runner: Runner, tools: Tools, sets: list[str]
) -> list[dict[str, Any]]:
    argv = [tools.helm, "template", unit.release, unit.chart, "--namespace", namespace, "--skip-tests"]
    if unit.version:
        argv += ["--version", unit.version]
    for path in unit.values:
        argv += ["--values", str(path)]
    for flag, value in unit.sets:
        argv += [flag, value]
    for value in sets:
        argv += ["--set", value]
    result = runner(argv, timeout=300, cwd=str(root))
    if not result.ok:
        raise CommandError(f"helm template of {unit.step} failed: {_short(result.message)}")
    return yamlio.load_documents(result.stdout, unit.step)


def render(
    stack: Stack,
    *,
    runner: Runner,
    tools: Tools | None = None,
    namespace: str = "default",
    variables: dict[str, str] | None = None,
    sets: list[str] | None = None,
) -> Rendered:
    """``namespace``, ``variables`` (``--var``) and ``sets`` (``--set``, for every Helm chart the guide
    installs) matter for a guide only."""
    tools = tools or Tools()
    if stack.kind == "manifests":
        return Rendered(*_manifests(stack.path, stack.id))
    if stack.kind == "kustomize":
        return Rendered(*_kustomize(stack.path, stack.id, runner, tools))
    root = stack.source.root
    guide = guides.load_guide(stack.path, stack.id)
    followed = guides.follow(guide, root, namespace=namespace, overrides=dict(variables or {}))
    documents: list[dict[str, Any]] = []
    commands = []
    if not any(finding.severity == "error" for finding in followed.findings):
        for unit in followed.units:
            if unit.kind == "helm":
                documents.extend(_helm(unit, namespace, root, runner, tools, list(sets or [])))
            elif unit.kind == "kustomize":
                label = unit.path.resolve().relative_to(root.resolve()).as_posix()
                documents.extend(_kustomize(unit.path, label, runner, tools)[0])
            else:
                label = unit.path.resolve().relative_to(root.resolve()).as_posix()
                documents.extend(_manifests(unit.path, label)[0])
            extra = "".join(f" --set {value}" for value in sets or []) if unit.kind == "helm" else ""
            commands.append(guides.describe_unit(unit, root) + extra)
    return Rendered(
        documents,
        "; ".join(commands) or f"guide {stack.id}: nothing rendered",
        steps=followed.steps,
        variables=followed.variables,
        findings=followed.findings,
    )
