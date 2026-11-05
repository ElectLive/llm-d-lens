"""A plan: what a stack renders to in one namespace, and every finding about it.

A plan is the same object offline (no cluster) and against a cluster; cluster checks add
findings to it (``preflight``). Its digest covers what would be applied, so ``standup
--expected-digest`` refuses to apply anything other than the plan that was reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import yamlio
from .catalog import Stack
from .model import Finding, object_ref, validate_instance_name, validate_namespace
from .policy import PolicyContext, PolicyResult, ScopeResolver, apply_policy, fallback_scope
from .process import Runner
from .render import Tools, render


@dataclass
class Options:
    """What a user may change about a plan; everything else comes from the stack's files."""

    namespace: str
    instance: str
    allow_privileged: bool = False
    skip_admin_parts: bool = False
    #: Let the instance move to this stack path from the one it runs (the objects stay where they are).
    switch_stack: bool = False
    #: A guide's variables (``--var NAME=VALUE``): they pick its variant (accelerator, model server,
    #: provider) as the README's ``export`` lines do.
    variables: dict[str, str] = field(default_factory=dict)
    #: ``--set KEY=VALUE`` for every Helm chart the guide installs (the router chart).
    sets: list[str] = field(default_factory=list)

    def choices(self) -> dict[str, Any]:
        """What the user chose about the render, kept with the instance."""
        return {
            **({"variables": dict(sorted(self.variables.items()))} if self.variables else {}),
            **({"sets": list(self.sets)} if self.sets else {}),
        }


@dataclass
class Plan:
    stack: Stack
    options: Options
    command: str
    policy: PolicyResult
    #: Results of the checks against the cluster, when there was one.
    cluster: dict[str, Any] = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return self.options.namespace

    @property
    def instance(self) -> str:
        return self.options.instance

    @property
    def documents(self) -> list[dict[str, Any]]:
        return self.policy.documents

    @property
    def findings(self) -> list[Finding]:
        return self.policy.findings

    @property
    def blocking(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    #: For a guide: how each of its steps was followed, and the variables they saw.
    steps: list[dict[str, str]] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        return yamlio.digest({"namespace": self.namespace, "instance": self.instance, "documents": self.documents})

    def stack_record(self) -> dict[str, Any]:
        """The stack as the instance's record keeps it: its path, source, and the user's choices."""
        return {"id": self.stack.id, "source": self.stack.source.describe(), **self.options.choices()}

    def objects(self) -> list[dict[str, str]]:
        return [
            {"apiVersion": doc["apiVersion"], "kind": doc["kind"], "name": doc["metadata"]["name"]}
            for doc in self.documents
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "stack": self.stack.id,
            "source": self.stack.source.describe(),
            "namespace": self.namespace,
            "instance": self.instance,
            "render": self.command,
            **({"steps": self.steps, "variables": self.variables} if self.stack.kind == "guide" else {}),
            "choices": self.options.choices(),
            "objects": [object_ref(doc) for doc in self.documents],
            "workloads": self.policy.workloads,
            "accelerators": self.policy.accelerators,
            "prerequisites": self.policy.prerequisites,
            "adminParts": [
                {"object": object_ref(part["document"]), "reason": part["reason"]} for part in self.policy.admin_parts
            ],
            "findings": [finding.as_dict() for finding in self.findings],
            "cluster": self.cluster,
            "digest": self.digest,
            "blocking": len(self.blocking),
        }


def make_plan(
    stack: Stack,
    options: Options,
    *,
    runner: Runner,
    tools: Tools | None = None,
    scope: ScopeResolver = fallback_scope,
    label: bool = True,
) -> Plan:
    """Render the stack and apply the namespace policy. Never contacts the cluster."""
    validate_namespace(options.namespace)
    validate_instance_name(options.instance)
    rendered = render(
        stack,
        runner=runner,
        tools=tools,
        namespace=options.namespace,
        variables=options.variables,
        sets=options.sets,
    )
    context = PolicyContext(
        namespace=options.namespace,
        instance=options.instance,
        allow_privileged=options.allow_privileged,
        skip_admin_parts=options.skip_admin_parts,
        label=label,
        scope=scope,
    )
    policy = apply_policy(rendered.documents, context)
    policy.findings[:0] = rendered.findings
    return Plan(
        stack=stack,
        options=options,
        command=rendered.command,
        policy=policy,
        steps=rendered.steps,
        variables=rendered.variables,
    )
