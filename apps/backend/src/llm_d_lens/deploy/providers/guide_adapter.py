"""Guide deployment contracts owned by the Deploy module."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol


def serving_arguments(model: dict[str, Any], component: dict[str, Any]) -> dict[str, int | float]:
    """Translate explicit serving controls; omitted fields retain Guide defaults."""
    result: dict[str, int | float] = {}
    for field_name, argument in (
        ("maxModelLen", "max-model-len"),
        ("maxNumSeqs", "max-num-seqs"),
        ("gpuMemoryUtilization", "gpu-memory-utilization"),
        ("blockSize", "block-size"),
        ("maxNumBatchedTokens", "max-num-batched-tokens"),
    ):
        value = component.get(field_name)
        if value is None and field_name == "maxModelLen":
            value = model.get(field_name)
        if value is None:
            continue
        if field_name == "gpuMemoryUtilization":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError(f"{field_name} must be a number greater than 0 and at most 1")
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        result[argument] = value
    return result


def set_vllm_argument(args: list[str], name: str, value: object) -> list[str]:
    """Replace all occurrences, including the two-token form used by Guides."""
    boolean = name in {"enable-auto-tool-choice", "enable-prefix-caching", "enforce-eager"} and str(value).lower() in {
        "true",
        "false",
    }
    flags = {f"--{name}", f"--no-{name}"} if boolean else {f"--{name}"}
    result = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in flags:
            index += 1 if boolean else 2
        else:
            if not any(item.startswith(f"{flag}=") for flag in flags):
                result.append(item)
            index += 1
    if boolean:
        result.append(f"--{name}" if str(value).lower() == "true" else f"--no-{name}")
    else:
        result.append(f"--{name}={value}")
    return result


def set_vllm_shell_argument(command: str, name: str, value: object) -> str:
    """Replace a vLLM argument inside a shell command string."""
    boolean = name in {"enable-auto-tool-choice", "enable-prefix-caching", "enforce-eager"} and str(value).lower() in {
        "true",
        "false",
    }
    pattern = (
        rf"--(?:no-)?{re.escape(name)}(?:=(?:true|false))?(?![\w-])"
        if boolean
        else rf"--{re.escape(name)}(?:=|\s+)(?:\"[^\"]*\"|'[^']*'|[^\s]+)"
    )
    replacement = (
        (f"--{name}" if str(value).lower() == "true" else f"--no-{name}")
        if boolean
        else f"--{name}={shlex.quote(str(value))}"
    )
    if re.search(pattern, command):
        return re.sub(pattern, lambda _: replacement, command)
    return f"{command.rstrip()} \\\n                {replacement}"


@dataclass(frozen=True)
class GuideDefinition:
    guide_id: str
    source_ref: str
    content_hash: str
    maturity: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuideDeploymentArtifact:
    guide_id: str
    artifact_hash: str
    manifest_ref: str | None = None
    values_checksum: str | None = None
    source_ref: str | None = None
    guide_content_hash: str | None = None
    manifest_checksum: str | None = None
    deployment_contract: dict[str, Any] = field(default_factory=dict)


class GuideAdapter(Protocol):
    """Registered provider lifecycle for one deployment Guide."""

    def discover(self) -> GuideDefinition: ...

    def validate_inputs(
        self,
        definition: GuideDefinition,
        cluster_snapshot: dict[str, Any],
        overrides: dict[str, Any],
    ) -> ValidationResult: ...

    async def render(self, definition: GuideDefinition, overrides: dict[str, Any]) -> GuideDeploymentArtifact: ...

    async def deploy(self, artifact: GuideDeploymentArtifact, execution_context: dict[str, Any]) -> dict[str, Any]: ...

    async def readiness(self, execution: dict[str, Any]) -> ValidationResult: ...

    async def diagnostics(self, execution: dict[str, Any]) -> dict[str, Any]: ...

    async def stop(self, execution: dict[str, Any], artifact: GuideDeploymentArtifact) -> dict[str, Any]: ...

    async def rollback(self, execution: dict[str, Any], artifact: GuideDeploymentArtifact) -> dict[str, Any]: ...

    async def cleanup(
        self, execution: dict[str, Any], artifact: GuideDeploymentArtifact, *, force: bool = False
    ) -> dict[str, Any]: ...
