"""Deploy checksum-bound manifests published by Configuration."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import shlex
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from llm_d_lens.configuration.manifest_facts import vllm_argument_value
from llm_d_lens.core.hashing import stable_hash
from llm_d_lens.deploy.providers.deployment_bundle import validate_deployment_bundle
from llm_d_lens.deploy.providers.guide_adapter import (
    GuideAdapter,
    GuideDefinition,
    GuideDeploymentArtifact,
    ValidationResult,
    set_vllm_argument,
    set_vllm_shell_argument,
)
from llm_d_lens.deploy.providers.hardware_profile import (
    accelerator_supported,
    discover_runtime_classes,
    prepare_runtime_class_manifest,
)
from llm_d_lens.deploy.providers.shared_memory import (
    _model_container,
    _pod_specs,
    prepare_shared_memory,
)
from llm_d_lens.deploy.providers.target_node import apply_placement, configured_target_node, write_pinned_manifest
from llm_d_lens.infrastructure.artifacts.manifests import (
    DEPLOYMENT_MANIFEST_DIR,
    text_checksum,
    validate_configuration_manifest_content,
)
from llm_d_lens.infrastructure.artifacts.store import register_artifacts
from llm_d_lens.monitoring.kv_trace.manifest import instrument_manifest
from llm_d_lens.storage.service import ensure_mount_in_namespace, get_ready_volume

logger = logging.getLogger(__name__)

KV_LOAD_FAILURE_POLICY = "recompute"
MODEL_SHUTDOWN_TIMEOUT_SECONDS = 90
MODEL_TERMINATION_GRACE_PERIOD_SECONDS = 120
MODEL_PRESTOP_SLEEP_SECONDS = 5
_MODEL_NAME_PREFIXES = ("modelserver", "model-server")
_VLLM_MARKERS = ("vllm serve", "vllm.entrypoints.")
_SGLANG_MARKERS = ("sglang.launch_server", "sglang serve")


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    return list(value or [])


def _container_invocation(container: dict[str, Any]) -> str:
    return " ".join(str(item) for item in [*_as_items(container.get("command")), *_as_items(container.get("args"))])


def _is_model_container(container: dict[str, Any]) -> bool:
    if _model_container(container):
        return True
    name = str(container.get("name") or "").lower()
    if name.startswith(_MODEL_NAME_PREFIXES):
        return True
    image = str(container.get("image") or "").lower()
    return "vllm" in image or "sglang" in image


def _runtime_kind(container: dict[str, Any]) -> str:
    invocation = _container_invocation(container).lower()
    if any(marker in invocation for marker in _VLLM_MARKERS):
        return "vllm"
    if any(marker in invocation for marker in _SGLANG_MARKERS):
        return "sglang"
    return "other"


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _existing_shutdown_timeout(container: dict[str, Any]) -> int | None:
    matches = re.findall(r"--shutdown-timeout(?:=|\s+)(\d+)", _container_invocation(container))
    return max(int(item) for item in matches) if matches else None


_SHUTDOWN_FLAG = "--shutdown-timeout"
_SHELLS = {"sh", "bash"}


def _shell_script_slot(container: dict[str, Any]) -> tuple[str, int] | None:
    """Locate the script of a validated `sh -c` invocation split across command and args."""
    command = container.get("command") or []
    if not command or PurePosixPath(command[0]).name not in _SHELLS:
        return None
    return ("command", 2) if len(command) >= 3 else ("args", 2 - len(command))


def _has_shutdown_flag(tokens: list[Any]) -> bool:
    return any(str(item) == _SHUTDOWN_FLAG or str(item).startswith(f"{_SHUTDOWN_FLAG}=") for item in tokens)


def _apply_shutdown_timeout(container: dict[str, Any]) -> int | None:
    """Add --shutdown-timeout only where the shared vLLM invocation parser will see it."""
    existing = _existing_shutdown_timeout(container)
    timeout = max(existing or 0, MODEL_SHUTDOWN_TIMEOUT_SECONDS)
    label = str(container.get("name") or "<unnamed>")
    try:
        current = vllm_argument_value(container, {_SHUTDOWN_FLAG}, label)
    except ValueError as error:
        logger.warning("Cannot safely apply %s to %s: %s", _SHUTDOWN_FLAG, label, error)
        return existing
    if current is not None and not current.isdigit():
        logger.warning("Keeping non-literal %s=%s for %s", _SHUTDOWN_FLAG, current, label)
        return None

    original = {key: copy.deepcopy(container[key]) for key in ("command", "args") if key in container}
    slot = _shell_script_slot(container)
    if slot is not None:
        field, index = slot
        values = list(container[field])
        values[index] = set_vllm_shell_argument(str(values[index]), "shutdown-timeout", timeout)
        container[field] = values
    else:
        command = list(container.get("command") or [])
        args = list(container.get("args") or [])
        command_has_flag = _has_shutdown_flag(command)
        if command_has_flag:
            container["command"] = set_vllm_argument(command, "shutdown-timeout", timeout)
        if not command_has_flag or _has_shutdown_flag(args):
            container["args"] = set_vllm_argument(args, "shutdown-timeout", timeout)

    try:
        applied = vllm_argument_value(container, {_SHUTDOWN_FLAG}, label)
    except ValueError:
        applied = None
    if applied == str(timeout):
        return timeout
    for key in ("command", "args"):
        container.pop(key, None)
    container.update(original)
    logger.warning("Cannot verify %s=%s for %s; leaving the invocation unchanged", _SHUTDOWN_FLAG, timeout, label)
    return existing


def _ensure_prestop(container: dict[str, Any]) -> str:
    lifecycle = container.get("lifecycle")
    if lifecycle is None:
        container["lifecycle"] = {
            "preStop": {"exec": {"command": ["/bin/sh", "-c", f"sleep {MODEL_PRESTOP_SLEEP_SECONDS}"]}}
        }
        return "added"
    if isinstance(lifecycle, dict) and "preStop" not in lifecycle:
        lifecycle["preStop"] = {"exec": {"command": ["/bin/sh", "-c", f"sleep {MODEL_PRESTOP_SLEEP_SECONDS}"]}}
        return "added"
    return "preserved"


def _update_kv_transfer_policy(raw: str) -> tuple[str, str]:
    try:
        config = json.loads(raw)
    except (TypeError, ValueError):
        return raw, "invalid"
    if not isinstance(config, dict):
        return raw, "invalid"
    if config.get("kv_load_failure_policy") == KV_LOAD_FAILURE_POLICY:
        return raw, "unchanged"
    config["kv_load_failure_policy"] = KV_LOAD_FAILURE_POLICY
    return json.dumps(config, separators=(",", ":")), "changed"


def _apply_kv_transfer_config(container: dict[str, Any], label: str) -> tuple[int, int, list[str]]:
    detected = 0
    updated = 0
    warnings: list[str] = []
    pattern = re.compile(r"(--kv-transfer-config(?:=|\s+))(?:'([^']*)'|\"([^\"]*)\"|(\S+))")

    def update(raw: str) -> str:
        nonlocal detected, updated
        detected += 1
        next_value, status = _update_kv_transfer_policy(raw)
        if status == "changed":
            updated += 1
            return next_value
        if status == "invalid":
            warnings.append(f"{label}: --kv-transfer-config is not a JSON object; left unchanged.")
        return raw

    def update_shell(command: str) -> str:
        def replace(match: re.Match[str]) -> str:
            raw = match.group(2) or match.group(3) or match.group(4) or ""
            next_value = update(raw)
            if next_value == raw:
                return match.group(0)
            return f"{match.group(1)}{shlex.quote(next_value)}"

        return pattern.sub(replace, command)

    def update_list(items: list[Any]) -> None:
        for index, item in enumerate(items):
            if not isinstance(item, str):
                continue
            if item == "--kv-transfer-config":
                if index + 1 < len(items):
                    items[index + 1] = update(str(items[index + 1]))
            elif item.startswith("--kv-transfer-config="):
                raw = item.split("=", 1)[1]
                next_value = update(raw)
                if next_value != raw:
                    items[index] = f"--kv-transfer-config={next_value}"
            elif "--kv-transfer-config" in item:
                next_value = update_shell(item)
                if next_value != item:
                    items[index] = next_value

    command = container.get("command")
    if isinstance(command, str):
        container["command"] = update_shell(command)
    elif isinstance(command, list):
        update_list(command)
    args = container.get("args")
    if isinstance(args, str):
        container["args"] = update_shell(args)
    elif isinstance(args, list):
        update_list(args)
    return detected, updated, warnings


def _ensure_termination_grace_period(pod: dict[str, Any], required_seconds: int) -> bool:
    current = _integer(pod.get("terminationGracePeriodSeconds"))
    if current is not None and current >= required_seconds:
        return False
    pod["terminationGracePeriodSeconds"] = required_seconds
    return True


def _graceful_shutdown_pod_specs(document: dict[str, Any]) -> list[dict[str, Any]]:
    pods = list(_pod_specs(document))
    if document.get("kind") != "DisaggregatedSet":
        return pods
    for role in (document.get("spec") or {}).get("roles") or []:
        template = (role.get("spec") or {}).get("leaderWorkerTemplate") or {}
        for key in ("leaderTemplate", "workerTemplate"):
            spec = (template.get(key) or {}).get("spec")
            if isinstance(spec, dict):
                pods.append(spec)
    return pods


def prepare_graceful_shutdown_manifest(manifest: str) -> tuple[str, dict[str, Any]]:
    documents = list(yaml.safe_load_all(manifest))
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    kv_detected = 0
    kv_updated = 0

    def visit(document: Any) -> None:
        nonlocal kv_detected, kv_updated
        if not isinstance(document, dict):
            return
        if document.get("kind") == "List":
            for item in document.get("items") or []:
                visit(item)
            return
        identity = f"{document.get('kind')}/{document.get('metadata', {}).get('name', '<unnamed>')}"
        for pod in _graceful_shutdown_pod_specs(document):
            for container in pod.get("containers") or []:
                label = f"{identity}/{container.get('name', '<unnamed>')}"
                detected, updated, kv_warnings = _apply_kv_transfer_config(container, label)
                kv_detected += detected
                kv_updated += updated
                warnings.extend(kv_warnings)
            model_containers = [
                container for container in pod.get("containers") or [] if _is_model_container(container)
            ]
            if not model_containers:
                continue
            pod_entries: list[dict[str, Any]] = []
            longest_shutdown_timeout = 0
            for container in model_containers:
                runtime = _runtime_kind(container)
                label = f"{identity}/{container.get('name', '<unnamed>')}"
                entry: dict[str, Any] = {
                    "resource": identity,
                    "container": container.get("name"),
                    "runtime": runtime,
                }
                if runtime == "vllm":
                    shutdown_timeout = _apply_shutdown_timeout(container)
                    if shutdown_timeout is not None:
                        longest_shutdown_timeout = max(longest_shutdown_timeout, shutdown_timeout)
                    entry["shutdownTimeoutSeconds"] = shutdown_timeout
                    if shutdown_timeout is None:
                        warnings.append(f"{label}: could not safely apply vLLM --shutdown-timeout")
                elif runtime == "sglang":
                    warnings.append(
                        f"{label}: SGLang does not expose vLLM --shutdown-timeout; "
                        "only endpoint removal and the termination grace period apply."
                    )
                entry["preStop"] = _ensure_prestop(container)
                pod_entries.append(entry)
            required_grace = max(
                MODEL_TERMINATION_GRACE_PERIOD_SECONDS,
                longest_shutdown_timeout + MODEL_PRESTOP_SLEEP_SECONDS + 10 if longest_shutdown_timeout else 0,
            )
            _ensure_termination_grace_period(pod, required_grace)
            for entry in pod_entries:
                entry["terminationGracePeriodSeconds"] = required_grace
            entries.extend(pod_entries)

    for document in documents:
        visit(document)
    report = {
        "kvLoadFailurePolicy": KV_LOAD_FAILURE_POLICY,
        "kvTransferConfigsDetected": kv_detected,
        "kvTransferConfigsUpdated": kv_updated,
        "shutdownTimeoutSeconds": MODEL_SHUTDOWN_TIMEOUT_SECONDS,
        "terminationGracePeriodSeconds": MODEL_TERMINATION_GRACE_PERIOD_SECONDS,
        "preStopSleepSeconds": MODEL_PRESTOP_SLEEP_SECONDS,
        "containers": entries,
        "warnings": warnings,
    }
    if not entries:
        return manifest, report
    return yaml.safe_dump_all(documents, sort_keys=False), report


def prepare_single_slice_disaggregated_set_manifest(manifest: str) -> tuple[str, dict[str, Any]]:
    documents = list(yaml.safe_load_all(manifest))
    disaggregated_sets = 0
    adjusted = 0

    def visit(document: Any) -> None:
        nonlocal disaggregated_sets, adjusted
        if not isinstance(document, dict):
            return
        if document.get("kind") == "List":
            for item in document.get("items") or []:
                visit(item)
            return
        if document.get("kind") != "DisaggregatedSet":
            return
        disaggregated_sets += 1
        spec = document.get("spec")
        if not isinstance(spec, dict):
            return
        if spec.get("slices") != 1:
            spec["slices"] = 1
            adjusted += 1

    for document in documents:
        visit(document)
    report = {
        "slices": 1,
        "disaggregatedSets": disaggregated_sets,
        "adjusted": adjusted,
    }
    if not adjusted:
        return manifest, report
    return yaml.safe_dump_all(documents, sort_keys=False), report


def prepare_single_slice_disaggregated_set(
    artifact: GuideDeploymentArtifact, output_root: Path
) -> GuideDeploymentArtifact:
    if not artifact.manifest_ref:
        return artifact
    reference = Path(artifact.manifest_ref)
    directory_artifact = reference.is_dir()
    path = reference / "manifest.yaml" if directory_artifact else reference
    if not path.is_file():
        return artifact
    manifest = path.read_text(encoding="utf-8")
    rendered, report = prepare_single_slice_disaggregated_set_manifest(manifest)
    if not report["disaggregatedSets"]:
        return artifact
    if report["adjusted"]:
        logger.warning(
            "Guide %s: forced DisaggregatedSet slices to 1 (Lens supports single-slice manual scaling)",
            artifact.guide_id,
        )
    checksum = text_checksum(rendered)
    if rendered != manifest:
        if directory_artifact:
            path.write_text(rendered, encoding="utf-8")
        else:
            directory = output_root / checksum.removeprefix("sha256:")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "manifest.yaml"
            if path.is_file() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError("checksum-addressed DisaggregatedSet manifest drift detected")
            path.write_text(rendered, encoding="utf-8")
            register_artifacts(
                directory,
                owner_type="deployment-manifest",
                owner_id=checksum.removeprefix("sha256:"),
                source_version=artifact.source_ref,
                retention_class="configuration",
            )
    return replace(
        artifact,
        manifest_ref=str(reference if directory_artifact else path),
        manifest_checksum=artifact.manifest_checksum if directory_artifact else checksum,
        artifact_hash=stable_hash({"provider": artifact.artifact_hash, "disaggregatedSet": report}),
        deployment_contract={**artifact.deployment_contract, "disaggregatedSet": report},
    )


def prepare_graceful_shutdown(artifact: GuideDeploymentArtifact, output_root: Path) -> GuideDeploymentArtifact:
    if not artifact.manifest_ref:
        return artifact
    reference = Path(artifact.manifest_ref)
    directory_artifact = reference.is_dir()
    path = reference / "manifest.yaml" if directory_artifact else reference
    if not path.is_file():
        return artifact
    manifest = path.read_text(encoding="utf-8")
    rendered, report = prepare_graceful_shutdown_manifest(manifest)
    if not report["containers"]:
        return artifact
    for warning in report["warnings"]:
        logger.warning("Guide %s: %s", artifact.guide_id, warning)
    checksum = text_checksum(rendered)
    if rendered != manifest:
        if directory_artifact:
            path.write_text(rendered, encoding="utf-8")
        else:
            directory = output_root / checksum.removeprefix("sha256:")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "manifest.yaml"
            if path.is_file() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError("checksum-addressed graceful-shutdown manifest drift detected")
            path.write_text(rendered, encoding="utf-8")
            register_artifacts(
                directory,
                owner_type="deployment-manifest",
                owner_id=checksum.removeprefix("sha256:"),
                source_version=artifact.source_ref,
                retention_class="configuration",
            )
    return replace(
        artifact,
        manifest_ref=str(reference if directory_artifact else path),
        manifest_checksum=artifact.manifest_checksum if directory_artifact else checksum,
        artifact_hash=stable_hash({"provider": artifact.artifact_hash, "gracefulShutdown": report}),
        deployment_contract={**artifact.deployment_contract, "gracefulShutdown": report},
    )


class ConfigurationManifestAdapter:
    """Use Configuration's manifest when present, otherwise retain legacy behavior."""

    def __init__(
        self,
        delegate: GuideAdapter,
        command_runner,
        namespace_prefix: str,
        readiness_timeout_seconds: int,
        output_root: Path = DEPLOYMENT_MANIFEST_DIR,
    ) -> None:
        self._delegate = delegate
        self._runner = command_runner
        self._namespace_prefix = namespace_prefix
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._output_root = output_root.resolve()

    def discover(self) -> GuideDefinition:
        return self._delegate.discover()

    def register_restored_namespace(self, namespace: str) -> None:
        register = getattr(self._runner, "register_restored_namespace", None)
        if callable(register):
            register(namespace)
        delegate_register = getattr(self._delegate, "register_restored_namespace", None)
        if callable(delegate_register):
            delegate_register(namespace)

    def validate_inputs(
        self, definition: GuideDefinition, cluster_snapshot: dict[str, Any], overrides: dict[str, Any]
    ) -> ValidationResult:
        if not self._is_exact(overrides):
            return self._delegate.validate_inputs(definition, cluster_snapshot, overrides)
        try:
            validate_configuration_manifest_content(overrides, definition.guide_id)
            source = (overrides.get("officialGuide") or {}).get("source") or {}
            if not accelerator_supported(source.get("accelerator")):
                raise ValueError(
                    f"{definition.guide_id} does not support accelerator "
                    f"{source.get('accelerator')!r} in this Prism release"
                )
            allowed_servers = {"vllm", "vllm-rdma"} if definition.guide_id == "pd-disaggregation" else {"vllm"}
            if source.get("modelServer") not in allowed_servers:
                raise ValueError(
                    f"{definition.guide_id} does not support model server {source.get('modelServer')} "
                    "in this Prism release"
                )
        except ValueError as error:
            return ValidationResult(False, [str(error)])
        return ValidationResult(True)

    async def render(self, definition: GuideDefinition, overrides: dict[str, Any]) -> GuideDeploymentArtifact:
        if not self._is_exact(overrides):
            artifact = await self._delegate.render(definition, overrides)
        else:
            manifest, source_checksum, deployment, source = validate_configuration_manifest_content(
                overrides, definition.guide_id
            )
            deployment_contract = dict(deployment)
            bundle = (overrides.get("officialGuide") or {}).get("deploymentBundle")
            if bundle is not None:
                validate_deployment_bundle(bundle, definition.guide_id, str(source.get("commit") or ""))
                deployment_contract["deploymentBundle"] = bundle
            router_profile = overrides.get("routerProfile")
            if isinstance(router_profile, str) and router_profile:
                deployment_contract["routerProfile"] = router_profile
            namespace = str(overrides.get("_deployment_namespace") or "")
            self._validate_namespace(namespace)
            rendered = manifest.replace("__PRISM_NAMESPACE__", namespace)
            if "__PRISM_NAMESPACE__" in rendered:
                raise ValueError("deployment manifest contains an unresolved namespace binding")
            manifest_checksum = text_checksum(rendered)
            directory = self._output_root / manifest_checksum.removeprefix("sha256:")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "manifest.yaml"
            if path.is_file() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError("checksum-addressed deployment manifest drift detected")
            if not path.is_file():
                path.write_text(rendered, encoding="utf-8")
            register_artifacts(
                directory,
                owner_type="deployment-manifest",
                owner_id=manifest_checksum.removeprefix("sha256:"),
                source_version=source.get("commit"),
                retention_class="configuration",
            )
            artifact = GuideDeploymentArtifact(
                guide_id=definition.guide_id,
                artifact_hash=stable_hash(
                    {
                        "source": source_checksum,
                        "manifest": manifest_checksum,
                        "namespace": namespace,
                        "deployment_contract": deployment_contract,
                    }
                ),
                manifest_ref=str(path),
                source_ref=str(source.get("commit")),
                guide_content_hash=source_checksum,
                manifest_checksum=manifest_checksum,
                deployment_contract=deployment_contract,
            )

        if artifact.manifest_ref and Path(artifact.manifest_ref).is_dir():
            reference = Path(artifact.manifest_ref)
            isolated = Path(tempfile.mkdtemp(prefix=f"{reference.name}-", dir=reference.parent))
            shutil.copytree(reference, isolated, dirs_exist_ok=True)
            artifact = replace(
                artifact,
                manifest_ref=str(isolated),
                artifact_hash=stable_hash({"provider_artifact": artifact.artifact_hash, "overlay": str(isolated)}),
            )
        artifact = prepare_graceful_shutdown(artifact, self._output_root)
        artifact = prepare_single_slice_disaggregated_set(artifact, self._output_root)
        artifact = prepare_shared_memory(artifact, self._output_root, exact=self._is_exact(overrides))
        if "_model_cache_dependencies" in overrides:
            from llm_d_lens.deploy.providers.storage_mount import validate_cached_model_arguments

            manifest_path = Path(artifact.manifest_ref or "")
            if manifest_path.is_dir():
                manifest_path = manifest_path / "manifest.yaml"
            if not manifest_path.is_file():
                raise ValueError("Model Cache preflight requires a rendered manifest file")
            validate_cached_model_arguments(overrides, manifest_path.read_text(encoding="utf-8"))
        artifact = instrument_manifest(artifact, self._output_root)
        placement = overrides.get("_placement")
        if placement is not None and not artifact.deployment_contract.get("placementApplied"):
            if not isinstance(placement, dict) or not artifact.manifest_ref or Path(artifact.manifest_ref).is_dir():
                raise ValueError("placement requires a rendered Kubernetes manifest file")
            rendered, changed = apply_placement(Path(artifact.manifest_ref).read_text(encoding="utf-8"), placement)
            checksum = text_checksum(rendered)
            directory = self._output_root / checksum.removeprefix("sha256:")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "manifest.yaml"
            if path.is_file() and path.read_text(encoding="utf-8") != rendered:
                raise ValueError("checksum-addressed placement manifest drift detected")
            path.write_text(rendered, encoding="utf-8")
            register_artifacts(
                directory,
                owner_type="deployment-manifest",
                owner_id=checksum.removeprefix("sha256:"),
                retention_class="configuration",
            )
            artifact = replace(
                artifact,
                manifest_ref=str(path),
                manifest_checksum=checksum,
                artifact_hash=stable_hash(
                    {"provider_artifact": artifact.artifact_hash, "placement": placement, "workloads": changed}
                ),
            )
        runtime_policy = overrides.get("_runtime_class_policy")
        cluster_id = overrides.get("_cluster_id")
        if runtime_policy is not None:
            capabilities = (
                {"items": [], "nodes": []}
                if runtime_policy.get("mode") == "node-default"
                else await discover_runtime_classes(str(cluster_id or ""))
            )
            reference = Path(artifact.manifest_ref or "")
            path = reference / "manifest.yaml" if reference.is_dir() else reference
            manifest = path.read_text(encoding="utf-8")
            source = (overrides.get("officialGuide") or {}).get("source") or {}
            rendered, report = prepare_runtime_class_manifest(
                manifest,
                runtime_policy,
                capabilities,
                accelerator=overrides.get("hardware_profile")
                or source.get("accelerator")
                or getattr(self._delegate, "_accelerator", None)
                or getattr(getattr(self._delegate, "_policy", None), "accelerator", None),
            )
            checksum = text_checksum(rendered)
            if rendered != manifest:
                if reference.is_dir():
                    path.write_text(rendered, encoding="utf-8")
                else:
                    directory = self._output_root / checksum.removeprefix("sha256:")
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "manifest.yaml"
                    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
                        raise ValueError("checksum-addressed RuntimeClass manifest drift detected")
                    path.write_text(rendered, encoding="utf-8")
            artifact = replace(
                artifact,
                manifest_ref=str(reference if reference.is_dir() else path),
                manifest_checksum=artifact.manifest_checksum if reference.is_dir() else checksum,
                artifact_hash=stable_hash({"provider_artifact": artifact.artifact_hash, "runtimeClass": report}),
                deployment_contract={**artifact.deployment_contract, "runtimeClass": report},
            )
        if artifact.manifest_ref and artifact.manifest_checksum:
            saved_path = Path(artifact.manifest_ref)
            if saved_path.is_file() and saved_path.resolve().is_relative_to(self._output_root.resolve()):
                register_artifacts(
                    saved_path.parent,
                    owner_type="deployment-manifest",
                    owner_id=artifact.manifest_checksum.removeprefix("sha256:"),
                    source_version=artifact.source_ref,
                    retention_class="configuration",
                )
        target_node = configured_target_node()
        if not target_node:
            return artifact
        if not artifact.manifest_ref:
            raise ValueError("target-node enforcement requires a rendered Kubernetes manifest")
        if Path(artifact.manifest_ref).is_dir():
            # Descriptor/Kustomize providers own directory overlays and must
            # apply node pinning while rendering their generated patch. The
            # generic pinning helper accepts a single rendered YAML file only.
            return artifact
        manifest_ref, manifest_checksum, pinned_workloads = write_pinned_manifest(
            artifact.manifest_ref, self._output_root, target_node
        )
        return replace(
            artifact,
            artifact_hash=stable_hash(
                {
                    "provider_artifact": artifact.artifact_hash,
                    "manifest": manifest_checksum,
                    "target_node": target_node,
                    "pinned_workloads": pinned_workloads,
                }
            ),
            manifest_ref=manifest_ref,
            manifest_checksum=manifest_checksum,
        )

    async def deploy(self, artifact: GuideDeploymentArtifact, execution_context: dict[str, Any]) -> dict[str, Any]:
        if not {"readinessDeployments", "endpoint"} <= artifact.deployment_contract.keys():
            return await self._delegate.deploy(artifact, execution_context)
        if Path(artifact.manifest_ref or "").is_dir():
            # Legacy Guide adapters may return a Kustomize overlay directory
            # rather than an immutable Configuration manifest file.
            return await self._delegate.deploy(artifact, execution_context)
        namespace = self._namespace(execution_context)
        status, stdout, stderr = await self._runner(["kubectl", "create", "namespace", namespace])
        if status != 0 and "alreadyexists" not in (stdout + stderr).lower():
            raise RuntimeError((stderr or stdout or "namespace creation failed").strip())
        manifest = Path(artifact.manifest_ref or "").read_text(encoding="utf-8")
        model_token = execution_context.get("model_token")
        if "name: llm-d-hf-token" in manifest and model_token:
            create_secret = getattr(self._runner, "create_model_secret_from_token", None)
            if not callable(create_secret):
                raise RuntimeError("deployment runner cannot provision the supplied model token")
            await create_secret(namespace, model_token)
        elif "name: llm-d-hf-token" in manifest:
            policy = execution_context.get("deployment_policy") or {}
            model_secret = policy.get("model_secret") or {"mode": "none"}
            if model_secret.get("mode") == "existing-secret":
                copy_secret = getattr(self._runner, "copy_model_secret", None)
                source_namespace = str(model_secret.get("sourceNamespace") or "")
                source_name = str(model_secret.get("sourceName") or "")
                if not callable(copy_secret) or not source_namespace or not source_name:
                    raise RuntimeError("existing model Secret source is not configured")
                await copy_secret(namespace, source_namespace, source_name)
            elif model_secret.get("mode") == "host":
                create_secret = getattr(self._runner, "create_model_secret", None)
                token_file = Path.home() / ".cache" / "huggingface" / "token"
                if not callable(create_secret) or not token_file.is_file():
                    raise RuntimeError(
                        "deployment requires a model token; select an existing Kubernetes Secret or "
                        "configure a host token"
                    )
                await create_secret(namespace, token_file)
        install_router = (
            getattr(self._delegate, "install_published_router", None)
            if callable(getattr(type(self._delegate), "install_published_router", None))
            else None
        )
        runtime = execution_context.get("runtime") or {}
        storage_volume_id = str(runtime.get("storageVolumeId") or "")
        cluster_id = str((execution_context.get("provenance") or {}).get("cluster_server_id") or "")
        if storage_volume_id:
            # Must run before any delegate-lifecycle early return below: the
            # rendered manifest (from Configuration) already embeds a
            # `persistentVolumeClaim` volume referencing the storage volume's
            # PVC name, but that PVC only exists in Storage's home namespace
            # (see storage.service.ensure_mount_in_namespace). Guides such as
            # pd-disaggregation delegate their whole `deploy()` to install
            # Guide-specific Router/EPP components before applying the
            # manifest, which would otherwise skip this and leave the Pod
            # referencing a PVC that doesn't exist in its own namespace.
            volume = await get_ready_volume(storage_volume_id, cluster_id=cluster_id or None)
            await ensure_mount_in_namespace(volume, namespace=namespace, cluster_id=cluster_id)
        if callable(install_router):
            await install_router(artifact, execution_context)
        elif self._uses_delegate_lifecycle(artifact):
            # Published manifests contain model-server resources, while these
            # providers also own Router or other Guide-specific components.
            return await self._delegate.deploy(artifact, execution_context)
        status, stdout, stderr = await self._runner(
            ["kubectl", "apply", "--namespace", namespace, "--filename", artifact.manifest_ref or ""]
        )
        if status != 0:
            raise RuntimeError((stderr or stdout or "manifest apply failed").strip())
        reproducibility = {
            key: value
            for key, value in execution_context.items()
            if key.startswith("router_effective_")
            or key.startswith("router_rendered_")
            or key
            in {
                "router_values_path",
                "router_chart",
                "router_version",
                "router_release_name",
                "calibration_script_path",
            }
        }
        return {
            "namespace": namespace,
            "artifact_hash": artifact.artifact_hash,
            "apply_output": stdout,
            **reproducibility,
        }

    async def readiness(self, execution: dict[str, Any]) -> ValidationResult:
        artifact = execution.get("_artifact")
        if not isinstance(artifact, GuideDeploymentArtifact) or not artifact.deployment_contract:
            return await self._delegate.readiness(execution)
        if self._uses_delegate_lifecycle(artifact):
            return await self._delegate.readiness(execution)
        namespace = self._namespace(execution)
        for name in artifact.deployment_contract["readinessDeployments"]:
            # A successful apply can return before the API server's subsequent
            # read path observes the Deployment.  Retry only this propagation
            # race; real rollout failures must still be reported immediately.
            for attempt in range(10):
                status, stdout, stderr = await self._runner(
                    [
                        "kubectl",
                        "rollout",
                        "status",
                        f"deployment/{name}",
                        "--namespace",
                        namespace,
                        f"--timeout={self._readiness_timeout_seconds}s",
                    ]
                )
                message = (stderr or stdout or f"deployment/{name} not ready").strip()
                if status == 0 or not self._is_transient_not_found(message) or attempt == 9:
                    break
                await asyncio.sleep(1)
            if status != 0:
                return ValidationResult(False, [message])
        endpoint = artifact.deployment_contract["endpoint"]
        execution["endpoint_url"] = (
            f"{endpoint['protocol']}://{endpoint['serviceName']}.{namespace}.svc.cluster.local:{endpoint['port']}"
        )
        return ValidationResult(True)

    @staticmethod
    def _is_transient_not_found(message: str) -> bool:
        """Recognize only the API propagation error that is safe to retry."""
        normalized = message.lower()
        return "deployments.apps" in normalized and "not found" in normalized

    async def diagnostics(self, execution: dict[str, Any]) -> dict[str, Any]:
        artifact = execution.get("_artifact")
        if (
            not isinstance(artifact, GuideDeploymentArtifact)
            or not artifact.deployment_contract
            or self._uses_delegate_lifecycle(artifact)
        ):
            diagnostics = await self._delegate.diagnostics(execution)
        else:
            namespace = self._namespace(execution)

            async def capture(arguments: list[str]) -> str:
                status, stdout, stderr = await self._runner(arguments)
                return (stdout if status == 0 else stderr).strip()

            diagnostics = {
                "namespace": namespace,
                "pods": await capture(["kubectl", "get", "pods", "--namespace", namespace, "-o", "wide"]),
                "events": await capture(
                    ["kubectl", "get", "events", "--namespace", namespace, "--sort-by=.lastTimestamp"]
                ),
                "resource_claims": await capture(
                    ["kubectl", "get", "resourceclaims.resource.k8s.io", "--namespace", namespace, "-o", "wide"]
                ),
            }
        if "running" in str(diagnostics.get("pods") or "").lower():
            status, stdout, stderr = await self._runner(
                ["kubectl", "get", "pods", "--namespace", self._namespace(execution), "-o", "json"]
            )
            if status != 0:
                diagnostics["pod_statuses_error"] = (stderr or stdout or "Pod status query failed").strip()
                logger.warning("Pod startup diagnostics failed: %s", diagnostics["pod_statuses_error"])
            else:
                pods = json.loads(stdout)["items"]
                timing_fields = ("initialDelaySeconds", "periodSeconds", "timeoutSeconds", "failureThreshold")
                diagnostics["pod_statuses"] = [
                    {
                        "name": pod["metadata"]["name"],
                        "deleting": bool(pod["metadata"].get("deletionTimestamp")),
                        "status": pod.get("status") or {},
                        "containers": [
                            {
                                "name": container["name"],
                                **{
                                    probe: {
                                        key: container[probe][key] for key in timing_fields if key in container[probe]
                                    }
                                    for probe in ("startupProbe", "readinessProbe")
                                    if probe in container
                                },
                            }
                            for container in pod.get("spec", {}).get("containers") or []
                        ],
                    }
                    for pod in pods
                ]
        if isinstance(artifact, GuideDeploymentArtifact) and "sharedMemory" in artifact.deployment_contract:
            diagnostics = {**diagnostics, "shared_memory": artifact.deployment_contract["sharedMemory"]}
        return diagnostics

    async def stop(self, execution, artifact):
        if not artifact.deployment_contract:
            return await self._delegate.stop(execution, artifact)
        if self._uses_delegate_lifecycle(artifact):
            return await self._delegate.stop(execution, artifact)
        namespace = self._namespace(execution)
        results = [
            await self._runner(["kubectl", "scale", f"deployment/{name}", "--namespace", namespace, "--replicas=0"])
            for name in artifact.deployment_contract["readinessDeployments"]
        ]
        return {
            "stopped": all(result[0] == 0 for result in results),
            "output": "\n".join(result[1] for result in results),
        }

    async def rollback(self, execution, artifact):
        return await self.stop(execution, artifact)

    async def cleanup(self, execution, artifact, *, force: bool = False):
        if not artifact.deployment_contract:
            return await self._delegate.cleanup(execution, artifact, force=force)
        if self._uses_delegate_lifecycle(artifact):
            return await self._delegate.cleanup(execution, artifact, force=force)
        namespace = self._namespace(execution)
        status, stdout, stderr = await self._runner(
            ["kubectl", "delete", "namespace", namespace, "--ignore-not-found=true"]
        )
        return {"cleaned_up": status == 0, "output": stdout, "error": stderr or None}

    @staticmethod
    def _is_exact(overrides: dict[str, Any]) -> bool:
        return isinstance(overrides.get("officialGuide"), dict)

    def _uses_delegate_lifecycle(self, artifact: GuideDeploymentArtifact) -> bool:
        return (
            not {"readinessDeployments", "endpoint"} <= artifact.deployment_contract.keys()
            or artifact.guide_id == "pd-disaggregation"
            or bool(getattr(type(self._delegate), "published_manifest_lifecycle", False))
        )

    def _namespace(self, context: dict[str, Any]) -> str:
        namespace = str(context.get("namespace") or "")
        self._validate_namespace(namespace)
        return namespace

    def _validate_namespace(self, namespace: str) -> None:
        if (
            not namespace.startswith(self._namespace_prefix)
            or namespace == self._namespace_prefix
            or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace)
        ):
            raise ValueError("deployment namespace is outside the configured allowlist")
