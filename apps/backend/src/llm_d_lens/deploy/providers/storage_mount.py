"""Resolve a Deploy ``runtime.storageVolumeId`` into a Pod volume/volumeMount.

Shared by all Deploy providers (``baseline_vllm.py``,
``optimized_baseline.py``, ``precise_prefix_cache_routing.py``) so a new
storage volume type only needs a change in ``llm_d_lens/storage/service.py``,
not in every provider. See ``docs/fern/pages/api-reference/storage.mdx`` for the full design.

Returns ``None`` when ``runtime`` has no ``storageVolumeId``, so providers keep
their existing ``runtime.mountPath``/``modelSource`` hostPath handling
untouched as the backward-compatible "ad-hoc" fallback.
"""

from __future__ import annotations

from typing import Any

_UNSUPPORTED_MODEL_ARGUMENTS = {"model", "model-path", "code-revision", "hf-config-path", "config", "download-dir"}


def _model_dependencies(model: str, arguments: dict[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    from llm_d_lens.model_cache.contracts import HuggingFaceSource

    revision = arguments.get("revision", "main")
    tokenizer = arguments.get("tokenizer", model)
    tokenizer_revision = arguments.get("tokenizer-revision", revision)
    dependencies = ((model, revision), (tokenizer, tokenizer_revision))
    for repository, required_revision in dependencies:
        HuggingFaceSource(repoId=repository, revision=required_revision)
    return dependencies


def _required_model_dependencies(
    overrides: dict[str, Any], provider_id: str
) -> dict[str, tuple[tuple[str, str], tuple[str, str]]]:
    model = str((overrides.get("model") or {}).get("name") or "").strip()
    if not model:
        raise ValueError("selected model revision is not ready in Model Cache; model.name is required")
    if (overrides.get("runtime") or {}).get("configurationSource"):
        raise ValueError("Model Cache preflight does not support runtime.configurationSource overrides")
    roles = ("prefill", "decode") if provider_id == "pd-disaggregation" else ("decode",)
    result = {}
    for role in roles:
        arguments: dict[str, str] = {}
        for item in overrides.get("customParameters") or []:
            if item.get("kind") != "argument" or item.get("target") not in {role, "both"}:
                continue
            name = str(item.get("name") or "").replace("_", "-")
            if name in _UNSUPPORTED_MODEL_ARGUMENTS:
                raise ValueError(f"Model Cache preflight does not support --{name}; use the selected cached model")
            if name not in {"revision", "tokenizer", "tokenizer-revision"}:
                continue
            if provider_id == "optimized-baseline" and not isinstance(overrides.get("officialGuide"), dict):
                raise ValueError(f"optimized-baseline does not apply custom --{name} during provider rendering")
            value = str(item.get("value") or "")
            if name in arguments and arguments[name] != value:
                raise ValueError(f"conflicting --{name} values for {role}")
            arguments[name] = value
        result[role] = _model_dependencies(model, arguments)
    return result


def validate_cached_model_arguments(overrides: dict[str, Any], manifest: str) -> None:
    from llm_d_lens.configuration.manifest_facts import (
        _documents,
        _invocation_tokens,
        _model_argument,
        _modelserver_containers,
        _option,
    )

    expected = overrides["_model_cache_dependencies"]
    seen = set()
    for workload, container in _modelserver_containers(_documents(manifest)):
        name = str(workload.get("name") or "<unnamed>")
        role = workload["role"] if "prefill" in expected else "decode"
        tokens = _invocation_tokens(container, name)
        for argument in _UNSUPPORTED_MODEL_ARGUMENTS:
            if _option(tokens, {f"--{argument}", f"--{argument.replace('-', '_')}"}, name) is not None:
                raise ValueError(f"Model Cache cannot verify rendered --{argument} in {name}")
        arguments = {}
        for argument in ("revision", "tokenizer", "tokenizer-revision"):
            value = _option(tokens, {f"--{argument}", f"--{argument.replace('-', '_')}"}, name)
            if value is not None:
                arguments[argument] = value
        actual = _model_dependencies(_model_argument(tokens, name), arguments)
        if role not in expected or actual != expected[role]:
            raise ValueError(f"rendered model/tokenizer for {name} does not match Model Cache preflight")
        seen.add(role)
    if seen != set(expected):
        raise ValueError("rendered manifest is missing model-server roles required by Model Cache preflight")


async def require_cached_model(
    overrides: dict[str, Any], *, cluster_id: str | None, provider_id: str = "baseline-vllm"
) -> None:
    """Require a ready, fully synchronized Model Cache entry before rendering."""
    from llm_d_lens.configuration.managed_environment import admin_managed_environment_names
    from llm_d_lens.model_cache.service import default_service
    from llm_d_lens.storage.contracts import StorageVolumeKind, StorageVolumePurpose
    from llm_d_lens.storage.service import get_ready_volume

    runtime = overrides.get("runtime") or {}
    volume_id = runtime.get("storageVolumeId")
    if not cluster_id or not isinstance(volume_id, str) or not volume_id.strip():
        raise ValueError(
            "Deployment requires a cluster and a Model Cache storage volume; download the model in Model Cache first"
        )
    if runtime.get("modelSource") not in (None, "auto-cache"):
        raise ValueError("Deployment modelSource must be auto-cache; download the model in Model Cache first")
    custom = overrides.get("customParameters") or []
    if admin_managed_environment_names(custom):
        raise ValueError("Deployment cannot override the managed Hugging Face environment")
    volume = await get_ready_volume(volume_id, cluster_id=cluster_id)
    if StorageVolumePurpose.MODEL_CACHE not in volume.purposes:
        raise ValueError("selected storage volume is not designated for model cache use")
    if volume.kind == StorageVolumeKind.DYNAMIC_PVC:
        raise ValueError("model-cache storage for deployments cannot use dynamic-pvc volumes")
    dependencies_by_role = _required_model_dependencies(overrides, provider_id)
    dependencies = set().union(*dependencies_by_role.values())
    cache = default_service()
    entries = await cache.list(cluster_id=cluster_id, storage_volume_id=volume_id)
    selected = [
        entry
        for entry in entries
        if entry.status == "ready"
        and entry.source.huggingface
        and (entry.source.huggingface.repo_id, entry.source.huggingface.revision) in dependencies
    ]
    available = {(entry.source.huggingface.repo_id, entry.source.huggingface.revision) for entry in selected}
    if available != dependencies:
        missing = ", ".join(f"{repository}@{revision}" for repository, revision in sorted(dependencies - available))
        raise ValueError(f"selected model/tokenizer revision is not ready in Model Cache: {missing}")
    if await cache.compute_pending_sync(selected):
        raise ValueError("selected model cache is missing on cluster nodes; sync it in Model Cache before deployment")
    runtime["modelSource"] = "auto-cache"
    runtime["cacheOffline"] = True
    runtime.pop("mountPath", None)
    runtime.pop("pvcName", None)
    overrides["runtime"] = runtime
    overrides["_model_cache_dependencies"] = dependencies_by_role


async def resolve_mount(
    runtime: dict[str, Any], *, cluster_id: str | None, namespace: str | None = None
) -> dict[str, Any] | None:
    """Resolve ``runtime.storageVolumeId`` into a mount descriptor, if present.

    The returned dict has keys:
      - ``volume_source``: the Pod ``volumes[]`` entry body (merged with a
        ``name`` key by the caller), e.g. ``{"persistentVolumeClaim": {...}}``.
      - ``mount_path``: container mount path (always ``/model-cache`` today).
      - ``read_only``: whether the mount should be read-only.
            - ``model_source``: ``"auto-cache"`` for a validated HF cache root, or
                ``"shared-path"`` for a legacy read-only model directory.

    Unlike an earlier revision, ``local-disk`` volumes are no longer pinned to
    a single node: the registered ``hostPath`` is expected to resolve to an
    identical directory on every node in the cluster (e.g. a pre-populated
    model cache synced to all nodes), so the Pod may be scheduled anywhere.

    ``namespace`` should be the actual namespace the rendered Pod will be
    applied into (Deploy providers pass ``overrides["_deployment_namespace"]``,
    set by ``deploy/service.py`` before calling ``render()``). PVCs are
    namespace-scoped, but that namespace is virtually never the fixed one the
    volume's own PVC was registered into by Storage, so without resolving a
    namespace-local PVC here, the rendered Pod would reference a PVC that
    doesn't exist in its own namespace and fail scheduling (see
    ``storage.service.ensure_mount_in_namespace``). ``namespace`` is optional,
    defaulting to ``None``, only so existing callers/tests that don't yet
    thread it through keep working -- passing it is required to actually mount
    successfully whenever the Deploy namespace differs from Storage's.
    """
    volume_id = runtime.get("storageVolumeId")
    if not volume_id or not isinstance(volume_id, str):
        return None
    from llm_d_lens.storage.service import ensure_mount_in_namespace, get_ready_volume

    volume = await get_ready_volume(volume_id, cluster_id=cluster_id)
    read_only = volume.read_only or runtime.get("modelSource") == "auto-cache"
    pvc_name = volume.pvc_name
    if namespace:
        pvc_name = await ensure_mount_in_namespace(volume, namespace=namespace, cluster_id=cluster_id or "")
    return {
        "volume_source": {"persistentVolumeClaim": {"claimName": pvc_name}},
        "mount_path": "/model-cache",
        "read_only": read_only,
        "model_source": "auto-cache"
        if runtime.get("modelSource") == "auto-cache"
        else ("shared-path" if read_only else "auto-cache"),
    }
