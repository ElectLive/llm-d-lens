"""Guide planning primitives shared by the Python compatibility API."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from copy import deepcopy

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_d_lens.configuration.manifest_facts import _invocation_tokens, workload_roles
from llm_d_lens.core.exceptions import DomainError
from llm_d_lens.hardware.registry import all_profiles
from llm_d_lens.hardware.resolver import resolve_by_accelerator_key, resolve_by_device_class, resolve_by_resource
from llm_d_lens.versions import router_pd_sidecar_image

from .guide_runtime import apply_runtime_overrides, configure_cpu_cache, configure_model_server, runtime_overrides
from .remote_deploy import RemoteError

router = APIRouter(prefix="/api/guide-planning", include_in_schema=False)
logger = logging.getLogger(__name__)


class GuideError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.public_message = message if status < 500 else "Guide source could not be prepared"


def profile_for_key(key: str):
    return resolve_by_accelerator_key(key) or next(
        (profile for profile in all_profiles() if profile.upstream_variant == key), None
    )


def overlay_root(profile, guide: str, model_server: str) -> str:
    if profile is None:
        raise GuideError("Hardware profile is unavailable")
    return profile.deployment.overlay_root.format(
        guide=guide,
        variant=profile.upstream_variant,
        model_server=model_server,
        vendor=profile.upstream_vendor,
    )


def select_manifest_paths(paths: list[str], guide: str, accelerator: str, model_server: str, variant: str = "") -> dict:
    profile = profile_for_key(accelerator)
    prefix = overlay_root(profile, guide, model_server) + "/"
    matching = [path for path in paths if path.startswith(prefix)]
    if not matching:
        raise GuideError("No official YAML manifests match the selected guide, accelerator, and model server", 404)
    normalized = variant.strip().strip("/")
    if normalized and (".." in normalized.split("/") or not re.fullmatch(r"[A-Za-z0-9._/-]+", normalized)):
        raise GuideError("Guide variant path is invalid")
    if normalized and not any(path.startswith(f"{prefix}{normalized}/") for path in matching):
        alternate, *parts = normalized.split("/")
        alternate_prefix = overlay_root(profile, guide, alternate) + "/"
        alternate_matching = [path for path in paths if path.startswith(alternate_prefix)]
        roots = [path for path in alternate_matching if "/" not in path[len(alternate_prefix) :]]
        if roots or parts and alternate_matching:
            model_server, prefix, matching, normalized = (
                alternate,
                alternate_prefix,
                alternate_matching,
                "/".join(parts),
            )
    entries = [
        path[len(prefix) :].rsplit("/", 1)[0] if "/" in path[len(prefix) :] else "."
        for path in matching
        if re.search(r"(?:^|/)(?:kustomization\.ya?ml|Kustomization)$", path, re.IGNORECASE)
    ]
    selected_variant = normalized or (
        "." if "." in entries else "base" if "base" in entries else min(entries, default="")
    )
    if selected_variant not in entries:
        raise GuideError("Selected variant has no Kustomization entry point", 404)
    selected = [
        path
        for path in matching
        if (
            "/" not in path[len(prefix) :]
            if selected_variant == "."
            else path.startswith(f"{prefix}{selected_variant}/")
        )
    ]
    return {
        "modelServer": model_server,
        "variant": selected_variant,
        "selected": selected,
        "guidePath": prefix.rstrip("/") if selected_variant == "." else prefix + selected_variant,
    }


def merge_patch(target, patch):
    if isinstance(patch, list):
        if not isinstance(target, list) or not all(
            isinstance(item, dict) and isinstance(item.get("name"), str) for item in [*target, *patch]
        ):
            return deepcopy(patch)
        result = deepcopy(target)
        for item in patch:
            index = next((index for index, value in enumerate(result) if value["name"] == item["name"]), None)
            if index is None:
                result.append(deepcopy(item))
            else:
                result[index] = merge_patch(result[index], item)
        return result
    if isinstance(patch, dict):
        result = deepcopy(target) if isinstance(target, dict) else {}
        for key, value in patch.items():
            result[key] = merge_patch(result.get(key), value)
        return result
    return deepcopy(patch)


def apply_custom_patches(
    documents: list[dict], patches: list[dict], errors: list[str], warnings: list[str]
) -> list[dict]:
    applied = []
    if len(patches) > 20:
        errors.append(f"Too many custom patch files ({len(patches)}); at most 20 are allowed per plan request.")
        return applied
    for patch in patches:
        label = str(patch.get("path") or "uploaded-patch.yaml").strip() or "uploaded-patch.yaml"
        content = str(patch.get("content") or "")
        if not content.strip():
            continue
        if len(content.encode()) > 262144:
            errors.append(f"Custom patch {label} is larger than the 256 KiB per-file limit.")
            continue
        try:
            parsed = [document for document in yaml.safe_load_all(content) if document]
        except yaml.YAMLError:
            logger.warning("Custom Guide patch could not be parsed as YAML", exc_info=True)
            errors.append(f"Custom patch {label} could not be parsed as YAML.")
            continue
        if not parsed:
            warnings.append(f"Custom patch {label} did not contain any YAML documents.")
        for document in parsed:
            if not isinstance(document, dict):
                errors.append(f"Custom patch {label} contains a document that is not a YAML mapping.")
                continue
            kind, name = document.get("kind"), (document.get("metadata") or {}).get("name")
            if not kind:
                errors.append(
                    f'Custom patch {label} has a document with no "kind"; '
                    "it cannot be matched to a base manifest resource."
                )
                continue
            matches = [
                index
                for index, value in enumerate(documents)
                if value.get("kind") == kind and (not name or value.get("metadata", {}).get("name") == name)
            ]
            if not name and len(matches) > 1:
                errors.append(
                    f'Custom patch {label} targets kind "{kind}" without metadata.name, '
                    f"and the base manifest has {len(matches)} matching resources. Add metadata.name to disambiguate."
                )
                continue
            if matches:
                documents[matches[0]] = merge_patch(documents[matches[0]], document)
            else:
                documents.append(deepcopy(document))
                identity = f"{kind}/{name}" if name else kind
                warnings.append(
                    f"Custom patch {label} ({identity}) did not match an existing resource in the base manifest; "
                    "it was added as a new resource."
                )
            applied.append(
                {"path": label, "kind": kind, "name": name or None, "action": "merged" if matches else "appended"}
            )
    return applied


def accelerator_total(resources: dict) -> float:
    return sum(
        float(value)
        for key, value in resources.items()
        if (profile := resolve_by_resource(key))
        and not any(key.endswith(suffix) for suffix in profile.monitor_resource_suffixes)
    )


def runtime_argument(container: dict, name: str):
    tokens = _invocation_tokens(container, "guide")
    found = []
    for index, token in enumerate(tokens):
        if token.startswith(f"--{name}="):
            found.append(token[len(name) + 3 :])
        elif token == f"--{name}" and index + 1 < len(tokens):
            found.append(tokens[index + 1])
    if len(found) > 1:
        raise GuideError(f"Duplicate --{name} arguments are ambiguous.")
    return found[0] if found else None


def validate_manifest_capacity(documents: list[dict], budget: float, basis: str, errors: list[str]) -> float:
    claims = [document for document in documents if document.get("kind") in {"ResourceClaimTemplate", "ResourceClaim"}]
    demand = 0

    def container_demand(container):
        resources = container.get("resources") or {}
        return accelerator_total({**(resources.get("limits") or {}), **(resources.get("requests") or {})})

    for workload in workload_roles(documents):
        document = workload["document"]
        pod = workload["pod"]
        containers = pod.get("containers", [])
        regular = sum(container_demand(container) for container in containers)
        persistent, peak = 0, 0
        for container in pod.get("initContainers", []):
            count = container_demand(container)
            if container.get("restartPolicy") == "Always":
                persistent += count
            peak = max(peak, persistent + (0 if container.get("restartPolicy") == "Always" else count))
        claim_demand = 0
        for reference in pod.get("resourceClaims", []):
            name = reference.get("resourceClaimTemplateName") or reference.get("resourceClaimName")
            namespace = document.get("metadata", {}).get("namespace", "default")
            claim = next(
                (
                    claim
                    for claim in claims
                    if claim.get("metadata", {}).get("name") == name
                    and claim.get("metadata", {}).get("namespace", "default") == namespace
                ),
                {},
            )
            spec = claim.get("spec", {})
            if claim.get("kind") == "ResourceClaimTemplate":
                spec = spec.get("spec", {})
            for request in spec.get("devices", {}).get("requests", []):
                choices = [request["exactly"]] if request.get("exactly") else request.get("firstAvailable", [request])
                claim_demand += max(
                    [
                        0,
                        *[
                            float(choice.get("count", 1))
                            for choice in choices
                            if resolve_by_device_class(choice.get("deviceClassName", ""))
                        ],
                    ]
                )
        model_tp = sum(
            float(runtime_argument(container, "tensor-parallel-size") or 1)
            for container in containers
            if container.get("name") == "modelserver"
        )
        demand += (
            float(workload.get("replicas", 1))
            * float(workload.get("slices", 1))
            * max(max(regular + persistent, peak) + claim_demand, model_tp)
        )
    errors[:] = [error for error in errors if not error.startswith("Insufficient resources:")]
    if not math.isfinite(demand) or demand > budget or (budget == 0 and demand == 0):
        label = "physical" if basis == "total" else "currently available"
        errors.append(
            f"Insufficient resources: this configuration requires {demand:g} accelerators; "
            f"the {label} budget is {budget:g}. Reduce replicas / TP or choose a sufficient resource budget "
            "before generating YAML."
        )
    return demand


def resolve_accelerator_variant(requested: str, cluster: dict | None) -> str:
    if not cluster:
        return requested
    resource_keys = cluster.get("allocatable", {}).get("accelerators", {})
    profiles = {profile.id for key in resource_keys if (profile := resolve_by_resource(key))}
    devices = (cluster.get("accelerator") or {}).get("devices") or [{}]
    for device_class in [*cluster.get("deviceClasses", []), devices[0].get("driver", "")]:
        profile = resolve_by_device_class(device_class)
        if profile:
            profiles.add(profile.id)
    return (
        next(profile.upstream_variant for profile in all_profiles() if profile.id in profiles)
        if len(profiles) == 1
        else requested
    )


def runtime_class_for_accelerator(cluster: dict | None, accelerator: str) -> str | None:
    profile = profile_for_key(accelerator)
    if not cluster or profile is None:
        return None
    for runtime in cluster.get("runtimeClasses", []):
        if any(
            pattern.lower() in str(runtime.get(key) or "").lower()
            for pattern in profile.deployment.runtime_class_matchers
            for key in ("name", "handler")
        ):
            return runtime["name"]
    return None


def quantity_number(value) -> float:
    text = str(value or "0")
    match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return 0
    number = float(match.group())
    for suffix, scale in (("Ki", 1 / 1024**2), ("Mi", 1 / 1024), ("Gi", 1), ("Ti", 1024), ("m", 0.001)):
        if text.endswith(suffix):
            return number * scale
    return number


def estimate_model(model: str) -> dict:
    match = re.search(r"(?:^|[-_])(\d+(?:\.\d+)?)b(?:[-_]|$)", model, re.IGNORECASE)
    billions = float(match[1]) if match else None
    return {
        "id": model,
        "parametersBillions": billions,
        "estimatedWeightMemoryGiB": round(billions * 2 * 1.2 * 1e9 / 1024**3, 1) if billions else None,
        "assumption": "BF16/FP16 weights plus 20% runtime overhead"
        if billions
        else "Model size could not be inferred from its ID",
    }


def normalize_guide_settings(guide: str, settings) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Guide settings must be an object.")
    if set(settings) - {"cacheCpuGiB", "rdmaNicCount", "routerValues"}:
        raise ValueError("Unknown Guide setting.")
    result = {}
    for field in ("cacheCpuGiB", "rdmaNicCount"):
        raw = settings.get(field)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            value = float(raw)
            if not math.isfinite(value) or value <= 0 or (field == "rdmaNicCount" and not value.is_integer()):
                raise ValueError
        except (TypeError, ValueError) as error:
            label = "integer" if field == "rdmaNicCount" else "number"
            raise ValueError(f"{field} must be a positive {label}.") from error
        result[field] = int(value) if value.is_integer() else value
    if "cacheCpuGiB" in result and guide != "tiered-prefix-cache":
        raise ValueError("CPU cache capacity requires a supported CPU offload variant.")
    if "rdmaNicCount" in result and guide != "pd-disaggregation":
        raise ValueError("NIC count requires the P/D RDMA variant.")
    values = settings.get("routerValues")
    if values is not None and not isinstance(values, str):
        raise ValueError("Router values must be YAML text.")
    if values and values.strip():
        result["routerValues"] = values.strip()
    return result


def resource_role(document: dict, fallback="serving") -> str:
    metadata = document.get("metadata") or {}
    label = str((metadata.get("labels") or {}).get("llm-d.ai/role") or "").lower()
    name = str(metadata.get("name") or "").lower()
    for role in ("prefill", "decode"):
        if label == role or role in name:
            return role
    return fallback


def pin_model_server_image(image: str) -> str:
    repository = image.split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    return next(
        (
            profile.deployment.runtime_image
            for profile in all_profiles()
            if repository in profile.deployment.managed_image_repositories and profile.deployment.runtime_image
        ),
        image,
    )


def plan_documents(
    documents: list[dict], model: str, machine: dict, cluster: dict | None, requested: dict | None = None
) -> dict:
    requested = requested or {}
    patches, warnings, errors = [], [], []
    settings = {}
    try:
        settings = normalize_guide_settings(requested.get("guide", ""), requested.get("guideSettings", {}))
    except ValueError:
        logger.warning("Invalid Guide settings", exc_info=True)
        errors.append("Invalid Guide settings; check cacheCpuGiB, rdmaNicCount and routerValues.")
    profile = profile_for_key(requested.get("accelerator", ""))
    workloads = workload_roles(documents)
    cluster_accel = (cluster or {}).get("accelerator") or {}
    machine_accel = machine.get("accelerator") or {}
    cluster_count = cluster_accel.get("count") or accelerator_total(
        (cluster or {}).get("allocatable", {}).get("accelerators", {})
    )
    addresses = {str(device.get("pciAddress") or "").strip().lower() for device in cluster_accel.get("devices", [])} - {
        ""
    }
    configured = [value.strip() for value in os.environ.get("PRISM_GPU_PCI_ALLOWLIST", "").split(",") if value.strip()]
    target_node = os.environ.get("PRISM_K8S_TARGET_NODE", "").strip()
    node_names = {node.get("name") for node in (cluster or {}).get("nodes", [])} - {None, ""}
    if cluster and target_node and node_names and target_node not in node_names:
        errors.append(
            f"Selected cluster does not contain configured target node {target_node}; "
            "choose the cluster that owns the configured GPU allowlist."
        )
    invalid = [value for value in configured if value.lower() not in addresses] if addresses else []
    if cluster and invalid:
        errors.append(f"Configured GPU PCI address(es) are not present in the selected cluster: {', '.join(invalid)}")
    if cluster and configured and not addresses:
        warnings.append(
            f"Kubernetes did not return DRA PCI-address inventory; using the {len(configured)}-device allowlist "
            f"pinned to {target_node or 'the Lens target node'}."
        )
    devices = [value for value in configured if value.lower() in addresses] if addresses else configured
    scoped = len(devices) or cluster_count or machine_accel.get("count", 0)
    total = cluster_count or machine_accel.get("count", 0)
    basis = "total" if requested.get("resourceBasis") == "total" else "available"
    observed = requested.get("recommendationGpuCount")
    available = total if basis == "total" else scoped
    if (
        basis != "total"
        and isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isfinite(observed)
        and int(observed) == observed
        and observed >= 0
    ):
        available = min(scoped, observed)
    if basis == "total" and devices and total > len(devices):
        warnings.append(
            f"Recommendations include all {total} physical cards; scheduling remains restricted to the "
            f"{len(devices)}-card allowlist until its scope is expanded."
        )
    model_estimate = estimate_model(model)
    model_source = requested.get("modelSource") or ("shared-path" if requested.get("modelPath") else "huggingface")
    host_path = str(requested.get("modelPath") or "").strip()
    if host_path and (not host_path.startswith("/") or ".." in host_path.split("/") or re.search(r"[\r\n]", host_path)):
        errors.append("Shared model path must be an absolute worker-node path without .. segments or newlines.")
    pvc_name = str(requested.get("modelPvcClaimName") or "").strip()
    uses_pvc = model_source == "auto-cache" and bool(pvc_name)
    model_argument = "/model-cache" if model_source == "shared-path" and host_path else model
    dual = requested.get("guide") == "pd-disaggregation"
    claims, tp_by_role, guide_by_role, planned_by_role = {}, {}, {}, {}
    guide_replicas = planned_replicas = accelerators_per_replica = cpu_per_replica = memory_per_replica = (
        nic_requests
    ) = 0
    tensor_parallel = 1

    def replace(target, key, value, pointer, reason):
        if key in target and target[key] == value:
            return
        patch = {"op": "replace", "path": pointer, "value": deepcopy(value), "reason": reason}
        if key in target:
            patch["from"] = deepcopy(target[key])
        patches.append(patch)
        target[key] = value

    def selected_value(role, suffix, flat):
        return float(requested.get(role + suffix) or (requested.get(flat) if role == "serving" or not dual else 0) or 0)

    for document_index, document in enumerate(documents):
        if document.get("kind") != "ResourceClaimTemplate":
            continue
        role = resource_role(document)
        tp = selected_value(role, "TensorParallelSize", "tensorParallelSize")
        requests = document.get("spec", {}).get("spec", {}).get("devices", {}).get("requests", [])
        name = document.get("metadata", {}).get("name", document_index)
        if not requests:
            errors.append(f"ResourceClaimTemplate {name} has no DRA device requests.")
        for request_index, request in enumerate(requests):
            exactly = request.get("exactly")
            if not exactly:
                errors.append(f"ResourceClaimTemplate {name} uses an unsupported DRA request shape.")
                continue
            device_class = exactly.get("deviceClassName", "")
            pointer = f"/documents/{document_index}/spec/spec/devices/requests/{request_index}/exactly"
            if cluster and device_class and device_class not in cluster.get("deviceClasses", []):
                errors.append(f"DRA device class {device_class} is not available in the selected cluster.")
            if profile and device_class == profile.dranet_device_class:
                nic_requests += 1
                if "rdmaNicCount" in settings:
                    replace(
                        exactly,
                        "count",
                        settings["rdmaNicCount"],
                        pointer + "/count",
                        "apply requested NIC count independently of GPU topology",
                    )
            if not resolve_by_device_class(device_class):
                continue
            if tp > 0:
                replace(
                    exactly,
                    "count",
                    int(tp),
                    pointer + "/count",
                    "match accelerator claim to requested tensor parallel size",
                )
            claims[role] = max(claims.get(role, 0), tp or exactly.get("count", 0))
            if devices:
                deployment = profile.deployment if profile else None
                if (
                    not deployment
                    or not deployment.supports_pci_allowlist
                    or not deployment.pci_attribute_domain
                    or not deployment.pci_attribute
                ):
                    errors.append("Selected hardware does not support PCI selectors")
                    continue
                expression = (
                    f"device.attributes[{json.dumps(deployment.pci_attribute_domain)}]"
                    f"[{json.dumps(deployment.pci_attribute)}] in {json.dumps(devices)}"
                )
                replace(
                    exactly,
                    "selectors",
                    [*exactly.get("selectors", []), {"cel": {"expression": expression}}],
                    pointer + "/selectors",
                    "restrict allocation to configured devices present in the selected cluster",
                )

    for workload in workloads:
        spec = workload["spec"]
        role = workload["role"]
        slices = int(workload["slices"] or 1)
        spec_pointer = workload["spec_pointer"]
        pod_pointer = workload["pod_pointer"]
        if workload["kind"] == "Deployment" and spec.get("progressDeadlineSeconds", 0) < 1800:
            replace(
                spec,
                "progressDeadlineSeconds",
                1800,
                spec_pointer + "/progressDeadlineSeconds",
                "allow accelerator allocation and model initialization to complete",
            )
        guide_count = spec.get("replicas") or 1
        guide_replicas += guide_count * slices
        guide_by_role[role] = guide_by_role.get(role, 0) + guide_count * slices
        pod = workload["pod"]
        if profile and profile.deployment.node_selector:
            replace(
                pod,
                "nodeSelector",
                {**profile.deployment.node_selector, **pod.get("nodeSelector", {})},
                pod_pointer + "/nodeSelector",
                "apply hardware scheduling defaults",
            )
        per_replica = cpu = memory = 0
        for container_index, container in enumerate(pod.get("containers", [])):
            container_pointer = pod_pointer + f"/containers/{container_index}"
            if container.get("name") == "modelserver":
                image = requested.get("runtimeImage") or (profile.deployment.runtime_image if profile else None)
                if image:
                    replace(
                        container,
                        "image",
                        pin_model_server_image(image),
                        container_pointer + "/image",
                        "apply selected runtime image",
                    )
                for entry in container.get("env", []):
                    secret = entry.get("valueFrom", {}).get("secretKeyRef", {})
                    if secret.get("name") == "llm-d-hf-token":
                        secret["optional"] = True
                if host_path or uses_pvc:
                    volumes = pod.setdefault("volumes", [])
                    volume = next((value for value in volumes if value.get("name") == "model-cache"), None)
                    if volume is None:
                        volume = {"name": "model-cache"}
                        volumes.append(volume)
                    volume.pop("hostPath", None)
                    volume.pop("persistentVolumeClaim", None)
                    volume.update(
                        {"persistentVolumeClaim": {"claimName": pvc_name}}
                        if uses_pvc
                        else {"hostPath": {"path": host_path, "type": "DirectoryOrCreate"}}
                    )
                    mounts = container.setdefault("volumeMounts", [])
                    if not any(value.get("name") == "model-cache" for value in mounts):
                        mounts.append(
                            {
                                "name": "model-cache",
                                "mountPath": "/model-cache",
                                "readOnly": model_source == "shared-path",
                            }
                        )
                    environment = container.setdefault("env", [])
                    if model_source != "auto-cache":
                        environment[:] = [
                            value
                            for value in environment
                            if value.get("valueFrom", {}).get("secretKeyRef", {}).get("name") != "llm-d-hf-token"
                        ]
                    injected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
                    if model_source == "auto-cache":
                        injected.update(HF_HOME="/model-cache", HF_HUB_DISABLE_XET="1")
                    for key, value in injected.items():
                        entry = next((entry for entry in environment if entry.get("name") == key), None)
                        if entry is not None:
                            entry["value"] = value
                        else:
                            environment.append({"name": key, "value": value})
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
                value = os.environ.get(key) or os.environ.get(key.lower())
                if not value:
                    continue
                if "env" not in container:
                    replace(container, "env", [], container_pointer + "/env", "add runtime environment")
                environment = container["env"]
                entry = next((entry for entry in environment if entry.get("name") == key), None)
                if entry is not None:
                    replace(
                        entry,
                        "value",
                        value,
                        container_pointer + f"/env/{environment.index(entry)}/value",
                        "apply deployment network proxy",
                    )
                else:
                    environment.append({"name": key, "value": value})
                    patches.append(
                        {
                            "op": "add",
                            "path": container_pointer + "/env/-",
                            "value": {"name": key, "value": value},
                            "reason": "apply deployment network proxy",
                        }
                    )
            resources = container.get("resources") or {}
            limits = resources.get("limits") or resources.get("requests") or {}
            per_replica += next((float(value) for key, value in limits.items() if resolve_by_resource(key)), 0)
            cpu += quantity_number(limits.get("cpu"))
            memory += quantity_number(limits.get("memory"))
        for index, container in enumerate(pod.get("initContainers", [])):
            if container.get("name") == "routing-proxy" and dual:
                location = pod_pointer + f"/initContainers/{index}"
                replace(
                    container,
                    "image",
                    router_pd_sidecar_image(),
                    location + "/image",
                    "pin routing proxy to the Guide release",
                )
                if container.get("imagePullPolicy") == "Always":
                    replace(
                        container,
                        "imagePullPolicy",
                        "IfNotPresent",
                        location + "/imagePullPolicy",
                        "avoid mutable nightly proxy pulls",
                    )
        per_replica = max(per_replica, claims.get(role) or claims.get("serving", 0))
        accelerators_per_replica, cpu_per_replica, memory_per_replica = (
            max(accelerators_per_replica, per_replica),
            max(cpu_per_replica, cpu),
            max(memory_per_replica, memory),
        )
        replicas = selected_value(role, "Replicas", "replicas")
        allocatable = (cluster or {}).get("allocatable") or {}
        capacity = math.inf
        if cpu > 0:
            capacity = min(
                capacity, math.floor((allocatable.get("cpu") or machine.get("cpu", {}).get("logicalCores", 0)) / cpu)
            )
        if memory > 0:
            capacity = min(capacity, math.floor((allocatable.get("memoryGiB") or machine.get("memoryGiB", 0)) / memory))
        capacity = max(0, capacity)
        if replicas > capacity:
            errors.append(
                f"{role} replicas={replicas:g} exceed selected cluster CPU/RAM capacity ({capacity}); "
                "reduce the topology in the configuration editor."
            )
        if available > 0 and per_replica > 0:
            capacity = min(capacity, math.floor(available / per_replica))
        if not math.isfinite(capacity):
            capacity = guide_count
        planned_count = int(replicas) if replicas > 0 else min(guide_count, capacity)
        if planned_count != guide_count:
            replace(
                spec,
                "replicas",
                planned_count,
                workload["replicas_pointer"],
                "apply requested topology" if replicas > 0 else "fit detected machine or cluster capacity",
            )
        planned_replicas += planned_count * slices
        planned_by_role[role] = planned_by_role.get(role, 0) + planned_count * slices

    def walk(value, pointer=""):
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}")
        elif isinstance(value, dict):
            for key, child in list(value.items()):
                if key == "model" and isinstance(child, str):
                    replace(
                        value,
                        key,
                        model_argument,
                        pointer + "/" + key,
                        "serve selected shared model directory" if host_path else "replace guide model",
                    )
                elif key == "llm-d.ai/model" and isinstance(child, str):
                    replace(value, key, model.rsplit("/", 1)[-1], pointer + "/" + key, "update model label")
                else:
                    walk(child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"))

    walk(documents, "/documents")
    try:
        overrides, applied = runtime_overrides(requested), set()
        for workload in workloads:
            role = workload["role"]
            pod_pointer = workload["pod_pointer"]
            for container_index, container in enumerate(workload["pod"].get("containers", [])):
                if container.get("name") != "modelserver":
                    continue
                selected = [item for item in overrides if item["target"] in {"both", role if dual else "decode"}]
                before = deepcopy(container)
                tp = selected_value(role, "TensorParallelSize", "tensorParallelSize")
                if profile and tp > 0:
                    for group in ("limits", "requests"):
                        for key in container.get("resources", {}).get(group, {}):
                            if resolve_by_resource(key) == profile:
                                container["resources"][group][key] = str(int(tp))
                configure_model_server(
                    container, model_argument, int(tp), model, requested.get("guide") == "precise-prefix-cache-routing"
                )
                if tp > 0:
                    tensor_parallel = int(tp)
                    tp_by_role[role] = int(tp)
                apply_runtime_overrides(container, selected)
                if requested.get("guide") == "precise-prefix-cache-routing":
                    events = json.loads(runtime_argument(container, "kv-events-config") or "{}")
                    topic = events.get("topic")
                    if (
                        not isinstance(topic, str)
                        or not re.match(r"^kv@[^@]+@.+$", topic)
                        or "@".join(topic.split("@")[2:]) != model
                    ):
                        raise ValueError("Precise KV event topic must match the configured model.")
                if "cacheCpuGiB" in settings:
                    if any(item["name"] in {"kv-transfer-config", "LMCACHE_MAX_LOCAL_CPU_SIZE"} for item in selected):
                        raise ValueError("CPU cache capacity conflicts with a custom connector/cache override.")
                    configure_cpu_cache(container, settings["cacheCpuGiB"])
                for key in ("command", "args", "env"):
                    if before.get(key) != container.get(key):
                        patch = {
                            "op": "add" if key not in before else "replace",
                            "path": f"{pod_pointer}/containers/{container_index}/{key}",
                            "value": deepcopy(container[key]),
                            "reason": "apply explicit runtime configuration",
                        }
                        if key in before:
                            patch["from"] = before[key]
                        patches.append(patch)
                applied.update(id(item) for item in selected)
        for item in overrides:
            if id(item) not in applied:
                errors.append(f"No model server matched runtime override {item['target']}:{item['name']}.")
    except (ValueError, TypeError, KeyError):
        logger.warning("Invalid model-server runtime configuration", exc_info=True)
        errors.append("Invalid model-server runtime configuration; check arguments, environment and cache settings.")
    if "rdmaNicCount" in settings and not nic_requests:
        errors.append("The selected Guide has no supported RDMA NIC request.")
    if not any(
        container.get("name") == "modelserver"
        for workload in workloads
        for container in workload["pod"].get("containers", [])
    ):
        errors.append("The selected Guide has no identifiable modelserver workload.")
    if not guide_replicas:
        warnings.append("No workload replica field was found; replica capacity was inherited from the guide unchanged.")
    for role, count in claims.items():
        if tp_by_role.get(role) and count and tp_by_role[role] != count:
            warnings.append(
                f"{role} TP={tp_by_role[role]}, while the manifest requests {count:g} accelerator(s) per replica."
            )
    if (
        tensor_parallel > 1
        and any(document.get("kind") == "ResourceClaimTemplate" for document in documents)
        and not claims
    ):
        errors.append(
            f"Requested TP={tensor_parallel}, but the selected Guide has no usable DRA ResourceClaimTemplate."
        )
    reported = cluster_accel.get("memoryGiB") or machine_accel.get("memoryGiB") or 0
    accelerator_model = cluster_accel.get("model") or machine_accel.get("model") or ""
    accelerator_memory = reported or next(
        (
            entry["gib"]
            for registered in all_profiles()
            for entry in registered.planning.memory_models
            if re.search(entry["pattern"], accelerator_model, re.IGNORECASE)
        ),
        0,
    )
    memory_source = "cluster-reported" if reported else "known-model-fallback" if accelerator_memory else None
    if memory_source == "known-model-fallback":
        warnings.append(
            f"Accelerator memory was not reported by DRA; using the known {accelerator_memory} GiB "
            f"capacity for {accelerator_model}."
        )
    weight_memory = model_estimate["estimatedWeightMemoryGiB"]
    recommended_tp = (
        max(1, math.ceil(weight_memory / (accelerator_memory * 0.9))) if weight_memory and accelerator_memory else None
    )
    if recommended_tp:
        roles = ("prefill", "decode") if dual else ("serving",)
        for role in roles:
            role_tp = (
                requested.get(role + "TensorParallelSize") or tp_by_role.get(role) or 1
                if dual
                else requested.get("tensorParallelSize") or tensor_parallel
            )
            label = role.capitalize() + " TP" if dual else "TP"
            if float(role_tp) < recommended_tp:
                errors.append(
                    f"{label}={role_tp} is too small for the estimated {weight_memory} GiB model footprint "
                    f"on {accelerator_memory} GiB accelerators. Recommended TP is at least {recommended_tp}."
                )
            elif weight_memory > accelerator_memory * float(role_tp):
                errors.append(
                    f"Estimated model weight memory {weight_memory} GiB exceeds {label}={role_tp} "
                    f"accelerator memory {accelerator_memory * float(role_tp)} GiB."
                )
    else:
        warnings.append(model_estimate["assumption"])
    runtime_class = runtime_class_for_accelerator(cluster, requested.get("accelerator", ""))
    if runtime_class:
        for workload in workloads:
            pod = workload["pod"]
            if pod.get("runtimeClassName"):
                continue
            pod["runtimeClassName"] = runtime_class
            patches.append(
                {
                    "op": "add",
                    "path": workload["pod_pointer"] + "/runtimeClassName",
                    "value": runtime_class,
                    "reason": (
                        f"run under the {runtime_class} container runtime so the container can access the accelerator"
                    ),
                }
            )
    demand = validate_manifest_capacity(documents, available, basis, errors)
    if cluster and not cluster.get("preflightPassed"):
        errors.append("Kubernetes preflight did not pass.")
    source = (
        {"mode": model_source, "path": host_path, "id": model}
        if host_path
        else (
            {"mode": model_source, "pvcClaimName": pvc_name, "id": model}
            if uses_pvc
            else {"mode": "huggingface", "id": model}
        )
    )
    return {
        "documents": documents,
        "planningPatch": patches,
        "validation": {
            "status": "invalid" if errors else "warnings" if warnings else "valid",
            "errors": errors,
            "warnings": warnings,
        },
        "summary": {
            "guideReplicas": guide_replicas,
            "plannedReplicas": planned_replicas or guide_replicas or None,
            "tensorParallelSize": tensor_parallel,
            "tensorParallelSizeByRole": tp_by_role,
            "guideReplicasByRole": guide_by_role,
            "plannedReplicasByRole": planned_by_role,
            "acceleratorsPerReplica": accelerators_per_replica,
            "cpuPerReplica": cpu_per_replica,
            "memoryPerReplicaGiB": memory_per_replica,
            "availableAccelerators": available,
            "resourceBasis": basis,
            "totalAccelerators": total,
            "acceleratorSource": "validated-pci-allowlist"
            if devices
            else cluster_accel.get("source") or "machine-discovery",
            "selectedDevicePciAddresses": devices,
            "modelEstimate": model_estimate,
            "modelSource": source,
            "acceleratorMemoryGiB": accelerator_memory or None,
            "acceleratorMemorySource": memory_source,
            "recommendedTensorParallelSize": recommended_tp,
            "maxRecommendedReplicas": math.floor(available / (recommended_tp or tensor_parallel)),
            "requestedAccelerators": demand,
        },
    }


def deployment_contract(documents: list[dict], guide: str) -> dict:
    workloads = workload_roles(documents)
    readiness = list(dict.fromkeys(workload["name"] for workload in workloads if workload["name"]))
    if not readiness:
        raise GuideError("Generated Guide YAML contains no model-server workload", 422)
    service = next(
        (
            document
            for document in documents
            if document.get("kind") == "Service"
            and document.get("metadata", {}).get("name")
            and document.get("spec", {}).get("ports")
        ),
        None,
    )
    if service is None:
        serving = next(
            (workload for workload in workloads if workload["role"] in {"decode", "serving"}),
            workloads[0],
        )
        container_port = next(
            (
                port
                for container in serving["pod"].get("containers", [])
                for port in container.get("ports", [])
                if float(port.get("containerPort") or 0) > 0
            ),
            {},
        )
        port = container_port.get("containerPort") or 8000
        if serving["kind"] == "Deployment":
            selector = (serving["spec"].get("selector") or {}).get("matchLabels") or serving["pod_labels"]
        else:
            selector = {
                key: serving["pod_labels"][key]
                for key in ("llm-d.ai/guide", "llm-d.ai/role")
                if serving["pod_labels"].get(key)
            } or serving["pod_labels"]
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{guide}-modelserver", "labels": {"llm-d.ai/guide": guide}},
            "spec": {
                "selector": selector,
                "ports": [{"name": "http", "port": port, "targetPort": container_port.get("name") or port}],
            },
        }
        documents.append(service)
    return {
        "readinessDeployments": readiness,
        "endpoint": {
            "protocol": "http",
            "serviceName": service["metadata"]["name"],
            "port": service["spec"]["ports"][0]["port"],
        },
    }


@router.get("/catalog")
async def catalog(request: Request):
    from .guide_sources import catalog_from_paths, load_guide_paths

    try:
        source = await load_guide_paths(request, dict(request.query_params))
        return {
            "repository": os.environ.get("LLM_D_GUIDES_REPOSITORY", "llm-d/llm-d"),
            "ref": source["ref"],
            "guides": catalog_from_paths(source["paths"]),
        }
    except DomainError:
        raise
    except GuideError as error:
        logger.warning("Guide catalog request failed", exc_info=True)
        return JSONResponse({"error": error.public_message}, status_code=502)
    except (OSError, ValueError):
        logger.exception("Guide catalog could not be loaded")
        return JSONResponse({"error": "Guide catalog could not be loaded"}, status_code=502)


@router.post("/prepare")
async def prepare(request: Request):
    from .guide_runtime import guide_settings_capabilities
    from .guide_sources import load_manifests

    body = await request.json()
    if not all(isinstance(body.get(key), str) and body[key].strip() for key in ("guide", "accelerator", "modelServer")):
        return JSONResponse({"error": "Guide, accelerator and model server are required"}, status_code=400)
    try:
        manifest = await load_manifests(request, body)
        documents = [document for document in yaml.safe_load_all(manifest["rendered"]) if isinstance(document, dict)]
        profile = profile_for_key(body["accelerator"])
        return {
            "ready": True,
            "commit": manifest["source"]["commit"],
            "capabilities": guide_settings_capabilities(documents, profile.dranet_device_class if profile else None),
        }
    except DomainError:
        raise
    except Exception:
        logger.exception("Guide preparation failed")
        return JSONResponse({"error": "Guide preparation failed; retry when generating configuration"}, status_code=502)


@router.post("/plan")
async def plan(request: Request):
    from .guide_bundle import build_guide_deployment_bundle
    from .guide_sources import (
        discover_local_cluster,
        discover_remote_environment,
        load_selected_manifests,
        machine_profile,
        validate_planning_session,
    )

    started, timings = time.monotonic(), {}

    def finish(payload, status=200):
        timings["total"] = round((time.monotonic() - started) * 1000)
        return JSONResponse(
            payload,
            status_code=status,
            headers={"Server-Timing": ", ".join(f"{key};dur={value}" for key, value in timings.items())},
        )

    async def measure(name, work):
        began = time.monotonic()
        try:
            return await work
        finally:
            timings[name] = round((time.monotonic() - began) * 1000)

    try:
        body = await request.json()
        guide, accelerator, model_server, model = (
            str(body.get(key) or "").strip() for key in ("guide", "accelerator", "modelServer", "model")
        )
        if not all((guide, accelerator, model_server, model)):
            return finish({"error": "Guide, accelerator, model server, and model are required"}, 400)
        environment = body.get("environment") or {}
        mode = environment.get("kubernetesMode")
        mode = mode if mode in {"auto", "required", "disabled"} else "auto"
        session_id = str(body.get("clusterSessionId") or "").strip()
        if session_id:
            await validate_planning_session(request, session_id)
        remote = (
            await discover_remote_environment(environment.get("target"), environment.get("credentials"))
            if environment.get("mode") == "remote"
            else None
        )
        try:
            detected_cluster = (
                None
                if mode == "disabled"
                else remote["cluster"]
                if remote
                else await measure("cluster_checks", discover_local_cluster(request, session_id or None))
            )
        except ValueError as error:
            if session_id:
                raise GuideError("The selected cluster session is no longer active", 409) from error
            raise
        if mode == "required" and not (detected_cluster or {}).get("preflightPassed"):
            raise GuideError(
                "No accessible Kubernetes cluster was found in the selected machine's current kubectl context"
            )
        resolved = resolve_accelerator_variant(accelerator, detected_cluster)
        body = dict(body, guide=guide, accelerator=resolved, modelServer=model_server)
        manifest = await measure("guide_source", load_selected_manifests(request, body))
        machine = remote["machine"] if remote else await machine_profile(not detected_cluster)
        if (detected_cluster or {}).get("accelerator"):
            machine = dict(machine, accelerator=detected_cluster["accelerator"])
        render_started = time.monotonic()
        documents = [document for document in yaml.safe_load_all(manifest["rendered"]) if document]
        planned = plan_documents(documents, model, machine, detected_cluster, body)
        if accelerator != resolved:
            planned["validation"]["warnings"].append(
                f'Guide accelerator variant corrected from "{accelerator}" to "{resolved}" '
                "to match the selected cluster's hardware."
            )
        patches = body.get("customPatches") if isinstance(body.get("customPatches"), list) else []
        applied = apply_custom_patches(
            planned["documents"], patches, planned["validation"]["errors"], planned["validation"]["warnings"]
        )
        planned["summary"]["requestedAccelerators"] = validate_manifest_capacity(
            planned["documents"],
            planned["summary"]["availableAccelerators"],
            planned["summary"]["resourceBasis"],
            planned["validation"]["errors"],
        )
        validation = planned["validation"]
        validation["status"] = "invalid" if validation["errors"] else "warnings" if validation["warnings"] else "valid"
        deployment = deployment_contract(planned["documents"], guide)
        bundle = None
        if not validation["errors"]:
            settings = normalize_guide_settings(guide, body.get("guideSettings", {}))
            servers = [
                container
                for workload in workload_roles(planned["documents"])
                for container in workload["pod"].get("containers", [])
                if container.get("name") == "modelserver"
            ]
            block_sizes = [
                int(value) for container in servers if (value := runtime_argument(container, "block-size")) is not None
            ]
            if guide == "precise-prefix-cache-routing" and (
                len(block_sizes) != len(servers) or len(set(block_sizes)) != 1
            ):
                raise ValueError("Precise routing requires matching block sizes on all model servers.")
            bundle = await measure(
                "bundle",
                build_guide_deployment_bundle(
                    guide,
                    manifest,
                    model,
                    resolved,
                    block_sizes[0] if block_sizes else None,
                    settings.get("routerValues", ""),
                ),
            )
        timings["render"] = round((time.monotonic() - render_started) * 1000)
        response = {
            "status": "VERIFIED"
            if (detected_cluster or {}).get("preflightPassed") and not validation["errors"]
            else "PLANNED",
            "timings": timings,
            "source": {
                **manifest["source"],
                "guide": guide,
                "accelerator": resolved,
                "modelServer": manifest.get("modelServer") or model_server,
                "variant": manifest["variant"],
                "files": [file["path"] for file in manifest["files"]],
            },
            "environment": {
                "machineMode": "remote" if environment.get("mode") == "remote" else "current",
                "kubernetesMode": mode,
                "kubernetesDetected": bool((detected_cluster or {}).get("preflightPassed")),
                "clusterSessionId": session_id or None,
            },
            "baseManifest": manifest["files"],
            "machineProfile": machine,
            "clusterProfile": detected_cluster,
            "modelConfig": planned["summary"]["modelEstimate"],
            "planningPatch": planned["planningPatch"],
            "appliedCustomPatches": applied,
            "plannedDeployment": {
                "format": "yaml",
                "content": yaml.safe_dump_all(planned["documents"], sort_keys=False),
            }
            if not validation["errors"]
            else None,
            "deployment": deployment,
            "validation": validation,
            "summary": planned["summary"],
        }
        if bundle is not None:
            response["deploymentBundle"] = bundle
        return finish(response)
    except DomainError:
        raise
    except (GuideError, RemoteError) as error:
        logger.warning("Guide planning request failed", exc_info=True)
        return finish({"error": error.public_message}, error.status)
    except Exception:
        logger.exception("Guide planning failed")
        return finish({"error": "Guide planning failed"}, 500)
