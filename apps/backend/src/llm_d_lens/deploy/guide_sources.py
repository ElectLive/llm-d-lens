"""Read pinned Guide inputs and discover planning hardware without mutating a cluster."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from urllib.parse import quote

import httpx

from llm_d_lens.auth.access import current_principal, require_cluster_access
from llm_d_lens.auth.context import require_route_permission
from llm_d_lens.cluster import registry
from llm_d_lens.cluster.deployment_source import resolve_cluster_deployment_source
from llm_d_lens.cluster.sessions import require_active_session
from llm_d_lens.hardware.registry import all_profiles
from llm_d_lens.hardware.resolver import resolve_by_device_class, resolve_by_resource
from llm_d_lens.infrastructure.hostos import current_host
from llm_d_lens.infrastructure.kubernetes import run_kubectl
from llm_d_lens.infrastructure.process import ShellClient

from .guide_planning import GuideError, overlay_root, quantity_number, select_manifest_paths
from .remote_deploy import execute_remote_inspection


async def command_output(argv: list[str], timeout=4) -> str:
    try:
        result = await ShellClient().run(argv, timeout=timeout)
        return result.stdout.strip() if result.ok else ""
    except (OSError, TimeoutError):
        return ""


async def require_internal_permission(request, path: str):
    from starlette.requests import Request

    await require_route_permission(Request(dict(request.scope, path=path, method="GET")))


async def validate_planning_session(request, session_id: str):
    await require_internal_permission(request, f"/api/cluster-overview/sessions/{session_id}")
    try:
        session = require_active_session(session_id)
    except ValueError as error:
        raise GuideError("The selected cluster session is no longer active", 409) from error
    require_cluster_access(current_principal(request), session.server_id)
    return session


async def load_guide_paths(request, selection: dict) -> dict:
    cluster_id = str(selection.get("clusterId") or "").strip()
    if selection.get("clusterSessionId"):
        session_id = selection["clusterSessionId"]
        await require_internal_permission(request, f"/api/cluster-overview/sessions/{session_id}")
        try:
            session = require_active_session(session_id)
        except ValueError as error:
            raise GuideError("Cluster session is no longer active", 404) from error
        if cluster_id and cluster_id != session.server_id:
            raise GuideError("Cluster and session do not match", 409)
        cluster_id = session.server_id
    if not cluster_id:
        raise GuideError("Select a cluster and download its Software Versions first")
    await require_internal_permission(request, "/api/cluster/clusters")
    require_cluster_access(current_principal(request), cluster_id)
    cluster = registry.get_cluster(cluster_id)
    if cluster is None:
        raise GuideError("Selected cluster was not found", 404)
    try:
        resolve_cluster_deployment_source(cluster)
        if not cluster.llm_d_repo_path:
            raise ValueError
    except ValueError as error:
        raise GuideError("Download the selected cluster's llm-d Software Versions first", 409) from error
    root = cluster.llm_d_repo_path
    commit, listing = await asyncio.gather(
        command_output(["git", "-C", root, "rev-parse", "HEAD"]),
        command_output(["git", "-C", root, "ls-files", "guides"]),
    )
    paths = [
        path for path in listing.splitlines() if re.search(r"^guides/.+(?:\.ya?ml|Kustomization)$", path, re.IGNORECASE)
    ]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not paths:
        raise GuideError("The cluster llm-d source has no usable guides; download its Software Versions", 409)
    return {"paths": paths, "commit": commit, "localRoot": root, "ref": cluster.llm_d_ref or ""}


def catalog_from_paths(paths: list[str]) -> list[dict]:
    guides = {}
    for path in paths:
        if not re.search(r"(?:^|/)(?:kustomization\.ya?ml|Kustomization)$", path, re.IGNORECASE):
            continue
        for profile in all_profiles():
            template = overlay_root(profile, "__GUIDE__", "__SERVER__")
            pattern = re.escape(template).replace("__GUIDE__", "([^/]+)").replace("__SERVER__", "([^/]+)")
            match = re.fullmatch(pattern + r"/(.*)", path)
            if not match:
                continue
            guide, server, suffix = match.groups()
            variant = suffix.rsplit("/", 1)[0] if "/" in suffix else "."
            guides.setdefault(guide, {}).setdefault(profile.upstream_variant, {}).setdefault(server, set()).add(variant)
            break
    return [
        {
            "id": guide,
            "label": " ".join(word[:1].upper() + word[1:] for word in guide.split("-")),
            "accelerators": [
                {
                    "id": accelerator,
                    "modelServers": [
                        {"id": server, "variants": sorted(variants, key=lambda value: (value != "base", value))}
                        for server, variants in sorted(servers.items())
                    ],
                }
                for accelerator, servers in sorted(accelerators.items())
            ],
        }
        for guide, accelerators in sorted(guides.items())
    ]


async def load_manifests(request, body: dict) -> dict:
    source = await load_guide_paths(request, body)
    selected = select_manifest_paths(
        source["paths"], body["guide"], body["accelerator"], body["modelServer"], str(body.get("guideVariant") or "")
    )
    root = Path(source["localRoot"])
    rendered = await command_output(["kubectl", "kustomize", str(root / selected["guidePath"])], 60)
    if not rendered:
        raise GuideError("Could not render the selected official guide with kubectl kustomize", 502)
    files = [{"path": path, "content": (root / path).read_text()} for path in selected["selected"]]
    return {
        "modelServer": selected["modelServer"],
        "localRoot": str(root),
        "variant": selected["variant"],
        "files": files,
        "rendered": rendered,
        "source": {
            "repository": os.environ.get("LLM_D_GUIDES_REPOSITORY", "llm-d/llm-d"),
            "requestedRef": source["ref"],
            "commit": source["commit"],
        },
    }


async def fetch_source(url: str, accept="application/vnd.github+json") -> str:
    headers = {"Accept": accept, "User-Agent": "llm-d-prism"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def github_content_url(repository: str, path: str, commit: str) -> str:
    return f"https://api.github.com/repos/{repository}/contents/{quote(path, safe='/')}?ref={quote(commit, safe='')}"


async def load_selected_manifests(request, body: dict) -> dict:
    source = body.get("source") or {"mode": "official"}
    mode = source.get("mode", "official")
    if mode == "official":
        return await load_manifests(request, body)
    if mode == "local":
        requested = Path(str(source.get("path") or "")).expanduser().resolve()
        roots = [
            Path(value).resolve()
            for value in os.environ.get("PRISM_GUIDE_LOCAL_ROOTS", os.getcwd()).split(os.pathsep)
            if value
        ]
        if not any(requested.is_relative_to(root) for root in roots):
            raise GuideError("Local YAML path must be under an allowed root: " + ", ".join(map(str, roots)), 403)
        if not requested.exists():
            raise GuideError("Local YAML/Kustomize path does not exist", 404)
        kustomization = (
            next(
                (
                    requested / name
                    for name in ("kustomization.yaml", "kustomization.yml")
                    if (requested / name).is_file()
                ),
                None,
            )
            if requested.is_dir()
            else (requested if requested.name.lower() in {"kustomization.yaml", "kustomization.yml"} else None)
        )
        input_file = kustomization or requested
        if not input_file.is_file() or input_file.suffix.lower() not in {".yaml", ".yml"}:
            raise GuideError("Local source must be a YAML file or a directory containing kustomization.yaml")
        content = input_file.read_text()
        rendered = (
            await command_output(["kubectl", "kustomize", str(input_file.parent)], 60) if kustomization else content
        )
        root, commit = await asyncio.gather(
            command_output(["git", "-C", str(input_file.parent), "rev-parse", "--show-toplevel"]),
            command_output(["git", "-C", str(input_file.parent), "rev-parse", "HEAD"]),
        )
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            commit = hashlib.sha1(rendered.encode(), usedforsecurity=False).hexdigest()
        return {
            "variant": "custom-local",
            "localRoot": root or str(input_file.parent),
            "files": [{"path": str(input_file), "content": content}],
            "rendered": rendered,
            "source": {
                "repository": root or str(input_file.parent),
                "requestedRef": str(source.get("ref") or "working-tree"),
                "commit": commit,
            },
        }
    if mode != "remote":
        raise GuideError("Guide source mode must be official, local, or remote")
    repository = str(source.get("repository") or "").strip().rstrip("/").removesuffix(".git")
    match = re.fullmatch(r"(?:https://github\.com/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", repository)
    if not match or any(part in {".", ".."} for part in match[1].split("/")):
        raise GuideError(
            "Remote repository must be a GitHub owner/repository name or https://github.com/owner/repository URL"
        )
    repository = match[1]
    path = str(source.get("path") or "").strip().lstrip("/").replace("\\", "/")
    if not path or ".." in path.split("/"):
        raise GuideError("Repository YAML path must be a relative path without .. segments")
    ref = str(source.get("ref") or source.get("branch") or "main").strip()
    if not ref or re.search(r"[\s\x00-\x1f]", ref):
        raise GuideError("Remote branch/ref is invalid")
    commit = json.loads(
        await fetch_source(f"https://api.github.com/repos/{repository}/commits/{quote(ref, safe='')}")
    ).get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise GuideError("Remote branch/ref could not be resolved to a commit", 404)
    content = await fetch_source(github_content_url(repository, path, commit), "application/vnd.github.raw+json")
    rendered = (
        await command_output(
            ["kubectl", "kustomize", f"https://github.com/{repository}//{str(Path(path).parent)}?ref={commit}"], 60
        )
        if Path(path).name.lower() in {"kustomization.yaml", "kustomization.yml"}
        else content
    )
    return {
        "variant": "custom-remote",
        "files": [{"path": path, "content": content}],
        "rendered": rendered,
        "source": {"repository": repository, "requestedRef": ref, "commit": commit},
    }


def accelerator_output(profile, output: str) -> dict | None:
    if not output.strip():
        return None
    settings = profile.planning.discovery
    if settings.get("format") == "csv":
        devices = [
            {"model": row[0].strip(), "memoryGiB": float(row[1]) / float(settings.get("memory_divisor", 1))}
            for row in csv.reader(output.strip().splitlines())
            if len(row) >= 2
        ]
        return {
            "vendor": profile.vendor,
            "model": devices[0]["model"] if devices else None,
            "count": len(devices),
            "memoryGiB": devices[0]["memoryGiB"] if devices else None,
            "devices": devices,
        }
    match = re.search(settings["model_pattern"], output, re.IGNORECASE) if settings.get("model_pattern") else None
    return {
        "vendor": profile.vendor,
        "model": match[1].strip() if match else profile.display_name,
        "count": len(re.findall(settings["device_pattern"], output, re.IGNORECASE))
        if settings.get("device_pattern")
        else 0,
        "raw": output[:2000],
    }


async def machine_profile(inspect=True) -> dict:
    accelerator = {"vendor": "unknown" if inspect else "none", "count": 0, "devices": []}
    if inspect:
        for profile in all_profiles():
            argv = profile.planning.discovery.get("command") or []
            if argv and (detected := accelerator_output(profile, await command_output(argv))):
                accelerator = detected
                break
    machine = current_host().machine_info()
    return {
        **machine,
        "accelerator": accelerator,
    }


def cluster_profile(
    version: dict, nodes: dict, device_classes=None, resource_slices=None, runtime_classes=None
) -> dict:
    node_profiles, totals = [], {"cpu": 0, "memoryGiB": 0, "accelerators": {}}
    for node in nodes.get("items", []):
        allocatable = node.get("status", {}).get("allocatable", {})
        node_profiles.append(
            {
                "name": node.get("metadata", {}).get("name"),
                "labels": node.get("metadata", {}).get("labels", {}),
                "taints": node.get("spec", {}).get("taints", []),
                "allocatable": allocatable,
            }
        )
        totals["cpu"] += quantity_number(allocatable.get("cpu"))
        totals["memoryGiB"] += quantity_number(allocatable.get("memory"))
        for key, value in allocatable.items():
            if resolve_by_resource(key):
                totals["accelerators"][key] = totals["accelerators"].get(key, 0) + quantity_number(value)
    devices, slices = [], []
    for item in (resource_slices or {}).get("items", []):
        spec = item.get("spec", {})
        slices.append(
            {
                "name": item.get("metadata", {}).get("name"),
                "nodeName": spec.get("nodeName"),
                "driver": spec.get("driver"),
                "deviceCount": len(spec.get("devices", [])),
            }
        )
        if not resolve_by_device_class(spec.get("driver", "")):
            continue
        for device in spec.get("devices", []):
            attributes = device.get("attributes", {})
            memory = device.get("capacity", {}).get("memory", 0)
            devices.append(
                {
                    "name": device.get("name"),
                    "driver": spec.get("driver"),
                    "model": attributes.get("model", {}).get("string") or attributes.get("family", {}).get("string"),
                    "family": attributes.get("family", {}).get("string"),
                    "pciAddress": attributes.get("pciAddress", {}).get("string")
                    or attributes.get("resource.kubernetes.io/pciBusID", {}).get("string"),
                    "memoryGiB": quantity_number(memory.get("value") if isinstance(memory, dict) else memory),
                    "health": attributes.get("health", {}).get("string"),
                }
            )
    accelerator = None
    if devices:
        accelerator = {
            "vendor": resolve_by_device_class(devices[0]["driver"]).vendor,
            "model": devices[0]["model"],
            "count": len(devices),
            "memoryGiB": devices[0]["memoryGiB"] or None,
            "devices": devices,
            "source": "kubernetes-dra",
        }
    return {
        "version": version.get("serverVersion", {}).get("gitVersion"),
        "nodes": node_profiles,
        "allocatable": totals,
        "accelerator": accelerator,
        "deviceClasses": [item.get("metadata", {}).get("name") for item in (device_classes or {}).get("items", [])],
        "runtimeClasses": [
            {"name": item.get("metadata", {}).get("name"), "handler": item.get("handler")}
            for item in (runtime_classes or {}).get("items", [])
        ],
        "resourceSlices": slices,
        "preflightPassed": True,
    }


async def discover_local_cluster(request, session_id=None):
    if session_id:
        session = require_active_session(session_id)
        require_cluster_access(current_principal(request), session.server_id)
        await require_internal_permission(request, f"/api/cluster/sessions/{session_id}/planning-discovery/nodes")

    async def query(argv):
        try:
            result = await run_kubectl(argv, timeout=10, cluster_id=session_id)
            return json.loads(result.stdout) if result.ok else None
        except (OSError, TimeoutError, ValueError):
            return None

    version, nodes, classes, slices, runtimes = await asyncio.gather(
        *[
            query(argv)
            for argv in [
                ["version", "-o", "json"],
                ["get", "nodes", "-o", "json"],
                ["get", "deviceclasses.resource.k8s.io", "-o", "json"],
                ["get", "resourceslices.resource.k8s.io", "-o", "json"],
                ["get", "runtimeclasses.node.k8s.io", "-o", "json"],
            ]
        ]
    )
    return cluster_profile(version, nodes, classes, slices, runtimes) if version and nodes else None


def hardware_inspection_commands() -> list[str]:
    return [
        f"printf '__PRISM_HARDWARE_{index}__\\n'; "
        f"({shlex.join(profile.planning.discovery['command'])} 2>/dev/null || true)"
        for index, profile in enumerate(all_profiles())
        if profile.planning.discovery.get("command")
    ]


async def discover_remote_environment(target, credentials):
    commands = [
        'printf "__PRISM_HOST__\\n"; hostname',
        'printf "__PRISM_PLATFORM__\\n"; uname -srm',
        'printf "__PRISM_CPU__\\n"; (lscpu -J 2>/dev/null || true)',
        'printf "__PRISM_CORES__\\n"; (getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 0)',
        "printf \"__PRISM_MEMORY__\\n\"; awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null",
        *hardware_inspection_commands(),
    ]
    for marker, command in (
        ("K8S_VERSION", "version"),
        ("K8S_NODES", "get nodes"),
        ("DEVICE_CLASSES", "get deviceclasses.resource.k8s.io"),
        ("RESOURCE_SLICES", "get resourceslices.resource.k8s.io"),
    ):
        commands.append(f'printf "__PRISM_{marker}__\\n"; (kubectl {command} -o json 2>/dev/null || true)')
    result = await execute_remote_inspection(target, credentials, "; ".join(commands))
    parts = re.split(r"^__PRISM_([A-Z_0-9]+)__\s*$", result["stdout"], flags=re.MULTILINE)
    sections = {parts[index]: parts[index + 1].strip() for index in range(1, len(parts) - 1, 2)}

    def parsed(name):
        try:
            return json.loads(sections.get(name, ""))
        except ValueError:
            return None

    accelerator = {"vendor": "unknown", "count": 0, "devices": []}
    for index, profile in enumerate(all_profiles()):
        if detected := accelerator_output(profile, sections.get(f"HARDWARE_{index}", "")):
            accelerator = detected
            break
    cpu = parsed("CPU") or {}
    machine = {
        "hostname": sections.get("HOST") or "remote-machine",
        "platform": sections.get("PLATFORM") or "unknown",
        "cpu": {
            "model": next(
                (item.get("data") for item in cpu.get("lscpu", []) if item.get("field") == "Model name:"), "unknown"
            ),
            "logicalCores": int(sections.get("CORES") or 0),
        },
        "memoryGiB": round(float(sections.get("MEMORY") or 0) / 1024**2, 1),
        "accelerator": accelerator,
        "connection": "ssh",
    }
    version, nodes = parsed("K8S_VERSION"), parsed("K8S_NODES")
    return {
        "machine": machine,
        "cluster": cluster_profile(version, nodes, parsed("DEVICE_CLASSES"), parsed("RESOURCE_SLICES"))
        if version and nodes
        else None,
    }
