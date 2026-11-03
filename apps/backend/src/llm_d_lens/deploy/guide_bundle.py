"""Build immutable Guide deployment inputs using the backend's pinned stack."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import yaml

from llm_d_lens.configuration.guide_settings import _mapping, _merge
from llm_d_lens.configuration.router_compatibility import compatible_router_values
from llm_d_lens.versions import router_chart, router_chart_version

from .guide_planning import profile_for_key
from .guide_sources import command_output, fetch_source, github_content_url
from .providers.deployment_bundle import text_checksum, validate_deployment_bundle

ROUTER_PATHS = {
    "optimized-baseline": "optimized-baseline.values.yaml",
    "pd-disaggregation": "pd-disaggregation.values.yaml",
    "tiered-prefix-cache": "tiered-prefix-cache-cpu.values.yaml",
    "precise-prefix-cache-routing": "precise-prefix-cache-routing.values.yaml",
}


def asset(name: str, content: str) -> dict:
    return {"name": name, "content": content, "checksum": text_checksum(content)}


def _needs_render_service(values: dict) -> bool:
    """Whether the Router config renders tokens in a separate Service.

    llm-d v1.0 moved vLLM tokenization onto the model-server pods through
    ``vllm.endpointDiscovery``; SGLang still points ``vllm.url`` at the
    dedicated render pool.
    """
    epp = (values.get("router") or {}).get("epp") or {}
    key = epp.get("pluginsConfigFile")
    custom = epp.get("pluginsCustomConfig") or {}
    try:
        document = yaml.safe_load(custom[key])
    except (KeyError, TypeError, yaml.YAMLError):
        return True
    if not isinstance(document, dict):
        return True
    token = next((item for item in document.get("plugins") or [] if item.get("type") == "token-producer"), None)
    if not isinstance(token, dict):
        return True
    vllm = (token.get("parameters") or {}).get("vllm") or {}
    return "endpointDiscovery" not in vllm


async def build_guide_deployment_bundle(
    guide: str, manifest: dict, model: str, accelerator: str, block_size=None, router_values=""
) -> dict:
    if guide not in ROUTER_PATHS:
        raise ValueError("This Guide does not have a supported deployment bundle.")
    source = manifest["source"]
    local_root = manifest.get("localRoot")

    async def read_source(path):
        if local_root:
            return (Path(local_root) / path).read_text()
        return await fetch_source(
            github_content_url(source["repository"], path, source["commit"]), "application/vnd.github.raw+json"
        )

    async def read_router():
        profile = profile_for_key(accelerator)
        topology = (profile.deployment.router_topology if profile else None) or "single-host"
        try:
            return await read_source(f"guides/{guide}/router/{topology}/{ROUTER_PATHS[guide]}")
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            return await read_source(f"guides/{guide}/router/{ROUTER_PATHS[guide]}")
        except (OSError, ValueError):
            return await read_source(f"guides/{guide}/router/{ROUTER_PATHS[guide]}")

    base_text, guide_text = await asyncio.gather(read_source("guides/recipes/router/base.values.yaml"), read_router())
    values = _merge(_mapping(base_text), _mapping(guide_text))
    if router_values.strip():
        _merge(values, _mapping(router_values))
    if guide == "precise-prefix-cache-routing":
        epp = values.get("router", {}).get("epp", {})
        key = epp.get("pluginsConfigFile")
        plugins = _mapping(epp.get("pluginsCustomConfig", {}).get(key, ""))
        token = next((item for item in plugins.get("plugins", []) if item.get("type") == "token-producer"), None)
        index = next(
            (
                item
                for item in plugins.get("plugins", [])
                if item.get("type") in {"precise-prefix-cache-producer", "precise-prefix-cache-scorer"}
            ),
            None,
        )
        if token is None or index is None:
            raise ValueError("Precise routing requires token-producer and precise-prefix-cache plugins.")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("Precise routing requires an explicit or inherited model-server block size.")
        token.setdefault("parameters", {})["modelName"] = model
        index.setdefault("parameters", {}).setdefault("tokenProcessorConfig", {})["blockSizeTokens"] = block_size
        epp["pluginsCustomConfig"][key] = yaml.safe_dump(plugins, sort_keys=False)
        if epp.get("replicas", 1) != 1:
            raise ValueError("Precise token-load routing currently requires one EPP replica.")
    values = compatible_router_values(values, router_chart_version())
    bundle = {
        "schemaVersion": "guide-deployment-bundle.v1",
        "guide": guide,
        "sourceCommit": source["commit"],
        "helm": {
            "chart": router_chart(),
            "version": router_chart_version(),
            "releaseName": guide,
            "values": [
                asset("router-base.yaml", base_text),
                asset("router-guide.yaml", guide_text),
                asset("router-effective.yaml", yaml.safe_dump(values, sort_keys=False)),
            ],
        },
        "resources": [],
    }
    if guide == "precise-prefix-cache-routing":
        relative = "guides/precise-prefix-cache-routing/render"
        variant = "xpu-sglang" if "xpu" in accelerator.lower() else "standalone"

        async def render_overlay(path: str) -> str:
            return await command_output(["kubectl", "kustomize", path], 60)

        async def render_asset() -> str:
            if not _needs_render_service(values):
                return ""
            if local_root:
                directory = Path(local_root) / relative
                selected = relative if (directory / "kustomization.yaml").is_file() else f"{relative}/{variant}"
                return await render_overlay(str(Path(local_root) / selected))
            rendered = await render_overlay(
                f"https://github.com/{source['repository']}//{relative}?ref={source['commit']}"
            )
            if rendered.strip():
                return rendered
            return await render_overlay(
                f"https://github.com/{source['repository']}//{relative}/{variant}?ref={source['commit']}"
            )

        rendered, script, job = await asyncio.gather(
            render_asset(),
            read_source("guides/recipes/router/calibration/calibrate.sh"),
            read_source("guides/recipes/router/calibration/calibration-peak-throughput.yaml"),
        )
        root = Path(__file__).parent / "providers/guide_overlays/precise-prefix-cache-routing/baseline"
        baseline, overlay = (
            _mapping((root / "service.yaml").read_text()),
            _mapping((root / "kustomization.yaml").read_text()),
        )
        baseline["metadata"]["name"] = overlay.get("namePrefix", "") + baseline["metadata"]["name"]
        for labels in overlay.get("labels", []):
            baseline["metadata"].setdefault("labels", {}).update(labels.get("pairs", {}))
        resources = [asset("baseline.yaml", yaml.safe_dump(baseline, sort_keys=False))]
        if rendered.strip():
            resources.insert(0, asset("render.yaml", rendered))
        bundle["resources"] = resources
        bundle["calibration"] = [asset("calibrate.sh", script), asset("calibration-peak-throughput.yaml", job)]
    return validate_deployment_bundle(bundle, guide, source["commit"])
