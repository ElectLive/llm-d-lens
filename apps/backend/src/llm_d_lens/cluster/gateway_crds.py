"""Install the pinned Gateway API, Gateway API Inference Extension and llm-d Router CRDs.

The llm-d gateway recipes apply upstream CRD bundles before installing a
Gateway. Lens pins their versions in the stack profile and applies the same
manifests, so the CRD versions follow the Lens release instead of whatever a
provider chart happens to bundle.

The llm-d router EPP registers the ``InferenceModelRewrite`` GVK only at
startup. The router bundle is therefore installed before a router deployment
starts, and the cluster CRD action also restarts Lens-managed EPPs so their pod
templates register the current CRD.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from llm_d_lens.infrastructure.kubernetes import kubeconfig_environment, scoped_runner
from llm_d_lens.infrastructure.process import invalidate_kubectl_discovery_cache
from llm_d_lens.versions import stack

logger = logging.getLogger(__name__)

_APPLY_TIMEOUT = 300.0
_QUERY_TIMEOUT = 60.0
#: The kubectl wait budget and the runner timeout that must exceed it.
_ROUTER_WAIT_TIMEOUT = 60
_ROUTER_RUN_TIMEOUT = 90.0

#: A representative CRD required by each bundle.
_REQUIRED_CRDS = {
    "gateway_api": ("gateways.gateway.networking.k8s.io", "httproutes.gateway.networking.k8s.io"),
    "gateway_api_inference_extension": ("inferencepools.inference.networking.k8s.io",),
    "llm_d_router": ("inferencemodelrewrites.llm-d.ai",),
}

_ROUTER_CRD = "inferencemodelrewrites.llm-d.ai"
_ROUTER_API_VERSION = "/apis/llm-d.ai/v1alpha2"
_ROUTER_RESOURCE = "inferencemodelrewrites"


def invalidate_cluster_discovery_cache(cluster_id: str | None) -> None:
    """Drop the kubectl discovery cache for the kubeconfig this cluster's runner uses.

    ``scoped_runner`` overlays :func:`kubeconfig_environment`, and the process
    layer keys its isolated ``--cache-dir`` off that environment's
    ``KUBECONFIG``. Recomputing the same value here ensures invalidation targets
    exactly the cache the next kubectl call reads. Best effort: cache cleanup
    must never fail the operation that applied the CRDs.
    """
    try:
        kubeconfig = kubeconfig_environment(cluster_id).get("KUBECONFIG")
        invalidate_kubectl_discovery_cache(kubeconfig)
    except Exception:  # noqa: BLE001 - cache invalidation is best effort
        logger.warning("could not invalidate the kubectl discovery cache for cluster %s", cluster_id, exc_info=True)


def manifest_urls() -> dict[str, str]:
    """The pinned upstream CRD manifest URLs."""
    current = stack()
    return {
        "gateway_api": (
            "https://github.com/kubernetes-sigs/gateway-api/releases/download/"
            f"{current.k8s_gateway_api}/standard-install.yaml"
        ),
        "gateway_api_inference_extension": (
            "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/"
            f"{current.k8s_gateway_api_inference_extension}/v1-manifests.yaml"
        ),
        "llm_d_router": (
            f"https://github.com/llm-d/llm-d-router/releases/download/{current.llm_d_router}/manifests.yaml"
        ),
    }


async def _router_api_served(runner) -> bool:
    """True when uncached discovery serves the InferenceModelRewrite resource.

    Object presence is not enough: the EPP registers the GVK from discovery at
    startup, so a stale discovery cache or a not-yet-established CRD must read
    as missing rather than ready.
    """
    try:
        result = await runner.run(["kubectl", "get", "--raw", _ROUTER_API_VERSION], timeout=_QUERY_TIMEOUT)
    except (FileNotFoundError, TimeoutError, OSError):
        return False
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    resources = payload.get("resources") or []
    return any(isinstance(item, dict) and item.get("name") == _ROUTER_RESOURCE for item in resources)


async def _apply_router_crds(cluster_id: str) -> dict[str, Any]:
    """Apply the llm-d router bundle, wait for Established, then verify discovery."""
    runner = scoped_runner(cluster_id)
    url = manifest_urls()["llm_d_router"]
    try:
        applied = await runner.run(
            ["kubectl", "apply", "--server-side", "--force-conflicts", "-f", url], timeout=_APPLY_TIMEOUT
        )
    except (FileNotFoundError, TimeoutError, OSError) as error:
        return {"ok": False, "message": f"llm-d router CRD apply failed: {error}"}
    if applied.returncode != 0:
        message = (applied.stderr or applied.stdout or "").strip()
        return {"ok": False, "message": (message or "llm-d router CRD apply failed")[:500]}
    # kubectl wait itself performs discovery and can re-cache an /apis listing
    # before llm-d.ai is served. Invalidate only after the wait and discovery
    # checks have finished, whether they succeed or fail.
    try:
        try:
            waited = await runner.run(
                [
                    "kubectl",
                    "wait",
                    "--for=condition=Established",
                    f"--timeout={_ROUTER_WAIT_TIMEOUT}s",
                    f"crd/{_ROUTER_CRD}",
                ],
                timeout=_ROUTER_RUN_TIMEOUT,
            )
        except (FileNotFoundError, TimeoutError, OSError) as error:
            return {"ok": False, "message": f"llm-d router CRD wait failed: {error}"}
        if waited.returncode != 0:
            message = (waited.stderr or waited.stdout or "").strip()
            return {"ok": False, "message": (message or "llm-d router CRD did not become Established")[:500]}
        if not await _router_api_served(runner):
            return {
                "ok": False,
                "message": f"{_ROUTER_CRD} is Established but {_ROUTER_API_VERSION} does not serve {_ROUTER_RESOURCE}",
            }
        return {"ok": True, "message": ""}
    finally:
        invalidate_cluster_discovery_cache(cluster_id)


async def apply_gateway_crds(cluster_id: str) -> dict[str, Any]:
    """Apply all pinned CRD bundles; returns a per-bundle result.

    A bundle can contain optional resources a given Kubernetes version rejects
    (e.g. Gateway API's TLS-routes CRD uses the CEL ``isIP`` function, which
    needs a newer control plane). The Gateway bundles are therefore judged by
    whether the *required* CRDs are present after the apply, not by ``kubectl``'s
    exit code. The llm-d router bundle is stricter: ``ok`` requires the apply,
    the Established wait and discovery to all succeed.
    """
    runner = scoped_runner(cluster_id)
    results: dict[str, Any] = {}
    for key, url in manifest_urls().items():
        if key == "llm_d_router":
            results[key] = await _apply_router_crds(cluster_id)
            continue
        # Server-side apply handles CRDs that already exist from an earlier
        # (client-side) install, which plain `kubectl apply` rejects with a
        # missing-resourceVersion update error.
        result = await runner.run(
            ["kubectl", "apply", "--server-side", "--force-conflicts", "-f", url], timeout=_APPLY_TIMEOUT
        )
        message = (result.stderr or result.stdout or "").strip()
        results[key] = {"ok": result.returncode == 0, "message": message[:500]}
    # Every bundle is applied now; make later kubectl calls discover the new
    # CRDs instead of the pre-install cache.
    invalidate_cluster_discovery_cache(cluster_id)
    status = await gateway_crds_status(cluster_id)
    for key, presence in status.items():
        if key == "llm_d_router":
            # Never upgrade a failed apply/wait/discovery to ok just because the
            # name is now visible; only a missing bundle can make it worse.
            if presence != "ready":
                results[key]["ok"] = False
            continue
        results[key]["ok"] = presence == "ready"
    return results


async def gateway_crds_status(cluster_id: str) -> dict[str, str]:
    """Report ``ready``/``missing`` per CRD bundle by listing the CRDs."""
    runner = scoped_runner(cluster_id)
    result = await runner.run(["kubectl", "get", "crd", "-o", "name"], timeout=_QUERY_TIMEOUT)
    names = result.stdout.split() if result.returncode == 0 else []
    status = {
        key: "ready" if all(any(name.endswith(crd) for name in names) for crd in required) else "missing"
        for key, required in _REQUIRED_CRDS.items()
    }
    if status["llm_d_router"] == "ready" and not await _router_api_served(runner):
        status["llm_d_router"] = "missing"
    return status


async def ensure_router_crds(cluster_id: str) -> dict[str, Any]:
    """Make sure the pinned llm-d router CRDs are usable on a cluster.

    Verify first: a ready bundle returns without downloading anything. Otherwise
    apply the router bundle only, wait for the model-rewrite CRD to be
    Established and confirm discovery before reporting success.
    """
    try:
        status = await gateway_crds_status(cluster_id)
    except (FileNotFoundError, TimeoutError, OSError) as error:
        return {"ok": False, "applied": False, "message": f"llm-d router CRD status unavailable: {error}"}
    if status.get("llm_d_router") == "ready":
        return {"ok": True, "applied": False, "message": "already present"}
    result = await _apply_router_crds(cluster_id)
    return {"ok": result["ok"], "applied": True, "message": result["message"]}
