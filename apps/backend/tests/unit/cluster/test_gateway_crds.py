"""Tests for the pinned Gateway API / GIE / llm-d Router CRD installer."""

import importlib
import json
from urllib.parse import urlsplit

import pytest

from llm_d_lens import versions
from llm_d_lens.cluster import gateway_crds, registry
from llm_d_lens.cluster.gateway_crds import _REQUIRED_CRDS, manifest_urls
from llm_d_lens.cluster.settings import ClusterSettings
from llm_d_lens.infrastructure.kubernetes import scoped_runner
from llm_d_lens.infrastructure.process import _with_kubectl_cache_dir, kubectl_cache_dir
from llm_d_lens.model_service.gateway_ops import GatewayOpsService

cluster_router = importlib.import_module("llm_d_lens.cluster.router")


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


_REQUIRED_NAMES = (
    "gateways.gateway.networking.k8s.io",
    "httproutes.gateway.networking.k8s.io",
    "inferencepools.inference.networking.k8s.io",
    "inferencemodelrewrites.llm-d.ai",
)


class _Runner:
    def __init__(
        self,
        *,
        crd_names=_REQUIRED_NAMES,
        raw=None,
        router_apply: _Result | None = None,
        wait: _Result | None = None,
    ) -> None:
        self.crd_names = list(crd_names)
        self.raw = raw
        self.router_apply = router_apply or _Result()
        self.wait = wait or _Result()
        self.calls: list[list[str]] = []

    async def run(self, argv, *, timeout=None, **_kwargs):
        self.calls.append(list(argv))
        if list(argv[:3]) == ["kubectl", "get", "crd"]:
            return _Result(stdout="\n".join(self.crd_names))
        if list(argv[:3]) == ["kubectl", "get", "--raw"]:
            if isinstance(self.raw, _Result):
                return self.raw
            return _Result(stdout=json.dumps(self.raw or {}))
        if list(argv[:3]) == ["kubectl", "apply", "--server-side"]:
            if any(str(arg).endswith("manifests.yaml") for arg in argv):
                return self.router_apply
            return _Result()
        if list(argv[:3]) == ["kubectl", "wait", "--for=condition=Established"]:
            return self.wait
        return _Result()


def _use_runner(monkeypatch, runner: _Runner) -> _Runner:
    monkeypatch.setattr(gateway_crds, "scoped_runner", lambda _cluster_id: runner)
    return runner


def _stub_cluster_kubeconfig(monkeypatch, kubeconfig: str) -> None:
    monkeypatch.setattr(gateway_crds, "kubeconfig_environment", lambda _cluster_id: {"KUBECONFIG": kubeconfig})
    monkeypatch.delenv("KUBECONFIG", raising=False)


def test_manifest_urls_follow_the_pinned_versions():
    current = versions.stack()
    urls = manifest_urls()

    assert urlsplit(urls["gateway_api"]).path.split("/")[-2] == current.k8s_gateway_api
    assert (
        urlsplit(urls["gateway_api_inference_extension"]).path.split("/")[-2]
        == current.k8s_gateway_api_inference_extension
    )
    assert urlsplit(urls["llm_d_router"]).path.split("/")[-2] == current.llm_d_router
    assert urlsplit(urls["gateway_api"]).path.split("/")[-1] == "standard-install.yaml"
    assert urlsplit(urls["gateway_api_inference_extension"]).path.split("/")[-1] == "v1-manifests.yaml"
    assert urlsplit(urls["llm_d_router"]).path.split("/")[-1] == "manifests.yaml"


def test_required_crds_cover_all_bundles():
    assert "gateways.gateway.networking.k8s.io" in _REQUIRED_CRDS["gateway_api"]
    assert "inferencepools.inference.networking.k8s.io" in _REQUIRED_CRDS["gateway_api_inference_extension"]
    assert _REQUIRED_CRDS["llm_d_router"] == ("inferencemodelrewrites.llm-d.ai",)


@pytest.mark.asyncio
async def test_router_status_requires_uncached_discovery(monkeypatch):
    runner = _use_runner(monkeypatch, _Runner(raw={"resources": [{"name": "inferenceobjectives"}]}))

    status = await gateway_crds.gateway_crds_status("cluster-a")

    assert status["llm_d_router"] == "missing"
    assert ["kubectl", "get", "--raw", "/apis/llm-d.ai/v1alpha2"] in runner.calls


@pytest.mark.asyncio
async def test_router_status_is_ready_when_discovery_serves_the_rewrite_resource(monkeypatch):
    runner = _use_runner(monkeypatch, _Runner(raw={"resources": [{"name": "inferencemodelrewrites"}]}))

    status = await gateway_crds.gateway_crds_status("cluster-a")

    assert status["llm_d_router"] == "ready"
    assert len(runner.calls) == 2


@pytest.mark.asyncio
async def test_router_status_treats_a_failed_discovery_query_as_missing(monkeypatch):
    _use_runner(
        monkeypatch, _Runner(raw=_Result(returncode=1, stderr="the server could not find the requested resource"))
    )

    status = await gateway_crds.gateway_crds_status("cluster-a")

    assert status["llm_d_router"] == "missing"


@pytest.mark.asyncio
async def test_ensure_router_crds_is_network_free_when_already_ready(monkeypatch):
    runner = _use_runner(monkeypatch, _Runner(raw={"resources": [{"name": "inferencemodelrewrites"}]}))

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result == {"ok": True, "applied": False, "message": "already present"}
    assert not any("apply" in call for call in runner.calls)


@pytest.mark.asyncio
async def test_ensure_router_crds_applies_waits_and_verifies_when_missing(monkeypatch):
    runner = _use_runner(
        monkeypatch,
        _Runner(
            crd_names=(),
            raw={"resources": [{"name": "inferencemodelrewrites"}]},
        ),
    )

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result == {"ok": True, "applied": True, "message": ""}
    apply_call = next(call for call in runner.calls if "apply" in call)
    assert any(str(arg).endswith("manifests.yaml") for arg in apply_call)
    wait_call = next(call for call in runner.calls if "wait" in call)
    assert "--for=condition=Established" in wait_call
    assert "--timeout=60s" in wait_call
    assert "crd/inferencemodelrewrites.llm-d.ai" in wait_call
    assert ["kubectl", "get", "--raw", "/apis/llm-d.ai/v1alpha2"] in runner.calls


@pytest.mark.asyncio
async def test_ensure_router_crds_preserves_a_wait_failure(monkeypatch):
    _use_runner(
        monkeypatch,
        _Runner(
            crd_names=(),
            raw={"resources": [{"name": "inferencemodelrewrites"}]},
            wait=_Result(returncode=1, stderr="timed out waiting for the condition"),
        ),
    )

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result["ok"] is False
    assert result["applied"] is True
    assert "timed out waiting for the condition" in result["message"]


@pytest.mark.asyncio
async def test_ensure_router_crds_preserves_a_discovery_failure(monkeypatch):
    _use_runner(
        monkeypatch,
        _Runner(
            crd_names=(),
            raw={"resources": [{"name": "inferenceobjectives"}]},
        ),
    )

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result["ok"] is False
    assert result["applied"] is True
    assert "does not serve inferencemodelrewrites" in result["message"]


@pytest.mark.asyncio
async def test_ensure_router_crds_invalidates_after_the_established_wait(monkeypatch):
    events: list[list[str]] = []
    _stub_cluster_kubeconfig(monkeypatch, "/kube/cluster-a")
    monkeypatch.setattr(
        gateway_crds, "invalidate_kubectl_discovery_cache", lambda _kubeconfig: events.append(["invalidate"])
    )
    runner = _use_runner(
        monkeypatch,
        _Runner(crd_names=(), raw={"resources": [{"name": "inferencemodelrewrites"}]}),
    )
    run = runner.run

    async def record(argv, **kwargs):
        events.append(list(argv))
        return await run(argv, **kwargs)

    runner.run = record

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result == {"ok": True, "applied": True, "message": ""}
    wait_index = next(
        index for index, event in enumerate(events) if event[:3] == ["kubectl", "wait", "--for=condition=Established"]
    )
    discovery_index = next(index for index, event in enumerate(events) if event[:3] == ["kubectl", "get", "--raw"])
    invalidate_index = events.index(["invalidate"])
    assert events.count(["invalidate"]) == 1
    assert invalidate_index > wait_index
    assert invalidate_index > discovery_index


@pytest.mark.asyncio
async def test_ensure_router_crds_invalidates_even_when_the_wait_fails(monkeypatch):
    invalidated: list[str | None] = []
    _stub_cluster_kubeconfig(monkeypatch, "/kube/cluster-a")
    monkeypatch.setattr(gateway_crds, "invalidate_kubectl_discovery_cache", invalidated.append)
    _use_runner(
        monkeypatch,
        _Runner(
            crd_names=(),
            raw={"resources": [{"name": "inferencemodelrewrites"}]},
            wait=_Result(returncode=1, stderr="timed out waiting for the condition"),
        ),
    )

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result["ok"] is False
    assert result["applied"] is True
    assert invalidated == ["/kube/cluster-a"]


@pytest.mark.asyncio
async def test_ensure_router_crds_does_not_invalidate_when_already_ready(monkeypatch):
    invalidated: list[str | None] = []
    _stub_cluster_kubeconfig(monkeypatch, "/kube/cluster-a")
    monkeypatch.setattr(gateway_crds, "invalidate_kubectl_discovery_cache", invalidated.append)
    _use_runner(monkeypatch, _Runner(raw={"resources": [{"name": "inferencemodelrewrites"}]}))

    result = await gateway_crds.ensure_router_crds("cluster-a")

    assert result == {"ok": True, "applied": False, "message": "already present"}
    assert invalidated == []


@pytest.mark.asyncio
async def test_apply_gateway_crds_invalidates_discovery_cache_after_applying(monkeypatch):
    invalidated: list[str | None] = []
    _stub_cluster_kubeconfig(monkeypatch, "/kube/cluster-a")
    monkeypatch.setattr(gateway_crds, "invalidate_kubectl_discovery_cache", invalidated.append)
    _use_runner(monkeypatch, _Runner(raw={"resources": [{"name": "inferencemodelrewrites"}]}))

    result = await gateway_crds.apply_gateway_crds("cluster-a")

    assert result["gateway_api"]["ok"] is True
    # Once after the router bundle (inside _apply_router_crds) and once after
    # the remaining bundles in apply_gateway_crds itself.
    assert invalidated == ["/kube/cluster-a", "/kube/cluster-a"]


def test_invalidate_cluster_discovery_cache_uses_the_runner_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "cluster_settings", ClusterSettings(tmp_path / "sessions", "llm-d-bench-", tmp_path))
    monkeypatch.setenv("PRISM_KUBERNETES_BACKEND", "cli")
    monkeypatch.delenv("KUBECONFIG", raising=False)
    cluster = registry.create_cluster("cluster-a", "", "kubeconfig: {}")

    runner = scoped_runner(cluster.id)
    runner_env = runner._merged_env(None)
    runner_cache_dir = next(
        arg.split("=", 1)[1]
        for arg in _with_kubectl_cache_dir(["kubectl"], runner_env)
        if arg.startswith("--cache-dir=")
    )

    invalidated: list[str | None] = []
    monkeypatch.setattr(gateway_crds, "invalidate_kubectl_discovery_cache", invalidated.append)

    gateway_crds.invalidate_cluster_discovery_cache(cluster.id)

    assert invalidated == [runner_env["KUBECONFIG"]]
    assert runner_cache_dir == str(kubectl_cache_dir(invalidated[0]))


def test_invalidate_cluster_discovery_cache_swallows_resolution_errors(monkeypatch):
    def fail(_cluster_id):
        raise FileNotFoundError("cluster not found")

    monkeypatch.setattr(gateway_crds, "kubeconfig_environment", fail)

    gateway_crds.invalidate_cluster_discovery_cache("cluster-a")


def test_invalidate_cluster_discovery_cache_swallows_removal_errors(monkeypatch):
    _stub_cluster_kubeconfig(monkeypatch, "/kube/cluster-a")

    def fail(_kubeconfig):
        raise OSError("cannot remove cache")

    monkeypatch.setattr(gateway_crds, "invalidate_kubectl_discovery_cache", fail)

    gateway_crds.invalidate_cluster_discovery_cache("cluster-a")


@pytest.mark.asyncio
async def test_apply_gateway_crds_reports_all_three_bundles(monkeypatch):
    _use_runner(
        monkeypatch,
        _Runner(raw={"resources": [{"name": "inferencemodelrewrites"}]}),
    )

    result = await gateway_crds.apply_gateway_crds("cluster-a")

    assert set(result) == set(_REQUIRED_CRDS)
    assert result["gateway_api"]["ok"] is True
    assert result["gateway_api_inference_extension"]["ok"] is True
    assert result["llm_d_router"]["ok"] is True


@pytest.mark.asyncio
async def test_apply_gateway_crds_never_upgrades_a_router_apply_failure(monkeypatch):
    _use_runner(
        monkeypatch,
        _Runner(
            raw={"resources": [{"name": "inferencemodelrewrites"}]},
            router_apply=_Result(returncode=1, stderr="apply failed"),
        ),
    )

    result = await gateway_crds.apply_gateway_crds("cluster-a")

    assert result["llm_d_router"]["ok"] is False
    assert "apply failed" in result["llm_d_router"]["message"]


@pytest.mark.asyncio
async def test_create_cluster_crds_repairs_router_epps_when_the_router_bundle_is_ready(monkeypatch):
    repaired: list[str] = []

    async def apply(_cluster_id):
        return {"llm_d_router": {"ok": True, "message": ""}}

    async def repair(_self, cluster_id):
        repaired.append(cluster_id)
        return {"ok": True, "restarted": ["deploy/qwen-epp"], "message": ""}

    monkeypatch.setattr(registry, "require_cluster", lambda _cluster_id: None)
    monkeypatch.setattr(gateway_crds, "apply_gateway_crds", apply)
    monkeypatch.setattr(GatewayOpsService, "repair_router_rewrites", repair)

    result = await cluster_router.create_cluster_crds("cluster-a")

    assert repaired == ["cluster-a"]
    assert result["router_epp_repair"] == {"ok": True, "restarted": ["deploy/qwen-epp"], "message": ""}


@pytest.mark.asyncio
async def test_create_cluster_crds_skips_the_router_repair_when_the_router_bundle_is_not_ready(monkeypatch):
    repaired: list[str] = []

    async def apply(_cluster_id):
        return {
            "gateway_api": {"ok": True, "message": ""},
            "gateway_api_inference_extension": {"ok": True, "message": ""},
            "llm_d_router": {"ok": False, "message": "not ready"},
        }

    async def repair(_self, cluster_id):
        repaired.append(cluster_id)
        return {"ok": True, "restarted": [], "message": ""}

    monkeypatch.setattr(registry, "require_cluster", lambda _cluster_id: None)
    monkeypatch.setattr(gateway_crds, "apply_gateway_crds", apply)
    monkeypatch.setattr(GatewayOpsService, "repair_router_rewrites", repair)

    result = await cluster_router.create_cluster_crds("cluster-a")

    assert repaired == []
    assert result["router_epp_repair"] == {"ok": False, "restarted": [], "message": "router CRDs are not ready"}
