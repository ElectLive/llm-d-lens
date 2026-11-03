"""Tests for portable Deploy runtime path discovery."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_d_lens.configuration.models import ModelSecretConfiguration
from llm_d_lens.deploy.data_plane import GATEWAY_MODE_VALUES_PATH, PLAINTEXT_EPP_VALUES_PATH
from llm_d_lens.deploy.runtime.composition import (
    RegisteredGuideCommandRunner,
    RegisteredGuideRuntime,
    RestrictedKubectlRunner,
    RuntimeConfigurationError,
    RuntimeEnvironment,
    _build_optimized_baseline_provider,
    _build_precise_prefix_cache_routing_provider,
    _build_tiered_prefix_cache_provider,
    build_deployment_service_from_environment,
    build_namespace_factory,
)


def _executable(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    root = tmp_path / "llm-d"
    (root / "guides").mkdir(parents=True)
    for relative in ("recipes/router/base.values.yaml", "optimized-baseline/router/optimized-baseline.values.yaml"):
        path = root / "guides" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for relative in (
        "tiered-prefix-cache/modelserver/xpu/vllm/base",
        "tiered-prefix-cache/modelserver/xpu/vllm/native/cpu/base",
        "tiered-prefix-cache/modelserver/xpu/vllm/lmcache-connector/cpu/base",
        "precise-prefix-cache-routing/modelserver/xpu/vllm",
        "precise-prefix-cache-routing/modelserver/gpu/vllm/base",
    ):
        overlay = root / "guides" / relative
        overlay.mkdir(parents=True)
        (overlay / "kustomization.yaml").write_text("resources: []")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = _executable(bin_dir, "kubectl")
    _executable(bin_dir, "helm")
    _executable(bin_dir, "docker")
    return {
        "PRISM_DEPLOY_ACCELERATOR": "xpu",
        "LLM_D_ROOT": str(root),
        "PATH": str(bin_dir),
        "LLM_D_BENCH_KUBECTL_PATH": "",
        "LLM_D_BENCH_HELM_PATH": "",
        "LLM_D_BENCH_DOCKER_PATH": "",
    }, kubectl


def test_runtime_discovers_executables_from_path(tmp_path: Path):
    environment, kubectl = _environment(tmp_path)

    runtime = RuntimeEnvironment.from_environment(environment)

    assert runtime is not None
    assert runtime.kubectl_path == kubectl
    assert runtime.helm_path == kubectl.with_name("helm")
    assert runtime.docker_path == kubectl.with_name("docker")


def test_runtime_rejects_relative_executable_override(tmp_path: Path):
    environment, _ = _environment(tmp_path)
    environment["LLM_D_BENCH_DOCKER_PATH"] = "bin/docker"

    with pytest.raises(RuntimeConfigurationError, match="executable absolute path"):
        RuntimeEnvironment.from_environment(environment)


@pytest.mark.parametrize("entries", [[], ["gke", "coreweave"]])
def test_optimized_baseline_reports_missing_generic_entry(tmp_path, entries):
    environment, _ = _environment(tmp_path)
    environment["PRISM_DEPLOY_ACCELERATOR"] = "nvidia"
    runtime = RuntimeEnvironment.from_environment(environment)
    for entry in entries:
        overlay = runtime.manifest_root / "guides/optimized-baseline/modelserver/gpu/vllm" / entry
        overlay.mkdir(parents=True)
        (overlay / "kustomization.yaml").write_text("resources: []")

    adapter = _build_optimized_baseline_provider(runtime)
    validation = adapter.validate_inputs(adapter.discover(), {}, {})
    assert not validation.accepted
    message = "; ".join(validation.reasons)
    assert "optimized-baseline" in message
    assert "nvidia" in message
    assert "expected '.' or 'base'" in message
    assert "available variants:" in message
    with pytest.raises(ValueError, match="no generic Kustomize entry point"):
        asyncio.run(adapter.render(adapter.discover(), {}))


@pytest.mark.parametrize("accelerator,tree,entry", [("nvidia", "gpu", "base"), ("intel-xpu", "xpu", ".")])
def test_optimized_baseline_resolves_registered_hardware_entry(tmp_path, accelerator, tree, entry):
    environment, _ = _environment(tmp_path)
    environment["PRISM_DEPLOY_ACCELERATOR"] = accelerator
    root = Path(environment["LLM_D_ROOT"])
    overlay = root / "guides/optimized-baseline/modelserver" / tree / "vllm" / entry
    overlay.mkdir(parents=True)
    (overlay / "kustomization.yaml").write_text("resources: []")

    adapter = _build_optimized_baseline_provider(RuntimeEnvironment.from_environment(environment))

    assert adapter._policy.overlay_path == overlay
    assert adapter._policy.source_error is None


def test_catalog_registration_does_not_require_resolved_hardware(tmp_path, monkeypatch):
    environment, _ = _environment(tmp_path)
    environment["PRISM_DEPLOY_ACCELERATOR"] = ""
    monkeypatch.delenv("PRISM_DEPLOY_ACCELERATOR", raising=False)

    configured = build_deployment_service_from_environment(environment)

    assert configured is not None


def test_tiered_prefix_cache_provider_registers_all_upstream_xpu_variants(tmp_path: Path):
    environment, _ = _environment(tmp_path)
    runtime = RuntimeEnvironment.from_environment(environment)

    adapter = _build_tiered_prefix_cache_provider(runtime)

    assert set(adapter._descriptor.variants) == {
        "base",
        "native/cpu/base",
        "lmcache-connector/cpu/base",
    }
    assert adapter._descriptor.variants["lmcache-connector/cpu/base"].name == "base"
    assert adapter._descriptor.variants["lmcache-connector/cpu/base"].parent.name == "cpu"
    # Native OffloadingConnector stays the recommended default; LMCache is opt-in.
    assert adapter._descriptor.default_variant == "native/cpu/base"


def test_existing_model_secret_requires_namespace_and_name():
    with pytest.raises(ValidationError, match="sourceNamespace and sourceName"):
        ModelSecretConfiguration(mode="existing-secret")


def test_model_secret_defaults_to_no_credential():
    assert ModelSecretConfiguration().mode == "none"


def test_precise_prefix_cache_routing_provider_registers_modelserver_render_and_baseline(tmp_path: Path):
    environment, _ = _environment(tmp_path)
    runtime = RuntimeEnvironment.from_environment(environment)

    adapter = _build_precise_prefix_cache_routing_provider(runtime)

    # _environment()'s fixture guides/ tree has no nested llm-d/ subdirectory, so the
    # provider's guide_root fallback resolves directly to manifest_root (same fallback
    # tiered-prefix-cache's builder uses).
    guide_root = runtime.manifest_root
    assert adapter._modelserver_source == guide_root / "guides/precise-prefix-cache-routing/modelserver/xpu/vllm"
    assert adapter._render_source == guide_root / "guides/precise-prefix-cache-routing/render"
    assert adapter._baseline_source.name == "baseline"
    assert adapter._baseline_source.is_dir()
    assert (
        adapter._router_values
        == guide_root / "guides/precise-prefix-cache-routing/router/precise-prefix-cache-routing.values.yaml"
    )
    assert adapter.discover().guide_id == "precise-prefix-cache-routing"

    configured = ModelSecretConfiguration(
        mode="existing-secret", sourceNamespace="model-secrets", sourceName="huggingface"
    )
    assert configured.source_namespace == "model-secrets"


def test_tiered_prefix_cache_provider_renders_gpu_when_the_run_recorded_an_nvidia_accelerator(tmp_path: Path):
    environment, _ = _environment(tmp_path)
    runtime = RuntimeEnvironment.from_environment({**environment, "PRISM_DEPLOY_ACCELERATOR": "gpu"})

    adapter = _build_tiered_prefix_cache_provider(runtime)

    assert all(str(path).count("/gpu/") == 1 for path in adapter._descriptor.variants.values())


def test_precise_prefix_cache_routing_provider_renders_gpu_when_the_run_recorded_an_nvidia_accelerator(
    tmp_path: Path,
):
    environment, _ = _environment(tmp_path)
    runtime = RuntimeEnvironment.from_environment({**environment, "PRISM_DEPLOY_ACCELERATOR": "gpu"})

    adapter = _build_precise_prefix_cache_routing_provider(runtime)

    guide_root = runtime.manifest_root
    assert adapter._modelserver_source == guide_root / "guides/precise-prefix-cache-routing/modelserver/gpu/vllm/base"


def test_concurrent_runs_with_different_accelerators_do_not_share_overlay_state(tmp_path: Path):
    """Two runs for different clusters must each render their own vendor's overlay.

    Nothing here is process-global (no shared env var is mutated), so building
    an XPU-run provider after a GPU-run provider must not leak the GPU choice,
    and vice versa -- this is the regression the per-run ``accelerator`` field
    protects against.
    """
    environment, _ = _environment(tmp_path)
    gpu_runtime = RuntimeEnvironment.from_environment({**environment, "PRISM_DEPLOY_ACCELERATOR": "gpu"})
    xpu_runtime = RuntimeEnvironment.from_environment({**environment, "PRISM_DEPLOY_ACCELERATOR": "xpu"})

    gpu_adapter = _build_precise_prefix_cache_routing_provider(gpu_runtime)
    xpu_adapter = _build_precise_prefix_cache_routing_provider(xpu_runtime)
    gpu_adapter_again = _build_precise_prefix_cache_routing_provider(gpu_runtime)

    guide_root = gpu_runtime.manifest_root
    assert (
        gpu_adapter._modelserver_source == guide_root / "guides/precise-prefix-cache-routing/modelserver/gpu/vllm/base"
    )
    assert xpu_adapter._modelserver_source == guide_root / "guides/precise-prefix-cache-routing/modelserver/xpu/vllm"
    assert gpu_adapter_again._modelserver_source == gpu_adapter._modelserver_source


def test_registered_guide_allows_idempotent_namespace_cleanup(tmp_path: Path):
    namespace = "llm-d-bench-optimized-baseline-test"
    runner = RegisteredGuideCommandRunner(
        kubectl_path=tmp_path / "kubectl",
        helm_path=tmp_path / "helm",
        namespace_prefix="llm-d-bench-",
        guide=RegisteredGuideRuntime(
            guide_id="optimized-baseline",
            source_ref="main",
            content_hash="test",
            maturity="supported",
            manifest_path=tmp_path / "kustomization.yaml",
            readiness_deployment_name="optimized-baseline-xpu-vllm-decode",
        ),
        timeout_seconds=30,
        rendered_overlay_root=tmp_path,
    )
    runner.register_restored_namespace(namespace)

    runner._validate_kustomize(
        [
            "kubectl",
            "delete",
            "namespace",
            namespace,
            "--wait=false",
            "--ignore-not-found=true",
        ]
    )
    runner._validate_kustomize(
        [
            "kubectl",
            "delete",
            "namespace",
            namespace,
            "--ignore-not-found=true",
        ]
    )
    runner._validate_kustomize(
        [
            "kubectl",
            "delete",
            "namespace",
            namespace,
        ]
    )


@pytest.mark.parametrize(
    ("requested_prefix", "expected_prefix"),
    [("llmd-team-", "llmd-team-2-4-"), (None, "llmd-baseline-vllm-2-4-")],
)
def test_namespace_factory_uses_only_allowed_request_prefix(requested_prefix, expected_prefix):
    request = type(
        "Request",
        (),
        {
            "provenance": {"guide_id": "baseline-vllm"},
            "deployment_policy": type(
                "Policy",
                (),
                {
                    "value": {"namespace_policy": {"prefix": requested_prefix}},
                },
            )(),
            "configuration_artifacts": [
                type(
                    "Artifact",
                    (),
                    {
                        "content": '{"decode":{"replicaCount":2,"tensorParallelSize":4}}',
                    },
                )()
            ],
        },
    )()

    namespace = build_namespace_factory("llmd-")(request)

    assert namespace.startswith(expected_prefix)


@pytest.mark.parametrize("requested_prefix", ["standard-", "other-llmd-", "LLMD-team-"])
def test_namespace_factory_rejects_prefix_outside_allowlist(requested_prefix):
    request = type(
        "Request",
        (),
        {
            "deployment_policy": type("Policy", (), {"value": {"namespace_policy": {"prefix": requested_prefix}}})(),
            "configuration_artifacts": [type("Artifact", (), {"content": "{}"})()],
        },
    )()

    with pytest.raises(RuntimeConfigurationError, match="namespace policy prefix is invalid"):
        build_namespace_factory("llmd-")(request)


def test_registered_guide_helm_allows_only_planned_data_plane_values(tmp_path: Path):
    namespace = "llm-d-bench-router"
    guide = RegisteredGuideRuntime(
        guide_id="optimized-baseline",
        source_ref="main",
        content_hash="test",
        maturity="supported",
        manifest_path=tmp_path / "kustomization.yaml",
        router_chart="oci://charts/router",
        router_chart_version="1.0.0",
        router_release_name="optimized-baseline",
        router_base_values_path=tmp_path / "base.yaml",
        router_values_path=tmp_path / "optimized.yaml",
        neutral_router_values_path=tmp_path / "neutral.yaml",
    )
    runner = RegisteredGuideCommandRunner(
        kubectl_path=tmp_path / "kubectl",
        helm_path=tmp_path / "helm",
        namespace_prefix="llm-d-bench-",
        guide=guide,
        timeout_seconds=30,
        rendered_overlay_root=tmp_path,
    )
    install = [
        "helm",
        "upgrade",
        "--install",
        guide.router_release_name,
        guide.router_chart,
        "--namespace",
        namespace,
        "--version",
        guide.router_chart_version,
        "--values",
        str(guide.router_base_values_path),
        "--values",
        str(guide.router_values_path),
    ]
    plaintext = ["--values", str(PLAINTEXT_EPP_VALUES_PATH)]
    gateway = ["--values", str(GATEWAY_MODE_VALUES_PATH)]

    runner._validate_helm(install)
    runner._validate_helm([*install, *plaintext])
    runner._validate_helm([*install, *plaintext, *gateway])
    neutral_install = [*install[:-1], str(guide.neutral_router_values_path)]
    runner._validate_helm([*neutral_install, *plaintext])
    with pytest.raises(RuntimeConfigurationError, match="outside the registered Guide plan"):
        runner._validate_helm([*install, *gateway])
    with pytest.raises(RuntimeConfigurationError, match="outside the registered Guide plan"):
        runner._validate_helm([*install, "--set", "router.enabled=false"])
    with pytest.raises(RuntimeConfigurationError, match="outside the registered Guide plan"):
        runner._validate_helm([*install, "--values", str(tmp_path / "unregistered.yaml")])


@pytest.mark.parametrize(
    "command_tail",
    [
        ["-l", "llm-d.ai/role=decode", "-c", "modelserver", "--tail=120", "--prefix=true"],
        ["-l", "llm-d.ai/role=prefill", "-c", "modelserver", "--tail=120", "--prefix=true"],
        ["-l", "llm-d.ai/role=decode", "-c", "routing-proxy", "--tail=120", "--prefix=true"],
        ["-l", "llm-d.ai/role=decode", "-c", "routing-proxy", "--previous", "--tail=120", "--prefix=true"],
    ],
)
def test_restricted_kubectl_allows_pd_diagnostic_logs(tmp_path: Path, command_tail: list[str]):
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llm-d-bench-")

    runner._validate(
        [
            "kubectl",
            "logs",
            "--namespace",
            "llm-d-bench-pd-test",
            *command_tail,
        ]
    )


def test_restricted_kubectl_allows_disaggregatedset_readiness_and_stop(tmp_path: Path):
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llm-d-bench-")
    namespace = "llm-d-bench-pd-test"

    runner._validate(
        [
            "kubectl",
            "wait",
            "disaggregatedset/pd-disagg-xpu",
            "--namespace",
            namespace,
            "--for=condition=Available",
            "--timeout=30s",
        ]
    )
    runner._validate(
        [
            "kubectl",
            "patch",
            "disaggregatedset/pd-disagg-xpu",
            "--namespace",
            namespace,
            "--type=json",
            "--patch",
            "[]",
        ]
    )
    with pytest.raises(RuntimeConfigurationError, match="outside the configured allowlist"):
        runner._validate(["kubectl", "wait", "deployment/other", "--namespace", namespace])


def test_structured_startup_diagnostics_remain_namespace_scoped(tmp_path: Path):
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llm-d-bench-")
    runner._validate(["kubectl", "get", "pods", "--namespace", "llm-d-bench-pd-test", "-o", "json"])
    with pytest.raises(RuntimeConfigurationError, match="outside the configured allowlist"):
        runner._validate(["kubectl", "get", "pods", "--namespace", "default", "-o", "json"])


@pytest.mark.asyncio
async def test_failed_cli_logs_selected_cluster_without_credentials(tmp_path, monkeypatch, caplog):
    from unittest.mock import AsyncMock

    from llm_d_lens.deploy.runtime import composition

    kubeconfig = tmp_path / "cluster.yaml"
    kubeconfig.write_text(
        """apiVersion: v1
kind: Config
current-context: selected
contexts:
- name: selected
  context: {cluster: target, user: admin}
clusters:
- name: unused
  cluster: {server: 'https://unused.example:6443'}
- name: target
  cluster:
    server: https://api-user:api-password@10.1.2.3:6443/private-path?token=api-token
    proxy-url: http://explicit-user:explicit-password@explicit.example:3128/private
users:
- name: admin
  user: {token: kube-token}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRISM_KUBERNETES_BACKEND", "cli")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy-user:proxy-password@proxy.example:3128/private")
    monkeypatch.setenv("NO_PROXY", "localhost,10.1.2.3")
    monkeypatch.setenv("no_proxy", "localhost")
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llmd-", kubeconfig=str(kubeconfig))
    monkeypatch.setenv("HTTPS_PROXY", "http://different.example:8080")
    process = AsyncMock()
    process.pid = 123
    process.returncode = 1
    process.communicate.return_value = (b"", b"couldn't get current server API group list")
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(composition, "spawn", spawn)

    result = await runner(["kubectl", "rollout", "status", "deployment/vllm", "--namespace", "llmd-test"])

    assert result == (1, "", "couldn't get current server API group list")
    spawn.assert_awaited_once()
    record = next(record for record in caplog.records if "deployment_kubernetes_command_failed" in record.message)
    details = json.loads(record.message.split(" ", 1)[1])
    assert details["transport"] == "cli"
    assert details["pid"] == 123
    assert details["namespace"] == "llmd-test"
    assert details["api_server"] == "https://10.1.2.3:6443"
    assert details["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert details["kubeconfig_proxy"] == "http://explicit.example:3128"
    assert details["NO_PROXY"]["api_host_explicit"] is True
    assert details["no_proxy"]["api_host_explicit"] is False
    for sensitive in (
        "api-user",
        "api-password",
        "api-token",
        "kube-token",
        "private-path",
        "explicit-user",
        "explicit-password",
        "proxy-user",
        "proxy-password",
    ):
        assert sensitive not in record.message


@pytest.mark.parametrize("content", [None, "", "clusters: [invalid"])
def test_failure_diagnostics_do_not_raise_for_invalid_kubeconfig(tmp_path, caplog, content):
    if content is not None:
        (tmp_path / "missing.yaml").write_text(content, encoding="utf-8")
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llmd-", kubeconfig=str(tmp_path / "missing.yaml"))

    runner._log_failure(["kubectl", "get", "pods", "--namespace", "llmd-test"], 1, transport="cli", pid=123)

    details = json.loads(caplog.records[-1].message.split(" ", 1)[1])
    assert details["kubeconfig_error"] in {"ConfigException", "ValueError", "ParserError", "TypeError"}
    assert details["returncode"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [0, 1, 124])
async def test_sdk_failure_diagnostics_use_live_proxy_environment(tmp_path, monkeypatch, caplog, status):
    from unittest.mock import AsyncMock

    from llm_d_lens.deploy.runtime import composition
    from llm_d_lens.infrastructure.kubernetes import commands
    from llm_d_lens.infrastructure.process import CommandResult

    monkeypatch.setenv("HTTPS_PROXY", "http://snapshot.example:3128")
    runner = RestrictedKubectlRunner(tmp_path / "kubectl", "llmd-", kubeconfig=str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("HTTPS_PROXY", "http://live.example:3128")
    monkeypatch.setattr(commands, "sdk_enabled", lambda: True)
    execute = AsyncMock(return_value=CommandResult(("kubectl",), status, "out", "err"))
    monkeypatch.setattr(commands, "execute_sdk_command", execute)
    spawn = AsyncMock()
    monkeypatch.setattr(composition, "spawn", spawn)

    result = await runner(["kubectl", "get", "pods", "--namespace", "llmd-test", "-o", "json"])

    assert result == (status, "out", "err")
    spawn.assert_not_awaited()
    execute.assert_awaited_once()
    records = [record for record in caplog.records if "deployment_kubernetes_command_failed" in record.message]
    assert len(records) == (1 if status else 0)
    if records:
        details = json.loads(records[0].message.split(" ", 1)[1])
        assert details["transport"] == "sdk"
        assert details["HTTPS_PROXY"] == "http://live.example:3128"


@pytest.mark.asyncio
async def test_published_deployment_names_scope_commands(tmp_path):
    namespace = "llm-d-bench-published"
    runner = RegisteredGuideCommandRunner(
        kubectl_path=tmp_path / "kubectl",
        helm_path=tmp_path / "helm",
        namespace_prefix="llm-d-bench-",
        guide=RegisteredGuideRuntime(
            guide_id="optimized-baseline",
            source_ref="main",
            content_hash="test",
            maturity="supported",
            manifest_path=tmp_path / "kustomization.yaml",
            readiness_deployment_name="old-decode",
            endpoint_service_name="epp",
            endpoint_service_port=80,
        ),
        timeout_seconds=30,
        rendered_overlay_root=tmp_path,
    )
    runner.register_published_deployments(namespace, ["published-decode"])
    runner._validate_kustomize(
        ["kubectl", "rollout", "status", "deployment/published-decode", "--namespace", namespace, "--timeout=30s"]
    )
    runner._validate_kustomize(
        ["kubectl", "scale", "deployment/published-decode", "--namespace", namespace, "--replicas=0"]
    )
    with pytest.raises(RuntimeConfigurationError):
        runner._validate_kustomize(
            ["kubectl", "scale", "deployment/unrelated", "--namespace", namespace, "--replicas=0"]
        )
    with pytest.raises(RuntimeConfigurationError):
        runner._validate_kustomize(
            ["kubectl", "scale", "deployment/published-decode", "--namespace", "llm-d-bench-other", "--replicas=0"]
        )
    with pytest.raises(RuntimeConfigurationError):
        runner.register_published_deployments("default", ["published-decode"])
