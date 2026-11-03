"""Structured P/D disaggregation deployment adapter (NVIDIA GPU and Intel XPU)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llm_d_lens.configuration.manifest_facts import workload_roles
from llm_d_lens.configuration.router_compatibility import compatible_router_values
from llm_d_lens.core.hashing import stable_hash
from llm_d_lens.deploy.data_plane import _deep_merge, router_data_plane_args, router_data_plane_effective_values
from llm_d_lens.deploy.providers.deployment_bundle import (
    install_deployment_bundle,
    subprocess_bundle_runner,
)
from llm_d_lens.deploy.providers.gpu_selection import gpu_device_selectors
from llm_d_lens.deploy.providers.guide_adapter import (
    GuideDefinition,
    GuideDeploymentArtifact,
    ValidationResult,
    serving_arguments,
    set_vllm_argument,
)
from llm_d_lens.deploy.providers.hardware_profile import (
    active_profile,
    apply_runtime_class,
    default_guide_variant,
    guide_overlays,
    requires_dra_claim,
    set_accelerator_request,
)
from llm_d_lens.deploy.providers.model_cache_environment import model_cache_environment
from llm_d_lens.deploy.providers.storage_mount import resolve_mount
from llm_d_lens.infrastructure.storage.paths import prism_temp_root
from llm_d_lens.versions import router_chart_version


class PdDisaggregationAdapter:
    published_manifest_lifecycle = True

    def __init__(
        self,
        command_runner,
        guide_root: Path,
        namespace_prefix: str,
        timeout: int,
        helm_path: Path,
        kubeconfig: str | None,
        accelerator: str | None = None,
    ) -> None:
        self._runner = command_runner
        self._accelerator = accelerator
        self._sources = self._guide_sources(guide_root, accelerator)
        self._root = prism_temp_root("prism-pd-overlays")
        self._namespace_prefix = namespace_prefix
        self._timeout = timeout
        self._helm_path = helm_path
        self._environment = {**os.environ, "KUBECONFIG": kubeconfig} if kubeconfig else None
        self._guide_root = guide_root
        self._bundle_command_runner = subprocess_bundle_runner(
            helm_path,
            Path(getattr(command_runner, "_kubectl_path", "kubectl")),
            self._environment,
        )
        self._definition = GuideDefinition(
            "pd-disaggregation",
            "local-pd-disaggregation",
            stable_hash({"sources": {name: str(path) for name, path in self._sources.items()}}),
            "supported-core",
            {
                "variants": sorted(self._sources),
                "default_variant": (
                    default_guide_variant(self._sources, guide="pd-disaggregation", accelerator=self._accelerator)
                    if any(name in self._sources for name in (".", "base"))
                    else ""
                ),
                "structured_custom_parameters": True,
            },
        )

    @staticmethod
    def _guide_sources(guide_root: Path, accelerator: str | None = None) -> dict[str, Path]:
        sources = guide_overlays(guide_root, "pd-disaggregation", accelerator=accelerator)
        profile = active_profile(accelerator)
        if profile:
            base = guide_root / profile.deployment.overlay_root.format(
                guide="pd-disaggregation", variant=profile.upstream_variant, model_server="vllm"
            )
            if base.parent.is_dir():
                for sibling in base.parent.iterdir():
                    if sibling.is_dir() and sibling.name.startswith("vllm-"):
                        for variant, path in guide_overlays(
                            guide_root, "pd-disaggregation", accelerator=accelerator, model_server=sibling.name
                        ).items():
                            sources[sibling.name if variant == "." else f"{sibling.name}/{variant}"] = path
        if "." in sources:
            sources["vllm"] = sources["."]
        elif "base" in sources:
            sources["vllm"] = sources["base"]
        return sources

    def discover(self):
        return self._definition

    def register_restored_namespace(self, namespace: str) -> None:
        register = getattr(self._runner, "register_restored_namespace", None)
        if callable(register):
            register(namespace)

    def validate_inputs(self, definition, cluster_snapshot, overrides):
        try:
            parameters = self._parameters(overrides)
        except ValueError as error:
            return ValidationResult(False, [str(error)])
        source = self._sources.get(parameters["variant"])
        if source is None:
            return ValidationResult(
                False, [f"unsupported pd-disaggregation variant for this accelerator: {parameters['variant']}"]
            )
        if not source.is_dir():
            return ValidationResult(False, ["pd-disaggregation variant source is unavailable for this accelerator"])
        return ValidationResult(True)

    async def render(self, definition, overrides):
        parameters = self._parameters(overrides)
        parameters["resolved_mount"] = await resolve_mount(
            overrides.get("runtime") or {},
            cluster_id=overrides.get("_cluster_id"),
            namespace=overrides.get("_deployment_namespace"),
        )
        digest = stable_hash(parameters)
        directory = self._root / digest
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        process = await asyncio.create_subprocess_exec(
            "kubectl",
            "kustomize",
            str(self._sources[parameters["variant"]]),
            env=self._environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise ValueError(f"Kustomize render failed: {stderr.decode(errors='replace').strip()}")
        documents = [item for item in yaml.safe_load_all(stdout) if isinstance(item, dict)]
        workloads = workload_roles(documents)
        workload_names = list(dict.fromkeys(str(item["name"]) for item in workloads if item["name"]))
        for workload in workloads:
            role = workload["role"]
            if role not in {"prefill", "decode"}:
                continue
            workload["spec"]["replicas"] = parameters[role]["replicas"]
            container = next(
                (item for item in workload["pod"].get("containers", []) if item.get("name") == "modelserver"),
                None,
            )
            if container is None:
                continue
            resolved_mount = parameters["resolved_mount"]
            model_source = resolved_mount["model_source"] if resolved_mount else "shared-path"
            cache_mounted = bool(resolved_mount or parameters["mount_path"])
            args = list(container.get("args") or [])
            args[0] = "/model-cache" if cache_mounted and model_source == "shared-path" else parameters["model"]
            args = self._set_argument(args, "tensor-parallel-size", str(parameters[role]["tensor_parallel_size"]))
            for name, value in parameters[role]["serving_arguments"].items():
                args = self._set_argument(args, name, value)
            environment = list(container.get("env") or [])
            for name, value in sorted(parameters["environment"].items()):
                if value:
                    environment = self._set_environment(environment, name, value)
            for custom in parameters["custom_parameters"]:
                if custom["target"] not in {role, "both"}:
                    continue
                if custom["kind"] == "argument":
                    args = self._set_argument(args, custom["name"], custom["value"])
                else:
                    environment = self._set_environment(environment, custom["name"], custom["value"])
            for name, value in model_cache_environment(model_source, cache_mounted=cache_mounted).items():
                environment = self._set_environment(environment, name, value)
            container["args"] = args
            container["env"] = environment
            if resolved_mount or parameters["mount_path"]:
                volume_source = (
                    resolved_mount["volume_source"]
                    if resolved_mount
                    else {"hostPath": {"path": parameters["mount_path"], "type": "DirectoryOrCreate"}}
                )
                workload["pod"].setdefault("volumes", []).append({"name": "model-cache", **volume_source})
                container.setdefault("volumeMounts", []).append(
                    {
                        "name": "model-cache",
                        "mountPath": "/model-cache",
                        "readOnly": resolved_mount["read_only"] if resolved_mount else True,
                    }
                )
        # Accelerator requests are profile-driven: DRA profiles (e.g. Intel XPU)
        # rewrite the ResourceClaimTemplate count; extended-resource profiles
        # (e.g. NVIDIA nvidia.com/gpu) set the container's resource limits.
        selectors = gpu_device_selectors(accelerator=self._accelerator)
        for claim in (item for item in documents if item.get("kind") == "ResourceClaimTemplate"):
            role = "prefill" if "prefill" in claim["metadata"]["name"] else "decode"
            requests = claim["spec"]["spec"]["devices"]["requests"]
            gpu_request = next(
                (
                    item
                    for item in requests
                    if item.get("exactly", {}).get("deviceClassName")
                    in active_profile(self._accelerator).device_classes
                ),
                None,
            )
            if gpu_request is None:
                continue
            gpu_request["exactly"]["count"] = parameters[role]["tensor_parallel_size"]
            if selectors:
                gpu_request["exactly"].setdefault("selectors", []).extend(selectors)
        for workload in workloads:
            role = workload["role"]
            if role not in {"prefill", "decode"}:
                continue
            container = next(
                (item for item in workload["pod"].get("containers", []) if item.get("name") == "modelserver"),
                None,
            )
            if container is None:
                continue
            container["image"] = parameters["image"]
            if not requires_dra_claim(accelerator=self._accelerator):
                set_accelerator_request(
                    container, None, parameters[role]["tensor_parallel_size"], accelerator=self._accelerator
                )
            apply_runtime_class(workload["pod"], accelerator=self._accelerator)
        manifest = directory / "manifest.yaml"
        manifest.write_text(yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8")
        return GuideDeploymentArtifact(
            guide_id=self._definition.guide_id,
            artifact_hash=digest,
            manifest_ref=str(manifest),
            source_ref=self._definition.source_ref,
            guide_content_hash=self._definition.content_hash,
            manifest_checksum=stable_hash({"parameters": parameters}),
            # llm-d router data plane: reached through the shared Gateway unless
            # the deployment is evaluation-owned (then it keeps its own proxy).
            # Readiness Deployment names are hardware/engine specific, so the
            # rendered names travel with the artifact.
            deployment_contract={
                "data_plane_kind": "llm-d-router",
                "readinessDeployments": workload_names,
            },
        )

    async def deploy(self, artifact, context):
        namespace = context["namespace"]
        status, stdout, stderr = await self._runner(["kubectl", "create", "namespace", namespace])
        if status != 0 and "AlreadyExists" not in stderr:
            raise RuntimeError((stderr or stdout).strip())
        model_secret = (context.get("deployment_policy") or {}).get("model_secret") or {}
        if "endpoint" not in artifact.deployment_contract and model_secret.get("mode") == "existing-secret":
            await self._runner.copy_model_secret(namespace, model_secret["sourceNamespace"], model_secret["sourceName"])
        bundle = artifact.deployment_contract.get("deploymentBundle")
        reproducibility = {}
        if bundle is not None:
            reproducibility = await install_deployment_bundle(
                bundle,
                guide=artifact.guide_id,
                source_commit=str(artifact.source_ref or ""),
                namespace=namespace,
                command_runner=self._bundle_command_runner,
                output_root=Path(artifact.manifest_ref or "").parent / "deployment-bundle",
                effective_values_content=router_data_plane_effective_values(
                    bundle["helm"]["values"][-1]["content"], context
                ),
                effective_values_name="router-effective-pd-disaggregation.yaml",
            )
        else:
            base_values = self._guide_root / "guides/recipes/router/base.values.yaml"
            pd_values = self._guide_root / "guides/pd-disaggregation/router/pd-disaggregation.values.yaml"
            values = _deep_merge(
                yaml.safe_load(base_values.read_text(encoding="utf-8")),
                yaml.safe_load(pd_values.read_text(encoding="utf-8")),
            )
            compatible = compatible_router_values(values, router_chart_version())
            if compatible != values:
                pd_values = Path(artifact.manifest_ref).parent / "router-compatible-pd.yaml"
                pd_values.write_text(yaml.safe_dump(compatible, sort_keys=False), encoding="utf-8")
            helm = await asyncio.create_subprocess_exec(
                str(self._helm_path),
                "upgrade",
                "--install",
                "pd-disaggregation",
                "oci://ghcr.io/llm-d/charts/llm-d-router-standalone",
                "--namespace",
                namespace,
                "--version",
                router_chart_version(),
                "--values",
                str(base_values),
                "--values",
                str(pd_values),
                *router_data_plane_args(context),
                env=self._environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            helm_stdout, helm_stderr = await helm.communicate()
            if helm.returncode:
                raise RuntimeError((helm_stderr or helm_stdout).decode(errors="replace").strip())
        status, stdout, stderr = await self._runner(
            ["kubectl", "apply", "--namespace", namespace, "--filename", artifact.manifest_ref]
        )
        if status != 0:
            raise RuntimeError((stderr or stdout).strip())
        return {
            "namespace": namespace,
            "artifact_hash": artifact.artifact_hash,
            "apply_output": stdout,
            **reproducibility,
        }

    @staticmethod
    def _workload_resources(artifact) -> list[tuple[str, str]]:
        """Readiness resources (kind, name) from the rendered workload manifest."""
        manifest_ref = getattr(artifact, "manifest_ref", None)
        if not manifest_ref or not Path(manifest_ref).is_file():
            return []
        documents = [item for item in yaml.safe_load_all(Path(manifest_ref).read_text(encoding="utf-8")) if item]
        resources = []
        for workload in workload_roles(documents):
            item = (str(workload["kind"]), str(workload["name"]))
            if item[1] and item not in resources:
                resources.append(item)
        return resources

    async def _readiness_deployments(self, execution) -> list[str]:
        artifact = execution.get("_artifact")
        resources = self._workload_resources(artifact) if artifact is not None else []
        if resources:
            return [name for _kind, name in resources]
        contract = artifact.deployment_contract if isinstance(artifact, GuideDeploymentArtifact) else {}
        names = [str(name) for name in contract.get("readinessDeployments") or []]
        if names:
            return names
        namespace = str(execution["namespace"])
        status, stdout, stderr = await self._runner(
            ["kubectl", "get", "deployments", "--namespace", namespace, "--output", "json"]
        )
        if status != 0:
            return []
        return [
            str(item["metadata"]["name"])
            for item in json.loads(stdout).get("items", [])
            if str(item["metadata"]["name"]).endswith(("prefill", "decode"))
        ]

    async def readiness(self, execution):
        namespace = execution["namespace"]
        artifact = execution.get("_artifact")
        resources = self._workload_resources(artifact) if artifact is not None else []
        if not resources:
            resources = [("Deployment", name) for name in await self._readiness_deployments(execution)]
        for kind, name in resources:
            if kind == "Deployment":
                command = [
                    "kubectl",
                    "rollout",
                    "status",
                    f"deployment/{name}",
                    "--namespace",
                    namespace,
                    f"--timeout={self._timeout}s",
                ]
            else:
                command = [
                    "kubectl",
                    "wait",
                    f"{kind.lower()}/{name}",
                    "--namespace",
                    namespace,
                    "--for=condition=Available",
                    f"--timeout={self._timeout}s",
                ]
            status, stdout, stderr = await self._runner(command)
            if status != 0:
                return ValidationResult(False, [(stderr or stdout).strip()])
        status, stdout, stderr = await self._runner(
            ["kubectl", "get", "service", "--namespace", namespace, "--output", "json"]
        )
        if status != 0:
            return ValidationResult(False, [(stderr or stdout).strip()])
        services = json.loads(stdout).get("items", [])
        service = next(
            (
                item
                for item in services
                if any(token in item["metadata"]["name"] for token in ("epp", "gateway", "inference", "router"))
            ),
            None,
        )
        if service is None:
            return ValidationResult(False, ["pd-disaggregation routing service was not discovered"])
        port = next(
            (
                item["port"]
                for item in service["spec"].get("ports", [])
                if item.get("name") in {"http", "default", "web"} or item.get("port") == 80
            ),
            None,
        )
        if port is None:
            return ValidationResult(False, ["pd-disaggregation routing service has no HTTP port"])
        execution["endpoint_url"] = f"http://{service['metadata']['name']}.{namespace}.svc:{port}"
        return ValidationResult(True)

    async def diagnostics(self, execution):
        namespace = execution["namespace"]

        async def capture(command):
            status, stdout, stderr = await self._runner(command)
            return (stdout if status == 0 else stderr).strip()

        return {
            "namespace": namespace,
            "pods": await capture(["kubectl", "get", "pods", "--namespace", namespace, "-o", "wide"]),
            "events": await capture(["kubectl", "get", "events", "--namespace", namespace, "--sort-by=.lastTimestamp"]),
            "decode_modelserver_logs": await capture(
                [
                    "kubectl",
                    "logs",
                    "--namespace",
                    namespace,
                    "-l",
                    "llm-d.ai/role=decode",
                    "-c",
                    "modelserver",
                    "--tail=120",
                    "--prefix=true",
                ]
            ),
            "prefill_modelserver_logs": await capture(
                [
                    "kubectl",
                    "logs",
                    "--namespace",
                    namespace,
                    "-l",
                    "llm-d.ai/role=prefill",
                    "-c",
                    "modelserver",
                    "--tail=120",
                    "--prefix=true",
                ]
            ),
            "routing_proxy_logs": await capture(
                [
                    "kubectl",
                    "logs",
                    "--namespace",
                    namespace,
                    "-l",
                    "llm-d.ai/role=decode",
                    "-c",
                    "routing-proxy",
                    "--tail=120",
                    "--prefix=true",
                ]
            ),
            "routing_proxy_previous_logs": await capture(
                [
                    "kubectl",
                    "logs",
                    "--namespace",
                    namespace,
                    "-l",
                    "llm-d.ai/role=decode",
                    "-c",
                    "routing-proxy",
                    "--previous",
                    "--tail=120",
                    "--prefix=true",
                ]
            ),
            **{
                key: value
                for key, value in execution.items()
                if key.startswith("router_effective_") or key.startswith("router_rendered_")
            },
            "captured_at": datetime.now(UTC).isoformat(),
        }

    async def stop(self, execution, artifact):
        namespace = execution["namespace"]
        resources = self._workload_resources(artifact)
        if not resources:
            names = await self._readiness_deployments({**execution, "_artifact": artifact})
            resources = [("Deployment", name) for name in names]
        outputs = []
        for kind, name in resources:
            if kind == "Deployment":
                command = ["kubectl", "scale", f"deployment/{name}", "--namespace", namespace, "--replicas=0"]
            elif kind == "DisaggregatedSet":
                manifest_ref = artifact.manifest_ref
                document = next(
                    (
                        item
                        for item in yaml.safe_load_all(Path(manifest_ref).read_text(encoding="utf-8"))
                        if item.get("kind") == kind and item.get("metadata", {}).get("name") == name
                    ),
                    None,
                )
                roles = (document or {}).get("spec", {}).get("roles") or []
                patch = [
                    {"op": "replace", "path": f"/spec/roles/{index}/spec/replicas", "value": 0}
                    for index, _role in enumerate(roles)
                ]
                command = [
                    "kubectl",
                    "patch",
                    f"disaggregatedset/{name}",
                    "--namespace",
                    namespace,
                    "--type=json",
                    "--patch",
                    json.dumps(patch),
                ]
            else:
                command = ["kubectl", "scale", f"{kind.lower()}/{name}", "--namespace", namespace, "--replicas=0"]
            status, stdout, stderr = await self._runner(command)
            outputs.append(stderr or stdout)
        return {"stopped": True, "output": "\n".join(outputs), "error": None}

    async def rollback(self, execution, artifact):
        return await self.stop(execution, artifact)

    async def cleanup(self, execution, artifact, *, force=False):
        helm = await asyncio.create_subprocess_exec(
            str(self._helm_path),
            "uninstall",
            "pd-disaggregation",
            "--namespace",
            execution["namespace"],
            env=self._environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await helm.communicate()
        status, stdout, stderr = await self._runner(
            ["kubectl", "delete", "namespace", execution["namespace"], "--ignore-not-found=true"]
        )
        if artifact.manifest_ref:
            shutil.rmtree(Path(artifact.manifest_ref).parent, ignore_errors=True)
        return {"cleaned_up": status == 0, "output": stdout, "error": stderr or None}

    @staticmethod
    def _set_argument(args, name, value):
        return set_vllm_argument(args, name, value)

    @staticmethod
    def _set_environment(items, name, value):
        return (
            [{"name": name, "value": value} if item.get("name") == name else item for item in items]
            if any(item.get("name") == name for item in items)
            else [*items, {"name": name, "value": value}]
        )

    def _parameters(self, overrides):
        model, runtime = overrides.get("model") or {}, overrides.get("runtime") or {}
        if not model.get("name") or not runtime.get("image") or ":" not in runtime["image"]:
            raise ValueError("pd-disaggregation requires model.name and a tagged runtime.image")
        # The variant id is accelerator-specific (XPU: vllm/vllm-rdma; NVIDIA GPU:
        # the vLLM infra-provider overlay such as base); the adapter validates it
        # against the overlays available for the active accelerator.
        variant = str(
            overrides.get("guideVariant")
            or default_guide_variant(self._sources, guide="pd-disaggregation", accelerator=self._accelerator)
        )
        result: dict[str, Any] = {
            "variant": variant,
            "model": model["name"],
            "image": runtime["image"],
            "mount_path": runtime.get("mountPath") or "",
            "environment": runtime.get("environment") if isinstance(runtime.get("environment"), dict) else {},
            "custom_parameters": overrides.get("customParameters") or [],
            "official_guide": overrides.get("officialGuide")
            if isinstance(overrides.get("officialGuide"), dict)
            else None,
        }
        if not isinstance(result["mount_path"], str) or (
            result["mount_path"] and not result["mount_path"].startswith("/")
        ):
            raise ValueError("runtime.mountPath must be an absolute host path")
        for role in ("prefill", "decode"):
            values = overrides.get(role) or {}
            if not isinstance(values.get("replicaCount"), int) or not isinstance(values.get("tensorParallelSize"), int):
                raise ValueError(f"pd-disaggregation requires {role} replicaCount and tensorParallelSize")
            result[role] = {
                "replicas": values["replicaCount"],
                "tensor_parallel_size": values["tensorParallelSize"],
                "serving_arguments": serving_arguments(model, values),
            }
        return result
