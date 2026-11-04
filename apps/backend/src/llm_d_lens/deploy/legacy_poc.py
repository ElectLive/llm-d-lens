"""Compatibility adapter for the original, explicitly selected PoC configurations."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
from datetime import UTC, datetime
from urllib.parse import quote

import httpx
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_d_lens.auth.access import current_principal, require_cluster_access
from llm_d_lens.cluster.sessions import require_active_session
from llm_d_lens.infrastructure.hostos import current_host
from llm_d_lens.infrastructure.process import ShellClient

router = APIRouter(prefix="/api/deploy-poc", include_in_schema=False)
KNOWN_GOOD = {
    "id": "optimized-baseline-xpu",
    "label": "Optimized Baseline \u00b7 Intel XPU (Qwen3-0.6B)",
    "guide": "optimized-baseline",
    "accelerator": "xpu",
    "model": "Qwen/Qwen3-0.6B",
    "defaultNamespace": "prism-deploy-poc",
}
SMOKE = {
    "id": "smoke-pause",
    "label": "Smoke Test \u00b7 pause (tiny, no image pull)",
    "image": "registry.k8s.io/pause:3.10.1",
    "defaultNamespace": "prism-smoke-test",
}
jobs: dict[str, dict] = {}
tasks: set[asyncio.Task] = set()


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def smoke_manifest(namespace: str) -> str:
    spec = {
        "restartPolicy": "Never",
        "priorityClassName": "system-node-critical",
        "tolerations": [{"key": "node.kubernetes.io/disk-pressure", "operator": "Exists", "effect": "NoSchedule"}],
        "containers": [
            {
                "name": "pause",
                "image": SMOKE["image"],
                "resources": {
                    "requests": {"cpu": "10m", "memory": "16Mi"},
                    "limits": {"cpu": "50m", "memory": "32Mi"},
                },
            }
        ],
    }
    target_node = os.environ.get("PRISM_K8S_TARGET_NODE", "").strip()
    if target_node:
        spec["nodeSelector"] = {"kubernetes.io/hostname": target_node}
    return yaml.safe_dump_all(
        [
            {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace}},
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "llm-d-smoke",
                    "namespace": namespace,
                    "labels": {"app": "llm-d-smoke", "llm-d.ai/role": "smoke"},
                },
                "spec": spec,
            },
        ],
        sort_keys=False,
    )


def child_environment(request: Request, session_id=None) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LLM_D_ROOT", None)
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy", "")
    cluster = os.environ.get("PRISM_CLUSTER_NO_PROXY", "localhost,127.0.0.1,.svc,.cluster.local")
    no_proxy = ",".join(value for value in (existing, cluster) if value)
    kubeconfig = os.environ.get("OPTIMALBENCH_KUBECONFIG", str(current_host().default_kubeconfig()))
    if session_id:
        session = require_active_session(session_id)
        require_cluster_access(current_principal(request), session.server_id)
        kubeconfig = str(session.kubeconfig_path)
    environment.update(KUBECONFIG=kubeconfig, NO_PROXY=no_proxy, no_proxy=no_proxy)
    return environment


def build_cli_deploy_command(kubectl: str, namespace: str, executable: str, root: str) -> str:
    query = shlex.join([kubectl, "get", "namespace", namespace])
    create = shlex.join([kubectl, "create", "namespace", namespace])
    deploy = shlex.join(
        [
            executable,
            "deploy",
            "guide",
            KNOWN_GOOD["guide"],
            "-e",
            KNOWN_GOOD["accelerator"],
            "-n",
            namespace,
            "--llm-d-root",
            root,
            "--no-wait",
        ]
    )
    return f"{query} >/dev/null 2>&1 || {create} ; exec {deploy}"


def push_log(job: dict, text: str) -> None:
    job["logTail"].extend(line for line in text.splitlines() if line.strip())
    del job["logTail"][:-200]


async def run_job(job: dict, executable: str, args: list[str], environment: dict[str, str], stdin: str | None) -> None:
    process = None
    try:
        process = await ShellClient().spawn_with_executable(
            executable,
            args,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **current_host().group_spawn_kwargs(),
        )

        async def read_stream(stream):
            while chunk := await stream.read(4096):
                push_log(job, chunk.decode(errors="replace"))

        async def feed_stdin():
            if stdin is not None:
                process.stdin.write(stdin.encode())
                await process.stdin.drain()
            process.stdin.close()

        await asyncio.gather(read_stream(process.stdout), read_stream(process.stderr), feed_stdin())
        job["exitCode"] = await process.wait()
        job["phase"] = "succeeded" if job["exitCode"] == 0 else "failed"
    except asyncio.CancelledError:
        job["phase"] = "failed"
        push_log(job, "deployment task cancelled")
        raise
    except OSError as error:
        job["phase"] = "failed"
        job["exitCode"] = -1
        push_log(job, f"spawn error: {error}")
    finally:
        if process is not None and process.returncode is None:
            host = current_host()
            with contextlib.suppress(ProcessLookupError):
                host.signal_group(process.pid, force=False)
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    host.signal_group(process.pid, force=True)
                await process.wait()
        job["finishedAt"] = timestamp()


def launch(
    namespace: str,
    action: str,
    executable: str,
    args: list[str],
    environment: dict[str, str],
    stdin: str | None = None,
):
    job = {"action": action, "phase": "running", "startedAt": timestamp(), "logTail": []}
    jobs[namespace] = job
    task = asyncio.create_task(run_job(job, executable, args, environment, stdin))
    tasks.add(task)
    task.add_done_callback(tasks.discard)


@router.on_event("shutdown")
async def stop_jobs():
    pending = list(tasks)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


@router.get("/config")
async def config():
    return {"configs": [SMOKE, KNOWN_GOOD]}


async def mutate(request: Request, action: str):
    body = await request.json()
    body = body if isinstance(body, dict) else {}
    descriptor = SMOKE if body.get("config") == SMOKE["id"] else KNOWN_GOOD
    namespace = str(body.get("namespace") or descriptor["defaultNamespace"]).strip()
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", namespace):
        return JSONResponse({"error": "invalid namespace"}, status_code=400)
    existing = jobs.get(namespace)
    if action == "deploy" and existing and existing["action"] == "deploy" and existing["phase"] == "running":
        return JSONResponse({"error": "deploy already running", "namespace": namespace}, status_code=409)
    try:
        environment = child_environment(request, body.get("clusterSessionId"))
    except ValueError:
        return JSONResponse({"error": "cluster session is no longer active"}, status_code=409)
    kubectl = os.environ.get("KUBECTL_BIN", "kubectl")
    executable = os.environ.get("OPTIMALBENCH_BIN", "llm-d-bench")
    root = os.environ.get("LLM_D_ROOT", "")
    stdin = None
    if descriptor is SMOKE:
        program = kubectl
        args = (
            ["apply", "-f", "-"]
            if action == "deploy"
            else [
                "delete",
                "namespace",
                namespace,
                "--ignore-not-found",
                "--wait=false",
            ]
        )
        stdin = smoke_manifest(namespace) if action == "deploy" else None
    elif action == "deploy":
        target_node = os.environ.get("PRISM_K8S_TARGET_NODE", "").strip()
        if target_node:
            return JSONResponse(
                {
                    "error": "legacy PoC deployment cannot enforce PRISM_K8S_TARGET_NODE; use the Deploy runtime",
                    "targetNode": target_node,
                },
                status_code=409,
            )
        program = "bash"
        args = ["-c", build_cli_deploy_command(kubectl, namespace, executable, root)]
    else:
        program = executable
        args = [
            "deploy",
            "destroy",
            KNOWN_GOOD["guide"],
            "-e",
            KNOWN_GOOD["accelerator"],
            "-n",
            namespace,
            f"--llm-d-root={root}",
            "--delete-namespace",
        ]
    launch(namespace, action, program, args, environment, stdin)
    response = {"namespace": namespace, "phase": "deploying" if action == "deploy" else "deleting"}
    if action == "deploy":
        response["config"] = descriptor["id"]
    return response


@router.post("/start")
async def start(request: Request):
    return await mutate(request, "deploy")


@router.post("/teardown")
async def teardown(request: Request):
    return await mutate(request, "teardown")


async def cluster_status(namespace: str):
    base = os.environ.get("OPTIMALBENCH_API", "http://127.0.0.1:8080").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(f"{base}/api/deployments/{quote(namespace, safe='')}/status")
            if response.is_success:
                return response.json(), None
            return None, f"status {response.status_code}"
    except (httpx.HTTPError, ValueError) as error:
        return None, str(error)


@router.get("/status")
async def status(namespace: str = KNOWN_GOOD["defaultNamespace"]):
    namespace = namespace.strip()
    job = jobs.get(namespace)
    state, error = await cluster_status(namespace)
    return {
        "namespace": namespace,
        "job": dict(job, logTail=job["logTail"][-100:]) if job else None,
        "k8s": state,
        "k8sError": error,
    }


@router.get("/validate")
async def validate(namespace: str = KNOWN_GOOD["defaultNamespace"]):
    namespace = namespace.strip()
    state, error = await cluster_status(namespace)
    state = state or {}
    total, ready = state.get("pods_total", 0), state.get("pods_ready", 0)
    checks = [
        {"name": "Namespace exists", "pass": bool(state.get("status")) and state["status"] != "not_found"},
        {"name": "Pods scheduled", "pass": total > 0},
        {"name": "All pods Ready", "pass": total > 0 and ready == total},
        {"name": "Gateway endpoint present", "pass": bool(state.get("gateway_endpoint"))},
    ]
    return {
        "namespace": namespace,
        "ready": total > 0 and ready == total,
        "checks": checks,
        "passed": sum(check["pass"] for check in checks),
        "total": len(checks),
        "endpoint": state.get("gateway_endpoint") or None,
        "error": error,
    }
