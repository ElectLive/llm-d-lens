"""Compatibility endpoints for explicitly confirmed remote SSH deployments."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path

import paramiko
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_d_lens.hardware.registry import all_profiles
from llm_d_lens.hardware.resolver import resolve_by_resource
from llm_d_lens.infrastructure.ssh import SshTarget, inspect_host_key, remote_exec

router = APIRouter(prefix="/api/remote-deploy", include_in_schema=False)
logger = logging.getLogger(__name__)
HOST_PATTERN = r"(?=.{1,253}$)(?:\[[0-9a-fA-F:]+\]|[a-zA-Z0-9](?:[a-zA-Z0-9.-]{0,251}[a-zA-Z0-9])?)"
NAMESPACE_PATTERN = r"(?=.{1,63}$)[a-z0-9]([-a-z0-9]*[a-z0-9])?"
TERMINAL_REASONS = {
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "RunContainerError",
}


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RemoteError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.public_message = message if status < 500 else "Remote deployment failed"


def assert_enabled():
    if os.environ.get("PRISM_REMOTE_DEPLOY_ENABLED") != "true" and os.environ.get("NODE_ENV") == "production":
        raise RemoteError("Remote deployment is disabled on this server", 503)


def parse_target(body, require_fingerprint=True, require_cards=True) -> dict:
    raw = body if isinstance(body, dict) else {}
    host, username = (str(raw.get(key) or "").strip() for key in ("host", "username"))
    auth_method = str(raw.get("authMethod") or "agent")
    fingerprint = str(raw.get("hostFingerprint") or "").strip()
    key_path = str(raw.get("keyPath") or "").strip()
    cards = (
        list(dict.fromkeys(str(value) for value in raw.get("cards", []))) if isinstance(raw.get("cards"), list) else []
    )
    if not re.fullmatch(HOST_PATTERN, host):
        raise RemoteError("Invalid SSH host")
    allowlist = [value.strip() for value in os.environ.get("PRISM_REMOTE_HOST_ALLOWLIST", "").split(",")]
    if os.environ.get("NODE_ENV") == "production" and host not in allowlist:
        raise RemoteError("SSH host is not in PRISM_REMOTE_HOST_ALLOWLIST", 403)
    try:
        port = float(raw.get("port") or 22)
        if not port.is_integer() or not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError) as error:
        raise RemoteError("Invalid SSH port") from error
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", username, re.IGNORECASE):
        raise RemoteError("Invalid SSH username")
    if auth_method not in {"agent", "key", "password"}:
        raise RemoteError("Invalid authentication method")
    if auth_method == "key" and not key_path:
        raise RemoteError("A server-side private key path is required")
    if require_fingerprint and not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}=?", fingerprint):
        raise RemoteError("A valid SSH host fingerprint is required")
    if require_cards and (not cards or any(not re.fullmatch(r"[0-9]{1,3}", value) for value in cards)):
        raise RemoteError("Select at least one valid accelerator card")
    return {
        "host": host,
        "port": int(port),
        "username": username,
        "authMethod": auth_method,
        "keyPath": key_path,
        "hostFingerprint": fingerprint,
        "cards": cards,
    }


def connection_target(target: dict, credentials: dict) -> SshTarget:
    password, key = None, None
    agent = target["authMethod"] == "agent"
    if target["authMethod"] == "password":
        password = credentials.get("password")
        if not password:
            raise RemoteError("Password is required")
    elif target["authMethod"] == "key":
        directory = Path(os.environ.get("PRISM_SSH_KEY_DIR", str(Path.home() / ".ssh"))).resolve()
        resolved = Path(target["keyPath"]).expanduser().resolve()
        if not resolved.is_relative_to(directory):
            raise RemoteError(f"Private key must be inside {directory}")
        key = str(resolved)
    elif not os.environ.get("SSH_AUTH_SOCK"):
        raise RemoteError("SSH_AUTH_SOCK is not available to the Lens server")
    return SshTarget(
        host=target["host"].strip("[]"),
        port=target["port"],
        username=target["username"],
        key_filename=key,
        password=password,
        host_key_fingerprint=target["hostFingerprint"].rstrip("="),
        allow_agent=agent,
    )


async def execute(target: dict, credentials: dict, command: str) -> dict:
    result = await asyncio.to_thread(remote_exec, connection_target(target, credentials), command, 600)
    return {"stdout": result.stdout[-32000:], "stderr": result.stderr[-32000:], "code": result.returncode}


async def execute_remote_inspection(target_body, credentials_body, command: str) -> dict:
    assert_enabled()
    target = parse_target(target_body, require_cards=False)
    return await execute(target, credentials_body if isinstance(credentials_body, dict) else {}, command)


def guide_for(plan: dict) -> str:
    guide = str(plan.get("guide") or "").strip()
    if guide:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", guide):
            raise RemoteError("Invalid deployment guide")
        return guide
    source = plan.get("source")
    if source == "epd":
        raise RemoteError("E/P/D deployment is not supported by the installed OptimalBench CLI")
    if source == "tiered-cache":
        return "tiered-prefix-cache"
    if source == "precise-prefix-cache":
        return "precise-prefix-cache-routing"
    return "pd-disaggregation" if float(plan.get("prefillReplicas") or 0) > 0 else "optimized-baseline"


def target_value(plan: dict, name: str) -> str:
    target = plan.get("deploymentTarget")
    return str(target.get(name) or "").strip() if isinstance(target, dict) else ""


def prepare_repository_command(plan: dict) -> dict:
    repository, branch = target_value(plan, "repository"), target_value(plan, "branch") or "main"
    if not repository:
        raise RemoteError("Deployment repository path or GitHub URL is required")
    if not re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", branch):
        raise RemoteError("Invalid repository branch")
    local = repository.startswith(("/", "~/"))
    github = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/?", repository)
    if not local and not github:
        raise RemoteError("Repository must be an absolute local path or a GitHub HTTPS URL")
    if local:
        root = f'"$HOME"/{shlex.quote(repository[2:])}' if repository.startswith("~/") else shlex.quote(repository)
        command = f'LLM_D_ROOT={root}; test -d "$LLM_D_ROOT/.git"; git -C "$LLM_D_ROOT" fetch origin --prune'
    else:
        owner, name = github.groups()
        name = name.removesuffix(".git")
        if owner in {".", ".."} or name in {"", ".", ".."}:
            raise RemoteError("Invalid GitHub repository path")
        cache = shlex.quote(f"deploy-repos/{owner}/{name}")
        command = (
            'LENS_DEPLOY_CACHE="${LENS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lens}"; '
            f'LLM_D_ROOT="$LENS_DEPLOY_CACHE"/{cache}; mkdir -p "$(dirname "$LLM_D_ROOT")"; '
            'if test -d "$LLM_D_ROOT/.git"; then git -C "$LLM_D_ROOT" fetch origin --prune; '
            f'else git clone {shlex.quote(repository)} "$LLM_D_ROOT"; fi'
        )
    return {
        "command": f'{command}; git -C "$LLM_D_ROOT" checkout {shlex.quote(branch)}',
        "rootVariable": '"$LLM_D_ROOT"',
    }


def parse_namespace(value) -> str:
    namespace = str(value or "").strip()
    if not re.fullmatch(NAMESPACE_PATTERN, namespace):
        raise RemoteError("Invalid namespace")
    return namespace


def modelserver_patch_command(namespace: str, mount_path: str, environment: list[dict]) -> str:
    encoded = base64.b64encode(json.dumps({"mountPath": mount_path, "environment": environment}).encode()).decode()
    script = """import base64,json,subprocess,sys,time
cfg=json.loads(base64.b64decode(sys.argv[1]))
names=[]
for attempt in range(60):
 try: names=subprocess.check_output(["kubectl","get","deployment","-n",sys.argv[2],"-o","name"],text=True).splitlines()
 except subprocess.CalledProcessError: names=[]
 if names: break
 time.sleep(1)
if not names: raise SystemExit("modelserver deployment was not created within 60 seconds")
for deployment in names:
 data=json.loads(subprocess.check_output(["kubectl","get",deployment,"-n",sys.argv[2],"-o","json"],text=True))
 containers=data["spec"]["template"]["spec"].get("containers",[])
 container=next((item for item in containers if item.get("name") in ("modelserver","vllm")),None)
 if not container: continue
 spec={"containers":[{"name":container["name"],"env":list(cfg["environment"])}]}
 if cfg["mountPath"]:
  spec["volumes"]=[{"name":"model-cache-hostpath","hostPath":{"path":cfg["mountPath"],"type":"DirectoryOrCreate"}}]
  spec["containers"][0]["volumeMounts"]=[{"name":"model-cache-hostpath","mountPath":"/model-cache","readOnly":True}]
  spec["containers"][0]["env"].append({"name":"HF_HOME","value":"/model-cache"})
 patch={"spec":{"template":{"spec":spec}}}
 subprocess.check_call(["kubectl","patch",deployment,"-n",sys.argv[2],"--type","strategic","--patch",json.dumps(patch)])
"""
    return shlex.join(["python3", "-c", script, encoded, namespace])


def deployment_command(plan: dict, target: dict, namespace: str, name: str) -> str:
    accelerator = str(plan.get("accelerator") or "")
    profile = next(
        (
            profile
            for profile in all_profiles()
            if profile.deployment.legacy_cli.get("accelerator_pattern")
            and re.search(profile.deployment.legacy_cli["accelerator_pattern"], accelerator, re.IGNORECASE)
        ),
        None,
    )
    if profile is None:
        raise RemoteError("The installed OptimalBench deploy backend currently supports Intel Data Center GPU/XPU only")
    if float(plan.get("nodes") or 1) != 1:
        raise RemoteError("Remote SSH deployment currently supports one Kubernetes control-plane machine only")
    values = {
        "prefillTp": plan.get("prefillTp") or 0,
        "prefillReplicas": plan.get("prefillReplicas") or 0,
        "decodeTp": plan.get("decodeTp") or 1,
        "decodeReplicas": plan.get("decodeReplicas") or plan.get("replicas") or 1,
    }
    for label, value in values.items():
        try:
            parsed = float(value)
            if not parsed.is_integer() or not 0 <= parsed <= 128 or (label.startswith("decode") and parsed < 1):
                raise ValueError
            values[label] = int(parsed)
        except (TypeError, ValueError) as error:
            raise RemoteError(f"Invalid {label}") from error
    required = values["prefillTp"] * values["prefillReplicas"] + values["decodeTp"] * values["decodeReplicas"]
    if required > len(target["cards"]):
        raise RemoteError(f"Topology requires {required} cards but only {len(target['cards'])} card IDs were selected")
    mount_path, mount_name = target_value(plan, "mountPath"), target_value(plan, "mountModelName")
    if bool(mount_path) != bool(mount_name):
        raise RemoteError("Model cache host path and mounted model name must be provided together")
    if mount_path and (not mount_path.startswith("/") or "\n" in mount_path or "\n" in mount_name):
        raise RemoteError("Model cache host path must be an absolute path and mounted model name must be one line")
    environment = [
        {"name": name, "value": target_value(plan, field)}
        for name, field in (("HTTP_PROXY", "httpProxy"), ("HTTPS_PROXY", "httpsProxy"), ("NO_PROXY", "noProxy"))
        if target_value(plan, field)
    ]
    repository = prepare_repository_command(plan)
    argv = [
        os.environ.get("PRISM_REMOTE_DEPLOY_BIN", "llm-d-bench"),
        "deploy",
        "custom",
        guide_for(plan),
        "-e",
        profile.deployment.legacy_cli["accelerator_key"],
        "-n",
        namespace,
        "--release-postfix",
        name,
    ]
    command = shlex.join(argv) + ' --llm-d-root "$LLM_D_ROOT" --no-wait'
    command += shlex.join(
        [
            "",
            "--prefill-replicas",
            str(values["prefillReplicas"]),
            "--prefill-tp",
            str(max(1, values["prefillTp"])),
            "--decode-replicas",
            str(values["decodeReplicas"]),
            "--decode-tp",
            str(values["decodeTp"]),
        ]
    )[2:]
    model = f"/model-cache/{mount_name}" if mount_path else str(plan.get("model") or "").strip()
    if model:
        effective = model if mount_path or model.startswith("hf://") else f"hf://{model}"
        command += " --model " + shlex.quote(effective)
    host_environment = " ".join(f"{item['name']}={shlex.quote(item['value'])}" for item in environment)
    patch_command = modelserver_patch_command(namespace, mount_path, environment)
    return f"{repository['command']}; {host_environment} {command}; {patch_command}"


async def start_remote(body: dict, target: dict, credentials: dict) -> dict:
    plan = body.get("plan") or {}
    namespace, name = (
        str(plan.get("namespace") or "llm-d-optimized").strip(),
        str(plan.get("name") or "llm-d-service").strip(),
    )
    if not re.fullmatch(NAMESPACE_PATTERN, namespace) or not re.fullmatch(NAMESPACE_PATTERN, name):
        raise RemoteError("Invalid service name or namespace")
    result = await execute(target, credentials, deployment_command(plan, target, namespace, name))
    if result["code"]:
        raise RemoteError(
            f"Remote deploy exited with code {result['code']}: {result['stderr'] or result['stdout']}", 502
        )
    response = {
        "id": f"remote-{int(time.time() * 1000)}",
        "name": name,
        "candidateId": plan.get("candidateId") or None,
        "candidateName": plan.get("candidateName") or "Standalone manual configuration",
        "cluster": f"{target['username']}@{target['host']}:{target['port']}",
        "remoteHost": target["host"],
        "nodes": 1,
        "gpusPerNode": len(target["cards"]),
        "selectedCards": target["cards"],
        "replicas": float(plan.get("replicas") or 1),
        "namespace": namespace,
        "status": "deploying",
        "createdAt": timestamp(),
        "endpoint": None,
        "pods": [],
        "logTail": combined(result).splitlines()[-100:],
    }
    response.update({key: plan[key] for key in ("machineType", "accelerator") if key in plan})
    return response


def combined(result: dict) -> str:
    return (result["stdout"] + result["stderr"]).strip()


def parse_json(result: dict, label: str) -> dict:
    if result["code"] != 0:
        raise RemoteError(f"{label}: {combined(result)}", 502)
    try:
        return json.loads(result["stdout"])
    except ValueError as error:
        raise RemoteError(f"{label} returned invalid JSON", 502) from error


async def remote_status(target: dict, credentials: dict, namespace: str) -> dict:
    quoted = shlex.quote(namespace)
    result = await execute(target, credentials, f"kubectl get pods -n {quoted} -o json")
    if result["code"] and "not found" in combined(result).lower():
        return {
            "status": "not_found",
            "pods": [],
            "podsReady": 0,
            "podsTotal": 0,
            "endpoint": None,
            "error": combined(result),
            "logTail": [],
        }
    pods = []
    for pod in parse_json(result, "kubectl get pods").get("items", []):
        statuses = pod.get("status", {}).get("containerStatuses", [])
        states = [
            item.get("state", {}).get("waiting") or item.get("state", {}).get("terminated") or {} for item in statuses
        ]
        resources = [
            container.get("resources", {}).get("limits", {}) for container in pod.get("spec", {}).get("containers", [])
        ]
        gpu_count = sum(
            float(value)
            for limits in resources
            for key, value in limits.items()
            if (profile := resolve_by_resource(key))
            and not any(key.endswith(suffix) for suffix in profile.monitor_resource_suffixes)
        )
        pods.append(
            {
                "name": pod.get("metadata", {}).get("name", "unknown"),
                "status": pod.get("status", {}).get("phase", "Unknown"),
                "ready": any(
                    item.get("type") == "Ready" and item.get("status") == "True"
                    for item in pod.get("status", {}).get("conditions", [])
                ),
                "restarts": sum(item.get("restartCount", 0) for item in statuses),
                "node": pod.get("spec", {}).get("nodeName"),
                "gpuCount": gpu_count,
                "reason": next((state["reason"] for state in states if state.get("reason")), None),
                "message": next((state["message"] for state in states if state.get("message")), None),
            }
        )
    failed = next((pod for pod in pods if pod["status"] == "Failed" or pod["reason"] in TERMINAL_REASONS), None)
    ready = sum(pod["ready"] for pod in pods)
    state = "failed" if failed else "ready" if pods and ready == len(pods) else "deploying"
    gateways = parse_json(
        await execute(
            target,
            credentials,
            f"kubectl get gateways.gateway.networking.k8s.io -n {quoted} -o json 2>/dev/null "
            "|| printf '{\"items\":[]}'",
        ),
        "kubectl get gateways",
    )
    services = parse_json(
        await execute(target, credentials, f"kubectl get services -n {quoted} -o json"), "kubectl get services"
    )
    address = next(
        (
            entry["value"]
            for gateway in gateways.get("items", [])
            for entry in gateway.get("status", {}).get("addresses", [])
            if entry.get("value")
        ),
        None,
    )
    load_balancer = next(
        (service for service in services.get("items", []) if service.get("spec", {}).get("type") == "LoadBalancer"), {}
    )
    ingress = (load_balancer.get("status", {}).get("loadBalancer", {}).get("ingress") or [{}])[0]
    host = address or ingress.get("ip") or ingress.get("hostname")
    port = (load_balancer.get("spec", {}).get("ports") or [{}])[0].get("port")
    gateway_service = next(
        (
            service
            for service in services.get("items", [])
            if service.get("metadata", {}).get("labels", {}).get("gateway.networking.k8s.io/gateway-name")
            or "gateway-istio" in service.get("metadata", {}).get("name", "").lower()
        ),
        {},
    )
    endpoint = None
    if host:
        endpoint = f"http://{host}" + (f":{port}" if port and port != 80 else "")
    elif gateway_service.get("metadata", {}).get("name"):
        port = (gateway_service.get("spec", {}).get("ports") or [{}])[0].get("port") or 80
        endpoint = f"http://{gateway_service['metadata']['name']}.{namespace}.svc.cluster.local:{port}"
    if state == "ready" and not endpoint:
        state = "ready_no_endpoint"
    error = None
    if failed:
        error = f"{failed['name']}: {failed['reason'] or failed['status']}"
        if failed["message"]:
            error += f" \u2014 {failed['message']}"
    return {
        "status": state,
        "pods": pods,
        "podsReady": ready,
        "podsTotal": len(pods),
        "endpoint": endpoint,
        "error": error,
        "logTail": [],
    }


@router.get("/defaults")
async def defaults():
    return {
        "repository": os.environ.get("LLM_D_ROOT") or os.environ.get("LLM_D_REPOSITORY", ""),
        "branch": os.environ.get("LLM_D_BRANCH", ""),
        "httpProxy": os.environ.get("HTTP_PROXY") or os.environ.get("LLM_D_HTTP_PROXY", ""),
        "httpsProxy": os.environ.get("HTTPS_PROXY") or os.environ.get("LLM_D_HTTPS_PROXY", ""),
        "noProxy": os.environ.get("NO_PROXY") or os.environ.get("LLM_D_NO_PROXY", ""),
        "mountPath": os.environ.get("LLM_D_MOUNT_PATH", ""),
        "mountModelName": os.environ.get("LLM_D_MOUNT_MODEL_NAME", ""),
    }


async def remote_operation(request: Request):
    try:
        assert_enabled()
        body = await request.json()
        operation = request.url.path.rsplit("/", 1)[1]
        raw_target = body.get("target") or {}
        if operation == "fingerprint":
            target = parse_target(dict(raw_target, username=raw_target.get("username") or "probe", cards=["0"]), False)
            key = await asyncio.to_thread(inspect_host_key, target["host"].strip("[]"), target["port"], 10)
            return {"fingerprint": key.fingerprint, "host": target["host"], "port": target["port"]}
        target = parse_target(raw_target)
        credentials = body.get("credentials") or {}
        if operation == "start":
            return await start_remote(body, target, credentials)
        if operation == "test":
            discovery = "; ".join(
                f"({shlex.join(profile.planning.discovery['command'])} 2>/dev/null || true)"
                for profile in all_profiles()
                if profile.planning.discovery.get("command")
            )
            command = (
                'set -o pipefail; printf "host="; hostname; printf "os="; uname -srm; '
                f"{discovery}; "
                "command -v llm-d-bench; llm-d-bench --version 2>/dev/null || true; "
                "kubectl version --client; kubectl cluster-info"
            )
            result = await execute(target, credentials, command)
            return {"ok": result["code"] == 0, "output": combined(result)[-12000:]}
        namespace = parse_namespace(body.get("namespace"))
        if operation == "status":
            return await remote_status(target, credentials, namespace)
        executable = shlex.quote(os.environ.get("PRISM_REMOTE_DEPLOY_BIN", "llm-d-bench"))
        if operation == "logs":
            pod = str(body.get("pod") or "").strip()
            if pod and not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", pod):
                raise RemoteError("Invalid pod name")
            try:
                tail = min(1000, max(1, int(body.get("tail") or 100)))
            except (ValueError, TypeError) as error:
                raise RemoteError("Invalid log tail") from error
            command = f"{executable} deploy logs -n {shlex.quote(namespace)}"
            if pod:
                command += f" -p {shlex.quote(pod)}"
            result = await execute(target, credentials, f"{command} --tail {tail}")
            if result["code"]:
                raise RemoteError(combined(result), 502)
            return {"logs": result["stdout"][-32000:], "truncated": len(result["stdout"]) >= 32000}
        result = await execute(target, credentials, f"{executable} deploy cleanup -n {shlex.quote(namespace)} -f")
        if result["code"]:
            raise RemoteError(combined(result), 502)
        return {"success": True, "namespace": namespace, "logTail": combined(result).splitlines()[-100:]}
    except RemoteError as error:
        logger.warning("Remote deployment request failed", exc_info=True)
        return JSONResponse(
            {"error": error.public_message},
            status_code=error.status,
        )
    except (OSError, paramiko.SSHException, EOFError):
        logger.exception("Remote deployment failed")
        return JSONResponse({"error": "Remote deployment failed"}, status_code=502)


for operation in ("fingerprint", "test", "start", "status", "logs", "teardown"):
    router.add_api_route(f"/{operation}", remote_operation, methods=["POST"])
