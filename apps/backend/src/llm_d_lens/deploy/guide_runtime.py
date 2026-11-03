"""Lossless edits to supported model-server invocations during Guide planning."""

from __future__ import annotations

import json
import math
import re
import shlex
from copy import deepcopy

from llm_d_lens.configuration.managed_environment import (
    admin_managed_environment_error,
    is_admin_managed_environment_variable,
)
from llm_d_lens.configuration.manifest_facts import _invocation_tokens, _shell_tokens, workload_roles

_SHELL_WORD = re.compile(r"""(?:[^\s'"\\]+|'[^']*'|"(?:[^"\\]|\\.)*"|\\[\s\S])+""")
DEDICATED = {"model", "model-path", "served-model-name", "tensor-parallel-size", "tensor_parallel_size", "tp_size"}


def scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def runtime_overrides(request: dict) -> list[dict]:
    custom = request.get("customParameters")
    if custom is None:
        custom = []
    if not isinstance(custom, list):
        raise ValueError("Custom runtime overrides must be an array.")
    result = []
    fields = {
        "maxModelLen": "max-model-len",
        "maxNumSeqs": "max-num-seqs",
        "gpuMemoryUtilization": "gpu-memory-utilization",
        "blockSize": "block-size",
        "maxNumBatchedTokens": "max-num-batched-tokens",
    }
    for field, name in fields.items():
        if request.get(field) is None or str(request[field]).strip() == "":
            continue
        if request.get("modelServer") and request["modelServer"] != "vllm":
            raise ValueError(f"{field} is a vLLM control; inherit the selected runtime or use its native arguments.")
        try:
            value = float(request[field])
            valid = (
                0 < value <= 1 if field == "gpuMemoryUtilization" else value.is_integer() and 1 <= value <= 2**53 - 1
            )
            if not valid:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be a positive integer.") from error
        result.append({"target": "both", "kind": "argument", "name": name, "value": scalar(value)})
    for item in custom:
        if (
            not isinstance(item, dict)
            or item.get("target") not in {"prefill", "decode", "both"}
            or item.get("kind") not in {"argument", "environment"}
            or not isinstance(item.get("name"), str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", item["name"])
            or not isinstance(item.get("value"), (str, int, float, bool))
        ):
            raise ValueError("Each runtime override requires a valid target, kind, name and scalar value.")
        name, kind, target = item["name"], item["kind"], item["target"]
        if kind == "argument" and name in DEDICATED:
            raise ValueError(f"Use the dedicated configuration control for {name}.")
        if kind == "environment":
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid environment variable {name}.")
            if is_admin_managed_environment_variable(name):
                raise ValueError(admin_managed_environment_error(name))
        if any(
            previous["kind"] == kind
            and previous["name"] == name
            and (previous["target"] == target or "both" in {target, previous["target"]})
            for previous in result
        ):
            raise ValueError(f"Overlapping runtime override: {name}.")
        result.append(dict(item, value=scalar(item["value"])))
    return result


def shell_script(container: dict):
    command = container.get("command") or []
    args = container.get("args") or []
    if not any(re.search(r"(?:^|/)(?:ba)?sh$", part) for part in command):
        return None
    option = next((index for index, value in enumerate(command) if value in {"-c", "-lc"}), None)
    if option is None:
        raise ValueError("Unsupported model-server shell command.")
    parts, index = (command, option + 1) if len(command) > option + 1 else (args, 0)
    if index >= len(parts):
        raise ValueError("Unsupported model-server shell command.")
    _shell_tokens(parts[index], "guide")
    return parts, index


def shell_words(script: str):
    _shell_tokens(script, "guide")
    words = []
    for match in _SHELL_WORD.finditer(script):
        raw = match.group().replace("\\\n", "")
        if not raw:
            continue
        values = shlex.split(raw)
        if len(values) != 1:
            raise ValueError("Ambiguous model-server shell argument.")
        words.append((values[0], match.start(), match.end()))
    return words


def runtime_argument(container: dict, name: str):
    values = _invocation_tokens(container, "guide")
    found = [
        value[len(name) + 3 :] if value.startswith(f"--{name}=") else values[index + 1]
        for index, value in enumerate(values)
        if value.startswith(f"--{name}=") or value == f"--{name}" and index + 1 < len(values)
    ]
    if len(found) > 1:
        raise ValueError(f"Duplicate --{name} arguments are ambiguous.")
    return found[0] if found else None


def apply_runtime_overrides(container: dict, overrides: list[dict]):
    for item in overrides:
        name, value = item["name"], item["value"]
        if item["kind"] == "environment":
            container["env"] = [entry for entry in container.get("env", []) if entry.get("name") != name]
            container["env"].append({"name": name, "value": value})
            continue
        script = shell_script(container)
        if script:
            parts, index = script
            text = parts[index]
            if name == "max-num-seqs":
                text = re.sub(r"--max-num-seq(?==|\s|$)", "--max-num-seqs", text)
            words = shell_words(text)
            spans = []
            for word_index, (word, start, end) in enumerate(words):
                if word.startswith(f"--{name}="):
                    spans.append((start, end))
                elif word == f"--{name}":
                    following = words[word_index + 1] if word_index + 1 < len(words) else None
                    has_value = following and not following[0].startswith("--")
                    if not has_value and value != "":
                        raise ValueError(
                            f"Existing --{name} is valueless; leave its value empty "
                            "or use the Guide's explicit disabling flag."
                        )
                    spans.append((start, following[2] if has_value else end))
            if len(spans) > 1:
                raise ValueError(f"Duplicate --{name} arguments are ambiguous.")
            flag = f"--{name}" + ("=" + shlex.quote(value) if value != "" else "")
            if spans:
                start, end = spans[0]
                parts[index] = text[:start] + flag + text[end:]
            else:
                tail = text.rstrip()
                parts[index] = tail + ("\n" if tail.endswith("\\") else " ") + " " + flag
        else:
            command, args = container.get("command") or [], container.get("args") or []
            for parts in (command, args):
                index = len(parts) - 1
                while index >= 0:
                    token = parts[index]
                    if token == f"--{name}":
                        has_value = index + 1 < len(parts) and not str(parts[index + 1]).startswith("--")
                        if not has_value and value != "":
                            raise ValueError(f"Existing --{name} is valueless; leave its value empty.")
                        del parts[index : index + (2 if has_value else 1)]
                    elif str(token).startswith(f"--{name}="):
                        del parts[index]
                    index -= 1
            container["args"] = [*args, f"--{name}" + (f"={value}" if value != "" else "")]


def configure_model_server(container: dict, model: str, tp: int, identity: str, precise: bool):
    script = shell_script(container)
    command, args = container.get("command") or [], container.get("args") or []
    if script:
        parts, script_index = script
        words = shell_words(parts[script_index])
        values = [word[0] for word in words]
    else:
        _invocation_tokens(container, "guide")
        values = [*command, *args]
    edits = {}
    model_index = None
    for index, token in enumerate(values):
        if re.match(r"^--model(?:-path)?=", token):
            edits[index] = token.split("=", 1)[0] + "=" + model
            model_index = index
            break
        if token in {"--model", "--model-path"} and index + 1 < len(values):
            model_index = index + 1
            edits[model_index] = model
            break
    if model_index is None:
        model_index = values.index("serve") + 1 if "serve" in values else len(command)
        if model_index >= len(values) or values[model_index].startswith("-"):
            raise ValueError("Cannot identify the model argument in the selected Guide.")
        edits[model_index] = model
    for index, argument in enumerate(values):
        if argument.startswith("--served-model-name="):
            edits[index] = "--served-model-name=" + identity
        elif argument == "--served-model-name" and index + 1 < len(values):
            edits[index + 1] = identity
        if precise and (argument == "--kv-events-config" or argument.startswith("--kv-events-config=")):
            inline = "=" in argument
            target = index if inline else index + 1
            try:
                events = json.loads(argument.split("=", 1)[1] if inline else values[target])
            except (ValueError, IndexError) as error:
                raise ValueError("Precise routing requires a valid JSON kv-events-config.") from error
            if not isinstance(events.get("topic"), str) or not re.match(r"^kv@[^@]+@.+$", events["topic"]):
                raise ValueError("Precise routing requires a kv@host:port@model event topic.")
            events["topic"] = events["topic"].rsplit("@", 1)[0] + "@" + identity
            edits[target] = ("--kv-events-config=" if inline else "") + json.dumps(events, separators=(",", ":"))
        for alias in ("tensor_parallel_size", "tp_size"):
            if argument == f"--{alias}" or argument.startswith(f"--{alias}="):
                edits[index] = argument.replace(f"--{alias}", "--tensor-parallel-size", 1)
    if script:
        text = parts[script_index]
        for index, value in sorted(edits.items(), reverse=True):
            _, start, end = words[index]
            text = text[:start] + shlex.quote(value) + text[end:]
        parts[script_index] = text
    else:
        for index, value in edits.items():
            if index < len(command):
                command[index] = value
            else:
                args[index - len(command)] = value
    overrides = []
    if tp > 0:
        overrides.append({"kind": "argument", "name": "tensor-parallel-size", "value": scalar(tp)})
    if model != identity:
        overrides.append({"kind": "argument", "name": "served-model-name", "value": identity})
    apply_runtime_overrides(container, overrides)


def configure_cpu_cache(container: dict, gib: float):
    connector = json.loads(runtime_argument(container, "kv-transfer-config") or "{}")
    if connector.get("kv_connector") == "OffloadingConnector":
        connector.setdefault("kv_connector_extra_config", {})["cpu_bytes_to_use"] = math.floor(gib * 1024**3)
        override = {
            "kind": "argument",
            "name": "kv-transfer-config",
            "value": json.dumps(connector, separators=(",", ":")),
        }
    else:
        if not str(connector.get("kv_connector") or "").startswith("LMCacheConnector"):
            raise ValueError("CPU capacity requires an LMCache connector.")
        if any(entry.get("name") == "LMCACHE_CONFIG_FILE" for entry in container.get("env", [])):
            raise ValueError(
                "LMCache file configuration must expose capacity in its Guide; "
                "environment capacity cannot override it reliably."
            )
        override = {"kind": "environment", "name": "LMCACHE_MAX_LOCAL_CPU_SIZE", "value": scalar(gib)}
    apply_runtime_overrides(container, [override])


def guide_settings_capabilities(documents: list[dict], device_class: str | None = None) -> dict:
    servers = [
        container
        for workload in workload_roles(documents)
        for container in workload["pod"].get("containers", [])
        if container.get("name") == "modelserver"
    ]
    cpu = bool(servers)
    for container in servers:
        try:
            configure_cpu_cache(deepcopy(container), 1)
        except (ValueError, TypeError, KeyError):
            cpu = False
    nic = bool(device_class) and any(
        request.get("exactly", {}).get("deviceClassName") == device_class
        for document in documents
        if document.get("kind") == "ResourceClaimTemplate"
        for request in document.get("spec", {}).get("spec", {}).get("devices", {}).get("requests", [])
    )
    return {"cacheCpuGiB": cpu, "rdmaNicCount": nic}
