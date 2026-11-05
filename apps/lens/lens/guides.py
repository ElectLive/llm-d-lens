"""llm-d guides: what a guide deploys into one namespace, read from its ``guide.yaml``.

``guides/<guide>/guide.yaml`` is the file llm-d generates each guide's README commands from
(``scripts/guide.py``): typed variables (``env``), and ordered ``prerequisites`` and
``deploy`` steps whose ``when:`` filters pick an accelerator, a model server and a
provider. llmd follows those steps without running a shell:

* ``export`` assignments set variables, as they do in the README;
* ``helm install`` of a chart becomes ``helm template`` of the same chart, version and values;
* ``kubectl apply -k DIR`` and ``kubectl apply -f FILE`` (also ``kubectl kustomize DIR |
  kubectl apply -f -``) render that directory or file of the checkout;
* creating the namespace or a Secret is left to the user: llmd does not create namespaces,
  and the plan checks that the Secrets the pods use exist;
* read-only commands (``kubectl wait``, ``kubectl get``, ``echo``) are not needed.

Only Standalone Mode is followed, since Gateway Mode needs a Gateway in the cluster.
Cluster-wide prerequisites (CRDs, operators) are listed for an administrator. A deploy step
llmd cannot follow (a pipe through ``sed``, a script) blocks the plan rather than being left
out, unless the guide's own CI skips it too (``skip_in: [ci]``).

A step that applies a directory the checkout does not have is another variant's step (the
TPU dynamic-slicing overlay when the accelerator is a GPU, say) and is left out; a ``deploy``
section left with none of its steps that way blocks the plan.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yamlio
from .catalog import GUIDE_FILE, inside, kustomization_file, manifest_files
from .errors import UsageError
from .model import Finding

#: The router mode llmd follows; the others need cluster-wide objects (a Gateway).
OTHER_MODES = {"gateway"}
#: Variables that pick the guide's variant, in the order the plan names them.
VARIANT_VARIABLES = ("ACCELERATOR_TYPE", "MODEL_SERVER", "INFRA_PROVIDER")
#: Set by llmd itself: the checkout, and the target namespace (``-n``).
FIXED = ("REPO_ROOT", "NAMESPACE")

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_EXPORT = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_IF_TEST = re.compile(
    r'^if\s+\[\[\s+"?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?"?\s+(==|!=)\s+"?([^"\s\]]*)"?\s+\]\]\s*;\s*then$'
)
_SEPARATORS = {";", "&&", "||", "&"}
_PIPES = {"|", "|&"}
#: Commands that change nothing in the cluster.
_NO_OPS = {
    "echo",
    "printf",
    "cd",
    "sleep",
    "true",
    "false",
    ":",
    "git",
    "source",
    ".",
    "set",
    "unset",
    "[[",
    "test",
    "exit",
    "return",
}
#: Filters a pipeline may pass output through when it only reads.
_FILTERS = {"grep", "tr", "sed", "awk", "head", "tail", "sort", "uniq", "jq", "yq", "base64", "cut", "wc", "cat"}
_KUBECTL_READS = {
    "get",
    "describe",
    "wait",
    "logs",
    "version",
    "api-resources",
    "api-versions",
    "explain",
    "top",
    "kustomize",
    "config",
    "auth",
    "rollout",
}
_HELM_READS = {"template", "show", "pull", "repo", "version", "list", "status", "get", "search", "dependency"}
_HELM_IGNORED = {"--wait", "--atomic", "--create-namespace", "--wait-for-jobs", "--debug", "--dependency-update"}


@dataclass(frozen=True)
class Guide:
    directory: Path
    #: The guide's path in the source, e.g. ``guides/optimized-baseline``.
    id: str
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.directory.name)

    def static(self) -> dict[str, Any]:
        env = self.data.get("env") if isinstance(self.data.get("env"), dict) else {}
        static = env.get("static")
        return static if isinstance(static, dict) else {}

    def variables(self) -> list[dict[str, Any]]:
        """The variables the guide declares, with their defaults and allowed values."""
        found = []
        for name, spec in self.static().items():
            spec = spec if isinstance(spec, dict) else {"default": spec}
            entry = {"name": str(name), "default": "" if spec.get("default") is None else _text(spec["default"])}
            if isinstance(spec.get("values"), list):
                entry["values"] = [_text(value) for value in spec["values"]]
            if spec.get("sensitive"):
                entry["sensitive"] = True
            found.append(entry)
        return found

    def support(self) -> dict[str, dict[str, str]]:
        """``{accelerator: {engine: status}}`` from the guide's support matrix."""
        accelerators = ((self.data.get("support") or {}).get("accelerators")) or {}
        found = {}
        for accelerator, info in accelerators.items() if isinstance(accelerators, dict) else []:
            engines = (info or {}).get("engines") or {}
            found[str(accelerator)] = {
                str(engine): str(cell.get("status") if isinstance(cell, dict) else cell)
                for engine, cell in engines.items()
            }
        return found


def load_guide(directory: Path, identifier: str) -> Guide:
    data = yamlio.load_file(directory / GUIDE_FILE)
    if not isinstance(data, dict):
        raise UsageError(f"{identifier}/{GUIDE_FILE} is not a mapping")
    return Guide(directory=directory, id=identifier, data=data)


@dataclass(frozen=True)
class Unit:
    """One thing a guide applies: a Helm chart, a kustomize directory, or manifest files."""

    kind: str  # helm | kustomize | manifests
    step: str
    path: Path | None = None
    #: For a Helm chart: the chart as the guide names it (an ``oci://`` reference or a local path).
    chart: str = ""
    release: str = ""
    version: str = ""
    values: tuple[Path, ...] = ()
    #: ``(flag, "key=value")`` pairs: ``--set``, ``--set-string``, ``--set-json``.
    sets: tuple[tuple[str, str], ...] = ()


@dataclass
class Followed:
    """The guide's steps as llmd follows them for one namespace."""

    units: list[Unit] = field(default_factory=list)
    #: One entry per step: ``{"step", "state", "detail"}``.
    steps: list[dict[str, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)


class _UnsetError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Environment:
    """Variables as the README's shell would see them, expanded when they are assigned."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        #: Variables that have no value, and why (sensitive, or set from a command's output).
        self.unknown: dict[str, str] = {}

    def assign(self, name: str, raw: str) -> None:
        try:
            self.values[name] = self.expand(raw)
            self.unknown.pop(name, None)
        except _UnsetError as unset:
            self.values.pop(name, None)
            self.unknown[name] = unset.reason

    def set(self, name: str, value: str) -> None:
        self.values[name] = value
        self.unknown.pop(name, None)

    def expand(self, text: str) -> str:
        """``$NAME``, ``${NAME}`` and ``${NAME:-default}``; a command substitution has no value."""
        if "`" in text:
            raise _UnsetError("it is set from a command's output")
        out, index, quoted = [], 0, False
        while index < len(text):
            char = text[index]
            following = text[index + 1 : index + 2]
            if char == '"':
                quoted = not quoted
            if char == "'" and not quoted:
                end = text.find("'", index + 1)
                end = len(text) - 1 if end < 0 else end
                out.append(text[index : end + 1])
                index = end + 1
            elif char != "$" or not following:
                out.append(char)
                index += 1
            elif following == "(":
                raise _UnsetError("it is set from a command's output")
            elif following == "{":
                end = _closing(text, index + 1)
                if end < 0:
                    raise _UnsetError(f"{text!r} has an unbalanced ${{")
                name, separator, default = text[index + 2 : end].partition(":-")
                if not _NAME.fullmatch(name):
                    raise _UnsetError(f"${{{text[index + 2 : end]}}} is not a plain variable")
                out.append(self._value(name, default if separator else None))
                index = end + 1
            elif match := _NAME.match(text, index + 1):
                out.append(self._value(match.group(0), None))
                index = match.end()
            else:
                out.append(char)
                index += 1
        return "".join(out)

    def _value(self, name: str, default: str | None) -> str:
        value = self.values.get(name)
        if value is not None and (value or default is None):
            return value
        if default is not None:
            return self.expand(default)
        raise _UnsetError(self.unknown.get(name) or f"${name} has no value")

    def get(self, name: str) -> str | None:
        return self.values.get(name)


def _closing(text: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return -1


def _text(value: Any) -> str:
    """A YAML scalar as the shell sees it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


# -- the variables -------------------------------------------------------------------------------------
def _env_files(guide: Guide, root: Path, env: Environment) -> list[Path]:
    """The files ``env.source`` sources (normally ``guides/env.sh``)."""
    files = []
    sources = (guide.data.get("env") or {}).get("source") or []
    for entry in sources if isinstance(sources, list) else [sources]:
        text = str(entry).split("#", 1)[0].strip()
        try:
            path = inside(root, Path(env.expand(text)).resolve().relative_to(root.resolve()).as_posix())
        except (_UnsetError, ValueError, UsageError):
            continue
        if path.is_file():
            files.append(path)
    return files


def _exported(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := _EXPORT.match(line.strip()):
            names.add(match.group(1))
    return names


def _run_env_lines(env: Environment, lines: list[str]) -> None:
    """Top-level ``export`` lines and ``if [[ $X == "v" ]]; then ... else ... fi`` blocks of an env
    file. Anything else (other tests, functions) is left out."""
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("if ") or line.startswith("if["):
            then_lines, else_lines, index = _if_block(lines, index)
            test = _IF_TEST.match(line)
            if test:
                value = env.get(test.group(1)) or ""
                matches = (value == test.group(3)) == (test.group(2) == "==")
                _run_env_lines(env, then_lines if matches else else_lines)
            continue
        if match := _EXPORT.match(line):
            value = match.group(2).split(" #", 1)[0]
            env.assign(match.group(1), _unquote(value))


def _if_block(lines: list[str], index: int) -> tuple[list[str], list[str], int]:
    depth, then_lines, else_lines, current = 1, [], [], None
    current = then_lines
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line.startswith("if ") or line.startswith("if["):
            depth += 1
        elif line == "fi" or line.startswith(("fi ", "fi;")):
            depth -= 1
            if depth == 0:
                return then_lines, else_lines, index
        elif line == "else" and depth == 1:
            current = else_lines
            continue
        current.append(line)
    return then_lines, else_lines, index


def environment(guide: Guide, root: Path, *, namespace: str, overrides: dict[str, str]) -> Environment:
    """The variables the guide's steps see: its declared defaults (``--var`` replaces them), the
    target namespace, the checkout, then the env files it sources."""
    env = Environment()
    env.set("REPO_ROOT", str(root))
    env.set("NAMESPACE", namespace)
    static = guide.static()
    for name in FIXED:
        if name in overrides:
            raise UsageError(
                f"--var {name} is set by llmd",
                hint="The namespace comes from -n, and REPO_ROOT is the source checkout.",
            )
    files = _env_files(guide, root, env)
    known = set(static) | {name for path in files for name in _exported(path)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise UsageError(
            f"guide {guide.id} has no variable {', '.join(unknown)}",
            hint="Its variables: " + ", ".join(sorted(name for name in static if name not in FIXED)),
        )
    for name, spec in static.items():
        if name in FIXED:
            continue
        if name in overrides:
            env.set(name, overrides[name])
            continue
        spec = spec if isinstance(spec, dict) else {"default": spec}
        if spec.get("sensitive"):
            env.unknown[name] = f"${name} is sensitive; llmd does not read it"
        elif spec.get("default") is not None:
            env.assign(name, _text(spec["default"]))
    for name, value in overrides.items():
        if name not in static:
            env.set(name, value)  # an env file's ``${NAME:-default}`` keeps it, as the shell would
    for path in files:
        _run_env_lines(env, path.read_text(encoding="utf-8").splitlines())
    return env


# -- the steps -----------------------------------------------------------------------------------------
def _steps(node: Any, path: str):
    if isinstance(node, dict) and "run" in node:
        yield path, node
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and "run" in item:
                yield path, item
    elif isinstance(node, dict):
        for key, child in node.items():
            yield from _steps(child, f"{path}.{key}")


def _when(step: dict[str, Any], env: Environment) -> bool:
    for name, allowed in (step.get("when") or {}).items():
        if env.get(str(name)) not in [_text(item) for item in allowed or []]:
            return False
    return True


def _lines(run: str) -> list[str]:
    lines = []
    for line in str(run).replace("\\\n", " ").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _commands(tokens: list[str]) -> list[list[list[str]]]:
    """Commands separated by ``;``, ``&&`` and ``||``, each a pipeline of words; redirections and
    braces are left out."""
    commands: list[list[list[str]]] = [[[]]]
    skip = False
    for token in tokens:
        if skip:
            skip = False
            continue
        if token in _SEPARATORS:
            commands.append([[]])
        elif token in _PIPES:
            commands[-1].append([])
        elif set(token) <= set("<>&") and ("<" in token or ">" in token):
            segment = commands[-1][-1]
            if segment and segment[-1] in ("0", "1", "2"):
                segment.pop()
            skip = True
        elif token not in ("{", "}"):
            commands[-1][-1].append(token)
    return [[segment for segment in pipeline if segment] for pipeline in commands if any(pipeline)]


def _verb(words: list[str]) -> tuple[str, list[str]]:
    """A kubectl or helm verb and the words after it, skipping global flags before the verb."""
    index = 1
    while index < len(words) and words[index].startswith("-"):
        index += 2 if words[index] in ("-n", "--namespace", "--context", "--kubeconfig") else 1
    return (words[index], words[index + 1 :]) if index < len(words) else ("", [])


@dataclass
class _Outcome:
    #: set | read | yours | admin | unit | variant | refused
    state: str
    detail: str = ""
    units: list[Unit] = field(default_factory=list)


class _Follower:
    def __init__(self, guide: Guide, root: Path, env: Environment, namespace: str) -> None:
        self.guide, self.root, self.env, self.namespace = guide, root, env, namespace

    def command(self, raw: list[list[str]], expanded: list[list[str]] | None, group: str, step: str):
        """What one command (a pipeline) of a step means."""
        first = raw[0]
        if first[0] == "export" or all(_ASSIGNMENT.match(word) for word in first):
            if len(raw) > 1:
                return _Outcome("refused", "an assignment in a pipeline")
            for word in first[1:] if first[0] == "export" else first:
                if match := _ASSIGNMENT.match(word):
                    self.env.assign(match.group(1), match.group(2))
            return _Outcome("set")
        if _ASSIGNMENT.match(first[0]):
            program = next(word for word in first if not _ASSIGNMENT.match(word))
            if group != "deploy":
                return _Outcome("admin", f"llmd does not run {program}")
            return _Outcome("refused", f"it runs {program}")
        tools = [segment[0] for segment in raw]
        if tools[0] in ("kubectl", "oc") and _verb(first)[0] == "create":
            kind = (_verb(first)[1] or [""])[0]
            rest = [segment for segment in raw[1:] if not _is_apply_stdin(segment)]
            if kind in ("namespace", "ns") and not rest:
                return _Outcome("yours", "llmd does not create namespaces; the namespace exists already")
            if kind == "secret" and not rest:
                return _Outcome("yours", "create the Secret yourself; the plan checks that it exists")
        if len(raw) == 2 and tools == ["kubectl", "kubectl"] and _verb(first)[0] == "kustomize":
            if expanded is None:
                return _Outcome("refused", "its variables have no value")
            return self.apply(expanded[1], group, step, piped=_verb(expanded[0])[1])
        if len(raw) == 1 and tools[0] in ("kubectl", "oc") and _verb(first)[0] == "apply":
            if expanded is None:
                return _Outcome("refused", "its variables have no value")
            return self.apply(expanded[0], group, step)
        if len(raw) == 1 and tools[0] == "helm" and _is_helm_install(first):
            if group != "deploy":
                return _Outcome("admin", "a cluster-wide install, for an administrator")
            if expanded is None:
                return _Outcome("refused", "its variables have no value")
            return self.helm(expanded[0], step)
        if all(_reads(segment) for segment in raw):
            return _Outcome("read")
        changing = next(segment for segment in raw if not _reads(segment))
        if group != "deploy":
            tool = changing[0] in ("kubectl", "oc", "helm")
            return _Outcome("admin", "for an administrator" if tool else f"llmd does not run {changing[0]}")
        return _Outcome("refused", f"it runs {' '.join(changing[:2])}")

    def apply(self, words: list[str], group: str, step: str, piped: list[str] | None = None) -> _Outcome:
        """``kubectl apply -k DIR`` / ``-f FILE`` (or ``-f -`` after ``kubectl kustomize DIR``)."""
        _, rest = _verb(words)
        namespace, target, kustomize = None, None, piped is not None
        index = 0
        while index < len(rest):
            word = rest[index]
            if word in ("-n", "--namespace") and index + 1 < len(rest):
                namespace, index = rest[index + 1], index + 2
            elif word.startswith("--namespace="):
                namespace, index = word.split("=", 1)[1], index + 1
            elif word in ("-k", "--kustomize", "-f", "--filename") and index + 1 < len(rest) and target is None:
                kustomize = kustomize or word in ("-k", "--kustomize")
                target, index = rest[index + 1], index + 2
            elif word in ("--server-side", "--server-side=true", "--force-conflicts"):
                index += 1
            else:
                return _Outcome("refused", f"kubectl apply {word}")
        if target == "-":
            if not piped:
                return _Outcome("refused", "it applies standard input")
            target = next((word for word in piped if not word.startswith("-")), None)
        if not target:
            return _Outcome("refused", "it names nothing to apply")
        if "://" in target:
            return _Outcome("admin", "a manifest from the network; llmd reads only the source's files")
        if namespace not in (None, self.namespace):
            if group != "deploy":
                return _Outcome("admin", f"it applies to namespace {namespace}")
            return _Outcome("refused", f"it applies to namespace {namespace}")
        if namespace is None and group != "deploy":
            return _Outcome("admin", "cluster-wide, for an administrator")
        path = self._local(target)
        if path is None:
            return _Outcome("refused", f"{target} is outside the source checkout")
        shown = self._shown(path)
        if not path.exists():
            return _Outcome("variant", f"{shown} is not in this checkout")
        if kustomize:
            if not path.is_dir() or kustomization_file(path) is None:
                return _Outcome("refused", f"{shown} is not a kustomize directory")
            return _Outcome("unit", f"kustomize build {shown}", [Unit("kustomize", step, path=path)])
        if not manifest_files(path):
            return _Outcome("refused", f"{shown} holds no Kubernetes manifests")
        return _Outcome("unit", f"read {shown}", [Unit("manifests", step, path=path)])

    def helm(self, words: list[str], step: str) -> _Outcome:
        rest = [word for word in words[2:] if word not in ("--install", "-i")]
        positionals: list[str] = []
        values: list[Path] = []
        sets: list[tuple[str, str]] = []
        version, namespace, index = "", None, 0
        while index < len(rest):
            word = rest[index]
            following = rest[index + 1] if index + 1 < len(rest) else None
            if word in ("-f", "--values") and following is not None:
                path = self._local(following) if "://" not in following else None
                if path is None:
                    return _Outcome("refused", f"values file {following} is not in the source checkout")
                if not path.is_file():
                    return _Outcome("variant", f"{self._shown(path)} is not in this checkout")
                values.append(path)
                index += 2
            elif word == "--version" and following is not None:
                version, index = following, index + 2
            elif word in ("-n", "--namespace") and following is not None:
                namespace, index = following, index + 2
            elif word in ("--set", "--set-string", "--set-json") and following is not None:
                sets.append((word, following))
                index += 2
            elif word == "--timeout" and following is not None:
                index += 2
            elif word in _HELM_IGNORED:
                index += 1
            elif word.startswith("-"):
                return _Outcome("refused", f"helm option {word}")
            else:
                positionals.append(word)
                index += 1
        if len(positionals) != 2:
            return _Outcome("refused", "helm install names no release and chart")
        if namespace not in (None, self.namespace):
            return _Outcome("refused", f"it installs into namespace {namespace}")
        release, chart = positionals
        if "://" not in chart and (chart.startswith((".", "/")) or (self.root / chart).exists()):
            path = self._local(chart)
            if path is None:
                return _Outcome("refused", f"chart {chart} is outside the source checkout")
            if not (path / "Chart.yaml").is_file():
                return _Outcome("variant", f"{self._shown(path)} is not in this checkout")
            chart = str(path)
        unit = Unit("helm", step, chart=chart, release=release, version=version, values=tuple(values), sets=tuple(sets))
        return _Outcome("unit", describe_unit(unit, self.root), [unit])

    def _local(self, target: str) -> Path | None:
        path = Path(target)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        return resolved if resolved.is_relative_to(self.root.resolve()) else None

    def _shown(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()


def _is_apply_stdin(segment: list[str]) -> bool:
    return segment[:1] in (["kubectl"], ["oc"]) and _verb(segment)[0] == "apply" and "-" in segment


def _is_helm_install(words: list[str]) -> bool:
    verb, rest = _verb(words)
    return verb == "install" or (verb == "upgrade" and ("--install" in rest or "-i" in rest))


def _reads(segment: list[str]) -> bool:
    """Whether a command only reads (or changes nothing in the cluster)."""
    tool = segment[0]
    if tool in _NO_OPS or tool in _FILTERS:
        return tool != "sed" or "-i" not in segment and not any(word.startswith("-i") for word in segment)
    if tool in ("kubectl", "oc"):
        verb, rest = _verb(segment)
        if verb == "rollout":
            return (rest or [""])[0] in ("status", "history")
        if verb == "auth":
            return (rest or [""])[0] in ("can-i", "whoami")
        return verb in _KUBECTL_READS
    if tool == "helm":
        return _verb(segment)[0] in _HELM_READS
    return False


def describe_unit(unit: Unit, root: Path) -> str:
    """A unit as the command llmd runs to render it, with paths relative to the checkout."""

    def shown(path: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    if unit.kind == "kustomize":
        return f"kustomize build {shown(unit.path)}"
    if unit.kind == "manifests":
        return f"read {shown(unit.path)}"
    chart = unit.chart
    if "://" not in chart and Path(chart).is_absolute():
        chart = shown(Path(chart))
    parts = ["helm template", unit.release, chart]
    if unit.version:
        parts += ["--version", unit.version]
    parts += [f"-f {shown(path)}" for path in unit.values]
    parts += [f"{flag} {value}" for flag, value in unit.sets]
    return " ".join(parts)


#: How a step's outcome is reported, from the one that matters most.
_PRIORITY = ["refused", "variant", "unit", "admin", "yours", "read", "set"]
_STATE = {
    "refused": "not followed",
    "variant": "other variant",
    "unit": "followed",
    "admin": "administrator",
    "yours": "yours",
    "read": "not needed",
    "set": "variables",
}


def _shown_command(raw_line: str) -> str:
    text = " ".join(raw_line.split())
    return text if len(text) <= 110 else text[:107] + "..."


def follow(guide: Guide, root: Path, *, namespace: str, overrides: dict[str, str]) -> Followed:
    """The units a guide's steps apply to ``namespace``, every step's outcome, and the findings."""
    env = environment(guide, root, namespace=namespace, overrides=overrides)
    result = Followed()
    follower = _Follower(guide, root, env, namespace)
    sections: dict[str, list[str]] = {}
    for group in ("prerequisites", "deploy"):
        for path, step in _steps(guide.data.get(group), group):
            if not _when(step, env):
                result.steps.append({"step": path, "state": "when", "detail": _when_text(step)})
                continue
            if group == "deploy" and path.split(".")[1] in OTHER_MODES:
                result.steps.append(
                    {"step": path, "state": "mode", "detail": "Gateway Mode needs a Gateway; llmd follows Standalone"}
                )
                continue
            outcomes = _follow_step(follower, step, group, path)
            worst = min(outcomes, key=lambda item: _PRIORITY.index(item[1].state))
            raw_line, outcome = worst
            manual = "ci" in (step.get("skip_in") or [])
            state = outcome.state
            detail = outcome.detail or _shown_command(raw_line)
            if state in ("refused", "variant", "admin", "yours"):
                detail = f"{_shown_command(raw_line)}  ({outcome.detail})" if outcome.detail else detail
            if state not in ("refused", "variant"):
                followed = [item for _, item in outcomes if item.state == "unit"]
                for item in followed:
                    result.units.extend(item.units)
                if state == "unit":
                    detail = "; ".join(item.detail for item in followed)
            result.steps.append({"step": path, "state": _STATE[state], "detail": detail})
            if group == "deploy":
                sections.setdefault(".".join(path.split(".")[:2]), []).append(state)
            if state == "refused":
                result.findings.append(_refused(path, raw_line, outcome.detail, manual=manual))
            elif state == "admin" and group == "deploy":
                result.findings.append(
                    Finding(
                        "warning",
                        "guide-administrator-step",
                        f"{path}: `{_shown_command(raw_line)}` is left to an administrator ({outcome.detail})",
                    )
                )
    variant = _variant(env)
    for section, states in sections.items():
        if states and all(state == "variant" for state in states):
            missing = [
                item["detail"]
                for item in result.steps
                if item["step"] == section or item["step"].startswith(section + ".")
            ]
            result.findings.append(
                Finding(
                    "error",
                    "guide-variant-missing",
                    f"no step of {section} applies to {variant or 'these variables'}: "
                    + "; ".join(item for item in missing if "not in this checkout" in item),
                    hint=f"Run `llmd stacks {guide.id}` for the guide's variables, and pick values with --var.",
                )
            )
    if not result.units and not any(f.code == "guide-variant-missing" for f in result.findings):
        result.findings.append(
            Finding("error", "guide-nothing-to-deploy", f"guide {guide.id} applies nothing llmd can follow")
        )
    result.findings[:0] = _variable_findings(guide, env, overrides)
    result.variables = {name: value for name, value in env.values.items() if name not in FIXED}
    return result


def _follow_step(follower: _Follower, step: dict[str, Any], group: str, path: str) -> list[tuple[str, _Outcome]]:
    outcomes = []
    for line in _lines(step.get("run") or ""):
        try:
            raw = _commands(_tokens(line))
        except ValueError:
            outcomes.append((line, _Outcome("refused", "its quoting cannot be read")))
            continue
        if "$(" in line or "`" in line:
            first = raw[0][0] if raw and raw[0] else []
            if first and (first[0] == "export" or all(_ASSIGNMENT.match(word) for word in first)):
                follower.command(raw, None, group, path)  # the variables are set without a value
                outcomes.append((line, _Outcome("set")))
            elif raw and all(_reads(segment) for pipeline in raw for segment in pipeline):
                outcomes.append((line, _Outcome("read")))
            else:
                outcomes.append((line, _Outcome("refused", "it runs a command substitution")))
            continue
        try:
            expanded = _commands(_tokens(follower.env.expand(line)))
        except _UnsetError as unset:
            expanded, reason = None, unset.reason
        except ValueError:
            expanded, reason = None, "its quoting cannot be read"
        else:
            reason = ""
            if len(expanded) != len(raw):
                expanded, reason = None, "a variable holds shell operators"
        for position, pipeline in enumerate(raw):
            outcome = follower.command(pipeline, expanded[position] if expanded else None, group, path)
            if outcome.state == "refused" and reason and outcome.detail == "its variables have no value":
                outcome.detail = reason
            outcomes.append((line, outcome))
    return outcomes or [("", _Outcome("read"))]


def _refused(path: str, line: str, reason: str, *, manual: bool) -> Finding:
    if manual:
        return Finding(
            "warning",
            "guide-step-left-out",
            f"{path} is left out: llmd cannot follow `{_shown_command(line)}` ({reason}); the guide's CI "
            "skips this step too",
        )
    return Finding(
        "error",
        "guide-step-not-followed",
        f"{path}: llmd cannot follow `{_shown_command(line)}` ({reason})",
        hint="Deploy the guide with its own commands and record it with `llmd import`, or pick variables "
        "(--var) whose steps llmd can follow.",
    )


def _when_text(step: dict[str, Any]) -> str:
    return ", ".join(f"{name} in {list(allowed or [])}" for name, allowed in (step.get("when") or {}).items())


def _variant(env: Environment) -> str:
    return ", ".join(f"{name}={env.get(name)}" for name in VARIANT_VARIABLES if env.get(name) is not None)


def _variable_findings(guide: Guide, env: Environment, overrides: dict[str, str]) -> list[Finding]:
    findings = []
    declared = {item["name"]: item for item in guide.variables()}
    for name, value in sorted(overrides.items()):
        allowed = (declared.get(name) or {}).get("values")
        if allowed and value not in allowed:
            findings.append(
                Finding(
                    "warning",
                    "guide-variable-value",
                    f"{name}={value} is not one of the guide's values ({', '.join(allowed)}); its steps are "
                    "picked with it anyway",
                )
            )
    support = guide.support()
    accelerator, engine = env.get("ACCELERATOR_TYPE"), env.get("MODEL_SERVER")
    if support and accelerator and engine:
        status = (support.get(accelerator) or {}).get(engine, "")
        if status not in ("validated", "community"):
            findings.append(
                Finding(
                    "warning",
                    "guide-variant-unsupported",
                    f"guide {guide.id} lists {engine} on {accelerator} as {status or 'not supported'}",
                )
            )
    return findings
