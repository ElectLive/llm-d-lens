"""Regression coverage for installing benchmark's unpublished workspace package."""

import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jinja2 import Template

router = importlib.import_module("llm_d_lens.evaluation.router")


@pytest.mark.parametrize("has_report_package", [True, False])
def test_install_supplies_bundled_report_to_pip(monkeypatch, tmp_path, has_report_package):
    root = tmp_path / "benchmark"
    root.mkdir()
    report = root / "benchmark-report"
    if has_report_package:
        report.mkdir()
        (report / "pyproject.toml").write_text('[project]\nname = "llmd-benchmark-report"\n')
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "pip" in command:
            if has_report_package and str(report) not in command:
                return SimpleNamespace(returncode=1, stderr="No matching distribution found for llmd-benchmark-report")
            assert (str(report) in command) == has_report_package
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(router.subprocess, "run", run)
    router._install_benchmark_checkout(root)
    assert any("-c" in command for command in commands)


def _write_buggy_kubernetes_configuration(venv: Path) -> Path:
    configuration_file = venv / "lib" / "python3.12" / "site-packages" / "kubernetes" / "client" / "configuration.py"
    configuration_file.parent.mkdir(parents=True, exist_ok=True)
    configuration_file.write_text(
        "import os\n\n"
        "class Configuration(object):\n"
        "    def __init__(self):\n"
        "        self.verify_ssl = True\n"
        "        self.proxy = None\n"
        '        if os.getenv("HTTPS_PROXY"): self.proxy = os.getenv("HTTPS_PROXY")\n'
        "        self.no_proxy = None\n"
        '        if os.getenv("NO_PROXY"): self.no_proxy = os.getenv("NO_PROXY")\n'
        '        if os.getenv("no_proxy"): self.no_proxy = os.getenv("no_proxy")\n'
        '        """Proxy URL\n'
        '        """\n'
        "        self.no_proxy = None\n"
        '        """bypass proxy for host in the no_proxy list.\n'
        '        """\n'
        "        self.proxy_headers = None\n",
        encoding="utf-8",
    )
    return configuration_file


def test_patch_kubernetes_client_no_proxy_bug_removes_clobbering_assignment(tmp_path):
    venv = tmp_path / ".venv"
    configuration_file = _write_buggy_kubernetes_configuration(venv)

    router._patch_kubernetes_client_no_proxy_bug(venv)

    patched = configuration_file.read_text(encoding="utf-8")
    assert patched.count("self.no_proxy = None") == 1
    assert 'if os.getenv("no_proxy"): self.no_proxy = os.getenv("no_proxy")' in patched


def test_patch_kubernetes_client_no_proxy_bug_is_idempotent_and_tolerates_missing_file(tmp_path):
    venv = tmp_path / ".venv"
    configuration_file = _write_buggy_kubernetes_configuration(venv)

    router._patch_kubernetes_client_no_proxy_bug(venv)
    once_patched = configuration_file.read_text(encoding="utf-8")
    router._patch_kubernetes_client_no_proxy_bug(venv)
    assert configuration_file.read_text(encoding="utf-8") == once_patched

    router._patch_kubernetes_client_no_proxy_bug(tmp_path / "does-not-exist")


def _write_harness_template(root: Path) -> Path:
    template = root / "config/templates/jinja/20_harness_pod.yaml.j2"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(
        'command: ["sh", "-c"]\nargs:\n- >-\n  {{ harness_command }}\n',
        encoding="utf-8",
    )
    return template


@pytest.mark.parametrize("already_fixed", [False, True])
@pytest.mark.parametrize("lowercase", [False, True])
def test_rendered_harness_repairs_client_before_running_original_command(tmp_path, already_fixed, lowercase):
    root = tmp_path / "benchmark"
    configuration = _write_buggy_kubernetes_configuration(root / ".venv")
    if already_fixed:
        router._patch_kubernetes_client_no_proxy_bug(root / ".venv")
    original = configuration.read_text(encoding="utf-8")
    original_mtime = configuration.stat().st_mtime_ns
    site_packages = configuration.parents[2]
    (site_packages / "kubernetes/__init__.py").write_text("", encoding="utf-8")
    (site_packages / "kubernetes/client/__init__.py").write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    template = _write_harness_template(root)
    router._patch_harness_kubernetes_no_proxy(root)
    once_patched = template.read_text(encoding="utf-8")
    router._patch_harness_kubernetes_no_proxy(root)
    assert template.read_text(encoding="utf-8") == once_patched

    probe = (
        "import json, os; from kubernetes.client.configuration import Configuration; "
        "from requests.utils import should_bypass_proxies; "
        "c = Configuration(); "
        "print(json.dumps([c.proxy, c.no_proxy, c.verify_ssl, "
        "should_bypass_proxies('https://10.233.0.1:443', no_proxy=c.no_proxy), "
        "os.environ['HTTP_PROXY'], os.environ['ORIGINAL_HARNESS']]))"
    )
    rendered = yaml.safe_load(
        Template(once_patched).render(
            harness_command=f"export ORIGINAL_HARNESS=preserved; python -c {shlex.quote(probe)}"
        )
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    }
    bypass = "localhost,10.233.0.0/18,.svc,.cluster.local"
    env.update(
        PATH=f"{bin_dir}:{os.defpath}",
        PYTHONPATH=str(site_packages),
        HTTP_PROXY="http://proxy.example:3128",
        HTTPS_PROXY="http://proxy.example:3128",
        NO_PROXY="ignored.example" if lowercase else bypass,
    )
    if lowercase:
        env["no_proxy"] = bypass
    result = subprocess.run(
        [*rendered["command"], *rendered["args"]], env=env, capture_output=True, text=True, timeout=30, check=True
    )

    assert json.loads(result.stdout) == [
        "http://proxy.example:3128",
        bypass,
        True,
        True,
        "http://proxy.example:3128",
        "preserved",
    ]
    assert "[Lens] Kubernetes NO_PROXY compatibility patch:" in result.stderr
    if already_fixed:
        assert configuration.read_text(encoding="utf-8") == original
        assert configuration.stat().st_mtime_ns == original_mtime
    else:
        assert configuration.read_text(encoding="utf-8").count("self.no_proxy = None") == 1


def test_harness_does_not_continue_after_bootstrap_failure(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\necho 'SDK is not writable' >&2\nexit 73\n", encoding="utf-8")
    python.chmod(0o755)
    template = _write_harness_template(tmp_path)
    router._patch_harness_kubernetes_no_proxy(tmp_path)
    rendered = yaml.safe_load(Template(template.read_text()).render(harness_command="echo first; echo second"))

    result = subprocess.run(
        [*rendered["command"], *rendered["args"]],
        env={**os.environ, "PATH": f"{bin_dir}:{os.defpath}"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 73
    assert "SDK is not writable" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("contents", ["args: []\n", "{{ harness_command }} {{ harness_command }}\n"])
def test_harness_rejects_unsupported_template_without_modifying_it(tmp_path, contents):
    template = _write_harness_template(tmp_path)
    template.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot install proxy fix"):
        router._patch_harness_kubernetes_no_proxy(tmp_path)

    assert template.read_text(encoding="utf-8") == contents


@pytest.mark.asyncio
async def test_prepare_runtime_repairs_host_and_harness_for_existing_checkout(monkeypatch, tmp_path):
    configuration = _write_buggy_kubernetes_configuration(tmp_path / ".venv")
    template = _write_harness_template(tmp_path)
    monkeypatch.setattr(router, "_benchmark_ready", lambda _root: True)

    cli, root = await router._prepare_benchmark_runtime({"resolved_repository": str(tmp_path)})

    assert (cli, root) == (str(tmp_path / ".venv/bin/llmdbenchmark"), tmp_path)
    assert configuration.read_text(encoding="utf-8").count("self.no_proxy = None") == 1
    assert router._harness_kubernetes_no_proxy_command() in template.read_text(encoding="utf-8")


def test_benchmark_python_floor_reads_requires_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.14"\n', encoding="utf-8")
    assert router._benchmark_python_floor(tmp_path) == (3, 14)


def test_benchmark_interpreter_uses_uv_when_required_python_is_missing(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.14"\n', encoding="utf-8")
    monkeypatch.delenv("LENS_BENCHMARK_PYTHON", raising=False)
    monkeypatch.setattr(router.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(router, "_python_version", lambda executable: (3, 12))
    assert router._benchmark_interpreter(tmp_path) == (None, "/usr/bin/uv")


def test_benchmark_ready_rebuilds_an_older_python_venv(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.14"\n', encoding="utf-8")
    script = tmp_path / ".venv" / "bin" / "llmdbenchmark"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(router, "_python_version", lambda executable: (3, 12))
    assert router._benchmark_ready(tmp_path) is False


def test_uv_install_uses_the_required_python(monkeypatch, tmp_path):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.14"\n', encoding="utf-8")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(router, "_benchmark_interpreter", lambda _root: (None, "/usr/bin/uv"))
    monkeypatch.setattr(router.subprocess, "run", run)
    router._install_benchmark_checkout(root)
    assert commands[0][:5] == ["/usr/bin/uv", "venv", "--python", "3.14", str(root / ".venv")]
    assert commands[1][:5] == ["/usr/bin/uv", "pip", "install", "--python", str(root / ".venv/bin/python")]
