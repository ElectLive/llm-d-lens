"""Command line output and exit-code tests."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_d_lens.infrastructure.hostos as hostos
from llm_d_lens.infrastructure.hostos import Host, SupportStatus
from llm_d_lens.infrastructure.hostos import __main__ as cli
from llm_d_lens.infrastructure.hostos.facts import HostFacts, detect_facts
from llm_d_lens.infrastructure.hostos.systems import UBUNTU

FACTS = HostFacts(
    family="linux",
    arch="amd64",
    machine="x86_64",
    distro_id="ubuntu",
    version="24.04",
    pretty_name="Ubuntu 24.04.4 LTS",
    in_container=False,
    in_wsl=False,
)
HOST = Host(FACTS, UBUNTU)
WINDOWS_FACTS = HostFacts(
    family="windows",
    arch="amd64",
    machine="AMD64",
    distro_id="windows",
    version="22631",
    pretty_name="Windows",
    in_container=False,
    in_wsl=False,
)


def _use(monkeypatch, host):
    monkeypatch.setattr(cli, "current_host", lambda: host)


@pytest.mark.parametrize(("supported", "exit_code", "stream"), [(True, 0, "out"), (False, 1, "err")])
def test_check(monkeypatch, capsys, supported, exit_code, stream):
    monkeypatch.setattr(cli, "check_host", lambda installer=None: SupportStatus(supported, "the message"))
    assert cli.main(["check"]) == exit_code
    captured = capsys.readouterr()
    assert getattr(captured, stream) == "the message\n"


def test_check_unknown_installer_is_usage_error(monkeypatch, capsys):
    def fail(installer=None):
        raise ValueError(f"unknown installer: {installer}")

    monkeypatch.setattr(cli, "check_host", fail)
    assert cli.main(["check", "--installer", "haiku"]) == 2
    assert "unknown installer: haiku" in capsys.readouterr().err


def test_info_json(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    assert cli.main(["info", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["facts"]["family"] == "linux"
    assert payload["facts"]["machine"] == "x86_64"
    assert payload["definition"] == {
        "id": "ubuntu",
        "display_name": "Ubuntu",
        "installer": "LensInstaller-Ubuntu-x86_64-online.sh",
    }
    assert payload["support"]["supported"] is True
    assert payload["capabilities"] == {"cluster-bootstrap": True}


def test_info_text(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    assert cli.main(["info"]) == 0
    output = capsys.readouterr().out
    assert "definition: ubuntu" in output
    assert "installer: LensInstaller-Ubuntu-x86_64-online.sh" in output
    assert "supported: True" in output
    assert "capability.cluster-bootstrap: True" in output


def test_info_succeeds_when_container_marker_is_inaccessible(monkeypatch, capsys, tmp_path):
    release = tmp_path / "etc" / "os-release"
    release.parent.mkdir(parents=True)
    release.write_text('ID=ubuntu\nVERSION_ID="22.04"\n', encoding="utf-8")
    blocked = tmp_path / ".dockerenv"
    original_exists = Path.exists

    def exists(path):
        if path == blocked:
            raise PermissionError(path)
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(
        cli,
        "current_host",
        lambda: Host(detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path), None),
    )
    assert cli.main(["info"]) == 0
    assert "distro_id: ubuntu" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["port-in-use", "1"],
        ["free-port", "3000"],
        ["port-owner", "22"],
        ["lan-ip"],
        ["tool-platform", "--style", "go"],
        ["packages", "missing"],
    ],
)
def test_unsupported_operation_returns_one(monkeypatch, capsys, arguments):
    monkeypatch.setattr(hostos.os, "name", "nt")
    _use(monkeypatch, Host(WINDOWS_FACTS, None))
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "is not supported" in captured.err


def test_packages_missing(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "missing_packages", lambda: ["curl", "python3-pip"])
    assert cli.main(["packages", "missing"]) == 0
    assert capsys.readouterr().out == "curl\npython3-pip\n"


def test_packages_missing_empty(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "missing_packages", lambda: [])
    assert cli.main(["packages", "missing"]) == 0
    assert capsys.readouterr().out == ""


def test_packages_missing_unsupported(monkeypatch, capsys):
    _use(monkeypatch, Host(HostFacts("linux", "amd64", "x86_64", None, None, None, False, False), None))
    assert cli.main(["packages", "missing"]) == 1
    assert "missing_packages is not supported" in capsys.readouterr().err


def test_packages_show(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "package_install_problem", lambda packages: None)
    commands = [
        ["sudo", "apt-get", "update", "-qq"],
        ["sudo", "apt-get", "install", "-y", "-qq", "curl", "two words"],
    ]
    monkeypatch.setattr(HOST, "package_commands", lambda packages: commands)
    assert cli.main(["packages", "show", "curl", "two words"]) == 0
    assert capsys.readouterr().out == " && ".join(shlex.join(command) for command in commands) + "\n"


def test_packages_show_reports_missing_manager(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(hostos.shutil, "which", lambda name: None)
    assert cli.main(["packages", "show", "curl"]) == 1
    assert "apt-get not found — install manually: curl" in capsys.readouterr().err


def test_packages_show_reports_missing_sudo(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(hostos.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else None)
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    assert cli.main(["packages", "show", "curl"]) == 1
    assert "sudo is required to install packages as non-root" in capsys.readouterr().err


def test_packages_install_refresh_failure_returns_one(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(hostos.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 0)
    monkeypatch.setattr(hostos.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(returncode=100))
    assert cli.main(["packages", "install", "curl"]) == 1
    assert "apt-get update failed (exit 100)" in capsys.readouterr().err


def test_packages_install_step_failure_returns_one(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(hostos.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 0)
    codes = iter([0, 100])
    monkeypatch.setattr(hostos.subprocess, "run", lambda argv, **kwargs: SimpleNamespace(returncode=next(codes)))
    assert cli.main(["packages", "install", "curl"]) == 1
    assert "apt-get install failed (exit 100)" in capsys.readouterr().err


def test_packages_install_unsupported(monkeypatch, capsys):
    _use(monkeypatch, Host(HostFacts("linux", "amd64", "x86_64", None, None, None, False, False), None))
    assert cli.main(["packages", "install", "curl"]) == 1
    assert "install_packages is not supported" in capsys.readouterr().err


def test_packages_hint(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    assert cli.main(["packages", "hint", "sshpass"]) == 0
    assert capsys.readouterr().out == "sudo apt-get install -y sshpass\n"


@pytest.mark.parametrize(("in_use", "exit_code"), [(True, 0), (False, 1)])
def test_port_in_use(monkeypatch, in_use, exit_code):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "port_in_use", lambda port: in_use)
    assert cli.main(["port-in-use", "3000"]) == exit_code


def test_free_port_finds_first_free(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "port_in_use", lambda port: port < 3003)
    assert cli.main(["free-port", "3000"]) == 0
    assert capsys.readouterr().out == "3003\n"


def test_free_port_fails_after_200_ports(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    checked = []

    def always_in_use(port):
        checked.append(port)
        return True

    monkeypatch.setattr(HOST, "port_in_use", always_in_use)
    assert cli.main(["free-port", "3000"]) == 1
    assert checked == list(range(3000, 3200))
    assert capsys.readouterr().err == "could not find a free port at or after 3000 (checked 200 ports)\n"


def test_port_owner(monkeypatch, capsys):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "describe_port_owner", lambda port: "an unknown process")
    assert cli.main(["port-owner", "22"]) == 0
    assert capsys.readouterr().out == "an unknown process\n"


@pytest.mark.parametrize(("address", "output"), [("10.1.2.3", "10.1.2.3\n"), ("", "")])
def test_lan_ip(monkeypatch, capsys, address, output):
    _use(monkeypatch, HOST)
    monkeypatch.setattr(HOST, "first_lan_address", lambda: address)
    assert cli.main(["lan-ip"]) == 0
    assert capsys.readouterr().out == output


@pytest.mark.parametrize(("style", "output"), [("node", "linux-x64\n"), ("go", "linux-amd64\n")])
def test_tool_platform(monkeypatch, capsys, style, output):
    _use(monkeypatch, HOST)
    assert cli.main(["tool-platform", "--style", style]) == 0
    assert capsys.readouterr().out == output


def _run_module(*arguments):
    source = Path(__file__).resolve().parents[4] / "src"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else str(source)
    return subprocess.run(
        [sys.executable, "-m", "llm_d_lens.infrastructure.hostos", *arguments],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_module_entry_point_real_subprocess():
    result = _run_module("info", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["facts"]["family"]


def test_fcntl_blocked_still_imports_and_checks():
    code = (
        "import sys\n"
        "sys.modules['fcntl'] = None\n"
        "from llm_d_lens.infrastructure.hostos.__main__ import main\n"
        "raise SystemExit(main(['check']))\n"
    )
    source = Path(__file__).resolve().parents[4] / "src"
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else str(source)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=False)
    assert result.returncode in (0, 1), result.stderr
    assert "Traceback" not in result.stderr
