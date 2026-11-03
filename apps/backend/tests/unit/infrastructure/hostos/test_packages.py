"""Ubuntu package definition and host package operations."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import llm_d_lens.infrastructure.hostos as hostos
from llm_d_lens.infrastructure.hostos import Host, UnsupportedHostOperation
from llm_d_lens.infrastructure.hostos.facts import HostFacts
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
ALL_COMMANDS = {"curl", "tar", "git", "python3", "ss", "openssl", "pip3", "rsync"}
HOST = Host(FACTS, UBUNTU)


def _which(monkeypatch, present):
    monkeypatch.setattr(hostos.shutil, "which", lambda name: f"/usr/bin/{name}" if name in present else None)


def _query(monkeypatch, *, installed=(), returncode=0, raises=False):
    def fake_run(argv, **kwargs):
        if raises:
            raise FileNotFoundError(argv[0])
        status = "install ok installed" if argv[-1] in installed else "unknown ok not-installed"
        return SimpleNamespace(returncode=returncode, stdout=status)

    monkeypatch.setattr(hostos.subprocess, "run", fake_run)


def _missing(monkeypatch, *, commands=ALL_COMMANDS, installed=("python3-venv",), **query):
    _which(monkeypatch, commands)
    _query(monkeypatch, installed=installed, **query)
    return HOST.missing_packages()


def test_no_missing_packages(monkeypatch):
    assert _missing(monkeypatch) == []


@pytest.mark.parametrize(
    ("commands", "installed", "query", "expected"),
    [
        (ALL_COMMANDS - {"curl"}, ("python3-venv",), {}, ["curl"]),
        (ALL_COMMANDS - {"tar"}, ("python3-venv",), {}, ["tar"]),
        (ALL_COMMANDS - {"git"}, ("python3-venv",), {}, ["git"]),
        (ALL_COMMANDS - {"python3"}, ("python3-venv",), {}, ["python3"]),
        (ALL_COMMANDS - {"ss"}, ("python3-venv",), {}, ["iproute2"]),
        (ALL_COMMANDS - {"openssl"}, ("python3-venv",), {}, ["openssl"]),
        (ALL_COMMANDS - {"pip3"}, ("python3-venv",), {}, ["python3-pip"]),
        (ALL_COMMANDS - {"rsync"}, ("python3-venv",), {}, ["rsync"]),
        (ALL_COMMANDS, (), {}, ["python3-pip", "python3-venv"]),
        (ALL_COMMANDS - {"pip3"}, (), {}, ["python3-pip", "python3-venv"]),
        (ALL_COMMANDS, ("python3-venv",), {"returncode": 1}, ["python3-pip", "python3-venv"]),
        (ALL_COMMANDS, ("python3-venv",), {"raises": True}, ["python3-pip", "python3-venv"]),
        (ALL_COMMANDS - {"curl", "rsync"}, ("python3-venv",), {}, ["curl", "rsync"]),
    ],
)
def test_missing_package_decisions(monkeypatch, commands, installed, query, expected):
    assert _missing(monkeypatch, commands=commands, installed=installed, **query) == expected


def test_package_commands_elevate_as_non_root(monkeypatch):
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    assert HOST.package_commands(["curl", "git"]) == [
        ["sudo", "apt-get", "update", "-qq"],
        ["sudo", "apt-get", "install", "-y", "-qq", "curl", "git"],
    ]


def test_package_commands_without_sudo_as_root(monkeypatch):
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 0)
    assert HOST.package_commands(["curl"]) == [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "curl"],
    ]


def test_package_commands_require_a_definition():
    with pytest.raises(UnsupportedHostOperation, match="package_commands"):
        Host(replace(FACTS, distro_id=None), None).package_commands(["curl"])


def test_install_hint_uses_manual_command():
    assert HOST.install_hint(["curl", "git"]) == "sudo apt-get install -y curl git"


def test_install_hint_as_root_keeps_sudo(monkeypatch):
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 0)
    assert HOST.install_hint(["curl"]) == "sudo apt-get install -y curl"


def test_install_hint_without_elevation():
    manager = replace(UBUNTU.package_manager, needs_elevation=False)
    host = Host(FACTS, replace(UBUNTU, package_manager=manager))
    assert host.install_hint(["curl"]) == "apt-get install -y curl"


def test_install_hint_without_definition():
    host = Host(replace(FACTS, distro_id=None), None)
    assert host.install_hint(["curl", "git"]) == "install curl git with the system package manager"


def test_tool_packages():
    assert HOST.tool_packages("sshpass") == ("sshpass",)
    assert HOST.tool_packages("curl") == ("curl",)
    assert Host(replace(FACTS, distro_id=None), None).tool_packages("sshpass") == ("sshpass",)


def test_install_packages_requires_a_definition():
    with pytest.raises(UnsupportedHostOperation, match="install_packages"):
        Host(replace(FACTS, distro_id=None), None).install_packages(["curl"])


def test_package_install_problem_ok(monkeypatch):
    _which(monkeypatch, {"apt-get", "sudo"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    assert HOST.package_install_problem(["curl"]) is None


def test_package_install_problem_missing_manager(monkeypatch):
    _which(monkeypatch, set())
    assert HOST.package_install_problem(["curl", "git"]) == "apt-get not found — install manually: curl git"


def test_package_install_problem_missing_sudo(monkeypatch):
    _which(monkeypatch, {"apt-get"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    assert HOST.package_install_problem(["curl"]) == "sudo is required to install packages as non-root"


def test_package_install_problem_requires_a_definition():
    with pytest.raises(UnsupportedHostOperation, match="install_packages"):
        Host(replace(FACTS, distro_id=None), None).package_install_problem(["curl"])


def test_install_packages_missing_manager(monkeypatch, capsys):
    _which(monkeypatch, set())
    assert HOST.install_packages(["curl", "git"]) == 1
    assert "apt-get not found — install manually: curl git" in capsys.readouterr().err


def test_install_packages_missing_sudo(monkeypatch, capsys):
    _which(monkeypatch, {"apt-get"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    assert HOST.install_packages(["curl"]) == 1
    assert "sudo is required to install packages as non-root" in capsys.readouterr().err


def _install_run(monkeypatch, codes):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(returncode=codes[len(calls) - 1])

    monkeypatch.setattr(hostos.subprocess, "run", fake_run)
    return calls


def test_install_packages_stops_when_refresh_fails(monkeypatch, capsys):
    _which(monkeypatch, {"apt-get", "sudo"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    calls = _install_run(monkeypatch, [7, 0])
    assert HOST.install_packages(["curl"]) == 1
    assert calls == [["sudo", "apt-get", "update", "-qq"]]
    assert "apt-get update failed (exit 7)" in capsys.readouterr().err


def test_install_packages_runs_install(monkeypatch):
    _which(monkeypatch, {"apt-get", "sudo"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    calls = _install_run(monkeypatch, [0, 0])
    assert HOST.install_packages(["curl", "git"]) == 0
    assert calls == [
        ["sudo", "apt-get", "update", "-qq"],
        ["sudo", "apt-get", "install", "-y", "-qq", "curl", "git"],
    ]


def test_install_packages_returns_install_failure(monkeypatch, capsys):
    _which(monkeypatch, {"apt-get", "sudo"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 1000)
    _install_run(monkeypatch, [0, 9])
    assert HOST.install_packages(["curl"]) == 1
    assert "apt-get install failed (exit 9)" in capsys.readouterr().err


def test_install_packages_as_root_skips_sudo(monkeypatch):
    _which(monkeypatch, {"apt-get"})
    monkeypatch.setattr(hostos.os, "geteuid", lambda: 0)
    calls = _install_run(monkeypatch, [0, 0])
    assert HOST.install_packages(["curl"]) == 0
    assert calls[0] == ["apt-get", "update", "-qq"]
    assert calls[1] == ["apt-get", "install", "-y", "-qq", "curl"]
