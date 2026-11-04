"""Support gate, definitions and unknown-host tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

import llm_d_lens.infrastructure.hostos as hostos
import llm_d_lens.infrastructure.hostos.systems as systems_module
from llm_d_lens.infrastructure.hostos import (
    Host,
    UnsupportedHost,
    UnsupportedHostOperation,
    check_host,
)
from llm_d_lens.infrastructure.hostos.facts import HostFacts
from llm_d_lens.infrastructure.hostos.systems import UBUNTU, find_definition

BASE = HostFacts(
    family="linux",
    arch="amd64",
    machine="x86_64",
    distro_id="ubuntu",
    version="24.04",
    pretty_name="Ubuntu 24.04.4 LTS",
    in_container=False,
    in_wsl=False,
)


def _facts(**overrides):
    return replace(BASE, **overrides)


def _host(facts):
    return Host(facts, find_definition(facts.distro_id))


@pytest.mark.parametrize("version", ["22.04", "24.04", "24.04.1", "26.10"])
def test_supported_ubuntu(version):
    status = _host(_facts(version=version, pretty_name=f"Ubuntu {version} LTS")).support
    assert status.supported
    assert f"Ubuntu {version} detected" in status.message
    assert status.message.endswith("architecture x86_64")


def test_supported_message_includes_pretty_name():
    status = _host(_facts()).support
    assert status.message == "Ubuntu 24.04 detected (Ubuntu 24.04.4 LTS), architecture x86_64"


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (_facts(version="20.04"), "Ubuntu 20.04 detected"),
        (_facts(version="21.10"), "Ubuntu 21.10 detected"),
        (_facts(version=None), "could not determine Ubuntu version"),
        (_facts(version="rolling"), "Ubuntu version 'rolling' detected"),
        (_facts(arch="arm64", machine="aarch64"), "architecture aarch64 detected"),
        (_facts(distro_id="debian", version="12", pretty_name="Debian GNU/Linux 12"), "Debian GNU/Linux 12 detected"),
        (
            _facts(distro_id="linuxmint", version="22.1", pretty_name="Linux Mint 22.1"),
            "Linux Mint 22.1 detected",
        ),
        (_facts(distro_id="fedora", version="40", pretty_name="Fedora Linux 40"), "Fedora Linux 40 detected"),
        (
            _facts(family="darwin", distro_id="macos", version="14.5", pretty_name="macOS 14.5", arch="arm64"),
            "macOS 14.5 detected",
        ),
        (
            _facts(family="windows", distro_id="windows", version="22631", pretty_name="Windows", arch="amd64"),
            "Windows detected",
        ),
        (_facts(family="other", distro_id=None, version=None, pretty_name=None), "unsupported platform other detected"),
    ],
)
def test_unsupported_messages(facts, expected):
    status = _host(facts).support
    assert not status.supported
    assert expected in status.message
    assert "Ubuntu 22.04+ (x86_64)" in status.message


def test_amd64_displayed_as_x86_64_in_supported_list():
    message = _host(_facts(version="20.04")).support.message
    assert "Ubuntu 22.04+ (x86_64)" in message


def test_definition_and_installer():
    host = _host(_facts())
    assert host.definition is UBUNTU
    assert host.installer == "LensInstaller-Ubuntu-x86_64-online.sh"
    assert host.capabilities == {"cluster-bootstrap": True}


def test_no_definition_keeps_posix_capabilities():
    host = Host(_facts(distro_id=None), None)
    assert host.definition is None
    assert host.installer is None
    assert host.capabilities == {"cluster-bootstrap": True}


def test_check_host_unknown_installer():
    with pytest.raises(ValueError, match="unknown installer"):
        check_host("haiku")


def test_check_host_matching_installer(monkeypatch):
    host = _host(_facts())
    monkeypatch.setattr(hostos, "current_host", lambda: host)
    assert check_host("ubuntu").supported


def test_check_host_installer_mismatch_names_the_right_installer(monkeypatch):
    fedora = replace(
        UBUNTU,
        id="fedora",
        display_name="Fedora",
        min_version="40",
        installer="LensInstaller-Fedora-x86_64-online.sh",
    )
    monkeypatch.setattr(hostos, "SYSTEMS", (UBUNTU, fedora))
    monkeypatch.setattr(systems_module, "SYSTEMS", (UBUNTU, fedora))
    host = _host(_facts())
    monkeypatch.setattr(hostos, "current_host", lambda: host)

    status = check_host("fedora")
    assert not status.supported
    assert "this installer is for Fedora" in status.message
    assert "LensInstaller-Ubuntu-x86_64-online.sh" in status.message


def test_check_host_mismatch_ignored_when_host_unsupported(monkeypatch):
    host = _host(_facts(version="20.04"))
    monkeypatch.setattr(hostos, "current_host", lambda: host)
    status = check_host("ubuntu")
    assert not status.supported
    assert "Ubuntu 20.04" in status.message


def test_unsupported_host_methods_raise():
    host = UnsupportedHost("windows")
    assert host.default_capabilities == {"cluster-bootstrap": False}
    assert host.null_device
    with pytest.raises(UnsupportedHostOperation, match="default_base.*windows"):
        host.default_base("data")
    with pytest.raises(UnsupportedHostOperation, match="try_lock.*windows"):
        host.try_lock(object())
    with pytest.raises(UnsupportedHostOperation, match="port_in_use.*windows"):
        host.port_in_use(80)


def test_host_selects_unsupported_family(monkeypatch):
    monkeypatch.setattr(hostos.os, "name", "nt")
    host = Host(_facts(distro_id=None), None)
    assert host.capabilities == {"cluster-bootstrap": False}
    with pytest.raises(UnsupportedHostOperation):
        host.default_base("data")
