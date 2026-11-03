"""PosixHost parity with today's POSIX behavior."""

from __future__ import annotations

import os
import platform
import shutil
import signal
import socket
import stat
from pathlib import Path

import pytest

import llm_d_lens.infrastructure.hostos.posix as posix
from llm_d_lens.infrastructure.hostos.facts import HostFacts
from llm_d_lens.infrastructure.hostos.posix import PosixHost, ToolPlatform

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


@pytest.fixture
def host():
    return PosixHost(FACTS)


def _xdg_paths(tmp_path):
    return {
        "data": tmp_path / "xdg-data" / "lens",
        "cache": tmp_path / "xdg-cache" / "lens",
        "log": tmp_path / "xdg-state" / "lens" / "logs",
    }


def test_default_base_uses_xdg(monkeypatch, tmp_path, host):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    expected = _xdg_paths(tmp_path)
    for area, path in expected.items():
        assert host.default_base(area) == path


def test_default_base_uses_home_without_xdg(monkeypatch, tmp_path, host):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    for name in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert host.default_base("data") == home / ".local/share/lens"
    assert host.default_base("cache") == home / ".cache/lens"
    assert host.default_base("log") == home / ".local/state/lens/logs"


def test_default_base_reads_environment_each_call(monkeypatch, tmp_path, host):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "first"))
    assert host.default_base("data") == tmp_path / "first/lens"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "second"))
    assert host.default_base("data") == tmp_path / "second/lens"


def test_default_runtime(monkeypatch, tmp_path, host):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))
    assert host.default_base("runtime") == tmp_path / "runtime/lens"
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert host.default_base("runtime") is None


def test_default_base_rejects_unknown_area(host):
    with pytest.raises(ValueError, match="Unknown storage area"):
        host.default_base("scratch")


def test_default_kubeconfig(monkeypatch, tmp_path, host):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert host.default_kubeconfig() == tmp_path / ".kube" / "config"


def test_venv_executable(host):
    assert host.venv_executable(Path("/opt/lens/.venv"), "python") == Path("/opt/lens/.venv/bin/python")


@pytest.mark.skipif(os.name != "posix", reason="flock is POSIX-only")
def test_try_lock_exclusive_contention(tmp_path, host):
    path = tmp_path / "lock"
    first = path.open("a+")
    second = path.open("a+")
    try:
        assert host.try_lock(first)
        assert not host.try_lock(second, blocking=False)
        first.close()
        assert host.try_lock(second, blocking=False)
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(os.name != "posix", reason="flock is POSIX-only")
def test_try_lock_shared_allows_shared_holders(tmp_path, host):
    path = tmp_path / "lock"
    first = path.open("a+")
    second = path.open("a+")
    try:
        assert host.try_lock(first, shared=True)
        assert host.try_lock(second, shared=True)
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(os.name != "posix", reason="flock is POSIX-only")
def test_try_lock_shared_blocks_exclusive(tmp_path, host):
    path = tmp_path / "lock"
    first = path.open("a+")
    second = path.open("a+")
    try:
        assert host.try_lock(first, shared=True)
        assert not host.try_lock(second, blocking=False)
    finally:
        first.close()
        second.close()


def test_restrict_to_owner_directory(tmp_path, host):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o755)
    host.restrict_to_owner(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_restrict_to_owner_file(tmp_path, host):
    path = tmp_path / "secret"
    path.write_text("value", encoding="utf-8")
    host.restrict_to_owner(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(("private", "mode"), [(True, 0o700), (False, 0o755)])
def test_make_executable(tmp_path, host, private, mode):
    path = tmp_path / "tool"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    host.make_executable(path, private=private)
    assert stat.S_IMODE(path.stat().st_mode) == mode


def test_group_spawn_kwargs(host):
    assert host.group_spawn_kwargs() == {"start_new_session": True}


def test_signal_group_uses_expected_signals(monkeypatch, host):
    calls = []
    monkeypatch.setattr(posix.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    host.signal_group(123, force=False)
    host.signal_group(456, force=True)
    assert calls == [(123, signal.SIGTERM), (456, signal.SIGKILL)]


def test_null_device(host):
    assert host.null_device == os.devnull


def test_tool_platform_amd64(host):
    go = host.tool_platform("go")
    node = host.tool_platform("node")
    assert go == ToolPlatform("linux", "amd64", "")
    assert go.joined() == "linux-amd64"
    assert node == ToolPlatform("linux", "x64", "")
    assert node.joined() == "linux-x64"


def test_tool_platform_arm64():
    host = PosixHost(HostFacts("linux", "arm64", "aarch64", "ubuntu", "24.04", "Ubuntu", False, False))
    assert host.tool_platform("go").joined() == "linux-arm64"
    assert host.tool_platform("node").joined() == "linux-arm64"


def test_tool_platform_unknown_style(host):
    with pytest.raises(ValueError, match="Unknown tool platform style"):
        host.tool_platform("ruby")


def test_tool_platform_unknown_arch_falls_back_to_amd64():
    host = PosixHost(HostFacts("linux", "i686", "i686", "ubuntu", "24.04", "Ubuntu", False, False))
    assert host.tool_platform("go").joined() == "linux-amd64"
    assert host.tool_platform("node").joined() == "linux-x64"


def test_machine_info_matches_platform_values(host):
    info = host.machine_info()
    assert set(info) == {"hostname", "platform", "cpu", "memoryGiB"}
    assert info["platform"] == f"{platform.system()} {platform.release()} {platform.machine()}"
    assert info["hostname"] == platform.node()
    assert info["cpu"]["logicalCores"] == (os.cpu_count() or 0)
    assert isinstance(info["cpu"]["model"], str)
    memory = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    assert info["memoryGiB"] == memory


def _listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock, sock.getsockname()[1]


@pytest.mark.skipif(shutil.which("ss") is None, reason="ss is not installed")
def test_port_in_use_ss_path(host):
    sock, port = _listener()
    try:
        assert host.port_in_use(port)
    finally:
        sock.close()
    assert not host.port_in_use(port)


def test_port_in_use_connect_path(monkeypatch, host):
    monkeypatch.setattr(posix.shutil, "which", lambda name: None)
    sock, port = _listener()
    try:
        assert host.port_in_use(port)
    finally:
        sock.close()
    assert not host.port_in_use(port)


def test_describe_port_owner_visible(monkeypatch, host):
    ss = 'LISTEN 0 128 0.0.0.0:8080 0.0.0.0:* users:(("nginx",pid=42,fd=6))\n'
    outputs = {
        ("ss", "-H", "-ltnp", "( sport = :8080 )"): ss,
        ("ps", "-o", "comm=,user=", "-p", "42"): "nginx nginx\n",
    }
    monkeypatch.setattr(posix, "_command_output", lambda argv: outputs.get(tuple(argv), ""))
    assert host.describe_port_owner(8080) == "nginx nginx (pid 42)"


def test_describe_port_owner_uses_first_line(monkeypatch, host):
    ss = "first line pid=7\nsecond line pid=9\n"
    outputs = {
        ("ss", "-H", "-ltnp", "( sport = :80 )"): ss,
        ("ps", "-o", "comm=,user=", "-p", "7"): "service user\n",
    }
    monkeypatch.setattr(posix, "_command_output", lambda argv: outputs.get(tuple(argv), ""))
    assert host.describe_port_owner(80) == "service user (pid 7)"


def test_describe_port_owner_shared_host(monkeypatch, host):
    outputs = {
        ("ss", "-H", "-ltnp", "( sport = :80 )"): 'LISTEN 0 128 0.0.0.0:80 0.0.0.0:* users:(("nginx",pid=42,fd=6))\n',
    }
    monkeypatch.setattr(posix, "_command_output", lambda argv: outputs.get(tuple(argv), ""))
    assert host.describe_port_owner(80) == "another process on this shared host (owner not visible without root)"


def test_describe_port_owner_without_pid(monkeypatch, host):
    monkeypatch.setattr(
        posix,
        "_command_output",
        lambda argv: "LISTEN 0 128 0.0.0.0:80 0.0.0.0:*\n" if argv[0] == "ss" else "",
    )
    assert host.describe_port_owner(80) == "another process on this shared host (owner not visible without root)"


def test_describe_port_owner_unknown(monkeypatch, host):
    monkeypatch.setattr(posix, "_command_output", lambda argv: "")
    assert host.describe_port_owner(80) == "an unknown process"


def test_first_lan_address_first_token(monkeypatch, host):
    monkeypatch.setattr(posix, "_command_output", lambda argv: "  10.1.2.3 10.4.5.6 \n")
    assert host.first_lan_address() == "10.1.2.3"


def test_first_lan_address_empty(monkeypatch, host):
    monkeypatch.setattr(posix, "_command_output", lambda argv: "")
    assert host.first_lan_address() == ""


def test_first_lan_address_failure(monkeypatch, host):
    def fail(argv, **kwargs):
        raise OSError("hostname is missing")

    monkeypatch.setattr(posix.subprocess, "run", fail)
    assert host.first_lan_address() == ""
