"""Host fact detection tests."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from llm_d_lens.infrastructure.hostos.facts import (
    detect_facts,
    family_for,
    normalize_arch,
    parse_os_release,
)


def _release(root, text, relative="etc/os-release"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("release", "distro_id", "version", "pretty_name"),
    [
        (
            'ID=ubuntu\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04.4 LTS"\n',
            "ubuntu",
            "22.04",
            "Ubuntu 22.04.4 LTS",
        ),
        (
            'ID=ubuntu\nVERSION_ID="20.04"\nPRETTY_NAME="Ubuntu 20.04.6 LTS"\n',
            "ubuntu",
            "20.04",
            "Ubuntu 20.04.6 LTS",
        ),
        (
            'ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.4 LTS"\n',
            "ubuntu",
            "24.04",
            "Ubuntu 24.04.4 LTS",
        ),
        ('ID=debian\nVERSION_ID="12"\nPRETTY_NAME="Debian GNU/Linux 12"\n', "debian", "12", "Debian GNU/Linux 12"),
        (
            'ID=linuxmint\nID_LIKE="ubuntu debian"\nVERSION_ID="22.1"\nPRETTY_NAME="Linux Mint 22.1"\n',
            "linuxmint",
            "22.1",
            "Linux Mint 22.1",
        ),
        ('ID=fedora\nVERSION_ID=40\nPRETTY_NAME="Fedora Linux 40"\n', "fedora", "40", "Fedora Linux 40"),
    ],
)
def test_detect_linux_release(tmp_path, release, distro_id, version, pretty_name):
    facts = detect_facts(sys_platform="linux", machine="x86_64", root=_release(tmp_path, release))
    assert facts.family == "linux"
    assert facts.arch == "amd64"
    assert facts.machine == "x86_64"
    assert facts.distro_id == distro_id
    assert facts.version == version
    assert facts.pretty_name == pretty_name


def test_detect_uses_usr_lib_os_release_fallback(tmp_path):
    _release(tmp_path, 'ID=ubuntu\nVERSION_ID="24.04"\n', relative="usr/lib/os-release")
    facts = detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path)
    assert facts.distro_id == "ubuntu"
    assert facts.version == "24.04"


def test_detect_missing_os_release_is_none(tmp_path):
    facts = detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path)
    assert facts.distro_id is None
    assert facts.version is None
    assert facts.pretty_name is None


def test_detect_malformed_version_is_preserved(tmp_path):
    facts = detect_facts(
        sys_platform="linux",
        machine="x86_64",
        root=_release(tmp_path, 'ID=ubuntu\nVERSION_ID="rolling"\n'),
    )
    assert facts.version == "rolling"


def test_detect_quoted_values_and_equals_sign(tmp_path):
    release = "ID='ubuntu'\nVERSION_ID=\"24.04\"\nPRETTY_NAME='Ubuntu \"Noble\" = Numbat'\n"
    facts = detect_facts(sys_platform="linux", machine="x86_64", root=_release(tmp_path, release))
    assert facts.distro_id == "ubuntu"
    assert facts.version == "24.04"
    assert facts.pretty_name == 'Ubuntu "Noble" = Numbat'


def test_parse_os_release_ignores_comments_and_blank_lines():
    parsed = parse_os_release("# comment\n\nID=ubuntu\nVERSION_ID=22.04\nNAME=some=value\n")
    assert parsed == {"ID": "ubuntu", "VERSION_ID": "22.04", "NAME": "some=value"}


def test_detect_container_dockerenv(tmp_path):
    _release(tmp_path, "ID=ubuntu\nVERSION_ID=24.04\n")
    (tmp_path / ".dockerenv").touch()
    assert detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path).in_container


def test_detect_container_containerenv(tmp_path):
    _release(tmp_path, "ID=ubuntu\nVERSION_ID=24.04\n")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / ".containerenv").touch()
    assert detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path).in_container


@pytest.mark.parametrize("marker", [".dockerenv", "run/.containerenv"])
def test_detect_inaccessible_container_marker_is_absent(monkeypatch, tmp_path, marker):
    _release(tmp_path, 'ID=ubuntu\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04.4 LTS"\n')
    blocked = tmp_path / marker
    original_exists = Path.exists

    def exists(path):
        if path == blocked:
            raise PermissionError(path)
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    facts = detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path)
    assert facts.distro_id == "ubuntu"
    assert facts.version == "22.04"
    assert facts.in_container is False


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_detect_unsearchable_run_directory_is_absent(tmp_path):
    _release(tmp_path, 'ID=ubuntu\nVERSION_ID="22.04"\nPRETTY_NAME="Ubuntu 22.04.4 LTS"\n')
    run = tmp_path / "run"
    run.mkdir()
    (run / ".containerenv").touch()
    run.chmod(0o000)
    try:
        facts = detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path)
    finally:
        run.chmod(0o700)
    assert facts.distro_id == "ubuntu"
    assert facts.version == "22.04"
    assert facts.in_container is False


def test_detect_container_cgroup(tmp_path):
    _release(tmp_path, "ID=ubuntu\nVERSION_ID=24.04\n")
    cgroup = tmp_path / "proc" / "1" / "cgroup"
    cgroup.parent.mkdir(parents=True)
    cgroup.write_text("1:name=systemd:/kubepods/burstable/pod123\n", encoding="utf-8")
    assert detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path).in_container


def test_detect_wsl_marker(tmp_path):
    _release(tmp_path, "ID=ubuntu\nVERSION_ID=24.04\n")
    osrelease = tmp_path / "proc" / "sys" / "kernel" / "osrelease"
    osrelease.parent.mkdir(parents=True)
    osrelease.write_text("5.15.153.1-microsoft-standard-WSL2\n", encoding="utf-8")
    assert detect_facts(sys_platform="linux", machine="x86_64", root=tmp_path).in_wsl


def test_detect_darwin(monkeypatch):
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.5", ("", "", ""), ""))
    facts = detect_facts(sys_platform="darwin", machine="arm64", root=Path("unused"))
    assert facts.family == "darwin"
    assert facts.distro_id == "macos"
    assert facts.version == "14.5"
    assert facts.pretty_name == "macOS 14.5"
    assert facts.arch == "arm64"


def test_detect_windows(monkeypatch):
    monkeypatch.setattr(platform, "win32_ver", lambda: ("10", "22631", "", ""))
    facts = detect_facts(sys_platform="win32", machine="AMD64", root=Path("unused"))
    assert facts.family == "windows"
    assert facts.distro_id == "windows"
    assert facts.version == "22631"
    assert facts.pretty_name == "Windows"
    assert facts.arch == "amd64"


def test_detect_other_platform():
    facts = detect_facts(sys_platform="freebsd14", machine="amd64", root=Path("unused"))
    assert facts.family == "other"
    assert facts.distro_id is None
    assert facts.version is None
    assert facts.pretty_name is None


@pytest.mark.parametrize(
    ("sys_platform", "family"),
    [
        ("linux", "linux"),
        ("linux2", "linux"),
        ("darwin", "darwin"),
        ("win32", "windows"),
        ("cygwin", "windows"),
        ("msys", "windows"),
        ("freebsd14", "other"),
    ],
)
def test_family_for(sys_platform, family):
    assert family_for(sys_platform) == family


@pytest.mark.parametrize(
    ("machine", "arch"),
    [
        ("x86_64", "amd64"),
        ("AMD64", "amd64"),
        ("amd64", "amd64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("ARM64", "arm64"),
        ("i686", "i686"),
        ("RISC-V", "risc-v"),
    ],
)
def test_normalize_arch(machine, arch):
    assert normalize_arch(machine) == arch
