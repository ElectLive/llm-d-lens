"""Host fact detection. Pure functions, standard library only.

Detection never raises: missing or unreadable files leave fields ``None`` so
imports and test collection work on any host.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

_CONTAINER_CGROUP_MARKERS = ("docker", "kubepods", "containerd", "podman")


@dataclass(frozen=True)
class HostFacts:
    """What we can tell about the machine running Lens."""

    family: str
    arch: str
    machine: str
    distro_id: str | None
    version: str | None
    pretty_name: str | None
    in_container: bool
    in_wsl: bool


def family_for(sys_platform: str) -> str:
    """Map ``sys.platform`` to a host family."""
    if sys_platform.startswith("linux"):
        return "linux"
    if sys_platform == "darwin":
        return "darwin"
    if sys_platform.startswith("win") or sys_platform in {"cygwin", "msys"}:
        return "windows"
    return "other"


def normalize_arch(machine: str) -> str:
    """Normalize common machine names to ``amd64`` / ``arm64``."""
    lowered = machine.lower()
    if lowered in {"x86_64", "amd64"}:
        return "amd64"
    if lowered in {"aarch64", "arm64"}:
        return "arm64"
    return lowered


def parse_os_release(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines, ignoring comments and blank lines.

    Values may be wrapped in single or double quotes.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _read_os_release(root: Path) -> dict[str, str]:
    for relative in ("etc/os-release", "usr/lib/os-release"):
        text = _read_text(root / relative)
        if text:
            return parse_os_release(text)
    return {}


def _container_markers(root: Path) -> bool:
    if _path_exists(root / ".dockerenv") or _path_exists(root / "run" / ".containerenv"):
        return True
    cgroup = _read_text(root / "proc" / "1" / "cgroup").lower()
    return any(marker in cgroup for marker in _CONTAINER_CGROUP_MARKERS)


def _wsl_marker(root: Path) -> bool:
    return "microsoft" in _read_text(root / "proc" / "sys" / "kernel" / "osrelease").lower()


def _windows_build() -> str | None:
    try:
        release = platform.win32_ver()
        build = release[1] if len(release) > 1 else ""
    except Exception:  # noqa: BLE001 - detection must never raise
        build = ""
    if build:
        return build
    try:
        version = platform.version()
    except Exception:  # noqa: BLE001 - detection must never raise
        return None
    if not version:
        return None
    parts = version.split(".")
    return parts[2] if len(parts) >= 3 else parts[-1]


def _macos_version() -> str | None:
    try:
        version = platform.mac_ver()[0]
    except Exception:  # noqa: BLE001 - detection must never raise
        return None
    return version or None


def detect_facts(
    *,
    sys_platform: str | None = None,
    machine: str | None = None,
    root: Path = Path("/"),
) -> HostFacts:
    """Detect host facts without raising.

    ``root`` is the filesystem root used for markers and ``os-release``; tests
    point it at a fixture tree.
    """
    sys_platform = sys.platform if sys_platform is None else sys_platform
    machine = platform.machine() if machine is None else machine
    family = family_for(sys_platform)
    arch = normalize_arch(machine)

    distro_id: str | None = None
    version: str | None = None
    pretty_name: str | None = None
    in_container = False
    in_wsl = False

    if family == "linux":
        release = _read_os_release(root)
        distro_id = release.get("ID", "").strip().lower() or None
        version = release.get("VERSION_ID") or None
        pretty_name = release.get("PRETTY_NAME") or None
        in_container = _container_markers(root)
        in_wsl = _wsl_marker(root)
    elif family == "darwin":
        distro_id = "macos"
        version = _macos_version()
        pretty_name = f"macOS {version}" if version else "macOS"
    elif family == "windows":
        distro_id = "windows"
        version = _windows_build()
        pretty_name = "Windows"

    return HostFacts(
        family=family,
        arch=arch,
        machine=machine,
        distro_id=distro_id,
        version=version,
        pretty_name=pretty_name,
        in_container=in_container,
        in_wsl=in_wsl,
    )
