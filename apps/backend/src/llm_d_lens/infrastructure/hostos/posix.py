"""POSIX family behavior: paths, locking, permissions, processes and probes.

Platform modules (``fcntl``, ``signal``) are imported inside the methods that
need them so this module imports on any host and ``hostos check`` works
everywhere.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import NamedTuple

from .facts import HostFacts


class ToolPlatform(NamedTuple):
    """Download platform tuple for a third-party tool artifact."""

    os: str
    arch: str
    exe_suffix: str

    def joined(self) -> str:
        return f"{self.os}-{self.arch}"


def _command_output(argv: list[str]) -> str:
    """Run a probe and return stdout, empty on any failure."""
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, text=True, check=False, stderr=subprocess.DEVNULL)
    except OSError:
        return ""
    return result.stdout


class PosixHost:
    """Today's POSIX host behavior, unchanged."""

    def __init__(self, facts: HostFacts) -> None:
        self._facts = facts

    def default_base(self, area: str) -> Path | None:
        homes = {
            "data": Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "lens",
            "cache": Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "lens",
            "log": Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "lens/logs",
        }
        if area in homes:
            return homes[area]
        if area == "runtime":
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            return Path(xdg) / "lens" if xdg else None
        raise ValueError(f"Unknown storage area: {area}")

    def default_kubeconfig(self) -> Path:
        return Path.home() / ".kube" / "config"

    def venv_executable(self, venv: Path, name: str) -> Path:
        return venv / "bin" / name

    def try_lock(self, handle: object, *, shared: bool = False, blocking: bool = True) -> bool:
        import fcntl  # noqa: PLC0415 - POSIX-only, imported lazily

        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle, operation)
        except BlockingIOError:
            return False
        return True

    def restrict_to_owner(self, path: Path) -> None:
        path.chmod(0o700 if path.is_dir() else 0o600)

    def make_executable(self, path: Path, *, private: bool) -> None:
        path.chmod(0o700 if private else 0o755)

    def group_spawn_kwargs(self) -> dict[str, bool]:
        return {"start_new_session": True}

    def signal_group(self, pid: int, *, force: bool) -> None:
        import signal  # noqa: PLC0415 - POSIX-only, imported lazily

        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)

    @property
    def null_device(self) -> str:
        return os.devnull

    def tool_platform(self, style: str) -> ToolPlatform:
        if style == "go":
            return ToolPlatform("linux", "arm64" if self._facts.arch == "arm64" else "amd64", "")
        if style == "node":
            return ToolPlatform("linux", "arm64" if self._facts.arch == "arm64" else "x64", "")
        raise ValueError(f"Unknown tool platform style: {style}")

    def machine_info(self) -> dict[str, object]:
        cpu_model = platform.processor() or "unknown"
        cpu_file = Path("/proc/cpuinfo")
        if cpu_file.is_file():
            cpu_model = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpu_file.read_text().splitlines()
                    if line.startswith("model name")
                ),
                cpu_model,
            )
        return {
            "hostname": platform.node(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "cpu": {"model": cpu_model, "logicalCores": os.cpu_count() or 0},
            "memoryGiB": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1),
        }

    def port_in_use(self, port: int) -> bool:
        if shutil.which("ss"):
            result = subprocess.run(
                ["ss", "-H", "-ltn", f"( sport = :{port} )"],
                stdout=subprocess.PIPE,
                text=True,
                check=False,
                stderr=subprocess.DEVNULL,
            )
            return bool(result.stdout.strip())
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except (OSError, OverflowError):
            return False

    def describe_port_owner(self, port: int) -> str:
        line = _command_output(["ss", "-H", "-ltnp", f"( sport = :{port} )"])
        lines = line.splitlines()
        first_line = lines[0] if lines else ""
        if first_line:
            match = re.search(r"pid=([0-9]+)", first_line)
            if match:
                pid = match.group(1)
                owner = _command_output(["ps", "-o", "comm=,user=", "-p", pid]).rstrip("\n")
                if owner:
                    return f"{owner} (pid {pid})"
            return "another process on this shared host (owner not visible without root)"
        return "an unknown process"

    def first_lan_address(self) -> str:
        tokens = _command_output(["hostname", "-I"]).split()
        return tokens[0] if tokens else ""

    @property
    def default_capabilities(self) -> dict[str, bool]:
        return {"cluster-bootstrap": True}
