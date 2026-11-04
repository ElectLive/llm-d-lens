"""Host OS abstraction: facts, support gate, family behavior and packages.

Only the standard library is used and nothing outside this package is
imported, so the layer can run before Lens's own Python environment exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NoReturn

from .definition import OsDefinition, PackageManager
from .facts import HostFacts, detect_facts
from .posix import PosixHost, ToolPlatform
from .systems import SYSTEMS, find_definition


class UnsupportedHostOperation(RuntimeError):  # noqa: N818 - public name fixed by the design
    """Raised when a host family cannot perform an operation."""


class UnsupportedHost:
    """Behavior methods for families without an implementation (yet)."""

    def __init__(self, family: str = "other") -> None:
        self._family = family

    def _unsupported(self, operation: str) -> NoReturn:
        raise UnsupportedHostOperation(f"{operation} is not supported on {self._family}")

    def default_base(self, area: str) -> Path | None:
        self._unsupported("default_base")

    def default_kubeconfig(self) -> Path:
        self._unsupported("default_kubeconfig")

    def venv_executable(self, venv: Path, name: str) -> Path:
        self._unsupported("venv_executable")

    def try_lock(self, handle: object, *, shared: bool = False, blocking: bool = True) -> bool:
        self._unsupported("try_lock")

    def restrict_to_owner(self, path: Path) -> None:
        self._unsupported("restrict_to_owner")

    def make_executable(self, path: Path, *, private: bool) -> None:
        self._unsupported("make_executable")

    def group_spawn_kwargs(self) -> dict[str, bool]:
        self._unsupported("group_spawn_kwargs")

    def signal_group(self, pid: int, *, force: bool) -> None:
        self._unsupported("signal_group")

    @property
    def null_device(self) -> str:
        return os.devnull

    def tool_platform(self, style: str) -> ToolPlatform:
        self._unsupported("tool_platform")

    def machine_info(self) -> dict[str, object]:
        self._unsupported("machine_info")

    def port_in_use(self, port: int) -> bool:
        self._unsupported("port_in_use")

    def describe_port_owner(self, port: int) -> str:
        self._unsupported("describe_port_owner")

    def first_lan_address(self) -> str:
        self._unsupported("first_lan_address")

    @property
    def default_capabilities(self) -> dict[str, bool]:
        return {"cluster-bootstrap": False}


@dataclass(frozen=True)
class SupportStatus:
    supported: bool
    message: str


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = value.split(".")
    if not all(part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _display_arch(arch: str) -> str:
    return "x86_64" if arch == "amd64" else arch


def _display_machine(facts: HostFacts) -> str:
    return "x86_64" if facts.arch == "amd64" else facts.machine


def _supported_description() -> str:
    return ", ".join(
        f"{definition.display_name} {definition.min_version}+ "
        f"({'/'.join(_display_arch(arch) for arch in definition.architectures)})"
        for definition in SYSTEMS
    )


def _support_status(facts: HostFacts, definition: OsDefinition | None) -> SupportStatus:
    supported = _supported_description()
    if facts.family not in {"linux", "darwin", "windows"}:
        return SupportStatus(False, f"unsupported platform {facts.family} detected — supported: {supported}")
    if definition is None:
        detected = facts.pretty_name or facts.distro_id or facts.family
        return SupportStatus(False, f"{detected} detected — supported: {supported}")
    version = _version_tuple(facts.version)
    if facts.version is None:
        return SupportStatus(False, f"could not determine {definition.display_name} version — supported: {supported}")
    if version is None:
        return SupportStatus(
            False, f"{definition.display_name} version {facts.version!r} detected — supported: {supported}"
        )
    minimum = _version_tuple(definition.min_version)
    if minimum is not None and version < minimum:
        return SupportStatus(False, f"{definition.display_name} {facts.version} detected — supported: {supported}")
    if facts.arch not in definition.architectures:
        return SupportStatus(False, f"architecture {_display_machine(facts)} detected — supported: {supported}")
    pretty = f" ({facts.pretty_name})" if facts.pretty_name else ""
    return SupportStatus(
        True,
        f"{definition.display_name} {facts.version} detected{pretty}, architecture {_display_machine(facts)}",
    )


def _is_root() -> bool:
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False


def _package_installed(package_manager: PackageManager, package: str | None) -> bool:
    if package is None:
        return False
    try:
        result = subprocess.run(
            [*package_manager.query, package],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and package_manager.installed_marker in result.stdout


class Host:
    """One detected host: facts, optional definition and family behavior."""

    def __init__(self, facts: HostFacts, definition: OsDefinition | None) -> None:
        self._facts = facts
        self._definition = definition
        self._impl: PosixHost | UnsupportedHost = (
            PosixHost(facts) if os.name == "posix" else UnsupportedHost(facts.family)
        )

    @property
    def facts(self) -> HostFacts:
        return self._facts

    @property
    def definition(self) -> OsDefinition | None:
        return self._definition

    @property
    def installer(self) -> str | None:
        return self._definition.installer if self._definition else None

    @property
    def capabilities(self) -> dict[str, bool]:
        capabilities = dict(self._impl.default_capabilities)
        if self._definition is not None:
            capabilities.update(self._definition.capabilities)
        return capabilities

    @property
    def support(self) -> SupportStatus:
        return _support_status(self._facts, self._definition)

    def default_base(self, area: str) -> Path | None:
        return self._impl.default_base(area)

    def default_kubeconfig(self) -> Path:
        return self._impl.default_kubeconfig()

    def venv_executable(self, venv: Path, name: str) -> Path:
        return self._impl.venv_executable(venv, name)

    def try_lock(self, handle: object, *, shared: bool = False, blocking: bool = True) -> bool:
        return self._impl.try_lock(handle, shared=shared, blocking=blocking)

    def restrict_to_owner(self, path: Path) -> None:
        self._impl.restrict_to_owner(path)

    def make_executable(self, path: Path, *, private: bool) -> None:
        self._impl.make_executable(path, private=private)

    def group_spawn_kwargs(self) -> dict[str, bool]:
        return self._impl.group_spawn_kwargs()

    def signal_group(self, pid: int, *, force: bool) -> None:
        self._impl.signal_group(pid, force=force)

    @property
    def null_device(self) -> str:
        return self._impl.null_device

    def tool_platform(self, style: str) -> ToolPlatform:
        return self._impl.tool_platform(style)

    def machine_info(self) -> dict[str, object]:
        return self._impl.machine_info()

    def port_in_use(self, port: int) -> bool:
        return self._impl.port_in_use(port)

    def describe_port_owner(self, port: int) -> str:
        return self._impl.describe_port_owner(port)

    def first_lan_address(self) -> str:
        return self._impl.first_lan_address()

    def _require_definition(self, operation: str) -> OsDefinition:
        if self._definition is None:
            raise UnsupportedHostOperation(f"{operation} is not supported on {self._facts.family}")
        return self._definition

    def missing_packages(self) -> list[str]:
        definition = self._require_definition("missing_packages")
        package_manager = definition.package_manager
        missing: list[str] = []
        for requirement in definition.requirements:
            if requirement.command is not None:
                if shutil.which(requirement.command) is None:
                    missing.extend(requirement.packages)
            elif not _package_installed(package_manager, requirement.package):
                missing.extend(requirement.packages)
        return sorted(set(missing))

    def package_commands(self, packages: list[str] | tuple[str, ...]) -> list[list[str]]:
        package_manager = self._require_definition("package_commands").package_manager
        commands = [list(package_manager.refresh), [*package_manager.install, *packages]]
        if package_manager.needs_elevation and not _is_root():
            return [["sudo", *command] for command in commands]
        return commands

    def install_hint(self, packages: list[str] | tuple[str, ...]) -> str:
        joined = " ".join(packages)
        if self._definition is None:
            return f"install {joined} with the system package manager"
        package_manager = self._definition.package_manager
        command = " ".join([*package_manager.manual_install, *packages])
        return f"sudo {command}" if package_manager.needs_elevation else command

    def tool_packages(self, name: str) -> tuple[str, ...]:
        if self._definition is None:
            return (name,)
        return self._definition.optional_tools.get(name, (name,))

    @staticmethod
    def _install_problem(definition: OsDefinition, packages: list[str] | tuple[str, ...]) -> str | None:
        package_manager = definition.package_manager
        binary = package_manager.install[0]
        if shutil.which(binary) is None:
            return f"{binary} not found — install manually: {' '.join(packages)}"
        if package_manager.needs_elevation and not _is_root() and shutil.which("sudo") is None:
            return "sudo is required to install packages as non-root"
        return None

    def package_install_problem(self, packages: list[str] | tuple[str, ...]) -> str | None:
        """Return why packages cannot be installed, or None when they can."""
        return self._install_problem(self._require_definition("install_packages"), packages)

    def install_packages(self, packages: list[str] | tuple[str, ...]) -> int:
        definition = self._require_definition("install_packages")
        problem = self._install_problem(definition, packages)
        if problem is not None:
            print(problem, file=sys.stderr)
            return 1
        package_manager = definition.package_manager
        refresh, install = self.package_commands(packages)
        refresh_result = subprocess.run(refresh, check=False)
        if refresh_result.returncode != 0:
            print(
                f"{package_manager.refresh[0]} update failed (exit {refresh_result.returncode})",
                file=sys.stderr,
            )
            return 1
        install_result = subprocess.run(install, check=False)
        if install_result.returncode != 0:
            print(
                f"{package_manager.install[0]} install failed (exit {install_result.returncode})",
                file=sys.stderr,
            )
            return 1
        return 0


@lru_cache(maxsize=1)
def current_host() -> Host:
    """Detect and cache this process's host. Never raises for unsupported hosts."""
    facts = detect_facts()
    return Host(facts, find_definition(facts.distro_id))


def check_host(installer: str | None = None) -> SupportStatus:
    """Return the support gate result, optionally checking the installer file."""
    host = current_host()
    requested: OsDefinition | None = None
    if installer is not None:
        requested = find_definition(installer)
        if requested is None:
            raise ValueError(f"unknown installer: {installer}")
    status = host.support
    if (
        requested is not None
        and status.supported
        and host.definition is not None
        and host.definition.id != requested.id
    ):
        return SupportStatus(
            False,
            f"this installer is for {requested.display_name}, but this host is "
            f"{host.definition.display_name}; use {host.definition.installer} instead",
        )
    return status
