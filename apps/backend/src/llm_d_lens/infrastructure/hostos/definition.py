"""Typed data definitions for supported host operating systems.

These structures carry OS identity and package knowledge only. Behavior lives
in a family implementation (see ``posix.py``), so a later OS is added as data
plus a definition module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class PackageManager:
    """Commands used to inspect, install and describe system packages."""

    name: str
    refresh: tuple[str, ...]
    install: tuple[str, ...]
    manual_install: tuple[str, ...]
    query: tuple[str, ...]
    installed_marker: str
    needs_elevation: bool


@dataclass(frozen=True)
class Requirement:
    """One installer system dependency and how to probe for it.

    Exactly one of ``command`` (probe PATH) or ``package`` (ask the package
    manager) is set.
    """

    packages: tuple[str, ...]
    command: str | None = None
    package: str | None = None

    def __post_init__(self) -> None:
        if (self.command is None) == (self.package is None):
            raise ValueError("Requirement must set exactly one of command or package")


@dataclass(frozen=True)
class OsDefinition:
    """Identity, version gate, package manager and capabilities for one OS."""

    id: str
    display_name: str
    family: str
    min_version: str
    architectures: tuple[str, ...]
    installer: str
    package_manager: PackageManager
    requirements: tuple[Requirement, ...]
    optional_tools: Mapping[str, tuple[str, ...]]
    capabilities: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "optional_tools", MappingProxyType(dict(self.optional_tools)))
        object.__setattr__(self, "capabilities", MappingProxyType(dict(self.capabilities)))
