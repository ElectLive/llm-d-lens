"""Built-in host OS definitions. No plugin or entry-point discovery."""

from __future__ import annotations

from ..definition import OsDefinition
from .ubuntu import UBUNTU

SYSTEMS: tuple[OsDefinition, ...] = (UBUNTU,)


def find_definition(distro_id: str | None) -> OsDefinition | None:
    """Return the definition whose id equals ``distro_id``, if any."""
    if distro_id is None:
        return None
    for definition in SYSTEMS:
        if definition.id == distro_id:
            return definition
    return None
