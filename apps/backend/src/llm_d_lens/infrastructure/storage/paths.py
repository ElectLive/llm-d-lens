"""Lens filesystem layout. Selection only: domains own lifecycle and cleanup."""

from __future__ import annotations

import os
from pathlib import Path

from llm_d_lens.infrastructure.hostos import current_host


def storage_path(area: str, *parts: str) -> Path:
    """Resolve a controlled path using only Lens roots and host defaults.

    LENS_* overrides win over the host's XDG defaults. No directories are
    created and no legacy data is moved on import. Use the explicit storage
    migration command.
    """
    host = current_host()
    defaults = {
        "data": host.default_base("data"),
        "cache": host.default_base("cache"),
        "log": host.default_base("log"),
    }
    if area in defaults:
        root = Path(os.environ.get(f"LENS_{area.upper()}_DIR") or defaults[area])
    elif area == "scratch":
        root = Path(os.environ.get("LENS_SCRATCH_DIR") or storage_path("cache", "tmp"))
    elif area == "runtime":
        override = os.environ.get("LENS_RUNTIME_DIR")
        if override:
            root = Path(override)
        else:
            runtime_base = host.default_base("runtime")
            root = runtime_base if runtime_base is not None else storage_path("scratch", "runtime")
    else:
        raise ValueError(f"Unknown storage area: {area}")
    root = root.expanduser().resolve()
    for part in parts:
        if Path(part).is_absolute() or ".." in Path(part).parts:
            raise ValueError("Storage path must be relative without parent traversal")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Storage path escapes its root")
    return target


def prism_temp_root(*parts: str) -> Path:
    """Compatibility entry point for provider scratch directories."""
    root = storage_path("scratch")
    for part in parts:
        if Path(part).is_absolute() or ".." in Path(part).parts:
            raise ValueError("Scratch path must be relative")
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Scratch path escapes its root")
    return target
