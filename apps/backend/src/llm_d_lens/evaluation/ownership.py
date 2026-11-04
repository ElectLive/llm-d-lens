"""Process ownership for evaluations sharing the same persistent data directory.

Locks are kernel-managed, not PID files or expiring leases. Keep the inode after
release: unlinking it would let two processes lock different files for one key.
All backends sharing evaluation records must also share this lock directory on
a filesystem supporting flock. Separate data roots are separate installations.
"""

import os
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
from inspect import signature
from typing import get_type_hints

from fastapi import HTTPException

from llm_d_lens.infrastructure.hostos import current_host
from llm_d_lens.infrastructure.storage.paths import storage_path

_held = {}


def _lock_path(kind, identifier):
    root = storage_path("data", "metadata", "evaluations", "locks")
    root.mkdir(parents=True, exist_ok=True)
    digest = sha256(f"{kind}:{identifier}".encode()).hexdigest()
    return root / f"{digest}.lock"


@contextmanager
def evaluation_lock(kind, identifier, *, allow_local=False, shared=False):
    """Try exclusive ownership; local control calls may retain an existing lock."""
    path = _lock_path(kind, identifier)
    key = (os.getpid(), path, shared)
    held = _held.get(key)
    if held is not None:
        if not (allow_local or shared):
            yield False
            return
        held[1] += 1
    else:
        handle = path.open("a+")
        try:
            if not current_host().try_lock(handle, shared=shared, blocking=False):
                handle.close()
                yield False
                return
        except BaseException:
            handle.close()
            raise
        held = [handle, 1]
        _held[key] = held
    try:
        yield True
    finally:
        held[1] -= 1
        if held[1] == 0:
            _held.pop(key, None)
            held[0].close()


def owned_execution(kind):
    """Hold ownership until execution and its finally/cleanup blocks settle."""

    def decorate(function):
        @wraps(function)
        async def execute(*args, **kwargs):
            identifier = args[0] if args else kwargs.get("workflow_id", kwargs.get("run_id"))
            with evaluation_lock(kind, identifier) as acquired:
                if acquired:
                    return await function(*args, **kwargs)
            return None

        return execute

    return decorate


def owned_control(kind):
    """Reject cross-process controls before they mutate records or resources."""

    def decorate(function):
        @wraps(function)
        async def control(*args, **kwargs):
            value = args[0] if args else kwargs.get("workflow_id", kwargs.get("run_id", kwargs.get("workflow")))
            identifier = value["id"] if isinstance(value, dict) else value
            with evaluation_lock(kind, identifier, allow_local=True) as acquired:
                if not acquired:
                    raise HTTPException(
                        status_code=409,
                        detail="Evaluation is active in another backend instance; use its control endpoint.",
                    )
                return await function(*args, **kwargs)

        hints = get_type_hints(function)
        original = signature(function)
        control.__signature__ = original.replace(
            parameters=[
                parameter.replace(annotation=hints.get(name, parameter.annotation))
                for name, parameter in original.parameters.items()
            ],
            return_annotation=None
            if hints.get("return") is type(None)
            else hints.get("return", original.return_annotation),
        )
        return control

    return decorate
