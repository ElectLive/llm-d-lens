"""Download and cache llm-d / llm-d-benchmark source checkouts pinned to a
specific ref (branch, tag, or commit SHA) selected in the cluster-creation
wizard's "Software Versions" step.

Here we only need the requested ref's source tree fetched once and its path
remembered against the cluster record, so any other module (evaluate,
monitoring, ...) that later needs "the llm-d ref this cluster was configured
with" can resolve it from the cluster record instead of re-downloading.

Downloads are keyed by source URL and resolved commit. Branches are refreshed;
immutable commits can be reused offline. A per-repository file lock serializes
publication across concurrent requests and processes.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal

from llm_d_lens.infrastructure.process import run_sync
from llm_d_lens.infrastructure.storage.paths import storage_path

logger = logging.getLogger(__name__)

RepoName = Literal["llm-d", "llm-d-benchmark"]
DownloadState = Literal["idle", "downloading", "ready", "failed"]

REPO_URLS: dict[RepoName, str] = {
    "llm-d": "https://github.com/llm-d/llm-d.git",
    "llm-d-benchmark": "https://github.com/llm-d/llm-d-benchmark.git",
}


_CLONE_TIMEOUT_SECONDS = 300.0
_COMMIT = re.compile(r"[0-9a-fA-F]{40}")


def cache_root() -> Path:
    """Root cache directory under LENS_CACHE_DIR."""
    return storage_path("cache", "repos")


def repo_dir(repo: RepoName, ref: str) -> Path:
    """URL-isolated location for a commit (or an unambiguous ref placeholder)."""
    source = hashlib.sha256(REPO_URLS[repo].encode()).hexdigest()
    name = ref.lower() if _COMMIT.fullmatch(ref) else "ref-" + hashlib.sha256(ref.encode()).hexdigest()
    return cache_root() / repo / source / name


def _git(*args: str) -> str:
    result = run_sync(["git", *args], timeout=_CLONE_TIMEOUT_SECONDS)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Git source download failed")
    return result.stdout.strip()


def _download_snapshot(repo: RepoName, ref: str) -> Path:
    destination = repo_dir(repo, ref)
    root = destination.parent
    root.mkdir(parents=True, exist_ok=True)
    with (root / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _COMMIT.fullmatch(ref) and (destination / ".git").is_dir():
            return destination
        mirror = root / "mirror.git"
        if not mirror.is_dir():
            with TemporaryDirectory(prefix="clone-", dir=root) as temporary:
                staging = Path(temporary) / "mirror.git"
                _git("clone", "--quiet", "--mirror", "--", REPO_URLS[repo], str(staging))
                staging.rename(mirror)
        else:
            _git("--git-dir", str(mirror), "fetch", "--quiet", "--force", "--prune", "origin")
        commit = _git("--git-dir", str(mirror), "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
        if not _COMMIT.fullmatch(commit):
            raise RuntimeError("Git did not resolve a commit")
        destination = repo_dir(repo, commit)
        if not (destination / ".git").is_dir():
            with TemporaryDirectory(prefix="checkout-", dir=root) as temporary:
                staging = Path(temporary) / "checkout"
                _git("clone", "--quiet", "--shared", "--no-checkout", "--", str(mirror), str(staging))
                _git("-C", str(staging), "checkout", "--quiet", "--detach", commit)
                staging.rename(destination)
        return destination


def download_snapshot(repo: RepoName, ref: str) -> Path:
    """Fetch a source snapshot synchronously using the shared managed cache."""
    ref = ref.strip()
    if not ref or ref.startswith("-"):
        raise ValueError("A nonempty Git ref is required")
    return _download_snapshot(repo, ref)


@dataclass
class DownloadStatus:
    repo: RepoName
    ref: str = ""
    state: DownloadState = "idle"
    path: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {"repo": self.repo, "ref": self.ref, "state": self.state, "path": self.path, "error": self.error}


_status_lock = Lock()
_status: dict[tuple[str, RepoName], DownloadStatus] = {}


def get_status(cluster_id: str, repo: RepoName) -> DownloadStatus:
    with _status_lock:
        existing = _status.get((cluster_id, repo))
        return existing if existing is not None else DownloadStatus(repo=repo)


def _set_status(cluster_id: str, repo: RepoName, **updates: object) -> None:
    with _status_lock:
        current = _status.get((cluster_id, repo)) or DownloadStatus(repo=repo)
        for key, value in updates.items():
            setattr(current, key, value)
        _status[(cluster_id, repo)] = current


async def ensure_downloaded(cluster_id: str, repo: RepoName, ref: str) -> Path:
    """Resolve a ref and publish an immutable checkout under a shared cache lock."""
    ref = ref.strip()
    _set_status(cluster_id, repo, ref=ref, state="downloading", path=None, error=None)
    try:
        dest = await asyncio.to_thread(download_snapshot, repo, ref)
    except Exception as error:
        logger.exception("event=repo_download_failed repo=%s ref=%s cluster_id=%s", repo, ref, cluster_id)
        _set_status(cluster_id, repo, ref=ref, state="failed", path=None, error=str(error))
        raise
    _set_status(cluster_id, repo, ref=ref, state="ready", path=str(dest), error=None)
    return dest
