import shutil
import subprocess
from pathlib import Path

import pytest


def git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("LENS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LENS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.delenv("LLMD_STATE_DIR", raising=False)
    monkeypatch.delenv("LLMD_NAMESPACE", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)


DATA = Path(__file__).parent / "data"


def committed_copy(target):
    """A copy of the test repository (llm-d's layout), committed so listing reads Git."""
    shutil.copytree(DATA / "llmd", target)
    git(target, "init", "-q", "-b", "main")
    git(target, "add", "-A")
    git(target, "commit", "-q", "-m", "stacks")
    return target


@pytest.fixture
def llmrepo(tmp_path):
    return committed_copy(tmp_path / "llmrepo")
