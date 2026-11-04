from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "llm_d_lens.agentic.public_benchmark_record.admin",
        "llm_d_lens.agentic.public_benchmark_record.case_retrieval",
        "llm_d_lens.db.dao.public_benchmark_record",
        "llm_d_lens.db.models.public_benchmark_record",
    ],
)
def test_entry_points_import_in_a_fresh_process(module_name: str):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; importlib.import_module(sys.argv[1])",
            module_name,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
