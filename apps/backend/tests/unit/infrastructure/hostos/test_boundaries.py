"""Import-boundary tests for the standalone host layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[4] / "src"


def _run_python(code: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{SOURCE}{os.pathsep}{existing}" if existing else str(SOURCE)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, check=False)


def test_hostos_imports_nothing_else_from_llm_d_lens():
    code = (
        "import json, sys\n"
        "import llm_d_lens.infrastructure.hostos\n"
        "allowed = {'llm_d_lens', 'llm_d_lens.infrastructure'}\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('llm_d_lens'))\n"
        "extra = [name for name in loaded if name not in allowed "
        "and not name.startswith('llm_d_lens.infrastructure.hostos')]\n"
        "forbidden = sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'fastapi', 'pydantic', 'sqlalchemy'})\n"
        "print(json.dumps({'extra': extra, 'forbidden': forbidden}))\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["extra"] == []
    assert payload["forbidden"] == []
