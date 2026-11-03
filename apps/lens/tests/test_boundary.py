"""The engine must stay independent of the legacy service and its stack."""

import subprocess
import sys

FORBIDDEN = ("llm_d_lens", "fastapi", "sqlalchemy", "alembic", "casbin", "ldap3")


def test_engine_imports_no_legacy_service_modules():
    # A fresh interpreter, so modules loaded by pytest or other tests do not count.
    probe = (
        f"import sys, lens, lens.__main__; print(sorted({{m.split('.')[0] for m in sys.modules}} & set({FORBIDDEN!r})))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", f"lens imported forbidden modules: {result.stdout.strip()}"
