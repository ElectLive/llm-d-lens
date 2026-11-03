"""llm-d Lens engine: lifecycle operations shared by the CLI and the dashboard.

This package must not import the legacy service (``llm_d_lens``), FastAPI,
the database layer or auth; ``tests/test_boundary.py`` enforces this.
"""

__version__ = "0.0.1"
