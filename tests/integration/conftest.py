"""Integration-test fixtures: one real simulator session per test module.

These are marked `sim` and require the BEHAVIOR dataset. The session is
built once per module; each test resets deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO = REPO_ROOT / "scenarios" / "knife_search_001" / "scenario.yaml"


@pytest.fixture(scope="session")
def sim_session(tmp_path_factory):
    if not SCENARIO.exists():
        pytest.skip(f"scenario not built yet: {SCENARIO}")
    os.environ.setdefault("RUMMAGEBENCH_ROOT", str(REPO_ROOT))

    from rummagebench.adapters.python_api import create_session

    run_dir = tmp_path_factory.mktemp("simrun")
    # legacy privileged session: scripted acceptance / reset checks (§5
    # requires oracle tests to opt in explicitly)
    session = create_session(SCENARIO, run_dir=run_dir, seed=0, mode="oracle")
    session.reset()
    yield session
    # no og.shutdown here: Kit teardown hangs/segfaults; pytest writes its
    # report and the process exit handles cleanup
