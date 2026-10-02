"""Every committed scenario file must pass the strict schema.

Guards the generated scenarios (scripts/generate_scenarios.py) against
schema drift: a batch re-render or a schema change that breaks any
scenarios/*/scenario.yaml fails here, on CPU.
"""
from pathlib import Path

import pytest

from rummagebench.core.scenario import load_scenario

SCENARIOS = sorted(
    p.parent for p in (Path(__file__).resolve().parents[2] / "scenarios").glob(
        "*/scenario.yaml"))


def test_scenarios_exist():
    assert SCENARIOS, "no scenarios/*/scenario.yaml found"


@pytest.mark.parametrize("scenario_dir", SCENARIOS, ids=[p.name for p in SCENARIOS])
def test_scenario_passes_strict_schema(scenario_dir: Path):
    scenario = load_scenario(str(scenario_dir / "scenario.yaml"))
    assert scenario.id == scenario_dir.name
    # a target must always be one of the spawned objects (schema cross-check
    # also enforces this; asserted here so the generator cannot regress it)
    assert scenario.target.entity in {o.name for o in scenario.objects}
    # the visual protocol requires the full renderer modality set
    assert {"rgb", "depth_linear", "seg_instance"} <= set(scenario.robot.obs_modalities)
