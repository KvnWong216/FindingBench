"""Round-6: strict production acceptance for the real-simulation smoke.

The smoke's trial-selection and acceptance-classification rules are pure
functions in scripts/smoke_place_postcondition.py; they are imported here
the same way test_adapter_contracts imports serve_play. The key rule under
test: a feasibility rejection followed by a fallback commit is classified
postcondition_only and can never enter the production pass statistics.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "smoke_place_postcondition", ROOT / "scripts/smoke_place_postcondition.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("smoke_place_postcondition", smoke)
_spec.loader.exec_module(smoke)


def _trial(**overrides):
    trial = {
        "receptacle": "counter",
        "feasible": True,
        "point_source": "feasibility",
        "approach": {"modelled": True, "clear": True, "held_geometry": True},
        "verdict": {"committed": True},
    }
    trial.update(overrides)
    return trial


def test_production_pass_requires_every_condition():
    assert smoke.classify_acceptance(_trial()) == "production_pass"
    # feasibility rejected + fallback commit -> NEVER a production pass
    fallback = _trial(feasible=False, point_source="fallback_targets[0]",
                      region_targets=8)
    assert smoke.classify_acceptance(fallback) == "postcondition_only"
    # approach not modelled -> committed but not a production pass
    assert smoke.classify_acceptance(
        _trial(approach={"modelled": False, "clear": True,
                         "held_geometry": False})) == "postcondition_only"
    # approach blocked -> not committed at all -> rejected
    assert smoke.classify_acceptance(
        _trial(approach={"modelled": True, "clear": False,
                         "held_geometry": True},
               verdict={"committed": False})) == "rejected"
    # nothing committed -> rejected
    assert smoke.classify_acceptance(_trial(verdict={"committed": False})) \
        == "rejected"
    # held geometry not evidenced -> not a production pass
    assert smoke.classify_acceptance(
        _trial(approach={"modelled": True, "clear": True,
                         "held_geometry": False})) == "postcondition_only"


def test_select_place_trial_forbids_fallback_by_default():
    feasible = _trial(receptacle="counter")
    fallback = _trial(receptacle="shelf", feasible=False,
                      point_source="fallback_targets[0]")
    # a feasible witness is always chosen
    assert smoke.select_place_trial([fallback, feasible],
                                    allow_fallback=False) is feasible
    # without the flag, a fallback-only candidate set selects NOTHING
    assert smoke.select_place_trial([fallback], allow_fallback=False) is None
    # with the explicit diagnostic flag it is eligible (postcondition-only)
    assert smoke.select_place_trial([fallback], allow_fallback=True) is fallback
    # empty set selects nothing
    assert smoke.select_place_trial([], allow_fallback=True) is None
