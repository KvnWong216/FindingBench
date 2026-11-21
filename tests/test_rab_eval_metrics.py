"""Unit tests for scripts/rab_eval_metrics.py (pure-Python, no simulator)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rab_eval_metrics as M  # noqa: E402


def _write_episode(root: Path, name: str, steps: list[tuple], terminal: dict) -> None:
    actions = root / name / "worker_trace" / "actions"
    actions.mkdir(parents=True)
    (root / name / "worker_trace" / "manifest.json").write_text(json.dumps({"id": name}))
    for i, (skill, code, err) in enumerate(steps, 1):
        last = i == len(steps)
        result = {"feedback": {"code": code}, "planning_step": i, "terminal": last,
                  "episode_status": terminal["status"] if last else "RUNNING"}
        meas = {}
        if last:
            result.update(success=terminal["status"] == "SUCCESS", restoration=terminal.get("res", 1.0))
            if terminal.get("attr"):
                meas["terminal_attribution"] = terminal["attr"]
        (actions / f"{i:04}.json").write_text(json.dumps(
            {"raw": {"skill": skill}, "result": result, "private_error": err, "measurements": meas}))


def test_prototypes_are_one_to_one():
    seen = {}
    for key, spec in M.CAPABILITIES.items():
        for p in spec["prototypes"]:
            assert p not in seen, f"{p} mapped to both {seen[p]} and {key}"
            seen[p] = key


def test_prototype_parsing_and_legacy():
    assert M.prototype_of("EXECUTED", None) == "EXECUTED"
    assert M.prototype_of("UNSAFE", "NAV_NO_CLEAR_ROUTE") == "NAV_NO_CLEAR_ROUTE"
    assert M.prototype_of("OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.2") == "GRASP_TOO_FAR"
    assert M.prototype_of("UNSAFE", "hand path blocked by /World/x") == "HAND_PATH_BLOCKED"
    assert M.prototype_of("UNSAFE", "INFRA_EXCEPTION:Traceback ...") == "INFRA_EXCEPTION"
    assert M.prototype_of("INVALID_ACTION", None) == "UNLABELED_INVALID_ACTION"


def test_aggregate_core_and_layers(tmp_path):
    _write_episode(tmp_path, "ep_succ", [
        ("VIEW", "EXECUTED", None),
        ("MOVE_TO", "UNSAFE", "NAV_NO_CLEAR_ROUTE"),
        ("GRASP", "OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.1"),
        ("GRASP", "EXECUTED", None),
        ("PLACE", "EXECUTED", None),
        ("REPORT_DONE", "EXECUTED", None),
    ], {"status": "SUCCESS", "res": 0.8})
    _write_episode(tmp_path, "ep_fail", [
        ("GRASP", "UNSAFE", "INFRA_EXCEPTION:boom"),
        ("PLACE", "INVALID_ACTION", "PRECONDITION_HAND_EMPTY"),
        ("REPORT_DONE", "EXECUTED", None),
    ], {"status": "FAIL_FALSE_COMPLETION", "attr": "PREMATURE_DONE", "res": 1.0})
    r = M.aggregate(M.load_episode(d) for d in M.find_episodes(tmp_path))
    c = r["core"]
    assert c["SR"] == 0.5 and c["ST_succ"] == 6 and c["ST_all"] == 4.5
    # infra call excluded: 8 agent calls, 5 executed; physical = MOVE_TO(x),GRASP(x),GRASP,PLACE,PLACE(x) -> 2/5
    assert r["calls"]["infra_excluded"] == 1 and r["calls"]["agent"] == 8
    assert c["EFF_all"] == round(5 / 8, 4) and c["EFF_phys"] == 0.4
    assert c["RES"] == 0.9
    la = r["layer_a"]
    assert la["collision"]["errors"] == 1 and la["spatial"]["errors"] == 1 and la["state"]["errors"] == 1
    # Layer A rates sum to 1 - EFF_all
    total = sum(v["rate_over_all_calls"] or 0 for v in la.values())
    assert abs(total - (1 - c["EFF_all"])) < 1e-3
    assert r["layer_b"]["PREMATURE_DONE"]["count"] == 1
    assert r["layer_b"]["PREMATURE_DONE"]["share_of_failed"] == 1.0


def test_incomplete_episode_excluded_from_sr(tmp_path):
    actions = tmp_path / "ep_crash" / "worker_trace" / "actions"
    actions.mkdir(parents=True)
    (actions / "0001.json").write_text(json.dumps(
        {"raw": {"skill": "VIEW"}, "result": {"feedback": {"code": "EXECUTED"}, "planning_step": 1,
                                              "terminal": False}, "private_error": None}))
    r = M.aggregate(M.load_episode(d) for d in M.find_episodes(tmp_path))
    assert r["episodes"]["incomplete_infra"] == 1 and r["core"]["SR"] is None
