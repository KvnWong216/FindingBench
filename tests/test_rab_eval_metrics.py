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


def test_proprioceptive_feedback_coverage_and_zero_hint_invariant():
    import re
    import rab_feedback as F

    # Every error prototype (plus infra) has an explicit proprioceptive mapping.
    all_protos = set(M.PROTOTYPE_TO_CAPABILITY) | set(M.INFRA_PROTOTYPES)
    assert all_protos <= set(F.PROPRIOCEPTIVE_BY_ERROR)

    forbidden = ("please", "try ", "move ", "click ", "open ", "close ", "remove ", "first", "should", "instead", "/world/")
    all_msgs = [msg for _, msg in F.PROPRIOCEPTIVE_BY_ERROR.values()] + list(F.SKILL_MESSAGE_OVERRIDES.values())
    for msg in all_msgs:
        low = msg.lower()
        assert not re.search(r"\d", msg), f"Numeric measurement leaked in public feedback: {msg!r}"
        for bad in forbidden:
            assert bad not in low, f"Directive/leakage token {bad!r} in public feedback: {msg!r}"

    # Mode switch: four_class_v1 omits detail_code; proprioceptive_v2 adds neutral detail_code
    fb_v1 = F.format_public_feedback("OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.42", skill="GRASP",
                                     default_message="Legacy", mode="four_class_v1")
    assert fb_v1 == {"code": "OUT_OF_CAPABILITY", "message": "Legacy"}
    fb_v2 = F.format_public_feedback("OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.42", skill="GRASP",
                                     default_message="Legacy", mode="proprioceptive_v2")
    assert fb_v2 == {"code": "OUT_OF_CAPABILITY", "detail_code": "OUT_OF_REACH",
                     "message": "Target point is out of reach."}
    fb_ok = F.format_public_feedback("EXECUTED", None, skill="REPORT_DONE",
                                     default_message="Action executed.", mode="proprioceptive_v2")
    assert fb_ok == {"code": "EXECUTED", "detail_code": None, "message": "Action executed."}


def test_task_funnel_and_rser(tmp_path):
    _write_episode(tmp_path, "ep_repeats", [
        ("OPEN", "INVALID_ACTION", "CLICK_NON_ARTICULATED"),
        ("OPEN", "INVALID_ACTION", "CLICK_NON_ARTICULATED"),       # consecutive repeat #1
        ("MOVE_TO", "UNSAFE", "NAV_NO_CLEAR_ROUTE"),
        ("MOVE_TO", "UNSAFE", "NAV_SEGMENT_CLEARANCE_BLOCKED"),    # same detail_code NAV_BLOCKED -> repeat #2
        ("GRASP", "OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.2"),
        ("REPORT_DONE", "EXECUTED", None),
    ], {"status": "FAIL_FALSE_COMPLETION", "attr": "PREMATURE_DONE", "res": 1.0})
    r = M.aggregate(M.load_episode(d) for d in M.find_episodes(tmp_path))
    assert r["task_funnel"]["stage2_articulation"]["errors"] == 2
    assert r["task_funnel"]["stage2_articulation"]["detail_codes"] == {"NO_ARTICULATION_HIT": 2}
    assert r["task_funnel"]["stage1_navigation"]["detail_codes"] == {"NAV_BLOCKED": 2}
    assert r["rser"]["repeat_errors"] == 2 and r["rser"]["refused_calls"] == 5
    assert r["core"]["RSER"] == 0.4
    assert r["rser"]["by_detail_code"] == {"NO_ARTICULATION_HIT": 1, "NAV_BLOCKED": 1}


def test_inter_step_agent_latency(tmp_path):
    _write_episode(tmp_path, "ep_timed", [
        ("VIEW", "EXECUTED", None),
        ("MOVE_TO", "EXECUTED", None),
        ("GRASP", "OUT_OF_CAPABILITY", "GRASP_TOO_FAR:dist=1.1"),
    ], {"status": "FAIL_MAX_STEPS", "attr": "TARGET_NEVER_ENGAGED", "res": 1.0})
    ep_root = tmp_path / "ep_timed"
    # Sibling mcp_private_events.jsonl next to worker_trace/
    events = [
        {"event": "reset_episode", "timestamp": 1000.0, "load_wall_s": 12.0},
        # Step 1 submitted at 1004.0, sim took 0.5s -> returned at 1004.5 => think = 4.0s
        {"event": "act", "planning_step": 1, "raw_action": {"skill": "VIEW"}, "step_wall_s": 0.5, "timestamp": 1004.5},
        # Intermediate observe took 0.3s sim
        {"event": "observe", "obs_wall_s": 0.3, "timestamp": 1006.8},
        # Step 2 submitted at 1010.8, sim took 0.7s -> returned at 1011.5 => think = (1010.8 - 1004.5 - 0.3) = 6.0s
        {"event": "act", "planning_step": 2, "raw_action": {"skill": "MOVE_TO"}, "step_wall_s": 0.7, "timestamp": 1011.5},
        # Step 3 submitted at 1019.5, sim took 1.0s -> returned at 1020.5 => think = 8.0s
        {"event": "act", "planning_step": 3, "raw_action": {"skill": "GRASP"}, "step_wall_s": 1.0, "timestamp": 1020.5},
    ]
    (ep_root / "mcp_private_events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    r = M.aggregate(M.load_episode(d) for d in M.find_episodes(tmp_path))
    assert r["core"]["LAT"] == 6.0
    assert r["latency"]["agent_think_mean_s"] == 6.0
    assert r["latency"]["agent_think_p50_s"] == 6.0
    assert r["latency"]["agent_think_total_mean_per_ep_s"] == 18.0
    assert r["latency"]["n_steps_timed"] == 3
    assert "LAT ↓ (s/step)" in M.to_markdown({"m": r})


