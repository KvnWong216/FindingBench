"""Real-simulation smoke for the PLACE placement-aware postcondition.

One Isaac Sim process per episode. Per trial, through the PRODUCTION
pipeline:

    session.reset -> [OPEN receptacle] -> GRASP target
    -> feasibility PLACE check selects the validated surface point
    -> backend.symbolic_place(held, receptacle, at=point)  (settle 10)
    -> MEASURE post-settle geometry (pose / AABB / region)
    -> restore snapshot -> PlaceSkill.execute -> commit-or-infra-fault

Recorded per trial: place_point, region kind/bounds, entity outer AABB,
post-settle object AABB bottom + xy centre, drifts vs the tolerances in
skills/place.py, and the production verdict. Results go to --out as JSON.

Usage:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=7 python \
        scripts/smoke_place_postcondition.py \
        --scenario assets/benchmark/easy_v1/scenarios/<id>.yaml --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef
from rummagebench.skills.place import _PLACE_POINT_XY_TOL, _PLACE_Z_TOL
from rummagebench.skills.place import PlaceSkill
from rummagebench.sim.base import ResolvedTarget
from rummagebench.sim.omnigibson.backend import OmniGibsonBackend


def _placement_of(scenario, entity):
    for p in scenario.placements:
        if p.entity == entity:
            return p
    return None


def run_episode(scenario_path: str, out_path: str) -> dict:
    scenario = load_scenario(scenario_path)
    target = scenario.target.entity
    placement = _placement_of(scenario, target)
    record: dict = {
        "episode": scenario.id,
        "scene": scenario.scene.model,
        "target": target,
        "receptacle": placement.receptacle if placement else None,
        "relation": placement.relation if placement else None,
        "tolerances": {"xy": _PLACE_POINT_XY_TOL, "z": _PLACE_Z_TOL},
    }
    if placement is None:
        record["error"] = "no placement for target"
        return record

    backend = OmniGibsonBackend(seed=0, validate_legacy_anchors=True)
    print("[smoke] setup begin", flush=True)
    backend.setup(scenario)
    session = BenchmarkSession(backend, scenario)
    session.reset()
    print("[smoke] setup done", flush=True)

    receptacle = placement.receptacle
    try:
        # certified plan: solve with the oracle planner (exactly what
        # certification does), then replay through the production session to
        # a held target — NAV/OPEN/GRASP sequences are scene-dependent
        from rummagebench.evaluation.oracle_planner import (
            OracleFeasibilityModel,
            initial_oracle_state,
            solve_full_information,
        )

        model = OracleFeasibilityModel(session._backend, scenario,
                                       session._feasibility)
        model.bind_robot(session.robot)
        plan = solve_full_information(scenario, initial_oracle_state(scenario),
                                      model)
        record["plan_solvable"] = plan.solvable
        print("[smoke] plan:", plan.solvable, plan.depth, plan.reason,
              flush=True)
        if not plan.solvable:
            record["error"] = f"oracle plan unsolvable: {plan.reason}"
            return record
        session.reset()  # leave the oracle's realized door states behind
        record["replay"] = []
        for act_dict in plan.actions:
            result = session.act(Action.from_dict(act_dict))
            record["replay"].append({
                "action": act_dict,
                "executed": result.executed,
                "failure_reason": result.failure_reason.value,
            })
        held = session.world_state.held_object
        record["held"] = held
        print("[smoke] replay done, held:", held, flush=True)
        if held != scenario.target.entity:
            record["error"] = (
                f"replay did not reach held target: {record['replay']}")
            return record

        # production point selection: try every receptacle the scenario
        # knows, in order (the target's own receptacle first); the PLACE
        # feasibility check picks the validated surface point exactly as
        # session._act does. Per-receptacle failures are recorded — on real
        # assets a fixed bottom cabinet's region is its root-link top, which
        # may be unreachable (e.g. under a countertop).
        from rummagebench.feasibility.interaction_target import (
            receptacle_place_targets)

        receptacles = [receptacle]
        for p in scenario.placements:
            if p.receptacle not in receptacles:
                receptacles.append(p.receptacle)
        record["receptacle_trials"] = []
        point = None
        chosen = None
        for rec in receptacles:
            trial = {"receptacle": rec}
            try:
                resolved = backend.resolve_entity(rec)
                info = resolved.info
                trial["info"] = {
                    "fixed_base": bool(getattr(info, "fixed_base", None)),
                    "openable": bool(getattr(info, "openable", None)),
                    "is_receptacle": bool(getattr(info, "is_receptacle", None)),
                    "category": getattr(info, "category", None),
                } if info is not None else None
                from rummagebench.object_interface.base import build_object_interface

                obj = build_object_interface(info, backend)
                trial["adapter"] = type(obj).__name__
                interface_targets = obj.interaction_targets(
                    "PLACE", session.world_state)
                trial["interface_targets"] = len(interface_targets or [])
                verdict = session._feasibility.check(
                    "PLACE", resolved, session.robot, session.world_state)
                trial["feasible"] = bool(verdict.feasible)
                trial["reason"] = verdict.reason
                trial["candidates_evaluated"] = (verdict.details or {}).get(
                    "candidates_evaluated")
                trial["evaluation_errors"] = (verdict.details or {}).get(
                    "evaluation_errors")
                if verdict.feasible:
                    point = (verdict.details or {}).get(
                        "interaction_target", {}).get("surface_point")
                    trial["point_source"] = "feasibility"
                else:
                    targets = receptacle_place_targets(rec, backend)
                    trial["region_targets"] = len(targets)
                    if targets:
                        point = list(targets[0].metadata["surface_point"])
                        trial["point_source"] = "fallback_targets[0]"
            except Exception as e:  # noqa: BLE001 - record, try next
                trial["error"] = f"{type(e).__name__}: {e}"
            record["receptacle_trials"].append(trial)
            print("[smoke] receptacle trial:", trial, flush=True)
            if point is not None:
                chosen = rec
                break
        if point is None:
            record["error"] = "no receptacle yielded a place point"
            return record
        receptacle = chosen
        record["receptacle_used"] = receptacle
        record["place_point"] = [round(float(v), 4) for v in point]
        record["point_source"] = next(
            t.get("point_source") for t in record["receptacle_trials"]
            if t["receptacle"] == receptacle)

        # realization under measurement: snapshot -> symbolic_place (settle
        # happens inside) -> measure BEFORE any rollback can destroy it
        snap = backend.capture_observe_state()
        realized = backend.symbolic_place(held, receptacle, at=point)
        record["symbolic_place_returned"] = bool(realized)

        pose = backend.entity_pose6d(held)
        aabb = backend.entity_aabb(held)
        region = backend.receptacle_region(receptacle)
        outer = backend.entity_aabb(receptacle)
        record["post_settle"] = {
            "origin": [round(float(v), 4) for v in pose.position],
            "aabb_lo": [round(float(v), 4) for v in aabb[0]],
            "aabb_hi": [round(float(v), 4) for v in aabb[1]],
        }
        if region is not None:
            record["region"] = {
                "kind": region.kind,
                "lo": [round(float(v), 4) for v in region.lo],
                "hi": [round(float(v), 4) for v in region.hi],
            }
        if outer is not None:
            record["entity_outer_aabb"] = {
                "lo": [round(float(v), 4) for v in outer[0]],
                "hi": [round(float(v), 4) for v in outer[1]],
            }
        if aabb is not None:
            cx = (float(aabb[0][0]) + float(aabb[1][0])) / 2.0
            cy = (float(aabb[0][1]) + float(aabb[1][1])) / 2.0
            xy_drift = math.hypot(cx - float(point[0]), cy - float(point[1]))
            bottom = float(aabb[0][2])
            z_vs_point = bottom - float(point[2])
            record["drift"] = {
                "xy_center_vs_point": round(xy_drift, 4),
                "bottom_vs_point_z": round(z_vs_point, 4),
                "xy_within_tol": xy_drift <= _PLACE_POINT_XY_TOL,
                "z_within_tol": abs(z_vs_point) <= _PLACE_Z_TOL + 1e-9,
            }
        if region is not None and outer is not None:
            record["region_vs_outer"] = {
                "support_plane": round(float(region.hi[2]), 4),
                "outer_top": round(float(outer[1][2]), 4),
                "root_link_plane_diff": round(
                    float(region.hi[2]) - float(outer[1][2]), 4),
                "interior_shrunk_xy": (
                    float(region.lo[0]) > float(outer[0][0]) + 1e-6
                    or float(region.lo[1]) > float(outer[0][1]) + 1e-6
                    or float(region.hi[0]) < float(outer[1][0]) - 1e-6
                    or float(region.hi[1]) < float(outer[1][1]) - 1e-6),
            }

        # production verdict on the restored (held) state
        backend.restore_observe_state(snap)
        state = session.world_state
        try:
            result = PlaceSkill().execute(
                backend,
                ResolvedTarget(kind=TargetKind.ENTITY, entity=receptacle,
                               info=backend.describe_entity(receptacle),
                               place_point=[float(v) for v in point]),
                state,
            )
            record["verdict"] = {
                "committed": bool(result.executed and result.postcondition_satisfied),
            }
        except FeasibilityBackendError as e:
            record["verdict"] = {"committed": False, "fault": str(e)}
        record["held_after_verdict"] = state.held_object
        return record
    except Exception as e:  # noqa: BLE001 - smoke must record, not die
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[-2000:]
    finally:
        # Isaac's app shutdown can hard-exit the process (os._exit) inside
        # backend.close(): the record MUST hit disk before that call
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_text(json.dumps(record, indent=2),
                                      encoding="utf-8")
        except Exception:
            traceback.print_exc()
        try:
            backend.close()
        except Exception:
            pass
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    record = run_episode(args.scenario, args.out)
    print("[smoke] result:", json.dumps(
        {k: record.get(k) for k in
         ("episode", "verdict", "drift", "error")}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
