"""vla_difficulty_v2 scene survey (GPU, one scene load). Evaluator-only.

Loads a BEHAVIOR scene with its clearance manifest and a probe object, then
dumps everything the location certifier needs to pick candidates on CPU:

  * every non-robot entity: category, model, fixed_base, openable, room,
    world AABB, support region (top surface) and native drawer interiors
  * the MOVE obstacle set exactly as FootprintChecker builds it (with the
    widened doorways), so reachable floor can be flood-filled on CPU

    python scripts/v2/survey_scene.py --scene Rs_int --start -0.365 0.275 -1.81 \
        --probe-on countertop_tpuwys_0

Writes runs/vla_difficulty_v2/_survey/<scene>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "assets/tasks/vla_difficulty_v1/common.yaml"
# copies of the teammate's build/behavior/clearance manifests (+ declared v2 removals)
CLEARANCE = ROOT / "build/vla_difficulty_v2/clearance"


def scene_block(scene: str) -> dict:
    """Scene spec; the clearance manifest only for the four widened layouts
    (other BEHAVIOR scenes are used unmodified)."""
    block = {"model": scene, "source": "behavior"}
    if (CLEARANCE / f"{scene}.json").exists():
        block["clearance_manifest"] = str(CLEARANCE / f"{scene}.json")
    return block


def survey_scenario(scene: str, start: list[float], probe_on: str) -> dict:
    """Minimal ScenarioSpec: v1 robot/camera block, clearance on, one probe mug."""
    common = yaml.safe_load(COMMON.read_text())
    x, y, yaw = start
    return {
        "id": f"v2_survey_{scene}",
        "instruction": "survey",
        "scene": scene_block(scene),
        "robot": {**common["robot"], "init_anchor": "survey_start"},
        "anchors": {"survey_start": {"position": [x, y, 0.0053],
                                     "orientation": [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]}},
        "target": {"entity": "probe_mug", "category": "mug"},
        "objects": [{"name": "probe_mug", "category": "mug", "model": "dhnxww"}],
        "placements": [{"entity": "probe_mug", "relation": "on_top", "receptacle": probe_on}],
        "termination": dict(common["termination"]),
        "feasibility": dict(common["feasibility"]),
        "skills": list(common["skills"]),
        "expose_available_skills": False,
        "goal": [{"kind": "holding", "entity": "probe_mug"}],
    }


def _box(b):
    return None if b is None else [[round(float(v), 3) for v in b[0]], [round(float(v), 3) for v in b[1]]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--start", nargs=3, type=float, required=True, metavar=("X", "Y", "YAW"))
    p.add_argument("--probe-on", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "runs/vla_difficulty_v2/_survey")
    p.add_argument('--unplaced-probe', action='store_true', help='Park the authoring probe outside the scene instead of requiring a valid support surface')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scen_path = ROOT / f"build/vla_difficulty_v2/_survey/{a.scene}.yaml"
    scen_path.parent.mkdir(parents=True, exist_ok=True)
    spec=survey_scenario(a.scene, a.start, a.probe_on)
    if a.unplaced_probe:
        spec['placements']=[]
        spec['objects'][0]['fixed_base']=True
    scen_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    out_path = a.out / f"{a.scene}.json"
    rep: dict = {"scene": a.scene, "scenario": str(scen_path), "start": a.start, "entities": {}}
    try:
        from rummagebench.authoring.tasks.navigation import FootprintChecker
        from rummagebench.authoring.tasks.room_map import RoomMap
        from rummagebench.environment.env import InteractiveSearchEnv

        if a.unplaced_probe:
            from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
            seed_pose=OmniGibsonBackend._seed_pose_for
            OmniGibsonBackend._seed_pose_for=lambda self,obj: ([1000.,1000.,10.] if obj.name=='probe_mug' else seed_pose(self,obj))
        env = InteractiveSearchEnv(scen_path, mode="agent")
        env.reset()
        b = env._backend
        root = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets"
        rm = RoomMap.load(root / "scenes" / a.scene, root / "metadata" / "room_categories.txt")
        rep["robot_pose_after_reset"] = b.robot_pose()
        robots = b.robot_entity_names()
        for name in b.entity_names():
            if name in robots or (a.unplaced_probe and name=='probe_mug'):
                continue
            info = b.describe_entity(name)
            if info is None:
                continue
            box = b.entity_aabb(name)
            row = {"category": info.category, "model": info.model, "fixed_base": info.fixed_base,
                   "openable": info.openable, "graspable": info.graspable,
                   "in_rooms": list(info.in_rooms), "aabb": _box(box)}
            if box is not None:
                row["room_at_center"] = rm.room_at((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
            if name != 'probe_mug':
                try:
                    reg = b.receptacle_region(name)
                    row["support"] = None if reg is None else {"kind": reg.kind, "lo": reg.lo, "hi": reg.hi}
                    row["interiors"] = [{"link": n, "lo": r.lo, "hi": r.hi}
                                        for n, r in b.receptacle_interior_regions(name)]
                    art = b.articulation_info(name)
                    row["articulated"] = art is not None and bool(art.joints)
                except Exception as e:  # noqa: BLE001
                    row["support_error"] = f"{type(e).__name__}: {e}"
            rep["entities"][name] = row
        chk = FootprintChecker.from_backend(b, 0.40, 0.03)
        # Membership is metadata, not a containment edge. Preserve native
        # binary predicates separately so nested objects do not get flattened.
        from omnigibson.object_states import Inside, OnTop
        objects = {o.name: o for o in b._env.scene.objects if o.name in rep['entities']}
        rep['relation_schema_version'] = 1
        rep['excluded_authoring_entities'] = ['probe_mug']
        rep['relations'] = []
        rep['relation_errors'] = []
        rep['relation_query_audit'] = {
            'objects': len([n for n in objects if n != 'probe_mug']),
            'attempted': 0, 'true': 0, 'false': 0, 'errors': 0,
            'unsupported_states': {
                n: [kind for state,kind in [(Inside,'inside'),(OnTop,'on_top')] if state not in obj.states]
                for n,obj in objects.items() if n != 'probe_mug'},
            'scope': 'All ordered native object pairs for each available Inside/OnTop state; authoring probe excluded',
        }
        for child, obj in objects.items():
            if child == 'probe_mug':
                continue
            for parent, other in objects.items():
                if parent in (child, 'probe_mug'):
                    continue
                for state, kind in [(Inside, 'inside'), (OnTop, 'on_top')]:
                    if state not in obj.states:
                        continue
                    rep['relation_query_audit']['attempted'] += 1
                    try:
                        if bool(obj.states[state].get_value(other)):
                            rep['relation_query_audit']['true'] += 1
                            rep['relations'].append(dict(parent=parent, child=child,
                                relation=kind, source='omnigibson_state'))
                        else:
                            rep['relation_query_audit']['false'] += 1
                    except Exception as exc:
                        rep['relation_query_audit']['errors'] += 1
                        rep['relation_errors'].append(dict(parent=parent, child=child,
                            relation=kind, error=str(exc)))
        rep['relations_complete'] = not rep['relation_errors']
        rep["footprint_boxes"] = [[*map(float, lo), *map(float, hi)] for lo, hi in zip(chk.lo, chk.hi)]
        rep["rooms"] = sorted(set(rm.names.values()))
        rep["ok"] = True
    except BaseException as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
    temporary = out_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    temporary.replace(out_path)
    print(json.dumps({"ok": rep.get("ok"), "error": rep.get("error"), "out": str(out_path)}), flush=True)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
