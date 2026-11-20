"""Compile vla_difficulty_v2 base tasks into standard scenario YAMLs.

    assets/tasks/vla_difficulty_v2/common.yaml          robot / camera / termination
    assets/tasks/vla_difficulty_v2/base_tasks/<id>.yaml one per base task (generate_tasks.py)

Every base task x L1-L3 x S1-S3 -> build/vla_difficulty_v2/<id>__<L>_<S>/scenario.yaml
(+ variant.json with the L/S metadata, never shown to the agent) and
build/vla_difficulty_v2/manifest.json.

  instruction = language.L1[S] | language.L2 | language.L3
  placements  = spatial[S].placement + background placements   (identical across L)
  anchors     = the task's start + evaluator-private interaction hints
                (<entity>__GRASP_k / <receptacle>__PLACE_k: certified base poses)

The three L variants of one (task, S) must be identical except for the
instruction: checked here, so scripts/v2/run_batch.py certifies one of them.

    python scripts/v2/compile_variants.py [--only B001 B002]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

from rummagebench.core.scenario import load_scenario

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_scene import scene_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "assets/tasks/vla_difficulty_v2"
OUT = ROOT / "build/vla_difficulty_v2"
LEVELS_L = ("L1", "L2", "L3")
LEVELS_S = ("S1", "S2", "S3")


def _quat(yaw: float) -> list[float]:
    return [0.0, 0.0, round(math.sin(yaw / 2), 5), round(math.cos(yaw / 2), 5)]


def instruction_for(task: dict, level_l: str, level_s: str) -> str:
    text = task["language"][level_l]
    if isinstance(text, dict):
        text = text[level_s]
    return " ".join(str(text).split())


def compose(common: dict, task: dict, level_l: str, level_s: str) -> tuple[str, dict, dict]:
    vid = f"{task['task_id']}__{level_l}_{level_s}"
    sp = task["spatial"][level_s]
    tgt = task["target"]
    start = task["start"]
    anchors = {"start": {"position": [start["x"], start["y"], 0.0053], "orientation": _quat(start["yaw"])}}
    for skill, entity, poses in ((("GRASP", tgt["entity"], sp.get("grasp_hints", []))),
                                 *((("PLACE", g["receptacle"], task.get("place_hints", []))
                                    for g in task["goal"] if g["kind"] in ("on_top", "inside")))):
        for k, (x, y, yaw) in enumerate(poses[:4]):
            anchors[f"{entity}__{skill}_{k}"] = {"position": [x, y, 0.0053], "orientation": _quat(yaw)}
    scenario = {
        "id": vid,
        "instruction": instruction_for(task, level_l, level_s),
        "scene": {**scene_block(task["scene"]),
                  **({"clearance_manifest": task["clearance_manifest"]}
                     if task.get("clearance_manifest") else {})},
        "robot": {**common["robot"], "init_anchor": "start"},
        "anchors": anchors,
        "target": {"entity": tgt["entity"], "category": tgt["category"]},
        "objects": [{"name": tgt["entity"], "category": tgt["category"], "model": tgt["model"]},
                    *[{k: o[k] for k in ("name", "category", "model")} for o in task.get("background_objects", [])]],
        "placements": [dict(sp["placement"]), *[dict(p) for p in task.get("background_placements", [])]],
        "initial_states": dict(task.get("initial_states", {})),
        "termination": dict(common["termination"]),
        "feasibility": dict(common["feasibility"]),
        "skills": list(common["skills"]),
        "expose_available_skills": False,
        "goal": [dict(g) for g in task["goal"]],
    }
    meta = {"variant_id": vid, "task_id": task["task_id"], "family": task["family"], "scene": task["scene"],
            "language_level": level_l, "spatial_level": level_s, "start": start,
            "target": tgt, "target_room": sp.get("room"), "location_phrase": sp.get("location_phrase"),
            "task_scope": sp.get("task_scope", "unspecified"), "target_placement": sp["placement"], "goal": task["goal"],
            "extra_ops_vs_S1": sp.get("extra_ops_vs_S1", []), "s2_kind": sp.get("s2_kind")}
    return vid, scenario, meta


def physical_hash(scenario: dict) -> str:
    body = {k: v for k, v in scenario.items() if k not in ("id", "instruction")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="base task ids (default: all)")
    p.add_argument("--no-validate", action="store_true", help="skip load_scenario (faster)")
    a = p.parse_args()
    common = yaml.safe_load((DESIGN / "common.yaml").read_text())
    manifest = {"design": str(DESIGN.relative_to(ROOT)), "variants": []}
    for path in sorted((DESIGN / "base_tasks").glob("*.yaml")):
        task = yaml.safe_load(path.read_text())
        if a.only and task["task_id"] not in a.only:
            continue
        for level_s in LEVELS_S:
            if level_s not in task["spatial"]:
                continue
            hashes = set()
            for level_l in LEVELS_L:
                text=task['language'].get(level_l)
                if text is None or (isinstance(text,dict) and level_s not in text):
                    continue
                vid, scenario, meta = compose(common, task, level_l, level_s)
                hashes.add(physical_hash(scenario))
                vdir = OUT / vid
                vdir.mkdir(parents=True, exist_ok=True)
                spath = vdir / "scenario.yaml"
                spath.write_text(f"# GENERATED by scripts/v2/compile_variants.py from {task['task_id']} "
                                 f"x {level_l} x {level_s}; do not edit.\n"
                                 + yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True))
                if not a.no_validate:
                    load_scenario(spath)
                meta["physical_config"] = f"{task['task_id']}__{level_s}"
                (vdir / "variant.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
                manifest["variants"].append({**meta, "scenario": str(spath.relative_to(ROOT))})
            if len(hashes) != 1:
                raise SystemExit(f"{task['task_id']} {level_s}: L variants differ beyond the instruction")
    if a.only and (OUT / 'manifest.json').exists():
        previous=json.loads((OUT / 'manifest.json').read_text())['variants']
        manifest['variants'] += [v for v in previous if v['task_id'] not in a.only]
        manifest['variants'].sort(key=lambda v:v['variant_id'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.tmp").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    (OUT / "manifest.tmp").replace(OUT / "manifest.json")
    print(f"compiled {len(manifest['variants'])} variants -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
