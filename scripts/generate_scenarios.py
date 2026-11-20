"""Batch scenario generator: one strict-schema YAML per target object.

Input: a batch spec (see scenarios/batch_kitchen_v1.yaml) listing scenarios
as compact data — target, receptacle, distractors, instruction. Each entry
is rendered from the kitchen template (the merged knife_search_001 layout:
854x480 camera, drawer-safe link placement, Beechwood anchors, scripted
ORACLE sequences) and validated through the strict scenario schema before
anything is written to disk.

    PYTHONPATH=src python scripts/generate_scenarios.py \
        --batch scenarios/batch_kitchen_v1.yaml [--dry-run]

Physical placement for a NEW receptacle/link still needs one GPU smoke run
before it may be used for evaluation; the batch spec marks those entries.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# verified models from the BEHAVIOR-1K object dataset (dataset-side lookup is
# avoided so generation stays CPU-only and deterministic)
DISTRACTOR_MODELS = {
    "bowl": "adciys",
    "teacup": "cpozxi",
    "tablespoon": "huudhe",
    "tablefork": "flexrc",
    "vegetable_peeler": "njgqao",
}

# counter/table props shared by every scenario (kept from knife_search_001)
COUNTER_ITEMS = [
    ("apple", "apple", "agveuv", "countertop_tpuwys_0"),
    ("cutting_board", "cutting_board", "aibvew", "countertop_tpuwys_0"),
    ("mug", "mug", "dhnxww", "countertop_tpuwys_0"),
    ("plate", "plate", "aewthq", "breakfast_table_uhrsex_0"),
]

# distractor cabinets, in scripted-opening order: first `split_at` categories
# land in cabinet A, the rest in drawer cabinet B
DISTRACTOR_CABINETS = ("bottom_cabinet_no_top_spojpj_0", 2,
                       "bottom_cabinet_rvpunw_0")

TARGET_DRAWER_LINK = "link_2"  # top drawer: proven reachable + visible


def build_scenario(entry: dict) -> dict:
    """Render one batch entry into a strict-schema scenario dict."""
    target_category = entry["target"]["category"]
    target_model = entry["target"].get("model")
    if not target_model:
        raise ValueError(f"{entry['id']}: target model must be explicit "
                         "(generation stays CPU-only, no dataset lookup)")
    entity = entry.get("entity") or "target_" + target_category
    receptacle = entry["receptacle"]
    link = entry.get("link", TARGET_DRAWER_LINK)
    distractors = [c for c in entry.get("distractors", [])
                   if c != target_category]

    doc = deepcopy(TEMPLATE)
    doc["id"] = entry["id"]
    doc["instruction"] = " ".join(entry["instruction"].split())

    doc["target"] = {"entity": entity, "category": target_category}
    objects = [{"name": entity, "category": target_category,
                "model": target_model}]
    placements = [{"entity": entity, "relation": "inside",
                   "receptacle": receptacle, "link": link}]
    for cat in distractors:
        name = "distractor_" + cat
        objects.append({"name": name, "category": cat,
                        "model": DISTRACTOR_MODELS[cat]})
    first_cab, split_at, second_cab = DISTRACTOR_CABINETS
    for i, cat in enumerate(distractors):
        placements.append({"entity": "distractor_" + cat, "relation": "inside",
                           "receptacle": first_cab if i < split_at else second_cab})
    for name, cat, model, rec in COUNTER_ITEMS:
        objects.append({"name": name, "category": cat, "model": model})
        placements.append({"entity": name, "relation": "on_top", "receptacle": rec})
    doc["objects"] = objects
    doc["placements"] = placements

    # scripted ORACLE sequences: open each involved cabinet exactly once,
    # in fixed order, then grasp the target
    cabinets = [first_cab, second_cab]
    if receptacle not in cabinets:
        cabinets.append(receptacle)
    success = [{"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}]
    for cab in cabinets:
        success.append({"skill": "NAV", "target": {"type": "place", "value": cab}})
        success.append({"skill": "OPEN", "target": {"type": "entity", "value": cab}})
    success.append({"skill": "GRASP",
                    "target": {"type": "entity", "value": entity}})
    doc["agent"]["scripted_success"] = success
    doc["agent"]["scripted_wrong_object"] = (
        success[:-1] + [{"skill": "GRASP",
                         "target": {"type": "entity",
                                    "value": "distractor_" + distractors[-1]}}]
        if distractors else [])
    doc["agent"]["unsafe_sequence"] = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}},
        {"skill": "GRASP", "target": {"type": "entity",
                                      "value": "countertop_tpuwys_0"}},
    ]
    return doc


TEMPLATE = {
    "scene": {"model": "Beechwood_0_int"},
    "robot": {
        "model": "r1pro",
        "obs_modalities": ["rgb", "depth_linear", "seg_instance"],
        "name": "robot_0",
        # 16:9 480p camera with a wider lens (~90 x 59 deg FOV); the OG default
        # 17 mm lens at 16:9 leaves only ~38 deg vertically.
        "image_width": 854,
        "image_height": 480,
        "focal_length_mm": 10.5,
        "init_anchor": "living_room",
        "kinematics": {
            "urdf_path": "build/robots/r1pro.urdf",
            "base_link": "base_link",
            "end_effector_link": "right_eef_link",
            "controlled_joints": "auto",
            "joint_limits_source": "urdf",
            "ik_seed": 0,
        },
        "reach_radius": 1.0,
        "z_min": 0.0,
        "z_max": 1.6,
        "hand_capacity": 1,
    },
    "feasibility": {"backend": "pinocchio", "mode": "endpoint"},
    "anchors": {
        "living_room": {
            "position": [5.35, -7.86, 0.0053],
            "orientation": [0.0, 0.0, 1.0, 0.0],
        },
        "kitchen": {
            "position": [0.75, -8.15, 0.0053],
            "orientation": [0.0, 0.0, 1.0, 0.0],
        },
        "bottom_cabinet_no_top_spojpj_0": {
            "position": [-2.99, -8.03, 0.0053],
            "orientation": [0.0, 0.0, 0.7071, 0.7071],
        },
        "bottom_cabinet_rvpunw_0": {
            "position": [-1.11, -6.20, 0.0053],
            "orientation": [0.0, 0.0, 0.7071, 0.7071],
        },
        "bottom_cabinet_no_top_qohxjq_0": {
            "position": [0.62, -6.15, 0.0053],
            "orientation": [0.0, 0.0, 0.7071, 0.7071],
        },
    },
    "initial_states": {
        "bottom_cabinet_no_top_spojpj_0": {"open": False},
        "bottom_cabinet_rvpunw_0": {"open": False},
        "bottom_cabinet_no_top_qohxjq_0": {"open": False},
    },
    "termination": {
        "max_planning_steps": 16,
        "fail_on_wrong_grasp": True,
        "fail_on_unsafe_action": True,
        "succeed_when_holding_target": True,
    },
    "skills": ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"],
    "safety": {"forbidden_categories": [], "grasping_fixed_base_unsafe": True},
    "agent": {},
}

HEADER = """# Generated by scripts/generate_scenarios.py — edit the batch spec, not
# this file. Physical placement for a NEW receptacle needs one GPU smoke run.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="scenarios/batch_kitchen_v1.yaml")
    parser.add_argument("--out-dir", default="scenarios")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = REPO / batch_path
    spec = yaml.safe_load(batch_path.read_text(encoding="utf-8"))
    entries = spec.get("scenarios") or []
    if not entries:
        print(f"no scenarios listed in {batch_path}")
        return 1

    # strict-schema validation imports the package
    sys.path.insert(0, str(REPO / "src"))
    from rummagebench.core.scenario import load_scenario  # noqa: E402

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = REPO / out_root

    written, failed = [], []
    for entry in entries:
        sid = entry["id"]
        try:
            doc = build_scenario(entry)
            payload = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
            target_dir = out_root / sid
            scenario_path = target_dir / "scenario.yaml"
            if not args.dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
                scenario_path.write_text(HEADER + payload, encoding="utf-8")
                load_scenario(str(scenario_path))  # strict schema re-check on disk
            else:
                from rummagebench.core.scenario import ScenarioSpec
                ScenarioSpec.model_validate(doc)  # validate in-memory dict
            written.append(sid)
            print(f"[ok] {sid}")
        except Exception as e:
            failed.append((sid, repr(e)))
            print(f"[FAIL] {sid}: {e}")

    print(f"\n{len(written)} written, {len(failed)} failed"
          + (" (dry-run)" if args.dry_run else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
