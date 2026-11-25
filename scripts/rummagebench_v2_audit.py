#!/usr/bin/env python3
"""Comprehensive 1,000-Episode Structural, Diversity & Domain-Randomization Audit for RummageBench v2."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path("/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace")


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/rummagebench_v2_20261124"
    cases_dir = run_dir / "cases"
    ep_files = sorted(cases_dir.glob("*/episode.json"))
    assert len(ep_files) == 1000, f"Expected 1000 episode.json files, found {len(ep_files)}"

    arch_c = Counter()
    layout_c = Counter()
    cfam_c = Counter()
    fix_c = Counter()
    para_c = Counter()
    light_c = Counter()
    pal_c = Counter()
    prior_c = Counter()
    visual_combos = set()
    obj_counts = []
    sm_vis_counts = []
    hh_vis_counts = []
    missing_scenes = 0

    for f in ep_files:
        d = json.loads(f.read_text())
        if not Path(d["scene"]).exists():
            missing_scenes += 1
        arch_c[d["scene_archetype"]] += 1
        layout_c[d["layout_name"]] += 1
        cfam_c[d["container_family"]] += 1
        fix_c[d["search_fixture"]] += 1
        para_c[d["cognitive_paradigm"]] += 1
        light_c[d["lighting_profile"]] += 1
        pal_c[d["material_palette"]] += 1
        prior_c[d["spatial_prior_mode"]] += 1
        visual_combos.add((d["scene_archetype"], d["layout_name"], d["lighting_profile"], d["material_palette"]))
        obj_counts.append(d["total_objects"])
        vtot = d.get("total_visual_objects", d["total_objects"])
        if d["scene_archetype"].startswith("S1_"):
            sm_vis_counts.append(vtot)
        else:
            hh_vis_counts.append(vtot)

    max_cfam = max(cfam_c.values()) / len(ep_files)
    cupboard_counts = {k: v for k, v in fix_c.items() if k.startswith("cupboard_") or k == "range_oven_cavity"}
    max_cupboard = max(cupboard_counts.values()) / len(ep_files) if cupboard_counts else 0.0

    report = {
        "total_episodes": len(ep_files),
        "missing_scene_files": missing_scenes,
        "unique_base_layouts": len(layout_c),
        "unique_scene_archetype_layout_light_palette_combos": len(visual_combos),
        "min_objects": min(obj_counts),
        "max_objects": max(obj_counts),
        "mean_objects": round(sum(obj_counts) / len(obj_counts), 2),
        "supermarket_visual_objects": {
            "episodes": len(sm_vis_counts),
            "min": min(sm_vis_counts),
            "mean": round(sum(sm_vis_counts) / len(sm_vis_counts), 2),
            "max": max(sm_vis_counts),
        },
        "household_visual_objects": {
            "episodes": len(hh_vis_counts),
            "min": min(hh_vis_counts),
            "mean": round(sum(hh_vis_counts) / len(hh_vis_counts), 2),
            "max": max(hh_vis_counts),
        },
        "max_container_family_share": round(max_cfam, 4),
        "max_single_kitchen_cupboard_share": round(max_cupboard, 4),
        "kitchen_cupboard_and_oven_breakdown": dict(sorted(cupboard_counts.items())),
        "scene_archetype_counts": dict(sorted(arch_c.items())),
        "container_family_counts": dict(sorted(cfam_c.items())),
        "search_fixture_counts": dict(fix_c.most_common()),
        "cognitive_paradigm_counts": dict(sorted(para_c.items())),
        "lighting_profile_counts": dict(sorted(light_c.items())),
        "material_palette_counts": dict(sorted(pal_c.items())),
        "spatial_prior_counts": dict(sorted(prior_c.items())),
    }
    assert missing_scenes == 0, "Missing scene.usda files!"
    assert max_cfam <= 0.15, f"Container family exceeded 15%: {max_cfam}"
    assert max_cupboard <= 0.025, f"Single kitchen cupboard exceeded 2.5%: {cupboard_counts}"

    out_p = run_dir / "full_1000_diversity_audit.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
