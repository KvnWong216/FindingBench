"""Probe a scene with the robot on GPU and write its SlotEmbodimentOverlay
(assets/tasks/embodiment/<scene>__<robot>.yaml): interaction anchors, OPEN /
GRASP-region feasibility, reveal evidence and start poses. Raw measurements
go to <build_root>/probe/<scene>/.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/tasks/probe_embodiment.py \
        --scene Beechwood_0_int [--rooms kitchen_0]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import yaml
from _common import ASSETS, REPO, host_config, load_yaml

from rummagebench.authoring.tasks.embodiment_slots import (
    SlotEmbodimentOverlay, SlotInteraction, StartPose, overlay_path, sha256_file)
from rummagebench.authoring.tasks.slots import SceneSlots

DEFAULT_PROBE_OBJECT = "apple/agveuv"  # small rigid body, imports cleanly on this host
# settled R1Pro base height on a BEHAVIOR floor (top at z=0), measured on
# Beechwood_0_int (scenarios/knife_search_001 anchors)
NOMINAL_BASE_Z = 0.0053


def probe_scenario(ss: SceneSlots, rooms: list[str], room_map, robot_doc: dict,
                   probe_cat: str, probe_model: str) -> dict:
    """Minimal scenario: the scene, the robot at a free pose of the first
    probed room, one probe object resting on a surface, every openable slot
    furniture of the probed rooms closed."""
    from rummagebench.authoring.tasks.probe import pose_from_xyyaw

    # spawn at the clearest traversable point of the whole scene (farthest
    # from any obstacle on the layout map); the real footprint check picks
    # the probe's own start afterwards (see run_probe)
    # (inside the first probed room: the whole-scene optimum can sit on a
    # porch / level change where the settled robot drifts off its anchor)
    clearance, (x, y) = room_map.clearest_points(1, room=rooms[0])[0]
    start = pose_from_xyyaw(x, y, NOMINAL_BASE_Z, 0.0).to_anchor()

    # park the probe object on the FLOOR of the first probed room (OnTop a
    # floor never fails; a small table can, which aborts the whole setup)
    scene_doc = json.loads(Path(ss.source["scene_json"]).read_text(encoding="utf-8"))
    floors = sorted(n for n, v in scene_doc["objects_info"]["init_info"].items()
                    if v["args"].get("category") == "floors"
                    and rooms[0] in (v["args"].get("in_rooms") or []))
    surfaces = [s for s in ss.slots if s.room in rooms and s.relation == "on_top"]
    rec = floors[0] if floors else (surfaces[0].parent_entity if surfaces else None)
    placements = ([{"entity": "probe_object", "relation": "on_top", "receptacle": rec}]
                  if rec else [])
    furn = ss.furniture_by_entity()
    initial = {e: {"open": False} for e, f in sorted(furn.items())
               if f.openable and f.room in rooms}
    return {
        "id": f"probe_{ss.scene}",
        "instruction": "embodiment probe (evaluator-only)",
        "scene": {"model": ss.scene},
        "robot": {**robot_doc["robot"], "init_anchor": "start"},
        "feasibility": dict(robot_doc["feasibility"]),
        "anchors": {"start": start},
        "target": {"entity": "probe_object", "category": probe_cat},
        "objects": [{"name": "probe_object", "category": probe_cat, "model": probe_model}],
        "placements": placements,
        "initial_states": initial,
        "skills": ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"],
    }


def _reanchor_start(backend, scenario, room_map, rooms, log) -> None:
    from rummagebench.authoring.tasks.probe import pose_from_xyyaw
    from rummagebench.core.scenario import AnchorSpec
    from rummagebench.skills.move import base_pose_collision_free

    # candidates INSIDE the probed rooms only (the whole-scene optimum can be
    # outdoors: the robot then drops onto the ground plane at z ~ -0.77 and
    # every later teleport inherits that height); nominal settled base height
    z = NOMINAL_BASE_Z
    cands = []
    for room in rooms:
        cands += [p for _, p in room_map.clearest_points(10, room=room)]
        cands += room_map.free_points(room, 0.5, 0.25)
    for x, y in cands:
        if not base_pose_collision_free(backend, x, y, 0.0, 0.40, 0.03):
            continue
        pose = pose_from_xyyaw(x, y, z, 0.0)
        anchor = AnchorSpec(position=list(pose.position), orientation=list(pose.orientation))
        backend.teleport_robot(anchor)
        backend.settle(15)
        backend.validate_physics_state()
        settled = backend.robot_pose()[0]
        if abs(settled[2] - z) > 0.05 or abs(settled[0] - x) > 0.1 or abs(settled[1] - y) > 0.1:
            log(f"[probe] spawn candidate ({x:.2f}, {y:.2f}) unstable: settled at {settled}")
            continue
        scenario.anchors[scenario.robot.init_anchor] = anchor
        backend._validate_spawn_clearance()
        backend._initial_state = backend.dump_state()
        log(f"[probe] spawn re-anchored at ({x:.2f}, {y:.2f}, z={settled[2]:.4f})")
        return
    raise RuntimeError("no stable footprint-free spawn pose inside the probed rooms")


def run_probe(args) -> int:
    from rummagebench.authoring.tasks.plan import git_state
    from rummagebench.authoring.tasks.probe import (
        PROBE_VERSION, ProbeConfig, SceneProber, anchor_candidates, slot_interaction)
    from rummagebench.authoring.tasks.room_map import RoomMap

    host = host_config(args.host_config)
    scene = args.scene
    ss = SceneSlots.load(host["build_root"] / "scenes" / f"{scene}.yaml")
    room_map = RoomMap.load(host["behavior_assets_root"] / "scenes" / scene,
                            host["behavior_assets_root"] / "metadata" / "room_categories.txt")
    rooms = args.rooms or sorted({s.room for s in ss.slots})
    unknown = [r for r in rooms if r not in ss.rooms]
    if unknown:
        raise SystemExit(f"unknown rooms {unknown}; scene rooms: {sorted(ss.rooms)}")
    robot_doc = load_yaml(REPO / "configs" / "tasks" / "robots" / f"{args.robot}.yaml")
    cat, model = args.probe_object.split("/")
    out_dir = host["build_root"] / "probe" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    scen_doc = probe_scenario(ss, rooms, room_map, robot_doc, cat, model)
    scen_path = out_dir / "probe_scenario.yaml"
    scen_path.write_text(yaml.safe_dump(scen_doc, sort_keys=False), encoding="utf-8")
    log_f = (out_dir / "probe.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    os.chdir(REPO)  # URDF + config paths are repo-relative
    from rummagebench.adapters.python_api import (
        install_renderer_grounding, prepare_renderer_grounding)
    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    scenario = load_scenario(scen_path)
    prepare_renderer_grounding(scenario)
    backend = OmniGibsonBackend(seed=0)
    report: dict = {"scene": scene, "rooms": rooms, "probe_object": args.probe_object,
                    "probe_version": PROBE_VERSION, "furniture": {}}
    cfg = ProbeConfig()
    try:
        log(f"[probe] {scene} rooms={rooms} setup ...")
        # the CPU spawn is only a guess: skip the builder's spawn-footprint
        # gate during setup, then re-anchor at a footprint-free pose and
        # re-snapshot (reset() enforces the gate from then on)
        backend._validate_spawn_clearance = lambda: None
        backend.setup(scenario)
        del backend._validate_spawn_clearance
        _reanchor_start(backend, scenario, room_map, rooms, log)
        install_renderer_grounding(backend)
        session = BenchmarkSession(backend, scenario)
        prober = SceneProber(backend, session, ss, room_map, "probe_object", cfg, log,
                             views_dir=out_dir / "views")

        starts: dict[str, list[StartPose]] = {}
        backend.reset()
        for room in rooms:
            starts[room] = prober.probe_start_poses(room)

        by_parent: dict[str, list] = {}
        for s in ss.slots:
            if s.room in rooms:
                by_parent.setdefault(s.parent_entity, []).append(s)
        furn = ss.furniture_by_entity()
        entities = sorted(by_parent)
        slots: dict[str, SlotInteraction] = {}
        for i, ent in enumerate(entities):
            fp = prober.probe_furniture(furn[ent], by_parent[ent])
            report["furniture"][ent] = {
                "anchors_tested": fp.anchors_tested,
                "anchors_footprint_free": fp.anchors_footprint_free,
                "open_feasible_from": fp.open_feasible_from,
                "chosen_anchor": None if fp.chosen_anchor is None
                else fp.chosen_anchor.to_anchor(),
                "fillable_links": fp.fillable_links, "slots": fp.slots,
                "error": fp.error, "note": fp.note, "seconds": fp.seconds,
                "furniture_px_by_anchor": fp.furniture_px_by_anchor,
                "open_route_steps": fp.open_route_steps,
                "free_after_open_from": fp.free_after_open_from,
                "anchor_candidates": [list(c) for c in anchor_candidates(furn[ent], cfg)]}
            for s in by_parent[ent]:
                inter = slot_interaction(s, fp)
                if inter is not None:
                    slots[s.slot_id] = inter
            ok = sum(1 for s in by_parent[ent]
                     if s.slot_id in slots and slots[s.slot_id].usable(s.requires_open))
            log(f"[probe] ({i + 1}/{len(entities)}) {ent}: {ok}/{len(by_parent[ent])} slots "
                f"usable, free anchors {fp.anchors_footprint_free}/{fp.anchors_tested}, "
                f"open from {fp.open_feasible_from}, error={fp.error} note={fp.note} "
                f"[{fp.seconds}s]")
            (out_dir / "probe_report.json").write_text(json.dumps(report, indent=1))

        complete = all(v["error"] is None for v in report["furniture"].values())
        robot_cfg = {k: v for k, v in robot_doc["robot"].items()}
        urdf = robot_cfg.get("kinematics", {}).get("urdf_path", "")
        gs = git_state(REPO)
        ov = SlotEmbodimentOverlay(
            scene=scene, robot_id=args.robot, urdf_path=urdf,
            urdf_hash=sha256_file(REPO / urdf) if urdf else None,
            status="probed" if complete else "partial",
            source=f"probe:v{PROBE_VERSION}:{dt.date.today().isoformat()}:"
                   f"{(gs.get('git_commit') or 'nogit')[:12]}",
            robot_config=robot_cfg, feasibility=dict(robot_doc["feasibility"]),
            start_poses=starts, slots=slots,
            notes=[f"probed rooms: {rooms}", f"probe object: {args.probe_object}",
                   f"probe config: {json.dumps(cfg.to_dict(), sort_keys=True)}"]
                  + [f"{e}: {v.get('error') or v.get('note')}"
                     for e, v in sorted(report['furniture'].items())
                     if v.get("error") or v.get("note")])
        out = overlay_path(ASSETS, scene, args.robot)
        if out.exists():
            backup = out_dir / f"{out.stem}.previous.yaml"
            backup.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            log(f"[probe] previous overlay backed up to {backup}")
        ov.save(out, "# Robot embodiment overlay probed on GPU by\n"
                     "# scripts/tasks/probe_embodiment.py. Do not hand-edit.\n")
        n_ok = sum(1 for sid, v in slots.items()
                   if v.usable(ss.by_id()[sid].requires_open))
        log(f"[probe] wrote {out} status={ov.status}: {n_ok}/{len(slots)} slots usable, "
            f"start poses { {r: len(p) for r, p in starts.items()} }")
        code = 0
    except BaseException as e:
        import traceback

        log(f"[probe] FAILED {type(e).__name__}: {e}\n{traceback.format_exc()}")
        code = 1
    finally:
        log_f.close()
        backend.close()
        sys.stdout.flush()
        os._exit(code if "code" in locals() else 1)  # Kit teardown may segfault


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rooms", nargs="+")
    ap.add_argument("--robot", default="r1pro")
    ap.add_argument("--probe-object", default=DEFAULT_PROBE_OBJECT,
                    help="category/model of the probe object")
    ap.add_argument("--host-config")
    return run_probe(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
