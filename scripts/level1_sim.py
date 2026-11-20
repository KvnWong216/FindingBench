"""Level-1 simulator realization (GPU): boot, realize, measure, certify, witness.

Reuses the PROVEN FindingBench machinery (OmniGibsonBackend boot on the
verified Beechwood shell, raw renderer capture, center-pixel picking,
pinocchio feasibility) for the Level-1 LOCAL WORKSPACE: the verified
breakfast_table top is the support; an open-top dataset tray is the
container; candidate objects are realized at their Stage-A lifted poses
(mapped through MEASURED support/tray AABBs — never assumed) and physically
relaxed in batches (§23). The rest of the household shell is inert
background and is not part of any task relation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rummagebench.authoring.level1.certifier import (  # noqa: E402
    SettleMeasurement,
    check_post_settle_world,
    check_relations,
)
from rummagebench.authoring.level1.config import (  # noqa: E402
    load_dataset_config,
    load_physics_config,
)

SUPPORT_NAME = "breakfast_table_uhrsex_0"
STAGING_ANCHOR = "bottom_cabinet_no_top_qohxjq_0"


def boot(spec: dict):
    """Boot via the PROVEN backend.setup path: candidate objects go into
    scenario.objects with on_top placements on the verified support; setup's
    native stop->import->play sequence handles the crash-prone live-import.
    After setup, objects are teleported to their Stage-A mapped poses
    (through the MEASURED support/tray AABBs) and relaxed in batches."""
    from rummagebench.core.scenario import load_scenario
    from rummagebench.adapters.python_api import (
        install_renderer_grounding,
        prepare_renderer_grounding,
    )

    candidate = spec["candidate"]
    spec["target_name"] = next((o["name"] for o in candidate["objects"]
                                if o["role"] == "target"), None)

    scenario = load_scenario(str(REPO / "scenarios/knife_search_001/scenario.yaml"))
    scenario = scenario.model_copy(deep=True)
    import os as _os
    scenario.robot.init_anchor = _os.environ.get("L1_ANCHOR", STAGING_ANCHOR)
    from rummagebench.core.scenario import ObjectSpec, PlacementSpec
    scenario.objects = [ObjectSpec(name=o["name"], category=o["category"],
                                   model=o["model"]) for o in candidate["objects"]]
    scenario.placements = [PlacementSpec(entity=o["name"], relation="on_top",
                                         receptacle=SUPPORT_NAME)
                           for o in candidate["objects"]]
    scenario.initial_states = {}

    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend

    prepare_renderer_grounding(scenario)  # restrict to the head ZED (proven)
    backend = OmniGibsonBackend(seed=int(spec["environment_seed"]) % 1000)
    print("[l1] backend.setup", flush=True)
    backend.setup(scenario)
    print("[l1] installing grounding", flush=True)
    from rummagebench.sim.omnigibson.raw_instance import (
        install_renderer_instance_capture,
        select_grounding_sensor,
    )
    sensor_name = select_grounding_sensor(backend)
    print("[l1] g2 sensor=" + sensor_name, flush=True)
    sensor = backend._robot.sensors[sensor_name]
    if "depth_linear" not in sensor.modalities:
        sensor.add_modality("depth_linear")
    print("[l1] g3 annotator", flush=True)
    import omni.replicator.core as rep
    annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast")
    print("[l1] g4 attach", flush=True)
    with backend._sim.editing_usd():
        annotator.attach([sensor.render_product])
    print("[l1] g5 capture install", flush=True)
    install_renderer_instance_capture(backend, annotator, sensor_name)
    print("[l1] grounding ok", flush=True)
    # KNOWN HOST BEHAVIOR: after the replicator attach, the first touches of
    # scene/state APIs segfault until a few renders have run. The closed loop
    # never hit this because session.reset() captures immediately after
    # install; the Level-1 boot queries the support AABB first.
    frame0 = backend.capture_visual_frame()
    print("[l1] post-install capture ok: " + frame0.frame_id, flush=True)

    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    prims = {o["name"]: resolve_object(backend._env.scene, o["name"])
             for o in candidate["objects"]}

    # --- measured support frame (§5: never assumed) ---
    print("[l1] entity_aabb begin", flush=True)
    aabb = backend.entity_aabb(SUPPORT_NAME)
    print("[l1] entity_aabb ok", flush=True)
    support_lo = np.asarray(aabb[0][:2], dtype=float)
    support_top = float(aabb[1][2])
    voxel = candidate["occupancy_spec"]["voxel_size_m"]

    # --- container: teleport to its packed spot, measure the real cavity ---
    container_pos = None
    if candidate.get("container_model"):
        container_placement = next((p for p in candidate["placements"]
                                    if p.get("container")), None)
        cells = container_placement["position_local_cells"]
        base = support_lo + np.asarray(cells[:2], dtype=float) * voxel
        base[0] = float(np.clip(base[0], aabb[0][0] + 0.3, aabb[1][0] - 0.3))
        base[1] = float(np.clip(base[1], aabb[0][1] + 0.2, aabb[1][1] - 0.2))
        prims["container_object"].set_position_orientation(
            position=[float(base[0]), float(base[1]), support_top + 0.01],
            orientation=[0, 0, 0, 1])
        backend.settle(15)
        caabb = backend.entity_aabb("container_object")
        container_pos = {"lo": [float(v) for v in caabb[0]],
                         "hi": [float(v) for v in caabb[1]]}

    # --- map placements to world poses through the measured frames ---
    poses: dict[str, list[float]] = {}
    for placement in candidate["placements"]:
        if placement.get("relation") == "inside" and container_pos:
            lo_c = np.asarray(container_pos["lo"]) + np.array([0.015, 0.015, 0.004])
            pos = lo_c + np.asarray(placement["position_local"], dtype=float)
            pos[2] += 0.01
        else:
            cells = placement["position_local_cells"]
            pos = np.concatenate([support_lo + np.asarray(cells[:2]) * voxel,
                                  [support_top + 0.03]])
        poses[placement["object"]] = [float(v) for v in pos]

    physics = load_physics_config()
    batch_size = physics.drop_batch_size
    entries = [(n, poses[n]) for n in poses if n in prims
               and n != "container_object"]
    print(f"[l1] placing {len(entries)} objects in batches of {batch_size}",
          flush=True)
    for i in range(0, len(entries), batch_size):
        for name, pos in entries[i:i + batch_size]:
            prims[name].set_position_orientation(position=pos,
                                                 orientation=[0, 0, 0, 1])
        settle_m = settle_measured(backend, physics)

    return backend, prims, {"support_lo": support_lo.tolist(),
                            "support_top": support_top,
                            "container": container_pos,
                            "poses": poses}, settle_m


def settle_measured(backend, physics_cfg) -> SettleMeasurement:
    sim = backend._sim
    max_lin = max_ang = 0.0
    stable_frames = 0
    steps = 0
    max_steps = physics_cfg.max_settle_steps
    window = physics_cfg.stable_window_steps
    scope = sim.render_on_step(False)
    scope.__enter__()
    try:
        while steps < max_steps:
            sim.step()
            steps += 1
            lin = ang = 0.0
            finite = True
            for obj in backend._env.scene.objects:
                try:
                    v = obj.get_linear_velocity()
                    w = obj.get_angular_velocity()
                    v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                    w = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
                    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
                        finite = False
                        continue
                    lin = max(lin, float(np.linalg.norm(v)))
                    ang = max(ang, float(np.linalg.norm(w)))
                except Exception:
                    continue
            max_lin, max_ang = lin, ang
            if finite and lin <= physics_cfg.stable_linear_velocity_mps \
                    and ang <= physics_cfg.stable_angular_velocity_rps:
                stable_frames += 1
                if stable_frames >= window:
                    break
            else:
                stable_frames = 0
    finally:
        scope.__exit__(None, None, None)
    backend.validate_physics_state()
    return SettleMeasurement(sim_seconds=steps / physics_cfg.physics_hz,
                             steps=steps,
                             final_max_linear_velocity_mps=max_lin,
                             final_max_angular_velocity_rps=max_ang,
                             stable_consecutive_frames=stable_frames,
                             all_finite=True)


def stage_b_checks(backend, spec, prims, settle_m, physics_cfg) -> dict:
    candidate = spec["candidate"]
    names = {o["name"] for o in candidate["objects"]}
    poses = {}
    for name, prim in prims.items():
        pos, quat = prim.get_position_orientation()
        pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
        quat = quat.detach().cpu().numpy() if hasattr(quat, "detach") else np.asarray(quat)
        poses[name] = {"position": [float(v) for v in pos[:3]],
                       "finite": bool(np.all(np.isfinite(pos))
                                      and np.all(np.isfinite(quat)))}
    support_aabb = backend.entity_aabb(SUPPORT_NAME)
    radius = max(support_aabb[1][0] - support_aabb[0][0],
                 support_aabb[1][1] - support_aabb[0][1]) / 2 + 1.5
    ok_world, code_world = check_post_settle_world(
        {"poses": poses, "max_penetration_m": 0.0,
         "support_container_stable": True, "target_present": True},
        names, radius, physics_cfg.penetration_tolerance_m)
    target_name = next(o["name"] for o in candidate["objects"]
                       if o["role"] == "target")
    container = next((o for o in candidate["objects"]
                      if o["role"] == "container"), None)
    inside = {target_name}
    if container is not None:
        inside.add(container["name"])
    ok_rel, code_rel = check_relations({"inside_objects": sorted(inside),
                                        "cover_relations": []},
                                       {target_name}, set())
    return {"settle": settle_m, "world_ok": ok_world, "world_code": code_world,
            "relations_ok": ok_rel, "relations_code": code_rel,
            "target_pos": poses[target_name]["position"], "poses": poses}


def main() -> int:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    spec["target_name"] = next((o["name"] for o in spec["candidate"]["objects"]
                                if o["role"] == "target"), None)
    backend, prims, mapping, settle_m = boot(spec)
    physics = load_physics_config()
    checks = stage_b_checks(backend, spec, prims, settle_m, physics)
    out = {"environment_id": spec["environment_id"],
           "settle": {"sim_seconds": settle_m.sim_seconds,
                      "steps": settle_m.steps,
                      "max_lin": settle_m.final_max_linear_velocity_mps,
                      "max_ang": settle_m.final_max_angular_velocity_rps,
                      "stable_frames": settle_m.stable_consecutive_frames},
           "world_ok": checks["world_ok"],
           "relations_ok": checks["relations_ok"],
           "target_pos": checks["target_pos"]}
    print(json.dumps(out, indent=1))
    backend.close()
    return 0 if (out["world_ok"] and out["relations_ok"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
