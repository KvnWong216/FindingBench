"""Evaluator-only household physical checks; simulation imports stay lazy.

The continuous sweep certificate uses actual collision bodies and a rigid
single-joint motion bound. It never treats an AABB overlap test, endpoint-only
check, or exhausting the subdivision budget as a collision-free proof.
"""
from __future__ import annotations

from copy import deepcopy
import math
import random

import numpy as np

from rummagebench.feasibility.ik_solver import Pose, quat_rotate, quat_multiply

SWEEP_VERSION = "rigid-single-joint-clearance-v1"
SUPPORTED_SWEEP_COLLIDERS = {("physics_collider_box","none"),("physics_collider_sphere","none"),
    ("physics_collider_cylinder","none"),("physics_convex_hull","convexHull"),("physics_mesh_bvh","none")}


def continuous_clearance(distance, motion_bound: float, *, margin=.005, max_queries=1025) -> dict:
    """Certify [0,1] by distance's supplied Lipschitz/Hausdorff motion bound.

    The caller must establish the bound for the complete moving geometry.
    Midpoint clearance > margin + bound * interval_halfwidth covers the
    entire interval. Otherwise subdivide finitely; no proof stays pending.
    """
    if (not math.isfinite(motion_bound) or motion_bound < 0 or not math.isfinite(margin)
            or margin <= 0 or not 3 <= max_queries <= 10001):
        raise ValueError("INVALID_SWEEP_CERTIFICATE_PARAMETERS")
    measurements, covered = [], []
    queue = [(0., 1.)]
    for endpoint in (0., 1.):
        value = float(distance(endpoint))
        if not math.isfinite(value) or value < 0:
            raise ValueError("SWEEP_DISTANCE_MISSING_OR_INVALID")
        measurements.append([endpoint, value])
        if value <= margin:
            return {"status": "failed", "reason": "SWEEP_CLEARANCE_VIOLATION", "samples": measurements,
                    "clearance_margin_m": margin, "motion_bound_m": motion_bound, "covered_intervals": []}
    while queue and len(measurements) < max_queries:
        lo, hi = queue.pop()
        mid = (lo + hi) / 2
        if mid==lo or mid==hi:
            queue.append((lo,hi))
            return {"status":"pending","reason":"SWEEP_NUMERIC_RESOLUTION_EXHAUSTED",
                    "samples":measurements,"clearance_margin_m":margin,"motion_bound_m":motion_bound,
                    "covered_intervals":sorted(covered),"unresolved_intervals":sorted(queue)}
        value = float(distance(mid))
        if not math.isfinite(value) or value < 0:
            raise ValueError("SWEEP_DISTANCE_MISSING_OR_INVALID")
        measurements.append([mid, value])
        if value <= margin:
            return {"status": "failed", "reason": "SWEEP_CLEARANCE_VIOLATION", "samples": measurements,
                    "clearance_margin_m": margin, "motion_bound_m": motion_bound, "covered_intervals": covered}
        if value > margin + motion_bound * (hi - lo) / 2:
            covered.append([lo, hi])
        else:
            queue.extend([(mid, hi), (lo, mid)])
    return {"status": "passed" if not queue else "pending",
            "reason": None if not queue else "SWEEP_SUBDIVISION_BUDGET_EXHAUSTED",
            "samples": measurements, "clearance_margin_m": margin, "motion_bound_m": motion_bound,
            "covered_intervals": sorted(covered), "unresolved_intervals": sorted(queue)}


def rigid_joint_pose(pose: Pose, kind: str, travel: float, pivot, axis, fraction: float) -> Pose:
    if kind not in ("revolute", "prismatic") or not 0 <= fraction <= 1:
        raise ValueError("UNSUPPORTED_SWEEP_MOTION")
    axis, pivot = np.asarray(axis, float), np.asarray(pivot, float)
    if (axis.shape != (3,) or pivot.shape != (3,) or not np.isfinite(axis).all()
            or not np.isfinite(pivot).all() or not math.isfinite(travel) or np.linalg.norm(axis) < 1e-8):
        raise ValueError("INVALID_SWEEP_JOINT_FRAME")
    axis = axis / np.linalg.norm(axis)
    if kind == "prismatic":
        return Pose(pose.position + axis * travel * fraction, pose.orientation)
    angle = travel * fraction
    rotation = np.r_[axis * math.sin(angle / 2), math.cos(angle / 2)]
    return Pose(pivot + quat_rotate(rotation, pose.position - pivot), quat_multiply(rotation, pose.orientation))


def _body_key(body, index):
    return (body.entity, body.link, index)


def placement_stability(poses, linear_velocity, angular_velocity) -> dict:
    values = np.asarray(poses,float)
    linear, angular = np.asarray(linear_velocity,float),np.asarray(angular_velocity,float)
    if (values.ndim!=2 or values.shape[1]!=7 or len(values)<2 or linear.shape!=(3,) or angular.shape!=(3,)
            or not np.isfinite(values).all() or not np.isfinite(linear).all() or not np.isfinite(angular).all()
            or not np.allclose(np.linalg.norm(values[:,3:],axis=1),1.,atol=1e-3)):
        raise ValueError("PLACEMENT_STABILITY_REAL_POSES_AND_VELOCITIES_REQUIRED")
    drift = float(np.linalg.norm(values[:,:3]-values[0,:3],axis=1).max())
    dots = np.clip(np.abs(values[:,3:] @ values[0,3:]),0.,1.)
    rotation = float(np.degrees(2*np.arccos(dots)).max())
    speed, spin = float(np.linalg.norm(linear)),float(np.linalg.norm(angular))
    return {"passed": drift<=.005 and rotation<=2. and speed<=.01 and spin<=.02,
            "position_drift_m":drift,"rotation_drift_deg":rotation,"linear_speed_m_s":speed,"angular_speed_rad_s":spin,
            "limits":{"position_m":.005,"rotation_deg":2.,"linear_m_s":.01,"angular_rad_s":.02}}


def verify_door_sweeps(backend, manifest) -> dict:
    """GPU only: all required containers, actual OPEN endpoints, bounded proof.

    Supported path: one changed revolute/prismatic joint, rigid descendant
    links, stationary foreign background. Multi-joint/carrying cases remain
    pending and cannot release this controlled direct-container pilot.
    """
    from pxr import UsdGeom, UsdPhysics
    from rummagebench.sim.omnigibson.robot_export import collect_articulation
    from rummagebench.sim.omnigibson.usd_collision import collect_entity_collision, physics_link_matrix
    from rummagebench.feasibility.fcl_compat import collision_backend, transform3
    from rummagebench.authoring.tasks.native_occupancy import plain
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object

    coal = collision_backend()
    snapshot = deepcopy(backend.capture_observe_state())
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    reports = {}
    originals = {obj.name: obj for obj in backend._env.scene.objects}
    robot_names = backend.robot_entity_names()
    expected_foreign = set(originals) - {manifest["target_asset"]["internal_entity"]} - robot_names
    def complete_collision(obj):
        from rummagebench.sim.omnigibson.usd_collision import iter_prims
        for body in [obj,*list((obj.links or {}).values())]:
            position,orientation=plain(body.get_position_orientation())
            if (np.asarray(position).shape!=(3,) or np.asarray(orientation).shape!=(4,)
                    or not np.isfinite(position).all() or not np.isfinite(orientation).all()
                    or not np.isclose(np.linalg.norm(orientation),1.,atol=1e-3)):
                raise ValueError(f"SWEEP_LIVE_BODY_POSE_INVALID:{obj.name}")
        expected = {str(p.GetPath()) for p in iter_prims(obj.prim)
                    if p.HasAPI(UsdPhysics.CollisionAPI)
                    and p.GetAttribute("physics:collisionEnabled").Get() is not False}
        bodies = collect_entity_collision(obj, backend._geom_cache)
        paths = [b.source_prim for b in bodies]
        if not expected or set(paths) != expected or len(paths) != len(set(paths)):
            raise ValueError(f"SWEEP_COLLIDER_EXPORT_INCOMPLETE:{obj.name}")
        if any((b.approximation,b.physics_approximation) not in SUPPORTED_SWEEP_COLLIDERS for b in bodies):
            raise ValueError(f"SWEEP_COLLIDER_FIDELITY_UNSUPPORTED:{obj.name}")
        if any(b.aabb is None or not np.isfinite(np.asarray(b.aabb)).all() for b in bodies):
            raise ValueError(f"SWEEP_COLLIDER_WORLD_GEOMETRY_UNAVAILABLE:{obj.name}")
        return bodies
    try:
        for candidate in manifest["candidate_bindings"]:
            entity = candidate["name"]
            try:
                backend.restore_observe_state(deepcopy(snapshot))
                obj = resolve_object(backend._env.scene, entity)
                if (getattr(obj,"metadata",None) or {}).get("openable_both_sides"):
                    raise ValueError("SWEEP_BIDIRECTIONAL_OPEN_REQUIRES_BOTH_LEGAL_PATHS")
                if backend.is_open(entity):
                    raise ValueError("SWEEP_REQUIRES_FROZEN_CLOSED_CONTAINER")
                art = collect_articulation(obj.prim)
                before = {name: float(plain(j.get_state()[0])[0]) for name, j in obj.joints.items()}
                closed = complete_collision(obj)
                foreign = []
                # This collector is intentionally NOT backend's permissive
                # loop, which logs and skips a failed object export.
                for name in sorted(expected_foreign - {entity} | robot_names):
                    bodies = complete_collision(originals[name])
                    if not bodies:
                        raise ValueError(f"SWEEP_FOREIGN_COLLIDERS_UNAVAILABLE:{name}")
                    foreign.extend(bodies)
                foreign_before = {name: plain(originals[name].get_position_orientation())
                                  for name in expected_foreign - {entity} | robot_names}
                if not backend.set_open(entity, True) or not backend.is_open(entity):
                    raise ValueError("SWEEP_OPEN_ENDPOINT_NOT_REALIZED")
                after = {name: float(plain(j.get_state()[0])[0]) for name, j in obj.joints.items()}
                changed = [name for name in before if abs(after[name] - before[name]) > 1e-5]
                if len(changed) != 1:
                    raise ValueError("SWEEP_SINGLE_CHANGED_JOINT_REQUIRED")
                joint = next(j for j in art["joints"] if j["name"] == changed[0])
                if joint["type"] not in ("revolute", "prismatic") or joint["limits"] is None:
                    raise ValueError("SWEEP_FINITE_JOINT_LIMITS_REQUIRED")
                if (not all(math.isfinite(v) for v in (*joint["limits"], before[changed[0]], after[changed[0]]))
                        or min(before[changed[0]], after[changed[0]]) < joint["limits"][0] - 1e-4
                        or max(before[changed[0]], after[changed[0]]) > joint["limits"][1] + 1e-4):
                    raise ValueError("SWEEP_JOINT_ENDPOINT_OUTSIDE_LIMITS")
                subtree = {joint["child"]}
                while True:
                    children = {j["child"] for j in art["joints"] if j["parent"] in subtree}
                    if children <= subtree: break
                    subtree |= children
                if any(j["name"] != changed[0] and j["type"] != "fixed" and j["parent"] in subtree for j in art["joints"]):
                    raise ValueError("SWEEP_RIGID_DESCENDANTS_REQUIRED")
                moving = [(i, b) for i, b in enumerate(closed) if b.link in subtree]
                if not moving or any(b.geometry is None or b.aabb is None for _, b in moving):
                    raise ValueError("SWEEP_MOVING_GEOMETRY_INCOMPLETE")
                # Actual parent link transform retains asset scale while
                # taking its rigid pose from physics, as the shared exporter.
                parent = obj.links[joint["parent"]]
                matrix = np.asarray(UsdGeom.Xformable(parent.prim).ComputeLocalToWorldTransform(0.), float)
                matrix = physics_link_matrix(obj, joint["parent"], matrix)
                pivot = (np.r_[joint["localPos0"], 1.] @ matrix)[:3]
                local_axis = quat_rotate(np.asarray(joint["localRot0"]), np.asarray(joint["axis"]))
                axis = np.r_[local_axis, 0.] @ matrix
                axis = axis[:3] / np.linalg.norm(axis[:3])
                travel = after[changed[0]] - before[changed[0]]
                opened = complete_collision(obj)
                if len(opened) != len(closed) or [_body_key(b,i) for i,b in enumerate(opened)] != [_body_key(b,i) for i,b in enumerate(closed)]:
                    raise ValueError("SWEEP_COLLIDER_TOPOLOGY_CHANGED")
                # Validate analytic kinematics against REAL open body poses;
                # unsupported scale/frame conventions cannot pass the bound.
                for i, body in moving:
                    predicted = rigid_joint_pose(body.pose, joint["type"], travel, pivot, axis, 1.)
                    actual = opened[i].pose
                    if (np.linalg.norm(predicted.position - actual.position) > 1e-4
                            or 1 - abs(float(np.dot(predicted.orientation, actual.orientation))) > 1e-6):
                        raise ValueError("SWEEP_ANALYTIC_ENDPOINT_MISMATCH")
                for name, poses in foreign_before.items():
                    actual = plain(originals[name].get_position_orientation())
                    if not np.allclose(np.asarray(poses[0]), actual[0], atol=1e-5, rtol=0):
                        raise ValueError("SWEEP_FOREIGN_BACKGROUND_MOVED")
                    if 1 - abs(float(np.dot(poses[1], actual[1]))) > 1e-6:
                        raise ValueError("SWEEP_FOREIGN_BACKGROUND_ROTATED")
                radii = [max(np.linalg.norm(np.array([x,y,z]) - pivot)
                             for x in (b.aabb[0][0],b.aabb[1][0]) for y in (b.aabb[0][1],b.aabb[1][1])
                             for z in (b.aabb[0][2],b.aabb[1][2])) for _,b in moving]
                bound = abs(travel) * (max(radii) if joint["type"] == "revolute" else 1.)
                def collision_object(body, pose=None):
                    p = pose or body.pose
                    # Quaternion-to-rotation is shared geometry, no URDF/IK
                    # model or approximate AABB geometry replaces the bodies.
                    rotation = np.column_stack([quat_rotate(p.orientation, np.eye(3)[i]) for i in range(3)])
                    return coal.CollisionObject(body.geometry, transform3(coal,p.position,rotation))
                stationary = [collision_object(b) for b in foreign]
                def distance(fraction):
                    value = math.inf
                    for _, b in moving:
                        current = collision_object(b, rigid_joint_pose(b.pose,joint["type"],travel,pivot,axis,fraction))
                        for other in stationary:
                            collision = coal.CollisionResult()
                            coal.collide(current,other,coal.CollisionRequest(),collision)
                            if collision.isCollision(): return 0.
                            result = coal.DistanceResult()
                            value = min(value,float(coal.distance(current,other,coal.DistanceRequest(),result)))
                    return value
                reports[entity] = {"joint": changed[0], "joint_type": joint["type"], "travel": travel,
                    "moving_links": sorted(subtree), "foreign_entities": sorted(expected_foreign - {entity} | robot_names),
                    "collider_fidelity": sorted({b.approximation for b in closed + foreign}),
                    **continuous_clearance(distance,bound)}
            except (ValueError, KeyError, StopIteration) as exc:
                reports[entity] = {"status": "pending", "reason": str(exc)}
    finally:
        try:
            backend.restore_observe_state(snapshot)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
    statuses = [r["status"] for r in reports.values()]
    return {"version": SWEEP_VERSION, "gpu_executed": True,
            "status": "failed" if "failed" in statuses else "passed" if statuses and set(statuses)=={"passed"} else "pending",
            "containers": reports, "restored": True,
            "scope": "single_joint_foreign_background_and_robot_clearance; excludes same-cabinet hinge/frame contacts and task target"}


def verify_temporary_place(backend, scenario, plan, slots, overlay, room_map, manifest) -> dict:
    """GPU only: actual public GRASP and PLACE, stable top-surface relation.

    Bounded geometry proposals only choose where to try. Every accepted route
    and interaction is executed through the existing visual public session.
    This feasibility witness is separate from the reference search trace.
    """
    from rummagebench.authoring.tasks.certify import TaskCertifier, EpisodeCertification, BASE_HALF_EXTENT, click_point, target_ring
    from rummagebench.core.visual_session import VisualProtocolSession
    from rummagebench.evaluation.goal_checker import check_placement_relation
    from rummagebench.sim.omnigibson.entity_resolver import resolve_object
    from omnigibson.object_states import OnTop

    snapshot = deepcopy(backend.capture_observe_state())
    result = {"version": 1, "gpu_executed": True, "status": "pending", "trace": [], "attempts": [],
              "scope": "public_target_grasp_then_temporary_top_surface_place_feasibility; not reference search cost"}
    certifier = TaskCertifier(slots, overlay, room_map)
    out = EpisodeCertification()
    vs = VisualProtocolSession(backend,scenario,base_half_extent=BASE_HALF_EXTENT)
    def step(action):
        res = vs.step(action)
        result["trace"].append({"action":action,"feedback":res.feedback.code.value,
                                "status":res.episode_status,"step":res.planning_step})
        return res.feedback.code.value
    def click(skill, entity):
        frame = vs._store.get(vs._store.current_id)
        point = click_point(frame,backend,entity)
        if point is None: return False
        return step({"skill":skill,"point":{"frame_id":frame.frame_id,"x":point[0],"y":point[1]}})=="EXECUTED"
    try:
        vs.reset()
        slot = slots.by_id()[plan.target.slot_id]
        if not certifier._drive(vs,backend,scenario,plan,slot.parent_entity,step,out):
            raise ValueError("TEMPORARY_PLACE_TARGET_ROUTE_FAILED")
        if not click("OPEN",slot.parent_entity):
            raise ValueError("TEMPORARY_PLACE_OPEN_FAILED")
        view = certifier._find_view_pose(backend,vs._session,scenario,plan,slot,scenario.target.entity,out)
        if view is None:
            raise ValueError("TEMPORARY_PLACE_GRASP_VIEW_UNAVAILABLE")
        if view != "here" and not certifier._drive_pose(vs,backend,plan,view,"temporary_grasp",step,out):
            raise ValueError("TEMPORARY_PLACE_GRASP_ROUTE_FAILED")
        if not click("GRASP",scenario.target.entity):
            raise ValueError("TEMPORARY_PLACE_GRASP_FAILED")
        surfaces = sorted({item["name"] for item in manifest["objects"]
            if {"Semantics.Furniture","Semantics.KitchenCounter","Semantics.Table"} & set(item.get("source_tags",[]))})
        # All tagged surfaces are merely proposals. Missing tags or an
        # unsupported support region cannot be relabelled usable space.
        surfaces = [name for name in surfaces if (region:=backend.receptacle_region(name)) is not None and region.kind=="top_surface"]
        if len(surfaces)>32:
            result["candidate_surfaces"] = surfaces
            raise ValueError("TEMPORARY_PLACE_SURFACE_PROPOSAL_BUDGET_EXCEEDED")
        for name in surfaces:
            box = backend.entity_aabb(name)
            if box is None: continue
            center = (np.asarray(box[0])+np.asarray(box[1]))/2
            # Up to six target-blind approach poses per actual support,
            # filtered/routed with the production checker in _drive_pose.
            candidates = ["here",*target_ring(center[:2],distances=(.8,1.1),bearings_deg=(0,120,240))]
            for pose in candidates:
                if vs.status() != "RUNNING": break
                if pose != "here" and not certifier._drive_pose(vs,backend,plan,pose,"temporary_support",step,out):
                    continue
                frame = vs._store.get(vs._store.current_id)
                if click_point(frame,backend,name) is None: continue
                feasible = certifier._feasible(vs._session,"PLACE",name)
                result["attempts"].append({"surface":name,"pose":pose,"feasible":feasible})
                if not feasible or not click("PLACE",name): continue
                # PLACE's production relation verifier establishes release
                # and support-region consistency. Add actual contact and a
                # finite stability window instead of relying on area budget.
                obj, support = resolve_object(backend._env.scene,scenario.target.entity),resolve_object(backend._env.scene,name)
                from rummagebench.authoring.tasks.native_occupancy import plain
                def pose():
                    position, orientation = plain(obj.get_position_orientation())
                    return [*position,*orientation]
                positions = [pose()]
                contact_samples = []
                contact = obj.states.get(OnTop)
                for _ in range(6):
                    backend.settle(5)
                    positions.append(pose())
                    contact_samples.append(contact is not None and bool(contact.get_value(support)))
                relation = check_placement_relation(backend,scenario.target.entity,name,
                                                   backend_held=backend.is_holding(scenario.target.entity))
                stability = placement_stability(positions,plain(obj.get_linear_velocity()),plain(obj.get_angular_velocity()))
                result.update(receptacle=name,relation=relation.to_dict(),physical_on_top_samples=contact_samples,
                              stability=stability,stability_steps=30,stability_poses=positions,
                              final_linear_velocity=plain(obj.get_linear_velocity()),final_angular_velocity=plain(obj.get_angular_velocity()))
                if relation.fault is None and relation.satisfied and relation.kind=="top_surface" and all(contact_samples) and stability["passed"]:
                    result["status"]="passed"
                    return result
                result.update(status="failed",reason="TEMPORARY_PLACE_RELATION_CONTACT_OR_STABILITY_FAILED")
                return result
        result.update(status="failed",reason="NO_PUBLICLY_REACHABLE_VERIFIED_TEMPORARY_PLACE")
    except ValueError as exc:
        result.update(status="failed",reason=str(exc))
    finally:
        # The temporary witness can end with an assisted grasp after a
        # failed PLACE. Reset releases runtime grasp constraints before the
        # original frozen tensor is restored, matching backend.reset's
        # existing lifecycle rather than leaving a dangling grasp joint.
        backend.reset()
        backend.restore_observe_state(snapshot)
        result["restored"] = True
        result["route_details"] = out.details
    return result


def validate_sweep_evidence(report, required_entities) -> None:
    if (report.get("version")!=SWEEP_VERSION or report.get("gpu_executed") is not True
            or report.get("status")!="passed" or report.get("restored") is not True
            or set(report.get("containers",{}))!=set(required_entities)):
        raise ValueError("COMPLETE_REAL_DOOR_SWEEP_EVIDENCE_REQUIRED")
    for name, proof in report["containers"].items():
        margin, bound = proof["clearance_margin_m"],proof["motion_bound_m"]
        if (proof.get("status")!="passed" or proof.get("reason") is not None or margin<.005
                or not math.isfinite(bound) or bound<0 or not math.isfinite(margin)
                or proof.get("unresolved_intervals")):
            raise ValueError(f"DOOR_SWEEP_CONTINUOUS_PROOF_REQUIRED:{name}")
        samples = dict(proof["samples"])
        intervals = sorted(proof["covered_intervals"])
        if (len(samples)!=len(proof["samples"]) or samples.get(0.,0)<=margin or samples.get(1.,0)<=margin
                or any(not 0<=t<=1 or not math.isfinite(d) or d<=margin for t,d in samples.items())):
            raise ValueError("DOOR_SWEEP_CLEARANCE_SAMPLES_INVALID")
        end = 0.
        for lo,hi in intervals:
            midpoint = (lo+hi)/2
            if lo!=end or not lo<hi<=1. or samples.get(midpoint,0)<=margin+bound*(hi-lo)/2:
                raise ValueError("DOOR_SWEEP_INTERVAL_GAP_OR_UNPROVED_CLEARANCE")
            end=hi
        if end!=1.:
            raise ValueError("DOOR_SWEEP_INCOMPLETE_CONTINUOUS_COVERAGE")


def validate_temporary_place_evidence(report, allowed_surfaces, horizon) -> None:
    from rummagebench.authoring.tasks.household_search import count_public_trace
    trace = report.get("trace",[])
    metrics = count_public_trace(trace)
    if (report.get("version")!=1 or report.get("gpu_executed") is not True or report.get("status")!="passed"
            or report.get("restored") is not True or report.get("receptacle") not in allowed_surfaces
            or not trace or trace[-1]["action"]["skill"]!="PLACE" or trace[-1]["feedback"]!="EXECUTED"
            or not any(r["action"]["skill"]=="GRASP" and r["feedback"]=="EXECUTED" for r in trace)
            or metrics["steps"]>horizon or report.get("physical_on_top_samples")!=[True]*6
            or report.get("stability_steps")!=30 or len(report.get("stability_poses",[]))!=7):
        raise ValueError("REAL_PUBLIC_TEMPORARY_PLACE_WITNESS_REQUIRED")
    relation = report.get("relation",{})
    if relation.get("kind")!="top_surface" or relation.get("satisfied") is not True or relation.get("fault") is not None:
        raise ValueError("TEMPORARY_PLACE_RELEASE_AND_SUPPORT_RELATION_REQUIRED")
    stability = placement_stability(report["stability_poses"],report["final_linear_velocity"],report["final_angular_velocity"])
    if not stability["passed"] or stability!=report.get("stability"):
        raise ValueError("TEMPORARY_PLACE_STABILITY_MEASUREMENT_INVALID")
