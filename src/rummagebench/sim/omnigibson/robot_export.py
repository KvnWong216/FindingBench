"""OmniGibson robot articulation -> URDF kinematics export.

The robot's kinematic model is DERIVED from the simulator asset — link
frames, joint tree, joint axes/limits and collision-bound boxes are read
from the USD articulation; nothing about the robot is hand-authored.

Outputs (written next to the requested urdf_path):

    <name>.urdf                fixed-base kinematics: link frames, joints,
                               limits, <collision> boxes (link collider
                               AABBs, approximation levels in the manifest)
    <name>.manifest.json       export provenance + approximation levels
    <name>_validation.json     §2 random-configuration cross-validation

Cross-validation (both are MANDATORY gates — a failed gate REJECTS the
export and the benchmark must not run):

    rest pose      every link pose (position AND orientation) compared
                   between the simulator and Pinocchio FK
    §2 random q    N >= 50 random joint configurations, simulator EEF pose
                   vs Pinocchio EEF pose; tolerance 1 cm / 3 deg

Run on the simulator host (needs omnigibson + pin):

    python scripts/export_robot_kinematics.py --robot r1pro
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_JOINT_TYPE_BY_USD = {
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsFixedJoint": "fixed",
    "PhysicsSphericalJoint": "spherical",
    "PhysicsD6Joint": "d6",
    "PhysicsJoint": "undefined",
}

VALIDATION_DEFAULT_SAMPLES = 50
VALIDATION_TRANS_TOL_M = 0.01
VALIDATION_ROT_TOL_DEG = 3.0


# ---------------------------------------------------------------------------
# USD articulation collection (pxr.UsdPhysics — stable schema-level API)
# ---------------------------------------------------------------------------


def iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from iter_prims(child)


def collect_articulation(root_prim) -> dict:
    """Walk the USD articulation under ``root_prim``.

    Returns {root, links, joints, handle_link} where each joint carries
    parent/child link names, joint frame offsets (localPos0/1, localRot0/1),
    axis and limits (radians; USD revolute degrees converted).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    root_name = root_prim.GetName()

    links: list[str] = []
    for prim in iter_prims(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.append(prim.GetName())

    joints: list[dict] = []
    for prim in iter_prims(root_prim):
        jtype = _JOINT_TYPE_BY_USD.get(prim.GetTypeName())
        if jtype is None:
            continue
        # runtime assisted-grasp constraint joints weld HELD OBJECTS into the
        # robot articulation; they are simulation bookkeeping, not morphology
        if "ag_constraint" in prim.GetName().lower():
            continue
        body0 = _rel_target_name(prim.GetRelationship("physics:body0"))
        body1 = _rel_target_name(prim.GetRelationship("physics:body1"))
        if body0 is None or body1 is None:
            continue

        def _attr(name):
            attr = prim.GetAttribute(name)
            return attr.Get() if attr else None

        lp0, lr0 = _attr("physics:localPos0"), _attr("physics:localRot0")
        lp1, lr1 = _attr("physics:localPos1"), _attr("physics:localRot1")
        axis = _attr("physics:axis")
        limits = None
        if jtype in ("revolute", "prismatic"):
            lo, hi = _attr("physics:lowerLimit"), _attr("physics:upperLimit")
            if lo is not None and hi is not None:
                scale = math.pi / 180.0 if jtype == "revolute" else 1.0
                limits = (float(lo) * scale, float(hi) * scale)

        joints.append({
            "name": prim.GetName(),
            "type": jtype,
            "parent": body0,
            "child": body1,
            "axis": _axis_vector(axis),
            "limits": limits,
            "localPos0": _gf_vec(lp0, default=(0.0, 0.0, 0.0)),
            "localRot0": _gf_quat_to_xyzw(lr0) if lr0 is not None else [0.0, 0.0, 0.0, 1.0],
            "localPos1": _gf_vec(lp1, default=(0.0, 0.0, 0.0)),
            "localRot1": _gf_quat_to_xyzw(lr1) if lr1 is not None else [0.0, 0.0, 0.0, 1.0],
        })

    handle_link = next((l for l in links if "handle" in l.lower()), None)
    return {
        "root": root_name,
        "links": links,
        "joints": joints,
        "handle_link": handle_link,
    }


def _rel_target_name(rel) -> str | None:
    targets = rel.GetTargets() if rel else []
    if not targets:
        return None
    return targets[0].name


def _axis_vector(axis) -> list[float] | None:
    if axis is None:
        return None
    return {
        "X": [1.0, 0.0, 0.0],
        "Y": [0.0, 1.0, 0.0],
        "Z": [0.0, 0.0, 1.0],
    }.get(str(axis))


def _gf_vec(v, default) -> list[float]:
    if v is None:
        return list(default)
    return [float(c) for c in v]


def _gf_quat_to_xyzw(q) -> list[float]:
    return [float(q.imaginary[0]), float(q.imaginary[1]), float(q.imaginary[2]), float(q.real)]


# ---------------------------------------------------------------------------
# per-link collision boxes (USD colliders -> local AABB boxes)
# ---------------------------------------------------------------------------


def collect_link_collision_boxes(root_prim) -> dict[str, list[dict]]:
    """For every link: the world-aligned AABBs of its collider prims,
    expressed in the LINK frame (USD), as {center, half, approximation}.

    Approximation levels: 'physics_collider_box' for axis-aligned USD box
    colliders read exactly, 'aabb_primitive' for the world-aligned bound of
    everything else (meshes/capsules). Unbounded or degenerate bounds are
    skipped with a warning.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    link_boxes: dict[str, list[dict]] = {}
    for prim in iter_prims(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        link_name = prim.GetName()
        link_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_to_link = link_xform.GetInverse()
        boxes: list[dict] = []

        for sub in iter_prims(prim):
            if sub == prim:
                continue
            if not sub.HasAPI(UsdPhysics.CollisionAPI):
                continue
            approx = "aabb_primitive"
            if sub.GetTypeName() == "Cube":
                approx = "physics_collider_box"
            try:
                rng = cache.ComputeWorldBound(sub).ComputeAlignedRange()
                lo, hi = np.array(rng.min, dtype=float), np.array(rng.max, dtype=float)
            except Exception as e:  # pragma: no cover - depends on stage
                logger.warning("bbox failed for %s: %s", sub.GetPath(), e)
                continue
            half = (hi - lo) / 2.0
            # skip unbounded/degenerate bounds (UsdGeom.BBoxCache returns
            # +-FLT_MAX ranges for colliders it cannot bound)
            if np.any(half <= 0.0) or np.any(half > 5.0):
                logger.warning(
                    "skipping invalid collider bound for %s: half=%s",
                    sub.GetPath(), half,
                )
                continue
            center_world = (lo + hi) / 2.0
            center_link = world_to_link.Transform(
                Gf.Vec3d(*[float(v) for v in center_world])
            )
            boxes.append({
                "center": [float(c) for c in center_link],
                "half": [float(c) for c in half],
                "approximation": approx,
                "source": str(sub.GetPath()),
            })
        link_boxes[link_name] = boxes
    return link_boxes


# ---------------------------------------------------------------------------
# minimal SE(3) helpers on (pos xyz, quat xyzw) tuples
# ---------------------------------------------------------------------------


def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_rot(q, v):
    t = _quat_mul(_quat_mul(q, np.array([*v, 0.0])), _quat_conj(q))
    return t[:3]


def _inv_pose(pose):
    q = _quat_conj(pose[1])
    return (-_quat_rot(q, pose[0]), q)


def _pose_mul_point(pose, p):
    return pose[0] + _quat_rot(pose[1], p)


def _quat_to_rpy(q) -> str:
    """xyzw quaternion -> URDF rpy (extrinsic XYZ / roll-pitch-yaw)."""
    x, y, z, w = q / np.linalg.norm(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_p = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sin_p) if abs(sin_p) >= 1 else math.asin(sin_p)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return f"{roll:.8g} {pitch:.8g} {yaw:.8g}"


def _make_T(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw / np.linalg.norm(quat_xyzw)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def _make_T_inv(base_T: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    return np.linalg.inv(base_T) @ _make_T(pos, quat_xyzw)


# ---------------------------------------------------------------------------
# URDF writer (frames/joints derived from USD; boxes from collider bounds)
# ---------------------------------------------------------------------------


def reachable_links(art: dict, base_link: str) -> set[str]:
    """Links connected to the joint tree rooted at base_link."""
    reachable = {base_link}
    frontier = [base_link]
    while frontier:
        current = frontier.pop()
        for j in art["joints"]:
            if j["parent"] == current and j["child"] not in reachable:
                reachable.add(j["child"])
                frontier.append(j["child"])
    return reachable


def choose_base_link(art: dict, base_link: str | None = None) -> str:
    """The URDF root: a rigid body named *base_link* when present (dropping
    the USD-root -> base_link virtual floating-base chain), else a root body
    named *base*, else the first root body."""
    children = {j["child"] for j in art["joints"]}
    root_bodies = [l for l in art["links"] if l not in children]
    if not root_bodies:
        raise RuntimeError(
            "articulation has no root rigid body (every link is a joint child)"
        )
    if base_link is not None:
        if base_link not in art["links"]:
            raise RuntimeError(f"base_link {base_link!r} not among the articulation links")
        return base_link
    named = [l for l in art["links"] if l.lower() == "base_link"]
    if named:
        return named[0]
    base_named = [l for l in root_bodies if "base" in l.lower()]
    return base_named[0] if base_named else root_bodies[0]


def write_urdf(art: dict, link_boxes: dict, out_path: Path, base_link: str | None = None) -> dict:
    """Compose the URDF from the collected articulation.

    Frame convention (the one that makes URDF FK match the simulator):

    - the URDF child link frame COINCIDES with the USD child link frame, so
      link content (collision boxes) needs no re-expression;
    - the URDF joint origin is T_P->J * T_J->C = (localPos0,rot0) * inv(localPos1,rot1);
    - the URDF joint axis (expressed in the child frame) is the USD axis
      rotated by localRot1's rotation.

    (T_P->J = localPos0/rot0 is the joint frame in the parent link; T_C->J =
    localPos1/rot1 is the joint frame in the child link. The naive conversion
    — assuming the child frame IS the joint frame — leaves a constant
    localRot1 rotation error, which the §2 random-configuration validation
    catches as large angular offsets.)
    """
    import xml.etree.ElementTree as ET

    joints = art["joints"]
    base = choose_base_link(art, base_link)

    # links connected to the joint tree starting at the base (FIXED-BASE export)
    reachable = reachable_links(art, base)
    disconnected = [l for l in art["links"] if l not in reachable]

    root = ET.Element("robot", {"name": art["root"]})
    manifest: dict = {
        "root": base,
        "robot_prim": art["root"],
        "links": {},
        "joints": {},
        "handle_link": art["handle_link"],
        "disconnected_links_skipped": disconnected,
        # raw USD joint frames kept in the manifest for offline debugging
        "raw_joints": joints,
    }

    def add_link(name: str):
        link = ET.SubElement(root, "link", {"name": name})
        # minimal inertial (kinematics-only model; dynamics are not evaluated)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": "1.0"})
        ET.SubElement(
            inertial, "inertia",
            {"ixx": "1e-4", "iyy": "1e-4", "izz": "1e-4", "ixy": "0", "ixz": "0", "iyz": "0"},
        )
        boxes = link_boxes.get(name, [])
        approx_levels = []
        for box in boxes:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", {
                "xyz": " ".join(f"{v:.8g}" for v in box["center"]),
                "rpy": "0 0 0",
            })
            ET.SubElement(collision, "geometry").append(
                ET.Element("box", {"size": " ".join(f"{2*v:.8g}" for v in box["half"])})
            )
            approx_levels.append(box["approximation"])
        manifest["links"][name] = {
            "collision_boxes": len(boxes),
            "approximation": sorted(set(approx_levels)) or ["none"],
        }

    for name in sorted(reachable):
        add_link(name)

    for j in joints:
        if j["child"] not in reachable or j["parent"] not in reachable:
            continue  # belongs to a dropped/disconnected subtree (virtual base)
        # URDF joint frame == USD child link frame:
        #   origin = T_P->J * T_J->C = (localPos0,rot0) * inv(localPos1,rot1)
        #   axis   = R(localRot1) * axis_J  (axis re-expressed in child frame)
        inv_b = _inv_pose((np.asarray(j["localPos1"], dtype=float),
                           np.asarray(j["localRot1"], dtype=float)))
        origin_pos = _pose_mul_point(
            (np.asarray(j["localPos0"], dtype=float),
             np.asarray(j["localRot0"], dtype=float)),
            inv_b[0],
        )
        origin_quat = _quat_mul(np.asarray(j["localRot0"], dtype=float), inv_b[1])
        axis_child = None
        if j["axis"] is not None:
            axis_child = _quat_rot(
                np.asarray(j["localRot1"], dtype=float),
                np.asarray(j["axis"], dtype=float),
            )

        jtype = j["type"]
        joint_el = ET.SubElement(root, "joint", {
            "name": j["name"],
            "type": "fixed" if jtype == "fixed" else jtype,
        })
        ET.SubElement(joint_el, "parent", {"link": j["parent"]})
        ET.SubElement(joint_el, "child", {"link": j["child"]})
        ET.SubElement(joint_el, "origin", {
            "xyz": " ".join(f"{v:.8g}" for v in origin_pos),
            "rpy": _quat_to_rpy(origin_quat),
        })
        if jtype in ("revolute", "prismatic", "continuous"):
            axis = axis_child if axis_child is not None else [0.0, 0.0, 1.0]
            norm = np.linalg.norm(axis)
            if norm > 1e-9:
                axis = axis / norm
            ET.SubElement(joint_el, "axis", {"xyz": " ".join(f"{v:.8g}" for v in axis)})
            lo, hi = j["limits"] if j["limits"] else (0.0, 0.0)
            # urdfdom rejects +-inf: clamp unbounded joints to huge-but-finite
            # ranges; such joints are locked by controlled_joints anyway
            if not math.isfinite(lo):
                lo = -1e9
            if not math.isfinite(hi):
                hi = 1e9
            ET.SubElement(joint_el, "limit", {
                "lower": f"{lo:.8g}", "upper": f"{hi:.8g}",
                "effort": "1e9", "velocity": "1e9",
            })
        manifest["joints"][j["name"]] = {
            "type": jtype, "parent": j["parent"], "child": j["child"],
        }

    ET.indent(root, space="  ")
    xml = ET.ElementTree(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml.write(out_path, encoding="utf-8", xml_declaration=True)
    return manifest


# ---------------------------------------------------------------------------
# cross-validation gates
# ---------------------------------------------------------------------------


def cross_validate_rest_pose(omni_robot, urdf_path: Path, art: dict, base_link: str) -> dict:
    """Pinocchio FK at q=0 vs OmniGibson link poses (all via the SAME
    OmniGibson world-frame API, relative to the URDF root link). Positions
    AND orientations are checked."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    # FK must run at the robot's CURRENT joint configuration (OmniGibson
    # loads robots in a home pose that is generally NOT the zero
    # configuration); zero-q comparison produced phantom accumulated errors.
    sim_joints = getattr(omni_robot, "joints", {}) or {}
    q = np.zeros(model.nq)
    for name, jprim in sim_joints.items():
        if name not in model.names:
            continue
        jid = model.getJointId(name)
        if jid == 0 or model.joints[jid].nv == 0:
            continue
        try:
            value = float(jprim.get_state()[0])
        except Exception:
            continue
        q[model.joints[jid].idx_q] = value
    # freeze the robot at exactly the read configuration: after a reset the
    # position controller is still settling, so link poses would drift
    # between the joint read and the pose read
    try:
        import torch

        n_dof = int(omni_robot.n_dof)
        full = np.asarray(omni_robot.get_joint_positions(), dtype=float).copy()
        if len(full) == n_dof:
            for name, jprim in sim_joints.items():
                if name not in model.names:
                    continue
                jid = model.getJointId(name)
                if jid == 0 or model.joints[jid].nv == 0:
                    continue
                idx = getattr(jprim, "dof_indices", None)
                if idx is None:
                    continue
                idx = idx[0] if isinstance(idx, (list, tuple)) else idx
                full[int(idx)] = q[model.joints[jid].idx_q]
            omni_robot.set_joint_positions(
                torch.as_tensor(full, dtype=torch.float32)
            )
    except Exception as e:  # pragma: no cover
        logger.warning("could not freeze joints for rest-pose validation: %s", e)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    def _sim_pose(link_name):
        pos, quat = omni_robot.links[link_name].get_position_orientation()
        for v in (pos, quat):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
        return np.asarray(pos, float), np.asarray(quat, float)

    base_pos, base_quat = _sim_pose(base_link)
    base_T = _make_T(base_pos, base_quat)

    errors: dict[str, float] = {}
    rot_errors: dict[str, float] = {}
    sim_links = getattr(omni_robot, "links", {}) or {}
    for joint in art["joints"]:
        link = joint["child"]
        if link not in reachable_links(art, base_link):
            continue
        if link not in sim_links:
            continue
        pos, quat = _sim_pose(link)
        sim_local = _make_T_inv(base_T, pos, quat)
        frame_id = model.getFrameId(link, pin.FrameType.BODY)
        if frame_id >= model.nframes:
            continue
        M = data.oMf[frame_id]
        d = float(np.linalg.norm(
            np.asarray(M.translation, dtype=float) - sim_local[:3, 3]
        ))
        R_rel = sim_local[:3, :3].T @ np.asarray(M.rotation, dtype=float)
        cos_a = min(1.0, max(-1.0, (np.trace(R_rel) - 1.0) / 2.0))
        errors[link] = round(d, 6)
        rot_errors[link] = round(float(np.degrees(np.arccos(cos_a))), 4)

    max_err = max(errors.values(), default=0.0)
    max_rot = max(rot_errors.values(), default=0.0)
    return {
        "rest_pose_ok": max_err < 0.05 and max_rot < 5.0,
        "max_link_error_m": max_err,
        "max_link_rotation_error_deg": max_rot,
        "per_link_error_m": errors,
        "per_link_rotation_error_deg": rot_errors,
        "note": "links resting >5cm / >5deg from the sim articulation indicate "
                "a frame-convention mismatch; the export must not be used",
    }


def cross_validate_random_configs(
    omni_robot,
    urdf_path: Path,
    eef_link: str,
    base_link: str,
    num_samples: int = VALIDATION_DEFAULT_SAMPLES,
    trans_tol_m: float = VALIDATION_TRANS_TOL_M,
    rot_tol_deg: float = VALIDATION_ROT_TOL_DEG,
    seed: int = 0,
) -> dict:
    """§2 production acceptance gate: sample N random joint configurations,
    drive the SIMULATOR articulation to each, and compare the EEF pose
    against Pinocchio FK on the exported URDF (base-relative).

    Any sample beyond (trans_tol_m, rot_tol_deg) REJECTS the export: a
    kinematically wrong model must never become the grounding truth.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    # movable joints present in BOTH the simulator and the exported URDF
    sim_joints = getattr(omni_robot, "joints", {}) or {}
    movable = []
    for name, jprim in sim_joints.items():
        if name not in model.names:
            continue
        jid = model.getJointId(name)
        if jid == 0 or model.joints[jid].nv == 0:
            continue
        limits = None
        try:
            lo, hi = float(jprim.lower_limit), float(jprim.upper_limit)
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                limits = (lo, hi)
        except Exception:
            limits = None
        movable.append((name, jid, limits))
    if not movable:
        raise RuntimeError(
            "cross_validate_random_configs: no movable joints shared between "
            "the simulator articulation and the exported URDF"
        )

    sim_links = getattr(omni_robot, "links", {}) or {}
    if eef_link not in sim_links:
        raise RuntimeError(
            f"eef_link {eef_link!r} not found among simulator links {sorted(sim_links)}"
        )

    def _read_sim_pose():
        link_pos, link_quat = sim_links[eef_link].get_position_orientation()
        base_pos, base_quat = sim_links[base_link].get_position_orientation()
        for v in (link_pos, link_quat, base_pos, base_quat):
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
        T_world_base = _make_T(np.asarray(base_pos, float), np.asarray(base_quat, float))
        T_world_link = _make_T(np.asarray(link_pos, float), np.asarray(link_quat, float))
        return np.linalg.inv(T_world_base) @ T_world_link

    # full-DOF positions: URDF joints get random samples, everything else
    # (e.g. virtual floating-base joints) stays put so the base does not move
    try:
        n_dof = int(omni_robot.n_dof)
    except Exception:
        n_dof = max(len(sim_joints), 1)
    try:
        original_full = np.asarray(omni_robot.get_joint_positions(), dtype=float).copy()
    except Exception:
        original_full = np.zeros(n_dof)

    def _dof_index(jprim) -> int:
        for attr in ("dof_indices", "_dof_indices", "dof_index", "_dof_index"):
            value = getattr(jprim, attr, None)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = value[0]
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        raise RuntimeError(
            "JointPrim exposes no dof index; cannot drive joints for validation"
        )

    dof_indices = [_dof_index(sim_joints[m[0]]) for m in movable]

    rng = np.random.default_rng(seed)
    trans_errs, rot_errs, samples = [], [], []
    try:
        for i in range(num_samples):
            q = np.zeros(model.nq)
            full_positions = (
                original_full.copy() if len(original_full) == n_dof else np.zeros(n_dof)
            )
            for name, jid, limits, dof_idx in zip(
                [m[0] for m in movable], [m[1] for m in movable],
                [m[2] for m in movable], dof_indices,
            ):
                if limits is None:
                    value = float(rng.uniform(-np.pi, np.pi))
                else:
                    value = float(rng.uniform(limits[0], limits[1]))
                q[model.joints[jid].idx_q] = value
                full_positions[dof_idx] = value
            try:
                import torch

                omni_robot.set_joint_positions(
                    torch.as_tensor(full_positions, dtype=torch.float32)
                )
            except Exception as e:
                raise RuntimeError(
                    "could not drive simulator joints for cross-validation "
                    f"(set_joint_positions failed: {e})"
                ) from e
            T_sim = _read_sim_pose()
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            frame_id = model.getFrameId(eef_link, pin.FrameType.BODY)
            if frame_id >= model.nframes:
                raise RuntimeError(f"eef frame {eef_link!r} missing from exported URDF")
            M = data.oMf[frame_id]
            trans_err = float(np.linalg.norm(M.translation - T_sim[:3, 3]))
            R_rel = T_sim[:3, :3].T @ np.asarray(M.rotation, dtype=float)
            cos_a = min(1.0, max(-1.0, (np.trace(R_rel) - 1.0) / 2.0))
            rot_err = float(np.degrees(np.arccos(cos_a)))
            trans_errs.append(trans_err)
            rot_errs.append(rot_err)
            samples.append({
                "sample": i,
                "translation_error_m": round(trans_err, 6),
                "rotation_error_deg": round(rot_err, 4),
            })
    finally:
        # restore the pre-validation joint state (best effort)
        try:
            import torch

            omni_robot.set_joint_positions(
                torch.as_tensor(original_full, dtype=torch.float32)
            )
        except Exception as e:  # pragma: no cover
            logger.warning("joint restore after validation failed: %s", e)

    report = {
        "num_samples": len(samples),
        "max_translation_error_m": round(max(trans_errs), 6) if trans_errs else 0.0,
        "mean_translation_error_m": round(sum(trans_errs) / len(trans_errs), 6) if trans_errs else 0.0,
        "max_rotation_error_deg": round(max(rot_errs), 4) if rot_errs else 0.0,
        "mean_rotation_error_deg": round(sum(rot_errs) / len(rot_errs), 4) if rot_errs else 0.0,
        "translational_tolerance_m": trans_tol_m,
        "rotational_tolerance_deg": rot_tol_deg,
        "eef_link": eef_link,
        "base_link": base_link,
        "sampled_joints": [m[0] for m in movable],
        "samples": samples,
    }
    report["passed"] = bool(
        report["max_translation_error_m"] <= trans_tol_m
        and report["max_rotation_error_deg"] <= rot_tol_deg
    )
    return report


# ---------------------------------------------------------------------------
# export entry point
# ---------------------------------------------------------------------------


def _fix_joints_in_urdf(urdf_path: Path, fix_joints: list[str]) -> None:
    """Write the named joints as FIXED (restricted morphology variant)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    fixed = []
    for joint in root.findall("joint"):
        if joint.get("name") in fix_joints:
            joint.set("type", "fixed")
            for child in list(joint):
                if child.tag in ("limit", "axis"):
                    joint.remove(child)
            fixed.append(joint.get("name"))
    missing = set(fix_joints) - set(fixed)
    if missing:
        raise RuntimeError(f"fix_joints not found in URDF: {sorted(missing)}")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def _default_eef_link(omni_robot, art: dict) -> str:
    """EEF for validation: the sim arm EEF link if present, else a
    gripper/eef-named link."""
    arms = list(getattr(omni_robot, "arm_names", []) or [])
    eef_links = getattr(omni_robot, "eef_links", {}) or {}
    if arms and arms[0] in eef_links:
        try:
            return eef_links[arms[0]].GetName()
        except Exception:
            pass
    for name in art["links"]:
        if "eef" in name.lower() or "gripper" in name.lower():
            return name
    raise RuntimeError(
        "could not determine an EEF link for validation; pass eef_link explicitly"
    )


def _print_chain_summary(manifest: dict, art: dict) -> None:
    """§3: print base link / controlled chain / locked joints after export."""
    print("Base link:", manifest["root"])
    print("EEF link:", manifest.get("eef_link"))
    print("Controlled chain (movable joints, URDF order):")
    for j in art["joints"]:
        if j["type"] != "fixed" and j["child"] in manifest["links"]:
            print(f"  {j['name']} ({j['type']}, {j['parent']} -> {j['child']})")
    print("Locked/fixed:")
    for j in art["joints"]:
        if j["type"] == "fixed" and j["child"] in manifest["links"]:
            print(f"  {j['name']} ({j['parent']} -> {j['child']})")


def export_robot_urdf(
    omni_robot,
    out_path: str | Path,
    eef_link: str | None = None,
    base_link: str | None = None,
    validate_samples: int = VALIDATION_DEFAULT_SAMPLES,
    fix_joints: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Export an OmniGibson robot articulation to URDF (+ manifest + both
    cross-validation gates). Raises RuntimeError on any validation failure —
    the caller must not run the benchmark on a rejected model."""
    prim = omni_robot.prim
    art = collect_articulation(prim)
    link_boxes = collect_link_collision_boxes(prim)
    out_path = Path(out_path)
    manifest = write_urdf(art, link_boxes, out_path, base_link=base_link)

    if fix_joints:
        _fix_joints_in_urdf(out_path, fix_joints)

    if eef_link is None:
        eef_link = _default_eef_link(omni_robot, art)
    manifest["eef_link"] = eef_link
    manifest["fixed_joints_variant"] = list(fix_joints or [])

    manifest_path = out_path.with_suffix(".manifest.json")
    validation_path = out_path.with_name(out_path.stem + "_validation.json")

    def _dump():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        validation_path.write_text(
            json.dumps(manifest.get("random_config_validation", {}), indent=2),
            encoding="utf-8",
        )

    # dump early AND after every gate: a rejected export keeps its evidence
    # so debugging never needs another Kit launch
    _dump()

    validation = cross_validate_rest_pose(omni_robot, out_path, art, base_link=manifest["root"])
    manifest["validation"] = validation
    _dump()
    if not validation.get("rest_pose_ok", False):
        raise RuntimeError(
            f"URDF export FK cross-validation FAILED for {out_path}: "
            f"max_link_error={validation.get('max_link_error_m')}m, "
            f"max_rotation={validation.get('max_link_rotation_error_deg')}deg — "
            "the exported kinematics do not match the simulator articulation; "
            "refusing to write a wrong model (evidence in the manifest)"
        )

    if validate_samples > 0:
        random_validation = cross_validate_random_configs(
            omni_robot, out_path, eef_link=eef_link,
            base_link=manifest["root"], num_samples=validate_samples,
        )
        manifest["random_config_validation"] = random_validation
        _dump()
        if not random_validation["passed"]:
            raise RuntimeError(
                f"URDF export REJECTED (§2): random-configuration cross-"
                f"validation exceeded tolerances: max_trans="
                f"{random_validation['max_translation_error_m']}m, max_rot="
                f"{random_validation['max_rotation_error_deg']}deg "
                f"(n={random_validation['num_samples']})"
            )
    if verbose:
        _print_chain_summary(manifest, art)
    return manifest
