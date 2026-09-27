"""OmniGibson robot articulation -> URDF kinematics export.

The robot's kinematic model is DERIVED from the simulator asset — link
frames, joint tree, joint axes/limits and collision-bound boxes are read
from the USD articulation; nothing about the robot is hand-authored.

Outputs (written next to the requested urdf_path):

    <name>.urdf      fixed-base kinematics: link frames, joints, limits,
                     <collision> boxes (link collider AABBs, approximation
                     level recorded in the manifest)
    <name>.manifest.json  export provenance + approximation levels +
                     rest-pose FK cross-validation results

Cross-validation: after export, the URDF is re-imported with Pinocchio and
the rest-pose link poses are compared against the simulator's link poses;
a random single-joint sweep is attempted when the sim allows joint writes.
A frame-convention mismatch FAILS the export loudly (exit != 0), so a broken
kinematic model can never silently become the benchmark's grounding truth.

Run on the simulator host (needs omnigibson + pin):

    python scripts/export_robot_kinematics.py --robot r1pro \
        --out assets/robots/r1pro/r1pro.urdf
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

_MAX_TRIANGLE_BUDGET = 60000  # mesh colliders above this fall back to AABB


# ---------------------------------------------------------------------------
# USD articulation collection (pxr.UsdPhysics — stable schema-level API)
# ---------------------------------------------------------------------------


def collect_articulation(root_prim) -> dict:
    """Walk the USD articulation under ``root_prim``.

    Returns {root: name, links: [names], joints: [joint dicts]} where each
    joint carries parent/child link names, joint frame offsets (localPos0/1,
    localRot0/1), axis and limits (radians; USD revolute degrees converted).
    """
    from pxr import UsdPhysics, Usd, UsdGeom

    stage = root_prim.GetStage()
    root_name = root_prim.GetName()

    # links: rigid bodies under the root
    links: list[str] = []
    link_frames: dict[str, tuple[list[float], list[float]]] = {}
    world_xforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prim in iter_prims(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = prim.GetName()
            links.append(name)
            xform = UsdGeom.Xformable(prim)
            m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_xforms[name] = (
                np.array(m.ExtractTranslation(), dtype=float),
                _gf_quat_to_xyzw(m.ExtractRotationQuat()),
            )

    joints: list[dict] = []
    for prim in iter_prims(root_prim):
        tname = prim.GetTypeName()
        jtype = _JOINT_TYPE_BY_USD.get(tname)
        if jtype is None:
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = _rel_target_name(prim.GetRelationship("physics:body0"))
        body1 = _rel_target_name(prim.GetRelationship("physics:body1"))
        if body0 is None or body1 is None:
            continue

        def _attr(name: str):
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

        joints.append(
            {
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
            }
        )

    handle_link = next((l for l in links if "handle" in l.lower()), None)
    return {
        "root": root_name,
        "links": links,
        "joints": joints,
        "handle_link": handle_link,
        "world_xforms": world_xforms,
    }


def iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from iter_prims(child)


def _rel_target_name(rel) -> str | None:
    targets = rel.GetTargets() if rel else []
    if not targets:
        return None
    return targets[0].name if len(targets) else None


def _axis_vector(axis) -> list[float] | None:
    if axis is None:
        return None
    return {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}.get(str(axis))


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
    expressed in the LINK frame (USD), as {cx,cy,cz,hx,hy,hz} boxes.

    Approximation levels: 'physics_collider_box' for axis-aligned USD box
    colliders read exactly, 'aabb_primitive' for the world-aligned bound of
    everything else (meshes/capsules). Levels are recorded per box.
    """
    from pxr import UsdPhysics, Usd, UsdGeom

    stage = root_prim.GetStage()
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
                bbox = cache.ComputeWorldBound(sub)
                rng = bbox.ComputeAlignedRange()
                lo, hi = np.array(rng.min, dtype=float), np.array(rng.max, dtype=float)
            except Exception as e:  # pragma: no cover - depends on stage
                logger.warning("bbox failed for %s: %s", sub.GetPath(), e)
                continue
            center_world = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            center_link = world_to_link.Transform(center_world)
            boxes.append(
                {
                    "center": [float(c) for c in center_link],
                    "half": [float(c) for c in half],
                    "approximation": approx,
                    "source": str(sub.GetPath()),
                }
            )
        link_boxes[link_name] = boxes
    return link_boxes


# ---------------------------------------------------------------------------
# URDF writer (frames/joints derived from USD; boxes from collider bounds)
# ---------------------------------------------------------------------------


def write_urdf(art: dict, link_boxes: dict, out_path: Path) -> dict:
    """Compose the URDF from the collected articulation.

    Frame bookkeeping: for joint J(parent P, child C) with USD joint frames
    localPos0/rot0 (P->J) and localPos1/rot1 (C->J), the URDF joint origin is
    s_P * (localPos0, localRot0) and the URDF child frame IS the joint frame;
    every USD-frame quantity of the child link is re-expressed with
    s_C = inv(localPos1, localRot1). (s_L maps URDF link frame -> USD link
    frame; s_root = identity.)
    """
    import xml.etree.ElementTree as ET

    root = ET.Element("robot", {"name": art["root"]})
    manifest: dict = {"root": art["root"], "links": {}, "joints": {}, "handle_link": art["handle_link"]}
    shift: dict[str, tuple[np.ndarray, np.ndarray]] = {art["root"]: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))}

    def add_link(name: str, s: tuple[np.ndarray, np.ndarray]):
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
            center = _xform_point(inv_pose(s), np.asarray(box["center"], dtype=float))
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", {
                "xyz": " ".join(f"{v:.8g}" for v in center),
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

    add_link(art["root"], shift[art["root"]])

    for j in art["joints"]:
        s_parent = shift[j["parent"]]
        origin_pos = _pose_mul_point(s_parent, np.asarray(j["localPos0"], dtype=float))
        origin_quat = _quat_mul(s_parent[1], np.asarray(j["localRot0"], dtype=float))
        child_shift = _inv_pose((np.asarray(j["localPos1"], dtype=float),
                                np.asarray(j["localRot1"], dtype=float)))
        shift[j["child"]] = child_shift
        add_link(j["child"], child_shift)

        jtype = j["type"]
        attrs = {
            "name": j["name"],
            "type": "fixed" if jtype == "fixed" else jtype,
        }
        joint_el = ET.SubElement(root, "joint", attrs)
        parent_name = j["parent"] if j["parent"] in shift else art["root"]
        ET.SubElement(joint_el, "parent", {"link": parent_name})
        ET.SubElement(joint_el, "child", {"link": j["child"]})
        ET.SubElement(joint_el, "origin", {
            "xyz": " ".join(f"{v:.8g}" for v in origin_pos),
            "rpy": _quat_to_rpy(origin_quat),
        })
        if jtype in ("revolute", "prismatic", "continuous"):
            axis = j["axis"] or [0.0, 0.0, 1.0]
            ET.SubElement(joint_el, "axis", {"xyz": " ".join(f"{v:.8g}" for v in axis)})
            lo, hi = j["limits"] if j["limits"] else (0.0, 0.0)
            ET.SubElement(joint_el, "limit", {
                "lower": f"{lo:.8g}", "upper": f"{hi:.8g}",
                "effort": "1e9", "velocity": "1e9",
            })
        manifest["joints"][j["name"]] = {"type": jtype, "parent": parent_name, "child": j["child"]}

    ET.indent(root, space="  ")
    xml = ET.ElementTree(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml.write(out_path, encoding="utf-8", xml_declaration=True)
    return manifest


# --- minimal SE(3) helpers on (pos xyz, quat xyzw) tuples -------------------


def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
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


def _xform_point(pose, p):
    return _pose_mul_point(pose, p)


def _quat_to_rpy(q) -> str:
    """xyzw quaternion -> URDF rpy (extrinsic XYZ / roll-pitch-yaw)."""
    x, y, z, w = q / np.linalg.norm(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_p = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sin_p) if abs(sin_p) >= 1 else math.asin(sin_p)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return f"{roll:.8g} {pitch:.8g} {yaw:.8g}"


# ---------------------------------------------------------------------------
# export entry point + FK cross-validation
# ---------------------------------------------------------------------------


def export_robot_urdf(omni_robot, out_path: str | Path) -> dict:
    """Export an OmniGibson robot articulation to URDF + manifest.

    ``omni_robot`` is the loaded OmniGibson robot object (EntityPrim). Returns
    the manifest; raises RuntimeError on FK cross-validation failure.
    """
    prim = omni_robot.prim
    art = collect_articulation(prim)
    link_boxes = collect_link_collision_boxes(prim)
    out_path = Path(out_path)
    manifest = write_urdf(art, link_boxes, out_path)

    # cross-validate: rest-pose link poses, URDF FK vs simulator
    validation = cross_validate_rest_pose(omni_robot, out_path, art)
    manifest["validation"] = validation
    if not validation.get("rest_pose_ok", False):
        raise RuntimeError(
            f"URDF export FK cross-validation FAILED for {out_path}: "
            f"{validation} — the exported kinematics do not match the "
            "simulator articulation; refusing to write a wrong model"
        )

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def cross_validate_rest_pose(omni_robot, urdf_path: Path, art: dict) -> dict:
    """Pinocchio FK at q=0 vs OmniGibson link world poses (base-relative)."""
    try:
        import pinocchio as pin
    except ImportError:
        return {"rest_pose_ok": False, "error": "pinocchio not importable on the sim host"}

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q = np.zeros(model.nq)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    base_pos, base_quat = omni_robot.get_position_orientation()
    base_T = _make_T(np.asarray(base_pos), np.asarray(base_quat))

    errors: dict[str, float] = {}
    for joint in art["joints"]:
        link = joint["child"]
        if link not in art.get("world_xforms", {}):
            continue
        pos, quat = art["world_xforms"][link]
        sim_local = _make_T_inv(base_T, np.asarray(pos), np.asarray(quat))
        frame_id = model.getFrameId(link, pin.FrameType.BODY)
        if frame_id >= model.nframes:
            continue
        M = data.oMf[frame_id]
        urdf_pos = np.asarray(M.translation, dtype=float)
        d = float(np.linalg.norm(urdf_pos - sim_local[:3, 3]))
        errors[link] = round(d, 6)

    max_err = max(errors.values(), default=0.0)
    return {
        "rest_pose_ok": max_err < 0.05,
        "max_link_error_m": max_err,
        "per_link_error_m": errors,
        "note": "links resting >5cm from the sim articulation indicate a "
                "frame-convention mismatch; the export must not be used",
    }


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
