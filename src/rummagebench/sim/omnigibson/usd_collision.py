"""USD collision geometry export: world bodies -> coal collision geometries.

Fidelity chain (recorded per body, never silent):

    existing physics collider (mesh BVH / exact primitive)
    -> world-aligned AABB primitive

The robot is excluded by the caller: its collision geometry comes from the
exported URDF (URDF <collision>), never from visual meshes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rummagebench.sim.base import WorldCollisionObject

logger = logging.getLogger(__name__)

_MAX_TRIANGLE_BUDGET = 60000


def _coal():
    try:
        import coal
        return coal
    except ImportError:  # pragma: no cover
        import pinocchio as pin
        return pin.hppfcl


def iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from iter_prims(child)


def _quat_xyzw(q) -> list[float]:
    return [float(q.imaginary[0]), float(q.imaginary[1]), float(q.imaginary[2]), float(q.real)]


def collect_entity_collision(obj, geom_cache: dict[str, Any]) -> list[WorldCollisionObject]:
    """All collision bodies of one scene object, with current world poses."""
    from pxr import UsdGeom, UsdPhysics

    coal = _coal()
    prim = obj.prim
    category = getattr(obj, "category", "")

    bodies: list[WorldCollisionObject] = []
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )

    # link decomposition: rigid bodies when present, else the object root
    link_prims = [p for p in iter_prims(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not link_prims:
        link_prims = [prim]

    for link_prim in link_prims:
        link_name = link_prim.GetName()
        is_root = link_prim == prim
        colliders = [
            p for p in iter_prims(link_prim)
            if p.HasAPI(UsdPhysics.CollisionAPI) and p != link_prim
        ] if link_prim != prim else [
            p for p in iter_prims(link_prim) if p.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not colliders:
            continue
        for collider in colliders:
            path = str(collider.GetPath())
            geometry = geom_cache.get(path)
            approximation = None
            if geometry is None:
                geometry, approximation = _build_geometry(collider, cache, coal)
                if geometry is None:
                    continue
                geom_cache[path] = (geometry, approximation)
            else:
                geometry, approximation = geometry

            xform = UsdGeom.Xformable(collider).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            pos = np.array(xform.ExtractTranslation(), dtype=float)
            quat = _quat_xyzw(xform.ExtractRotationQuat())
            try:
                rng = cache.ComputeWorldBound(collider).ComputeAlignedRange()
                aabb = (
                    [float(v) for v in rng.min],
                    [float(v) for v in rng.max],
                )
            except Exception:
                aabb = None

            bodies.append(
                WorldCollisionObject(
                    entity=obj.name,
                    link=None if is_root and len(link_prims) == 1 else link_name,
                    geometry=geometry,
                    pose=_pose_from(pos, quat),
                    category=category,
                    approximation=approximation or "aabb_primitive",
                    aabb=aabb,
                )
            )
    return bodies


def _pose_from(pos: np.ndarray, quat_xyzw: list[float]):
    from rummagebench.feasibility.ik_solver import Pose

    return Pose.from_lists([float(v) for v in pos], quat_xyzw)


def _build_geometry(collider, cache, coal):
    """coal geometry for one USD collider + its approximation level."""
    from pxr import UsdGeom

    tname = collider.GetTypeName()
    try:
        if tname == "Cube":
            size_attr = UsdGeom.Cube(collider).GetAttr("size")
            size = float(size_attr.Get() or 1.0)
            return coal.Box(size / 2, size / 2, size / 2), "physics_collider_box"
        if tname == "Sphere":
            r = float(UsdGeom.Sphere(collider).GetRadiusAttr().Get() or 0.1)
            return coal.Sphere(r), "physics_collider_sphere"
        if tname == "Cylinder":
            cyl = UsdGeom.Cylinder(collider)
            r = float(cyl.GetRadiusAttr().Get() or 0.1)
            h = float(cyl.GetHeightAttr().Get() or 0.2)
            return coal.Cylinder(r, h), "physics_collider_cylinder"
        if tname == "Mesh":
            return _mesh_geometry(UsdGeom.Mesh(collider), coal)
    except Exception as e:
        logger.warning("exact collider build failed for %s: %s", collider.GetPath(), e)
    # fallback: world-aligned AABB primitive
    try:
        rng = cache.ComputeWorldBound(collider).ComputeAlignedRange()
        lo, hi = np.array(rng.min, dtype=float), np.array(rng.max, dtype=float)
        half = np.maximum((hi - lo) / 2.0, 1e-4)
        return coal.Box(float(half[0]), float(half[1]), float(half[2])), "aabb_primitive"
    except Exception as e:
        logger.warning("aabb fallback failed for %s: %s", collider.GetPath(), e)
        return None, None


def _mesh_geometry(mesh, coal):
    """Exact triangle-mesh collider (existing physics collision) within a
    triangle budget; above it the caller's AABB fallback applies."""
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    idx = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    points = mesh.GetPointsAttr().Get()
    if not counts or not idx or points is None:
        return None, None
    tris = []
    i = 0
    for n in counts:
        for k in range(1, n - 1):
            tris.append((idx[i], idx[i + k], idx[i + k + 1]))
        i += n
    if len(tris) > _MAX_TRIANGLE_BUDGET:
        return None, None
    verts_std = coal.StdVec_Vec3s()
    for p in points:
        verts_std.append(np.array([float(p[0]), float(p[1]), float(p[2])]))
    tris_std = coal.StdVec_Triangle()
    for a, b, c in tris:
        tris_std.append(coal.Triangle(int(a), int(b), int(c)))
    bvh = coal.BVHModelOBB()
    bvh.beginModel(len(verts_std), len(tris_std))
    bvh.addSubModel(verts_std, tris_std)
    bvh.endModel()
    return bvh, "physics_mesh_bvh"


def compute_link_aabb(link_prim):
    """World-frame AABB (lo, hi) of a link's collider bounds."""
    from pxr import UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    try:
        rng = cache.ComputeWorldBound(link_prim).ComputeAlignedRange()
        return ([float(v) for v in rng.min], [float(v) for v in rng.max])
    except Exception as e:
        logger.warning("link aabb failed for %s: %s", link_prim.GetPath(), e)
        return None
