"""Scaled USD colliders -> coal geometry.

BEHAVIOR colliders are unit meshes / primitives under scaled xforms. The
export must bake the stretch into the geometry and keep only the rigid part
in the pose.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rummagebench.feasibility.fcl_compat import collision_backend
from rummagebench.sim.omnigibson.usd_collision import (
    _BOX_TRIS, _bvh, bake_points, box_corners, decompose_world_transform,
    mesh_triangles, primitive_geometry)

coal = collision_backend()


def usd_matrix(scale, rot_xyzw, t, parent_scale=(1, 1, 1), parent_rot=(0, 0, 0, 1)):
    """Row-vector world matrix of a collider: local scale, local rotation,
    translation, under a scaled + rotated parent (USD composition order)."""
    def m(lin, tr=(0, 0, 0)):
        M = np.eye(4)
        M[:3, :3] = lin
        M[3, :3] = tr
        return M
    local = m(np.diag(scale)) @ m(Rotation.from_quat(rot_xyzw).as_matrix().T, t)
    parent = m(np.diag(parent_scale)) @ m(Rotation.from_quat(parent_rot).as_matrix().T)
    return local @ parent


def world_points(M, pts):
    return np.c_[pts, np.ones(len(pts))] @ M


def test_decomposition_reproduces_the_usd_transform():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = usd_matrix(rng.uniform(0.01, 2, 3), Rotation.random(random_state=rng).as_quat(),
                       rng.uniform(-5, 5, 3), rng.uniform(0.5, 1.5, 3),
                       Rotation.random(random_state=rng).as_quat())
        P, R, t = decompose_world_transform(M)
        assert np.allclose(P, P.T)
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
        pts = rng.uniform(-0.5, 0.5, (20, 3))
        ref = world_points(M, pts)[:, :3]
        assert np.allclose(bake_points(pts, P) @ R.T + t, ref, atol=1e-9)


def test_cabinet_panel_collider_keeps_its_real_size():
    """Unit cube under scale [0.017, 0.46, 0.04] (measured on a BEHAVIOR
    cabinet) must become a 1.7 cm x 46 cm x 4 cm panel, not a 1 m cube."""
    M = usd_matrix([0.0171, 0.4618, 0.0385], [0, 0, 0.7071068, 0.7071068], [0.6, -5.3, 0.4])
    P, R, t = decompose_world_transform(M)
    geom, approx, local = primitive_geometry("Cube", {"size": 1.0}, P, coal)
    assert approx == "physics_collider_box"
    assert np.allclose(sorted(2 * geom.halfSide), sorted([0.0171, 0.4618, 0.0385]), atol=1e-6)
    world = local @ R.T + t
    ref = world_points(M, box_corners(np.full(3, 0.5)))[:, :3]
    assert np.allclose(world.min(0), ref.min(0), atol=1e-6)
    assert np.allclose(world.max(0), ref.max(0), atol=1e-6)


def _collides(g1, T1, g2, T2) -> bool:
    def obj(g, T):
        tf = coal.Transform3s(T[:3, :3], T[:3, 3]) if hasattr(coal, "Transform3s") \
            else coal.Transform3f(T[:3, :3], T[:3, 3])
        return coal.CollisionObject(g, tf)
    res = coal.CollisionResult()
    coal.collide(obj(g1, T1), obj(g2, T2), coal.CollisionRequest(), res)
    return res.isCollision()


def test_scaled_mesh_collides_exactly_where_the_usd_shape_is():
    """A unit-cube MESH collider under a thin-panel scale: a probe sphere
    just outside the panel misses it, one touching the panel hits it."""
    pts = box_corners(np.full(3, 0.5))
    M = usd_matrix([0.02, 0.5, 0.4], [0, 0, 0, 1], [1.0, 0.0, 0.5])
    P, R, t = decompose_world_transform(M)
    geom = _bvh(bake_points(pts, P), _BOX_TRIS, coal)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    sphere = coal.Sphere(0.05)

    def at(x, y, z):
        S = np.eye(4)
        S[:3, 3] = [x, y, z]
        return S
    assert _collides(geom, T, sphere, at(1.0 + 0.01 + 0.04, 0.0, 0.5))      # touches
    assert not _collides(geom, T, sphere, at(1.0 + 0.01 + 0.07, 0.0, 0.5))  # 2 cm gap
    assert not _collides(geom, T, sphere, at(1.0, 0.0, 0.5 + 0.2 + 0.08))   # above
    # the old export (unit mesh, no stretch) put a surface 0.5 m out, where
    # the real 2 cm panel is nowhere near
    old = _bvh(pts, _BOX_TRIS, coal)
    assert _collides(old, T, sphere, at(1.5, 0.0, 0.5))
    assert not _collides(geom, T, sphere, at(1.5, 0.0, 0.5))


def test_convex_hull_colliders_are_solid():
    """PhysX convexHull colliders are solid: a body fully inside must
    collide (a BVH surface mesh would miss it)."""
    from rummagebench.sim.omnigibson.usd_collision import mesh_geometry_from_points

    pts = bake_points(box_corners(np.full(3, 0.5)), np.diag([1.0, 1.0, 1.0]))
    tris = _BOX_TRIS
    solid, a1 = mesh_geometry_from_points(pts, tris, "convexHull", coal)
    hollow, a2 = mesh_geometry_from_points(pts, tris, "convexDecomposition", coal)
    assert a1 == "physics_convex_hull" and a2 == "physics_mesh_bvh"
    inside = np.eye(4)
    sphere = coal.Sphere(0.05)
    assert _collides(solid, np.eye(4), sphere, inside)
    assert not _collides(hollow, np.eye(4), sphere, inside)


def test_primitives_under_non_uniform_scale():
    P = np.diag([2.0, 2.0, 0.5])
    g, a, _ = primitive_geometry("Cylinder", {"radius": 0.1, "height": 1.0, "axis": "Z"},
                                 P, coal)
    assert a == "physics_collider_cylinder"
    assert g.radius == pytest.approx(0.2) and g.halfLength == pytest.approx(0.25)
    g, a, _ = primitive_geometry("Sphere", {"radius": 0.1}, np.eye(3) * 3, coal)
    assert a == "physics_collider_sphere" and g.radius == pytest.approx(0.3)
    g, a, _ = primitive_geometry("Sphere", {"radius": 0.1}, np.diag([1, 2, 3.0]), coal)
    assert a in ("physics_collider_ellipsoid", "scaled_sphere_bounding_box")
    sheared = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
    g, a, local = primitive_geometry("Cube", {"size": 1.0}, sheared, coal)
    assert a == "physics_collider_box_sheared" and len(local) == 8


def test_polygon_triangulation():
    assert mesh_triangles([4, 3], [0, 1, 2, 3, 4, 5, 6]) == [(0, 1, 2), (0, 2, 3), (4, 5, 6)]


def test_obb_in_parent_is_exact_for_a_rotated_scaled_box():
    from rummagebench.sim.omnigibson.usd_collision import obb_in_parent

    rot = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_quat()
    M = usd_matrix([0.04, 0.1, 0.3], rot, [0.2, -0.1, 0.5])
    P, R, t = decompose_world_transform(M)
    _, _, pts = primitive_geometry("Cube", {"size": 1.0}, P, coal)
    box = obb_in_parent(pts, R, t)
    assert np.allclose(sorted(box["half"]), [0.02, 0.05, 0.15], atol=1e-9)
    assert np.allclose(box["center"], [0.2, -0.1, 0.5], atol=1e-9)
    # URDF rpy (fixed-axis xyz) reproduces the rotation
    Rb = Rotation.from_euler("xyz", box["rpy"]).as_matrix()
    corners = box_corners(np.array(box["half"])) @ Rb.T + box["center"]
    ref = world_points(M, box_corners(np.full(3, 0.5)))[:, :3]
    assert np.allclose(np.sort(corners, 0), np.sort(ref, 0), atol=1e-9)


def test_physics_pose_replaces_stale_usd_link_pose():
    """A collider follows its link's PHYSICS pose (drawer slid out, object
    fallen), keeping the static collider-to-link offset and the stretch."""
    from rummagebench.sim.omnigibson.usd_collision import compose_link_matrix

    obj_scale = [1.3, 1.2, 1.1]
    M_link_usd = usd_matrix(obj_scale, [0, 0, 0, 1], [0, 0, 0])  # stale: closed
    M_coll_usd = usd_matrix([0.02, 0.4, 0.1], [0, 0, 0, 1], [0.25, 0, 0.3]) @ \
        np.diag([1.3, 1.2, 1.1, 1.0])  # panel 0.25 m in front, under the scaled root
    q = Rotation.from_euler("z", 30, degrees=True).as_quat()
    M_link = compose_link_matrix(M_link_usd, np.array([0.4, 0.0, 0.0]), q)
    M = M_coll_usd @ (np.linalg.inv(M_link_usd) @ M_link)
    P, R, t = decompose_world_transform(M)
    # rigid part follows the physics pose
    expect_t = Rotation.from_quat(q).apply(np.array([0.25, 0, 0.3]) * obj_scale) + [0.4, 0, 0]
    assert np.allclose(t, expect_t, atol=1e-9)
    # the panel keeps its real size
    _, _, pts = primitive_geometry("Cube", {"size": 1.0}, P, coal)
    ext = pts.max(0) - pts.min(0)
    assert np.allclose(sorted(ext), sorted([0.02 * 1.3, 0.4 * 1.2, 0.1 * 1.1]), atol=1e-9)
