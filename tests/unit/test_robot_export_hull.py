"""Robot collider export as convex-hull meshes."""
from __future__ import annotations

import numpy as np
import pytest

from rummagebench.sim.omnigibson.robot_export import write_hull_obj

pin = pytest.importorskip("pinocchio")

URDF = """<robot name="r">
  <link name="base_link"><collision><origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry><mesh filename="r_meshes/base_link__0.obj"/></geometry></collision></link>
  <link name="tip"/>
  <joint name="j" type="revolute"><parent link="base_link"/><child link="tip"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""


def test_hull_obj_is_outward_and_loads_as_solid_convex(tmp_path):
    rng = np.random.default_rng(0)
    cube = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.2, 0.2) for z in (0, 0.3)])
    pts = np.vstack([cube, rng.uniform([-0.1, -0.2, 0], [0.1, 0.2, 0.3], (50, 3))])
    assert write_hull_obj(pts, tmp_path / "r_meshes" / "base_link__0.obj")
    text = (tmp_path / "r_meshes" / "base_link__0.obj").read_text()
    assert text.count("\nv ") + text.startswith("v ") == 8  # interior points dropped
    (tmp_path / "r.urdf").write_text(URDF)

    from rummagebench.feasibility.fcl_compat import collision_backend
    from rummagebench.feasibility.pinocchio_solver import PinocchioKinematics

    kin = PinocchioKinematics(urdf_path=str(tmp_path / "r.urdf"), base_link="base_link",
                              eef_link="tip", controlled_joints="auto")
    geom = kin.geom_model.geometryObjects[0].geometry
    coal = collision_backend()
    assert "Convex" in type(geom).__name__
    # a small sphere fully INSIDE the hull collides (solid, not a shell)
    inside = coal.CollisionObject(coal.Sphere(0.01), coal.Transform3s(np.eye(3), np.array([0, 0, 0.15]))
                                  if hasattr(coal, "Transform3s") else
                                  coal.Transform3f(np.eye(3), np.array([0, 0, 0.15])))
    hull = coal.CollisionObject(geom, coal.Transform3s() if hasattr(coal, "Transform3s")
                                else coal.Transform3f())
    res = coal.CollisionResult()
    coal.collide(hull, inside, coal.CollisionRequest(), res)
    assert res.isCollision()


def test_degenerate_point_sets_fall_back(tmp_path):
    flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    assert not write_hull_obj(flat, tmp_path / "f.obj")
    assert not write_hull_obj(flat[:3], tmp_path / "g.obj")
