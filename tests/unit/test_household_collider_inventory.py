"""Real CPU OpenUSD schema, explicitly synthetic test collider geometry only."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_disabled_colliders_are_skipped_and_enabled_prim_provenance_is_kept(monkeypatch):
    pytest.importorskip('pxr',reason='standalone usd-core needed, no simulator')
    from pxr import Usd,UsdGeom,UsdPhysics
    from rummagebench.sim.omnigibson import usd_collision as uc
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/unit_object').GetPrim()
    link=UsdGeom.Xform.Define(stage,'/unit_object/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link)
    for name,enabled in [('enabled',True),('disabled',False)]:
        cube=UsdGeom.Cube.Define(stage,f'/unit_object/base_link/{name}').GetPrim()
        UsdPhysics.CollisionAPI.Apply(cube).CreateCollisionEnabledAttr(enabled)
    monkeypatch.setattr(uc,'_coal',lambda:SimpleNamespace())
    monkeypatch.setattr(uc,'_build_geometry',lambda *args:(object(),'synthetic_unit_geometry_only',np.array([[-1,-1,-1],[1,1,1]])))
    obj=SimpleNamespace(name='unit_object',category='synthetic_unit_only',prim=root,links={})
    bodies=uc.collect_entity_collision(obj,{})
    assert len(bodies)==1 and bodies[0].source_prim=='/unit_object/base_link/enabled'
    assert bodies[0].physics_approximation=='none'
    assert bodies[0].approximation=='synthetic_unit_geometry_only'  # cannot satisfy sweep fidelity gate
