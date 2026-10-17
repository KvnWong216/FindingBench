"""Synthetic USD lights ONLY. No renderer or physical acceptance claimed."""
import math

import pytest

from rummagebench.authoring.tasks.household import UPSTREAM_COMMIT
from rummagebench.sim.omnigibson.infinigen_adapter import normalize_infinigen_point_photometry


def point_fixture():
    from pxr import Usd,UsdLux,Sdf
    stage=Usd.Stage.CreateInMemory()
    stage.DefinePrim('/source','Xform')
    point=UsdLux.SphereLight.Define(stage,'/source/point')
    radius=.02072676084935665
    energy=59.384735107421875
    factor=20/(4*math.pi)*1000/(4*math.pi*radius**2)*100
    point.CreateRadiusAttr(radius)
    point.CreateNormalizeAttr(True)
    point.CreateIntensityAttr(energy*factor/math.pi)
    point.GetPrim().CreateAttribute('userProperties:blender:data_name',Sdf.ValueTypeNames.String).Set('UNIT_SOURCE_POINT')
    rect=UsdLux.RectLight.Define(stage,'/source/area');rect.CreateIntensityAttr(3.)
    return stage,point,rect,energy


def test_inverse_matches_original_blender_usd_energy_conversion_and_leaves_area_lights():
    stage,point,rect,energy=point_fixture()
    report=normalize_infinigen_point_photometry(stage,'/source',UPSTREAM_COMMIT)
    assert point.GetIntensityAttr().Get()==pytest.approx(energy/math.pi,rel=2e-7)
    assert rect.GetIntensityAttr().Get()==3.
    assert report['geometry_or_physics_changed'] is False and report['image_acceptance']=='pending'
    assert len(report['point_lights'])==1
    with pytest.raises(ValueError,match='DOUBLE_NORMALIZATION'):
        normalize_infinigen_point_photometry(stage,'/source',UPSTREAM_COMMIT)


@pytest.mark.parametrize('problem',['unrecognized_exporter','missing_source','non_normalized','invalid_radius'])
def test_unproven_point_contract_is_rejected(problem):
    stage,point,_,_=point_fixture()
    commit=UPSTREAM_COMMIT
    if problem=='unrecognized_exporter':commit='unrecognized'
    elif problem=='missing_source':point.GetPrim().RemoveProperty('userProperties:blender:data_name')
    elif problem=='non_normalized':point.GetNormalizeAttr().Set(False)
    else:point.GetRadiusAttr().Set(0.)
    with pytest.raises(ValueError,match='PHOTOMETRY'):
        normalize_infinigen_point_photometry(stage,'/source',commit)
