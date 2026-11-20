"""CPU mathematics and explicit synthetic physics fixtures; never real data."""
from copy import deepcopy
import math
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from rummagebench.authoring.tasks.household_physics import (
    continuous_clearance, rigid_joint_pose, placement_stability,
    validate_sweep_evidence, validate_temporary_place_evidence, SWEEP_VERSION,
)
from rummagebench.feasibility.ik_solver import Pose, quat_rotate


@pytest.mark.kinematics
@pytest.mark.parametrize('height,distance,depth',[(1.1,.1,0.),(1.,0.,0.),(.99,0.,.01)])
def test_actual_fcl_replacement_pair_measurements(height,distance,depth):
    """Real FCL boxes, synthetic CPU fixtures only; no simulator imports."""
    from rummagebench.feasibility.fcl_compat import collision_backend
    from rummagebench.authoring.tasks.household_physics import replacement_pair_measurement
    from rummagebench.sim.base import WorldCollisionObject
    try:coal=collision_backend()
    except ImportError:pytest.skip('optional coal/hpp-fcl required for real CPU collider query')
    cabinet=WorldCollisionObject('unit_cab',None,coal.Box(1.,1.,1.),Pose.from_lists([0.,0.,height]))
    floor=WorldCollisionObject('unit_floor',None,coal.Box(4.,4.,1.),Pose.from_lists([0.,0.,0.]))
    measured=replacement_pair_measurement(cabinet,floor)
    assert measured['distance_m']==pytest.approx(distance,abs=1e-7)
    assert measured['max_penetration_m']==pytest.approx(depth,abs=1e-7)
    if distance:assert measured['nearest_points'] is not None
    else:assert measured['contacts']


@pytest.mark.parametrize('corruption',[None,'floating','penetration','missing_pair','unknown_foreign','fake_points','limit','flag'])
def test_replacement_support_requires_complete_numeric_geometry(corruption):
    from rummagebench.authoring.tasks.household_physics import validate_replacement_support_evidence
    pair={'first':'unit_cab/collider','second':'unit_floor/collider','foreign_entity':'unit_floor',
        'distance_m':.003,'max_penetration_m':0.,'contacts':[],'nearest_points':[[0.,0.,.003],[0.,0.,0.]]}
    record={'status':'passed','fixed_base':True,'bottom_z_m':.003,'foreign_entities':['unit_floor'],
        'collider_counts':{'unit_cab':1,'unit_floor':1},'pair_measurements':[pair],
        'support':[{'floor':'unit_floor','point':[0.,0.,0.]}]}
    report={'version':1,'status':'passed','gpu_executed':True,'max_penetration_m':.002,
        'support_clearance_m':.005,'containers':{'unit_cab':record},'unit_fixture_only':True}
    if corruption=='floating':pair['distance_m']=.1;pair['nearest_points'][0][2]=.1;record['bottom_z_m']=.1
    if corruption=='penetration':pair['max_penetration_m']=.02
    if corruption=='missing_pair':record['pair_measurements']=[]
    if corruption=='unknown_foreign':pair['foreign_entity']='unit_unknown'
    if corruption=='fake_points':pair['nearest_points'][1][0]=3.
    if corruption=='limit':report['max_penetration_m']=1.
    if corruption=='flag':report['gpu_executed']=False
    args=(report,{'unit_cab'},{'unit_cab','unit_floor'},{'unit_floor'})
    if corruption:
        with pytest.raises(ValueError):validate_replacement_support_evidence(*args)
    else:validate_replacement_support_evidence(*args)


def test_continuous_clearance_finds_collision_between_clear_endpoints_and_midpoint():
    distance=lambda t:max(0.,abs(t-.375)-.01)
    assert all(distance(t)>.005 for t in (0.,.5,1.))
    result=continuous_clearance(distance,1.)
    assert result["status"]=="failed"
    assert any(d==0 for t,d in result["samples"])


def test_bounded_sweep_does_not_turn_unresolved_samples_into_a_pass():
    result=continuous_clearance(lambda t:.006,100.,max_queries=3)
    assert result["status"]=="pending" and result["unresolved_intervals"]
    assert result["reason"]=="SWEEP_SUBDIVISION_BUDGET_EXHAUSTED"


def test_continuous_interval_cover_proves_safe_geometry_and_rejects_gaps():
    proof=continuous_clearance(lambda t:.1,.2)
    assert proof["status"]=="passed" and len(proof["covered_intervals"])>1
    report={"version":SWEEP_VERSION,"gpu_executed":True,"status":"passed","restored":True,"containers":{"test_only":proof}}
    validate_sweep_evidence(report,["test_only"])
    report["containers"]["test_only"]["covered_intervals"].pop()
    with pytest.raises(ValueError,match="COVERAGE|GAP"):
        validate_sweep_evidence(report,["test_only"])


@pytest.mark.parametrize("value",[float('nan'),float('inf'),-1.])
def test_invalid_distance_never_means_sweep_passed(value):
    with pytest.raises(ValueError):continuous_clearance(lambda t:value,1.)


@pytest.mark.parametrize("mutate",["margin","bound","sample","container","gpu"])
def test_sweep_flags_do_not_replace_a_complete_numeric_proof(mutate):
    report={"version":SWEEP_VERSION,"gpu_executed":True,"status":"passed","restored":True,
            "containers":{"test_only":continuous_clearance(lambda t:.2,.1)}}
    proof=report["containers"]["test_only"]
    if mutate=="margin":proof["clearance_margin_m"]=0.
    if mutate=="bound":proof["motion_bound_m"]=float('nan')
    if mutate=="sample":proof["samples"][-1][1]=.005
    if mutate=="container":report["containers"]["extra"]=proof
    if mutate=="gpu":report["gpu_executed"]=False
    with pytest.raises(ValueError):validate_sweep_evidence(report,["test_only"])


def test_rigid_rotation_and_translation_use_joint_frame_not_object_center():
    pose=Pose.from_lists([2.,1.,0.])
    rotated=rigid_joint_pose(pose,"revolute",math.pi/2,[1.,1.,0.],[0.,0.,1.],1.)
    assert rotated.position==pytest.approx([1.,2.,0.])
    assert quat_rotate(rotated.orientation,[1.,0.,0.])==pytest.approx([0.,1.,0.])
    translated=rigid_joint_pose(pose,"prismatic",.4,[100.,0.,0.],[0.,2.,0.],.5)
    assert translated.position==pytest.approx([2.,1.2,0.])


@pytest.mark.parametrize("case",["drift","rotation","speed","spin","nan"])
def test_contact_and_area_are_not_a_stability_certificate(case):
    poses=np.tile([0.,0.,1.,0.,0.,0.,1.],(7,1))
    velocity,spin=np.zeros(3),np.zeros(3)
    if case=="drift":poses[3,0]=.01
    if case=="rotation":poses[3,3:]=[0.,0.,math.sin(math.radians(2)),math.cos(math.radians(2))]
    if case=="speed":velocity[0]=.1
    if case=="spin":spin[0]=.1
    if case=="nan":poses[3,0]=float('nan')
    if case=="nan":
        with pytest.raises(ValueError):placement_stability(poses,velocity,spin)
    else:assert not placement_stability(poses,velocity,spin)["passed"]


@pytest.mark.parametrize('angle,passed',[(0.,True),(.005,True),(2.1,False)])
def test_rotation_uses_normalized_quaternions_without_erasing_real_motion(angle,passed):
    q=np.array([0.,0.,math.sin(math.radians(angle)/2),math.cos(math.radians(angle)/2)])
    poses=[[0.,0.,1.,0.,0.,0.,.9999],[0.,0.,1.,*(-q*.9999)]]
    result=placement_stability(poses,[0.,0.,0.],[0.,0.,0.])
    assert result['rotation_drift_deg']==pytest.approx(angle,abs=1e-8)
    assert result['passed']==passed


def test_temporary_place_requires_actual_contact_stability_and_public_trace():
    from test_household_dataset import trace
    poses=[[0.,0.,1.,0.,0.,0.,1.]]*7
    report={"version":1,"gpu_executed":True,"status":"passed","restored":True,"receptacle":"unit_table",
        "trace":trace(["GRASP","PLACE"]),"relation":{"kind":"top_surface","satisfied":True,"fault":None},
        "physical_on_top_samples":[True]*6,"stability_steps":30,"stability_poses":poses,
        "final_linear_velocity":[0.,0.,0.],"final_angular_velocity":[0.,0.,0.],
        "stability":placement_stability(poses,[0.,0.,0.],[0.,0.,0.])}
    validate_temporary_place_evidence(report,{"unit_table"},16)
    bad=deepcopy(report);bad["physical_on_top_samples"][2]=False
    with pytest.raises(ValueError):validate_temporary_place_evidence(bad,{"unit_table"},16)
    bad=deepcopy(report);bad["trace"][0]["action"]["skill"]="OPEN"
    with pytest.raises(ValueError):validate_temporary_place_evidence(bad,{"unit_table"},16)
    bad=deepcopy(report);bad["stability_poses"][2]=[1.,0.,1.,0.,0.,0.,1.]
    with pytest.raises(ValueError):validate_temporary_place_evidence(bad,{"unit_table"},16)


def test_temporary_place_failure_restores_world_and_preserves_diagnostic(monkeypatch):
    """Minimal explicit unit backend; not a physics acceptance result."""
    from rummagebench.authoring.tasks.household_physics import verify_temporary_place
    from rummagebench.core import visual_session
    from rummagebench.authoring.tasks import certify
    ogstates=ModuleType('omnigibson.object_states');ogstates.OnTop=type('OnTop',(),{})
    monkeypatch.setitem(sys.modules,'omnigibson.object_states',ogstates)
    class UnitSession:
        def __init__(self,*a,**k):pass
        def reset(self):pass
    monkeypatch.setattr(visual_session,'VisualProtocolSession',UnitSession)
    monkeypatch.setattr(certify.TaskCertifier,'_drive',lambda *a:False)
    restored=[]
    backend=SimpleNamespace(capture_observe_state=lambda:{"unit_only_state":1},reset=lambda:None,restore_observe_state=lambda s:restored.append(s))
    plan=SimpleNamespace(target=SimpleNamespace(slot_id='s'))
    slots=SimpleNamespace(by_id=lambda:{'s':SimpleNamespace(parent_entity='cab')})
    result=verify_temporary_place(backend,None,plan,slots,None,None,{})
    assert result['status']=='failed' and result['reason']=='TEMPORARY_PLACE_TARGET_ROUTE_FAILED'
    assert result['restored'] is True and restored==[{"unit_only_state":1}]


def test_invalid_live_pose_cannot_fall_back_to_stale_usd_and_pass_sweep(monkeypatch):
    """Standalone USD and a fake runtime object, exclusively a unit fixture."""
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom
    from rummagebench.authoring.tasks.household_physics import verify_door_sweeps
    from rummagebench.feasibility import fcl_compat
    stage=Usd.Stage.CreateInMemory();prim=UsdGeom.Xform.Define(stage,'/unit_cab').GetPrim()
    obj=SimpleNamespace(name='unit_cab',prim=prim,links={},joints={},metadata={},
                        get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,0.]))
    scene=SimpleNamespace(objects=[obj],object_registry=lambda key,name:obj)
    backend=SimpleNamespace(_env=SimpleNamespace(scene=scene),_geom_cache={},robot_entity_names=lambda:set(),
        is_open=lambda name:False,capture_observe_state=lambda:{'unit_only_state':1},
        restore_observe_state=lambda state:None)
    from rummagebench.sim.omnigibson import entity_resolver
    monkeypatch.setattr(entity_resolver,'resolve_object',lambda *args:obj)
    monkeypatch.setattr(fcl_compat,'collision_backend',lambda:SimpleNamespace())
    report=verify_door_sweeps(backend,{'target_asset':{'internal_entity':'unit_target'},'candidate_bindings':[{'name':'unit_cab'}]})
    assert report['status']=='pending' and report['restored'] is True
    assert report['containers']['unit_cab']['reason']=='SWEEP_LIVE_BODY_POSE_INVALID:unit_cab'


@pytest.mark.kinematics
@pytest.mark.parametrize("obstacle,expected",[([.7,.7,0.],"failed"),([4.,4.,0.],"passed")])
def test_real_fcl_door_sweep_not_just_endpoint_boxes(obstacle,expected):
    from rummagebench.feasibility.fcl_compat import collision_backend,transform3
    try:coal=collision_backend()
    except ImportError:pytest.skip("optional coal/hpp-fcl needed for actual CPU collision geometry")
    body=coal.Box(1.8,.08,1.)
    stationary=coal.CollisionObject(coal.Box(.08,.08,.08),transform3(coal,obstacle))
    def distance(t):
        pose=rigid_joint_pose(Pose.from_lists([1.,0.,0.]),"revolute",math.pi/2,[0.,0.,0.],[0.,0.,1.],t)
        rotation=np.column_stack([quat_rotate(pose.orientation,np.eye(3)[i]) for i in range(3)])
        moving=coal.CollisionObject(body,transform3(coal,pose.position,rotation))
        collision=coal.CollisionResult()
        coal.collide(moving,stationary,coal.CollisionRequest(),collision)
        if collision.isCollision():return 0.
        return float(coal.distance(moving,stationary,coal.DistanceRequest(),coal.DistanceResult()))
    assert distance(0.)>.005 and distance(1.)>.005
    result=continuous_clearance(distance,2.*math.pi/2)
    assert result['status']==expected
