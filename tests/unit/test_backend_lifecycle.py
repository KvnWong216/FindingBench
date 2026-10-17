"""Object loading must finish before physics/rendering resumes."""
import sys
import types
import pytest

from rummagebench.sim.omnigibson import backend as module


@pytest.mark.parametrize("frozen", [False, True])
def test_setup_loads_all_objects_before_resuming(monkeypatch, frozen):
    events = []
    imported = []
    class Sim:
        playing = True
        def is_playing(self):
            return self.playing
        def stop(self):
            events.append("stop")
            self.playing = False
        def play(self):
            assert len(imported) == 2
            events.append("play")
            self.playing = True
        def step(self):
            assert self.playing and len(imported) == 2
            events.append("step")
    sim = Sim()
    class Obj:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
        def set_position_orientation(self, **kwargs):
            assert not sim.playing
    def add_object(obj):
        assert not sim.playing
        imported.append(obj)
        events.append("add")
    scene = types.SimpleNamespace(robots=[object()], objects=[], add_object=add_object)
    og = types.ModuleType("omnigibson")
    og.sim = sim
    objects = types.ModuleType("omnigibson.objects")
    objects.DatasetObject = Obj
    states = types.ModuleType("omnigibson.object_states")
    states.Inside = type("Inside", (), {})
    states.OnTop = type("OnTop", (), {})
    states.Open = type("Open", (), {})
    for mod in (og, objects, states):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(module, "build_env", lambda _: (types.SimpleNamespace(scene=scene), og))
    monkeypatch.setattr(module, "seed_everything", lambda _: None)
    monkeypatch.setattr(module, "apply_runtime_env", lambda: None)
    monkeypatch.setattr(module, "dump_state", lambda _: {})
    backend = module.OmniGibsonBackend()
    monkeypatch.setattr(backend, "_validate_spawn_clearance", lambda: None)
    monkeypatch.setattr(backend, "_seed_pose_for", lambda _: [0, 0, 1])
    monkeypatch.setattr(backend, "teleport_robot", lambda _: None)
    monkeypatch.setattr(backend, "robot_pose", lambda: ([0, 0, 0], [0, 0, 0, 1]))
    anchor = types.SimpleNamespace(position=[0, 0, 0], orientation=[0, 0, 0, 1])
    scenario = types.SimpleNamespace(
        scene=types.SimpleNamespace(model="synthetic", frozen_member_manifest="unit_member.json" if frozen else None),
        robot=types.SimpleNamespace(init_anchor="start"),
        objects=[types.SimpleNamespace(name=n, category="cube", model="fixed", fixed_base=False)
                 for n in ("one", "two")],
        placements=[], initial_states={}, anchors={"start": anchor, "legacy": anchor})
    if frozen:
        from rummagebench.authoring.tasks import household_state
        member = {"parking": [100.,100.,10.], "builder": {"placements": [{"verified": True}]},
                  "manifest_sha256": "synthetic unit evidence only"}
        monkeypatch.setattr(household_state, "load_frozen_member", lambda *args: member)
        def restore(*args):
            assert sim.playing and len(imported) == 2
            events.append("restore_member")
            return member
        monkeypatch.setattr(household_state, "restore_frozen_member", restore)
        monkeypatch.setattr(backend, "_apply_placements", lambda *a, **kw: pytest.fail("frozen member was resampled"))
    report = backend.setup(scenario)
    assert events[:4] == ["stop", "add", "add", "play"]
    assert len(report["spawned"]) == 2
    assert report["anchor_validation_scope"] == "initial_spawn"
    assert report["untested_anchors"] == ["legacy"]
    assert len(scenario.anchors) == 2
    if frozen:
        assert events.count("restore_member") == 2
        assert report["frozen_member_manifest_sha256"] == member["manifest_sha256"]

def test_corrupt_robot_pose_never_returns_commanded_anchor():
    import numpy as np
    import pytest
    from rummagebench.core.errors import SimBackendError
    backend=module.OmniGibsonBackend()
    backend._commanded_pose=([1,2,3],[0,0,0,1])
    backend._robot=types.SimpleNamespace(get_position_orientation=lambda **kw: (np.zeros(3),np.full(4,np.nan)))
    with pytest.raises(SimBackendError): backend.robot_pose()
    def invalid(**kw): raise AssertionError("invalid physical quaternion")
    backend._robot.get_position_orientation=invalid
    with pytest.raises(SimBackendError): backend.robot_pose()


def test_world_validation_rejects_nonfinite_articulation():
    import numpy as np
    import pytest
    from rummagebench.core.errors import SimBackendError
    obj=types.SimpleNamespace(
        name='unit_private_articulation',
        get_position_orientation=lambda:(np.zeros(3),np.array([0,0,0,1])),
        n_joints=1,get_joint_positions=lambda:np.array([np.nan]),links={})
    backend=module.OmniGibsonBackend()
    backend._env=types.SimpleNamespace(scene=types.SimpleNamespace(objects=[obj]))
    with pytest.raises(SimBackendError) as error:backend.validate_physics_state()
    assert 'unit_private_articulation' not in str(error.value)
    assert backend._last_physics_failure=={'entity':'unit_private_articulation',
        'component':'joint_positions','reason':'nonfinite joint positions'}


def test_unmapped_static_mesh_uses_real_geometry_for_navigation(monkeypatch):
    from rummagebench.sim.omnigibson import usd_collision
    b=module.OmniGibsonBackend()
    b._scenario=types.SimpleNamespace(scene=types.SimpleNamespace(source='infinigen'))
    b._env=types.SimpleNamespace(scene='unit_scene',_findingbench_floor_z=.1,
        _findingbench_assembly_manifest={'objects':[{'name':'unit_skirting','role':'static_background','source_component':None}]})
    position=[0,0,0]
    obj=types.SimpleNamespace(scale=[1,1,1],links={'base':types.SimpleNamespace(scale=[1,1,1],
        get_position_orientation=lambda:(position,[0,0,0,1]))})
    monkeypatch.setattr(module,'resolve_object',lambda *a:obj)
    called=[]
    monkeypatch.setattr(usd_collision,'static_architecture_navigation_aabbs',lambda o,z:called.append((o,z)) or [])
    assert b.navigation_obstacle_aabbs('unit_skirting')==[] and called==[(obj,.1)]
    assert b.navigation_obstacle_aabbs('unit_skirting')==[] and len(called)==1
    position[0]=1
    assert b.navigation_obstacle_aabbs('unit_skirting')==[] and len(called)==2
    obj.scale[0]=2
    assert b.navigation_obstacle_aabbs('unit_skirting')==[] and len(called)==3


def test_spawn_rejection_records_blocker_only_in_private_diagnostic(monkeypatch):
    from rummagebench.core.errors import SimBackendError
    b=module.OmniGibsonBackend()
    monkeypatch.setattr(b,'robot_pose',lambda:([0,0,0],[0,0,0,1]))
    monkeypatch.setattr(b,'robot_entity_names',lambda:set())
    monkeypatch.setattr(b,'entity_names',lambda:['unit_private_blocker'])
    monkeypatch.setattr(b,'describe_entity',lambda n:object())
    monkeypatch.setattr(b,'is_holding',lambda n:False)
    monkeypatch.setattr(b,'navigation_obstacle_aabbs',lambda n:[([-.1,-.1,0],[.1,.1,1])])
    with pytest.raises(SimBackendError) as exc:b._validate_spawn_clearance()
    assert 'unit_private_blocker' not in str(exc.value)
    assert b._last_spawn_failure['blockers'][0]['entity']=='unit_private_blocker'


def test_sdk_shutdown_system_exit_does_not_mask_physics_failure():
    import pytest
    from rummagebench.core.errors import SimBackendError
    backend=module.OmniGibsonBackend()
    def shutdown():raise SystemExit(0)
    backend._og=types.SimpleNamespace(shutdown=shutdown)
    backend._env=object()
    with pytest.raises(SimBackendError,match='unit physics failure'):
        try:raise SimBackendError('unit physics failure')
        finally:backend.close()
    assert backend._env is None

def test_unverified_nav_fails_before_mutating_world():
    from rummagebench.skills.nav import NavSkill
    from rummagebench.sim.base import ResolvedTarget
    from rummagebench.core.types import TargetKind
    backend=module.OmniGibsonBackend()
    anchor=types.SimpleNamespace(position=[0,0,0],orientation=[0,0,0,1])
    result=NavSkill().execute(backend,ResolvedTarget(kind=TargetKind.PLACE,place="unchecked",anchor=anchor),None)
    assert not result.executed
    assert result.details["reason"]=="UNVALIDATED_ANCHOR"
    backend._validated_anchor_poses["checked"]=((0,0,0),(0,0,0,1))
    assert backend.is_anchor_validated("checked",anchor)
    anchor.position[0]=1
    assert not backend.is_anchor_validated("checked",anchor)

def test_initial_spawn_config_uses_world_anchor():
    from rummagebench.sim.omnigibson.env_factory import build_env_config
    from rummagebench.core.scenario import load_scenario
    scenario = load_scenario("scenarios/knife_search_001/scenario.yaml")
    config = build_env_config(scenario)["robots"][0]
    anchor = scenario.anchors[scenario.robot.init_anchor]
    assert config["position"] == list(anchor.position)
    assert config["orientation"] == list(anchor.orientation)
    assert config["pose_frame"] == "world"

def test_spawn_collision_rejected_before_snapshot(monkeypatch):
    import pytest
    from rummagebench.core.errors import SimBackendError
    from rummagebench.skills import move
    backend = module.OmniGibsonBackend()
    monkeypatch.setattr(backend, "robot_pose", lambda: ([0,0,.0053],[0,0,0,1]))
    monkeypatch.setattr(move, "base_pose_collision_free", lambda *a: False)
    with pytest.raises(SimBackendError, match="invalid spawn"):
        backend._validate_spawn_clearance()


@pytest.mark.parametrize('category,fixed,walkable', [
    ('floors', True, True), ('paver', True, True),
    ('paver', False, False), ('walls', True, False),
    ('stairs', True, False), ('coffee_table', True, False),
])
def test_native_ground_support_does_not_block_base_skills(monkeypatch, category, fixed, walkable):
    from rummagebench.sim.omnigibson import usd_collision
    from rummagebench.skills.move import execute_move, execute_turn
    box = ([-2., -2., -.2], [2., 2., .16])
    obj = types.SimpleNamespace(category=category, fixed_base=fixed, links={})
    backend = module.OmniGibsonBackend()
    backend._scenario = types.SimpleNamespace(behavior_task=object())
    backend._env = types.SimpleNamespace(scene=object())
    monkeypatch.setattr(module, 'resolve_object', lambda *args: obj)
    monkeypatch.setattr(usd_collision, 'collider_navigation_aabbs', lambda obj: [box])
    monkeypatch.setattr(backend, 'entity_names', lambda: ['surface'])
    monkeypatch.setattr(backend, 'describe_entity', lambda name: {})
    monkeypatch.setattr(backend, 'robot_entity_names', lambda: set())
    monkeypatch.setattr(backend, 'is_holding', lambda name: False)
    poses = []
    monkeypatch.setattr(backend, 'teleport_robot', poses.append)
    pose = ([0., 0., 0.], [0., 0., 0., 1.])
    assert execute_move(backend, pose, .2, .4, .05, .03)[0] is walkable
    assert execute_turn(backend, pose, 30., .4, 5., .03)[0] is walkable
    assert len(poses) == (2 if walkable else 0)
