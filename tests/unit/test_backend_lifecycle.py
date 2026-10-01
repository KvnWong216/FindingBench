"""Object loading must finish before physics/rendering resumes."""
import sys
import types

from rummagebench.sim.omnigibson import backend as module


def test_setup_loads_all_objects_before_resuming(monkeypatch):
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
        scene=types.SimpleNamespace(model="synthetic"),
        robot=types.SimpleNamespace(init_anchor="start"),
        objects=[types.SimpleNamespace(name=n, category="cube", model="fixed", fixed_base=False)
                 for n in ("one", "two")],
        placements=[], initial_states={}, anchors={"start": anchor, "legacy": anchor})
    report = backend.setup(scenario)
    assert events[:4] == ["stop", "add", "add", "play"]
    assert len(report["spawned"]) == 2
    assert report["anchor_validation_scope"] == "initial_spawn"
    assert report["untested_anchors"] == ["legacy"]
    assert len(scenario.anchors) == 2

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
        get_position_orientation=lambda:(np.zeros(3),np.array([0,0,0,1])),
        n_joints=1,get_joint_positions=lambda:np.array([np.nan]),links={})
    backend=module.OmniGibsonBackend()
    backend._env=types.SimpleNamespace(scene=types.SimpleNamespace(objects=[obj]))
    with pytest.raises(SimBackendError):backend.validate_physics_state()

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
