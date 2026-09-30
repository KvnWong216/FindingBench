"""Default-adapter renderer instance grounding (raw_instance integration).

CPU tests with a faked replicator/Kit surface: the verified raw renderer-ID
recipe must install on the shared default path (Python API / MCP / UI) and
produce complete, real-ID frames through capture_visual_frame.
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from rummagebench.sim.omnigibson.raw_instance import (
    GROUNDING_MODALITIES,
    enable_renderer_grounding,
    install_physics_only_settle,
)

HEAD = "zed_link"
WRIST = "wrist_realsense"


class FakeSensor:
    def __init__(self, position=(0.1, 0.2, 0.3)):
        self.modalities = ["rgb"]
        self.render_product = object()
        self._position = np.asarray(position, dtype=float)
        self.camera_parameters = {
            # USD world-to-camera view transform of the identity pose,
            # flattened; capture reshapes and transposes before comparing
            # against inverse(world).
            "cameraViewTransform": np.linalg.inv(_world(self._position)).T.reshape(-1),
        }

    def add_modality(self, modality):
        self.modalities.append(modality)

    def get_position_orientation(self):
        return self._position, np.array([0.0, 0.0, 0.0, 1.0])

    @property
    def intrinsic_matrix(self):
        return np.array([[224.0, 0, 112], [0, 224.0, 112], [0, 0, 1.0]])


def _world(position):
    world = np.eye(4)
    world[:3, 3] = position
    return world


class FakeAnnotator:
    def __init__(self, raw):
        self.raw = raw
        self.attached = []

    def attach(self, render_products):
        self.attached.extend(render_products)

    def get_data(self):
        return self.raw


RAW = {
    "data": np.full((4, 4), 5, dtype=np.uint32),
    "info": {"idToLabels": {"5": "/World/counter/front", "9": "INVALID"}},
}


@pytest.fixture
def grounding_world(monkeypatch):
    """Fake backend + replicator module surface for the grounding install."""
    zed = FakeSensor()
    wrist = FakeSensor()

    def _camera_pose(sensor_name):
        pos, quat = (zed if sensor_name == HEAD else wrist).get_position_orientation()
        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)

    backend = SimpleNamespace(
        _env=SimpleNamespace(
            get_obs=lambda: (
                [{"robot_0": {WRIST: {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
                              HEAD: {"rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                                     "depth_linear": np.full((4, 4), 1.0)}}}],
                {},
            ),
            scene=SimpleNamespace(
                objects=[SimpleNamespace(prim_path="/World/counter", name="cabinet_A")]
            ),
        ),
        _robot=SimpleNamespace(name="robot_0", sensors={HEAD: zed, WRIST: wrist}),
        _sim=SimpleNamespace(
            renders=0,
            render=lambda self=None: None,
            editing_usd=lambda: _nullcontext(),
            render_on_step=lambda flag: _nullcontext(),
        ),
        validate_physics_state=lambda: None,
        _camera_pose=_camera_pose,
    )
    annotator = FakeAnnotator(RAW)

    omni = types.ModuleType("omni")
    replicator = types.ModuleType("omni.replicator")
    core = types.ModuleType("omni.replicator.core")
    core.AnnotatorRegistry = SimpleNamespace(
        get_annotator=lambda name: annotator
    )
    omni.replicator = replicator
    replicator.core = core
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)
    return backend, zed, wrist, annotator


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_grounding_installs_on_head_sensor_identity(grounding_world):
    backend, zed, wrist, annotator = grounding_world
    name = enable_renderer_grounding(backend)
    assert name == HEAD
    assert annotator.attached == [zed.render_product]  # never the wrist camera
    assert zed.modalities == ["rgb", "depth_linear"]
    assert wrist.modalities == ["rgb"]


def test_installed_capture_returns_real_renderer_frame(grounding_world):
    backend, _zed, _wrist, _annotator = grounding_world
    enable_renderer_grounding(backend)
    frame = backend.capture_visual_frame()
    assert frame.meta["sensor_name"] == HEAD
    np.testing.assert_array_equal(frame.instance_segmentation, RAW["data"])
    # renderer IDs copied, labels canonicalized to the owning scene object
    assert frame.meta["instance_source"] == "renderer:instance_id_segmentation_fast"
    assert frame.meta["instance_labels"] == {"5": "cabinet_A", "9": "unlabelled"}
    assert frame.meta["visual_grounding_supported"] is True
    assert frame.meta["bridge"] == "segmentation"
    assert frame.meta["unsupported_reason"] == ""
    # ownership snapshot for instance_to_entity diagnostics
    assert backend._instance_labels == {"5": "cabinet_A", "9": "unlabelled"}


def test_physics_only_settle_never_renders(grounding_world):
    backend, _zed, _wrist, _annotator = grounding_world
    steps, renders, flags = [], [], []
    backend._sim.step = lambda: steps.append(1)
    backend._sim.render = lambda: renders.append(1)

    def render_on_step(flag):
        flags.append(flag)
        return _nullcontext()

    backend._sim.render_on_step = render_on_step
    backend._settle_steps = 3
    install_physics_only_settle(backend)
    backend.settle()
    assert len(steps) == 3 and renders == [] and flags == [False]
    backend.settle(1)
    assert len(steps) == 4


def test_build_env_config_routes_grounding_modalities_to_raw_capture():
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.env_factory import build_env_config

    scenario = load_scenario("tests/unit/fixtures/mini.yaml")
    scenario.robot.obs_modalities = ["rgb", "depth_linear", "seg_instance"]
    assert GROUNDING_MODALITIES & set(scenario.robot.obs_modalities)
    config = build_env_config(scenario)
    # OG never receives the crashing instance chain; the raw capture supplies it
    assert config["robots"][0]["obs_modalities"] == ["rgb", "depth_linear"]


def test_build_env_config_keeps_non_grounding_modalities():
    from rummagebench.core.scenario import load_scenario
    from rummagebench.sim.omnigibson.env_factory import build_env_config

    scenario = load_scenario("tests/unit/fixtures/mini.yaml")
    scenario.robot.obs_modalities = ["rgb", "depth_linear"]
    config = build_env_config(scenario)
    assert config["robots"][0]["obs_modalities"] == ["rgb", "depth_linear"]


def test_prepare_renderer_grounding_restricts_r1pro_sensors():
    from rummagebench.adapters.python_api import prepare_renderer_grounding
    from rummagebench.core.scenario import load_scenario

    scenario = load_scenario("tests/unit/fixtures/mini.yaml")
    scenario.robot.obs_modalities = ["rgb", "seg_instance"]
    assert scenario.robot.include_sensor_names is None
    assert prepare_renderer_grounding(scenario) is True
    assert scenario.robot.include_sensor_names == ["zed_link"]

    rgb_only = load_scenario("tests/unit/fixtures/mini.yaml")
    assert prepare_renderer_grounding(rgb_only) is False
    assert rgb_only.robot.include_sensor_names is None
