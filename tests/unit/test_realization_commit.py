"""R01 fault injection: realization commit discipline.

For GRASP / PLACE / OPEN / CLOSE, a backend that reports failure (or cannot
establish the postcondition) must produce an infrastructure fault — never a
committed semantic state, never a returned EXECUTED, never a half-committed
world.
"""

from pathlib import Path

import pytest

from conftest import FakeBackend

from rummagebench.core.errors import FeasibilityBackendError
from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(backend) -> BenchmarkSession:
    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = False
    scenario.termination.fail_on_wrong_grasp = False
    return BenchmarkSession(backend, scenario)


def _act(session, skill: str, value: str):
    kind = TargetKind.PLACE if skill == "NAV" else TargetKind.ENTITY
    return session.act(Action(skill=skill, target=TargetRef(type=kind, value=value)))


def _clone_world(backend_cls, fake: FakeBackend):
    """A fault-injecting backend of ``backend_cls`` over the same world."""
    backend = backend_cls(
        entities=fake.entities,
        anchors=fake.anchors,
        poses=fake.poses,
        aabbs=fake.aabbs,
        articulations=fake.articulations,
        link_poses=fake.link_poses,
        link_aabbs=fake.link_aabbs_map,
    )
    backend._last_anchor_position = getattr(fake, "_last_anchor_position", None)
    return backend


class _FailingGraspBackend(FakeBackend):
    """symbolic_grasp returns False; the world never believes it holds."""

    def symbolic_grasp(self, entity: str) -> bool:
        return False

    def is_holding(self, entity: str) -> bool:
        return False


class _RaisingGraspBackend(FakeBackend):
    """symbolic_grasp raises mid-realization (backend crash)."""

    def symbolic_grasp(self, entity: str) -> bool:
        raise RuntimeError("assisted-grasp joint exploded")


class _FailingPlaceBackend(FakeBackend):
    """symbolic_place reports failure after having released the joint
    (half-committed world: the rollback must undo the release)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.place_released = False

    def symbolic_place(self, entity: str, receptacle: str, at=None) -> bool:
        # mimic the real backend: release happens before the placement write
        self.place_released = True
        self.holding_entity = None  # half-commit: joint released
        return False  # ...but the placement write fails


class _FailingOpenBackend(FakeBackend):
    """set_open reports failure without changing the world."""

    def set_open(self, entity: str, open_value: bool) -> bool:
        return False


def test_grasp_backend_false_is_infrastructure_fault(fake_backend):
    backend = _clone_world(_FailingGraspBackend, fake_backend)
    session = _session(backend)
    session.reset()
    _act(session, "NAV", "kitchen")
    _act(session, "OPEN", "cabinet_B")
    with pytest.raises(FeasibilityBackendError, match="GRASP"):
        _act(session, "GRASP", "target_knife")
    # no pseudo-success: semantic state untouched, episode still running
    assert session.world_state.held_object is None
    assert session.status().value == "RUNNING"


def test_grasp_backend_exception_propagates_after_rollback(fake_backend):
    backend = _clone_world(_RaisingGraspBackend, fake_backend)
    session = _session(backend)
    session.reset()
    _act(session, "NAV", "kitchen")
    _act(session, "OPEN", "cabinet_B")
    with pytest.raises(RuntimeError, match="exploded"):
        _act(session, "GRASP", "target_knife")
    assert session.world_state.held_object is None


def test_place_backend_false_rolls_back_release(fake_backend):
    backend = _clone_world(_FailingPlaceBackend, fake_backend)
    session = _session(backend)
    session.reset()
    _act(session, "NAV", "kitchen")
    _act(session, "OPEN", "cabinet_B")
    _act(session, "GRASP", "target_knife")
    assert session.world_state.held_object == "target_knife"
    with pytest.raises(FeasibilityBackendError, match="PLACE"):
        _act(session, "PLACE", "cabinet_B")
    # the half-committed release was rolled back: still holding
    assert session.world_state.held_object == "target_knife"
    assert backend.holding_entity == "target_knife"
    assert backend.is_holding("target_knife")


def test_open_backend_false_is_infrastructure_fault(fake_backend):
    backend = _clone_world(_FailingOpenBackend, fake_backend)
    session = _session(backend)
    session.reset()
    _act(session, "NAV", "kitchen")
    with pytest.raises(FeasibilityBackendError, match="OPEN"):
        _act(session, "OPEN", "cabinet_B")
    assert not backend.is_open("cabinet_B")
    assert session.status().value == "RUNNING"


def test_visual_session_maps_commit_fault_to_engine_error(visual_session,
                                                          visual_backend,
                                                          monkeypatch):
    """The visual session's ENGINE_ERROR path must invalidate the run on a
    realization fault — the model never sees a verdict for that step."""
    from rummagebench.core.scenario import AnchorSpec

    monkeypatch.setattr(type(visual_backend), "symbolic_grasp",
                        lambda self, entity: False)
    monkeypatch.setattr(type(visual_backend), "is_holding",
                        lambda self, entity: False)
    # into reach of the spoon, container open (same recipe as the
    # acceptance tests so the grasp reaches the execution stage)
    visual_backend.teleport_robot(
        AnchorSpec(position=[3.0, 1.0, 0.0], orientation=[0.0, 0.0, 0.0, 1.0]))
    visual_backend.set_open("cabinet_B", True)
    point = {"frame_id": visual_session.observe().frame_id, "x": 0.79, "y": 0.5}
    with pytest.raises(RuntimeError, match="invalidated"):
        visual_session.step({"skill": "GRASP", "point": point})
    assert visual_session.status() == "ENGINE_ERROR"
