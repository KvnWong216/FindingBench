"""§21.8 / acceptance E: benchmark-owned holding state.

GRASP A -> held=A; GRASP B while holding -> INVALID_STATE; PLACE -> held
None; GRASP B then valid. The truth lives in BenchmarkWorldState — never in
the backend's assisted-grasp internals — and task success reads it.
"""

from pathlib import Path

from conftest import FakeBackend

from rummagebench.core.scenario import load_scenario
from rummagebench.core.session import BenchmarkSession
from rummagebench.core.types import Action, TargetKind, TargetRef

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"


def _session(fake_backend) -> BenchmarkSession:
    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = False
    scenario.termination.fail_on_wrong_grasp = False
    return BenchmarkSession(fake_backend, scenario)


def _act(session, skill: str, value: str):
    kind = TargetKind.PLACE if skill == "NAV" else TargetKind.ENTITY
    return session.act(Action(skill=skill, target=TargetRef(type=kind, value=value)))


def test_holding_state_lifecycle(fake_backend):
    session = _session(fake_backend)
    session.reset()

    assert session.world_state.held_object is None
    _act(session, "NAV", "kitchen")
    _act(session, "OPEN", "cabinet_B")

    # GRASP A -> held=A (benchmark state, regardless of backend realization)
    r = _act(session, "GRASP", "target_knife")
    assert r.executed
    assert session.world_state.held_object == "target_knife"
    assert session.world_state.held_count == 1

    # GRASP B while holding -> structured INVALID_STATE, state unchanged
    r = _act(session, "GRASP", "distractor_spoon")
    assert not r.executed
    assert r.failure_reason.value == "INVALID_STATE"
    assert session.world_state.held_object == "target_knife"

    # PLACE -> held=None
    r = _act(session, "PLACE", "cabinet_B")
    assert r.executed
    assert session.world_state.held_object is None
    assert session.world_state.held_count == 0

    # GRASP B is valid again
    r = _act(session, "GRASP", "distractor_spoon")
    assert r.executed
    assert session.world_state.held_object == "distractor_spoon"


def test_success_is_evaluated_against_benchmark_state(fake_backend):
    """Even when the assisted-grasp realization fails, the benchmark state
    (and only it) decides task success."""
    scenario = load_scenario(FIXTURE)
    scenario.termination.succeed_when_holding_target = True
    backend = _BreakingGraspBackend.from_fake(fake_backend)
    session = BenchmarkSession(backend, scenario)
    session.reset()
    _act(session, "NAV", "kitchen")
    _act(session, "OPEN", "cabinet_B")
    r = _act(session, "GRASP", "target_knife")
    assert r.executed
    assert session.world_state.held_object == "target_knife"
    assert session.status().value == "SUCCESS"


class _BreakingGraspBackend(FakeBackend):
    """FakeBackend whose grasp REALIZATION always fails (benchmark truth
    must not care)."""

    @classmethod
    def from_fake(cls, fake: FakeBackend):
        backend = cls(
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

    def symbolic_grasp(self, entity: str) -> bool:
        self.grasp_base_pose = self.robot_pose()
        return False  # realization breaks

    def is_holding(self, entity: str) -> bool:
        return False  # simulator never believes it holds anything
