"""SearchStructureCertificate: the partial-observation search structure of a task.

Measures partial-observation SEARCH structure; complementary to the oracle
d* certificate (full-information EXECUTION depth), which stays untouched.

Two layers:
  * structural fields — computed on CPU from the TaskPlan + SceneSlots +
    SlotEmbodimentOverlay (status "structural");
  * SimulatorEvidence — measured by the GPU certifier (visibility, reveal
    causality, graspability, oracle solvability, replay). With evidence
    attached and every gate passed, status becomes "certified".
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

CERTIFICATE_VERSION = 1


@dataclass
class SimulatorEvidence:
    """GPU-measured facts. None = not measured (never treated as a pass)."""

    builder_ok: Optional[bool] = None
    placements_verified: Optional[bool] = None
    settle_ok: Optional[bool] = None
    # target visibility ratio from the start pose
    start_visibility: Optional[float] = None
    # target visibility with the target container still closed, measured
    # from the reveal viewpoint (proves OPEN is causally necessary)
    closed_visibility: Optional[float] = None
    # GRASP(target) physically feasible before / after the tier's minimal
    # required reveal (OPEN for easy in_container, rearrangement for medium)
    pre_reveal_graspable: Optional[bool] = None
    post_reveal_graspable: Optional[bool] = None
    # visibility after the minimal required reveal
    revealed_visibility: Optional[float] = None
    # candidate slots the embodiment can actually serve
    reachable_candidate_slot_count: Optional[int] = None
    # measured minimal rearrangement (easy must be 0)
    rearrangement_depth: Optional[int] = None
    oracle_solvable: Optional[bool] = None
    oracle_depth: Optional[int] = None
    # non-target objects the full-information oracle plan moves (GRASPs of a
    # non-target): medium requires the ORACLE to uncover too, not only the
    # RGB witness — an invisible target the oracle can side-grasp under its
    # cover makes d* understate the task (A.12)
    oracle_rearrangement_depth: Optional[int] = None
    replay_success: Optional[bool] = None
    # planning steps of the certified feasible execution
    certified_execution_steps: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchStructureCertificate:
    tier_proposed: str
    mode: str
    room_count: int
    candidate_slot_count: int
    containment_depth: int
    open_depth: int
    rearrangement_depth: int
    lookalike_count: int
    target_slot_type: str
    # full-information task-structure lower bound, in planning steps
    navigation_lower_bound: int
    minimum_required_interactions: int
    # filled from SimulatorEvidence once measured
    start_visibility: Optional[float] = None
    revealed_visibility: Optional[float] = None
    reachable_candidate_slot_count: Optional[int] = None
    evidence: Optional[SimulatorEvidence] = None
    status: str = "structural"  # structural | certified | rejected
    tier_certified: Optional[str] = None
    candidate_slots: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)
    version: int = CERTIFICATE_VERSION

    def attach_evidence(self, evidence: SimulatorEvidence) -> None:
        self.evidence = evidence
        self.start_visibility = evidence.start_visibility
        self.revealed_visibility = evidence.revealed_visibility
        self.reachable_candidate_slot_count = evidence.reachable_candidate_slot_count
        if evidence.rearrangement_depth is not None:
            self.rearrangement_depth = evidence.rearrangement_depth

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = None if self.evidence is None else self.evidence.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchStructureCertificate":
        d = dict(d)
        ev = d.pop("evidence", None)
        cert = cls(**d)
        cert.evidence = None if ev is None else SimulatorEvidence(**ev)
        return cert


# ---------------------------------------------------------------------------
# horizon lower bound
# ---------------------------------------------------------------------------

def navigation_lower_bound(start_xy: tuple[float, float], start_yaw: float,
                           goal_xy: tuple[float, float], goal_yaw: float,
                           max_move_m: float, max_turn_deg: float,
                           tolerance_m: float = 0.05,
                           tolerance_deg: float = 5.0) -> int:
    """Admissible lower bound on MOVE/TURN planning steps start -> goal pose.

    MOVE translates along +/- heading only (<= max_move_m per step), TURN
    rotates in place (<= max_turn_deg per step). The straight-line distance
    under-estimates any collision-free path, and at most one heading change
    is assumed for the whole trip, so the result never exceeds the true
    optimum (it is a pruning bound, not an estimate).
    """
    dx, dy = goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]
    dist = math.hypot(dx, dy)
    moves = 0 if dist <= tolerance_m else math.ceil((dist - tolerance_m) / max_move_m)
    turns = 0
    if moves:
        travel = math.degrees(math.atan2(dy, dx))
        # forward or backward travel both work: compare modulo 180 deg
        mis = abs(_wrap_deg(travel - math.degrees(start_yaw)))
        mis = min(mis, 180.0 - mis)
        if mis > tolerance_deg:
            turns = 1
    else:
        mis = abs(_wrap_deg(math.degrees(goal_yaw - start_yaw)))
        if mis > tolerance_deg:
            turns = math.ceil(mis / max_turn_deg)
    return moves + turns


def minimum_required_interactions(open_depth: int, rearrangement_depth: int,
                                  navigation_steps: int) -> int:
    """Required OPEN + distractor GRASP/PLACE pairs + target GRASP +
    REPORT_DONE + minimum navigation allowance."""
    return (int(open_depth) + 2 * int(rearrangement_depth) + 1 + 1
            + int(navigation_steps))


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
