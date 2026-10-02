"""§42-§43: certification pipeline contracts and pure validation checks.

Stage B/C/D/E measurements are produced by the GPU certify script; the
checks here are pure functions so they are unit-testable on CPU and cannot
drift between generator and simulator. OCCUPANCY/STRUCTURAL VALIDITY IS
NEVER SUFFICIENT (§24) — every acceptance decision consumes simulator
measurements.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RejectionCode(enum.Enum):
    """§43: generator diagnostics — NEVER agent-facing feedback."""
    ASSET_LOAD_FAILED = "ASSET_LOAD_FAILED"
    INVALID_ASSET_GEOMETRY = "INVALID_ASSET_GEOMETRY"
    NO_CONTAINER_FIT = "NO_CONTAINER_FIT"
    OCCUPANCY_PACK_FAILED = "OCCUPANCY_PACK_FAILED"
    PHYSICS_UNSTABLE = "PHYSICS_UNSTABLE"
    OBJECT_ESCAPED = "OBJECT_ESCAPED"
    OBJECT_PENETRATION = "OBJECT_PENETRATION"
    TARGET_RELATION_LOST = "TARGET_RELATION_LOST"
    TARGET_VISIBILITY_OUT_OF_RANGE = "TARGET_VISIBILITY_OUT_OF_RANGE"
    COVER_RELATION_FAILED = "COVER_RELATION_FAILED"
    ROBOT_SPAWN_FAILED = "ROBOT_SPAWN_FAILED"
    ROBOT_CAMERA_FAILED = "ROBOT_CAMERA_FAILED"
    TARGET_EVENTUALLY_UNREACHABLE = "TARGET_EVENTUALLY_UNREACHABLE"
    RESET_MISMATCH = "RESET_MISMATCH"
    # infrastructure (not scene semantics): retried, then INVALID_RUN
    SIM_CRASH = "SIM_CRASH"
    WITNESS_NOT_FOUND = "WITNESS_NOT_FOUND"


@dataclass
class SettleMeasurement:
    """§8: measured settling statistics (simulator seconds + steps)."""
    sim_seconds: float
    steps: int
    final_max_linear_velocity_mps: float
    final_max_angular_velocity_rps: float
    stable_consecutive_frames: int
    all_finite: bool


def check_settle(measurement: SettleMeasurement,
                 stable_linear_mps: float, stable_angular_rps: float,
                 stable_window_frames: int, max_settle_frames: int,
                 ) -> tuple[bool, RejectionCode | None]:
    """§23/rev§8: stability is necessary, not sufficient. Contacts in a pile
    are legitimate; velocities must be under threshold for the configured
    window and the state must be finite."""
    if not measurement.all_finite:
        return False, RejectionCode.PHYSICS_UNSTABLE
    if measurement.steps >= max_settle_frames and (
            measurement.final_max_linear_velocity_mps > stable_linear_mps
            or measurement.final_max_angular_velocity_rps > stable_angular_rps):
        return False, RejectionCode.PHYSICS_UNSTABLE
    if measurement.stable_consecutive_frames < stable_window_frames:
        return False, RejectionCode.PHYSICS_UNSTABLE
    if (measurement.final_max_linear_velocity_mps > stable_linear_mps
            or measurement.final_max_angular_velocity_rps > stable_angular_rps):
        return False, RejectionCode.PHYSICS_UNSTABLE
    return True, None


def check_post_settle_world(world: dict[str, Any], expected_names: set[str],
                            workspace_radius_m: float,
                            penetration_tol_m: float,
                            ) -> tuple[bool, RejectionCode | None]:
    """§24 exact post-settle validation over simulator measurements.

    world: {"poses": {name: {"position": [x,y,z], "finite": bool}},
            "contacts_valid": bool, "support_container_stable": bool,
            "target_present": bool, "max_penetration_m": float}
    """
    poses = world.get("poses", {})
    if set(poses) != expected_names:
        return False, RejectionCode.OBJECT_ESCAPED
    for name, pose in poses.items():
        if not pose.get("finite", False):
            return False, RejectionCode.PHYSICS_UNSTABLE
        pos = pose["position"]
        if abs(pos[0]) > workspace_radius_m or abs(pos[1]) > workspace_radius_m:
            return False, RejectionCode.OBJECT_ESCAPED
        if pos[2] < -penetration_tol_m:
            return False, RejectionCode.OBJECT_ESCAPED
    if world.get("max_penetration_m", 0.0) > penetration_tol_m:
        return False, RejectionCode.OBJECT_PENETRATION
    if not world.get("support_container_stable", False):
        return False, RejectionCode.TARGET_RELATION_LOST
    if not world.get("target_present", False):
        return False, RejectionCode.TARGET_RELATION_LOST
    return True, None


def check_relations(world: dict[str, Any], planned_inside: set[str],
                    planned_cover: set[str],
                    ) -> tuple[bool, RejectionCode | None]:
    """§24: intended INSIDE/COVER relations must survive settling."""
    inside = set(world.get("inside_objects", []))
    if planned_inside and not inside.issuperset(planned_inside):
        return False, RejectionCode.TARGET_RELATION_LOST
    covers = set(world.get("cover_relations", []))
    if planned_cover and not planned_cover.issubset(covers):
        return False, RejectionCode.COVER_RELATION_FAILED
    return True, None


def check_visibility(ratio: float | None, band: tuple[float, float],
                     numerator: int | None, denominator: int | None,
                     ) -> tuple[bool, RejectionCode | None, str]:
    """§27/§28/rev§13: visibility from private renderer segmentation; the
    ratio is valid only when the isolated reference projects measurably.
    An undefined measurement is invalid, not zero."""
    if ratio is None or denominator is None or numerator is None:
        return False, RejectionCode.TARGET_VISIBILITY_OUT_OF_RANGE, "undefined"
    if denominator <= 0:
        return False, RejectionCode.TARGET_VISIBILITY_OUT_OF_RANGE, "empty_reference"
    if not (band[0] - 1e-9 <= ratio <= band[1] + 1e-9):
        return False, RejectionCode.TARGET_VISIBILITY_OUT_OF_RANGE, "out_of_band"
    return True, None, "ok"


@dataclass
class StageResult:
    stage: str                     # A|B|C|D|E
    passed: bool
    rejection: RejectionCode | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentCertificate:
    environment_id: str
    environment_seed: int
    paradigm: str
    code_commit: str
    dataset_version: str
    grounding_version: str
    physics_profile_hash: str
    stages: list[StageResult] = field(default_factory=list)
    accepted: bool = False
    rejection: RejectionCode | None = None
    canonical_state_hash: str | None = None

    def record(self, result: StageResult) -> None:
        self.stages.append(result)
        if not result.passed:
            self.accepted = False
            self.rejection = result.rejection


@dataclass
class RobotCertificate:
    robot_id: str
    environment_id: str
    loads: bool
    camera_valid: bool
    local_spawn_found: bool
    workspace_observable: bool
    target_stage_a_feasible_after_clutter_removal: bool
    details: dict[str, Any] = field(default_factory=dict)

    def passed(self) -> bool:
        return all([self.loads, self.camera_valid, self.local_spawn_found,
                    self.workspace_observable,
                    self.target_stage_a_feasible_after_clutter_removal])


@dataclass
class WitnessRecord:
    """§10/rev§10: a recorded protocol-solvable witness.

    Every executed action went through the production AGENT interface; the
    certifier may inspect private state for planning but may not delete
    blockers, disable collision, inject hidden targets, teleport-navigate,
    or inflate the action budget.
    """
    environment_id: str
    robot_id: str
    success: bool
    actions: list[dict] = field(default_factory=list)
    frame_provenance: list[str] = field(default_factory=list)
    private_resolved_targets: list[dict] = field(default_factory=list)
    gate_outcomes: list[dict] = field(default_factory=list)
    goal_evidence: dict[str, Any] = field(default_factory=dict)
    length: int = 0
    grounding_version: str = ""
    replayed_from_fresh_reset: bool = False
