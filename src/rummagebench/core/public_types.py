"""Final visual interaction protocol — public types (AGENT mode).

The evaluated agent receives ONLY: visual observations, the task instruction,
the public skill library, and four-class feedback from its own previous
action. Everything else (entity names, poses, segmentation, admissible action
lists, oracle state) stays evaluator-private.

    AGENT CHOOSES FIRST. SIMULATOR GROUNDS AND VALIDATES SECOND.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------- skill library

PUBLIC_SKILLS = [
    "MOVE_TO",
    "TURN",
    "OPEN",
    "CLOSE",
    "GRASP",
    "PLACE",
    "VIEW",
    "CHECK",
    "REPORT_DONE",
]

# ------------------------------------------------------------------- versions

# Remediation R05: the public surface is versioned. PROTOCOL_VERSION covers
# the action schema and observation shape; FEEDBACK_VERSION covers the
# feedback mapping. The four-class boundary below is replayable as
# four-class-v1; the fine-grained codes stay evaluator-private.
PROTOCOL_VERSION = "agent-visual-v2"
FEEDBACK_VERSION = "four-class-v1"

# R05: optional public alias for the report skill, normalized at the parse
# boundary to REPORT_DONE (no second report executor exists).
REPORT_COMPLETE_ALIAS = "REPORT_COMPLETE"

# R05: fixed fine-grained feedback codes (evaluator-private, fixed
# priority per failure branch). The public boundary stays the four classes;
# these names land in the evaluator-private event log as `fine_reason` and
# never leak hidden answers, positions or action recommendations.
FINE_FEEDBACK_CODES = frozenset({
    "INVALID_PARAMETER",
    "NO_VISUAL_TARGET",
    "ACTION_NOT_APPLICABLE",
    "OUT_OF_CAPABILITY",
    "ACTION_UNSAFE",
    "MOVE_BLOCKED",
    "STALE_FRAME",
    "OBSERVE_NOT_FEASIBLE",
    "TASK_COMPLETED",
    "FALSE_COMPLETION",
    "MAX_STEPS",
})

# private_reason -> canonical fine code (unique per branch; checked against
# FINE_FEEDBACK_CODES)
FINE_REASON_MAP = {
    "SCHEMA": "INVALID_PARAMETER",
    "MOVE_PARAMETER_RANGE": "INVALID_PARAMETER",
    "TURN_PARAMETER_RANGE": "INVALID_PARAMETER",
    "STALE_FRAME": "STALE_FRAME",
    "VISIBILITY_UNAVAILABLE": "NO_VISUAL_TARGET",
    "NO_VISUAL_TARGET": "NO_VISUAL_TARGET",
    "INVALID_DEPTH_GEOMETRY": "NO_VISUAL_TARGET",
    "NO_VALID_DEPTH": "NO_VISUAL_TARGET",
    "NO_OWNING_ENTITY": "NO_VISUAL_TARGET",
    "SKILL_NOT_APPLICABLE_TO_ENTITY": "ACTION_NOT_APPLICABLE",
    "CHECK_UNSUPPORTED": "ACTION_NOT_APPLICABLE",
    "MOVE_BLOCKED": "MOVE_BLOCKED",
    "TURN_BLOCKED": "MOVE_BLOCKED",
    "NAV_NOT_FLOOR": "ACTION_NOT_APPLICABLE",
    "NAV_NO_PATH": "MOVE_BLOCKED",
    "NAV_START_BLOCKED": "MOVE_BLOCKED",
    "NAV_GOAL_BLOCKED": "MOVE_BLOCKED",
    "NO_VALID_OBSERVE_VIEWPOINT": "OBSERVE_NOT_FEASIBLE",
    "GOAL_UNSATISFIED": "FALSE_COMPLETION",
}


def fine_reason_for(private_reason: str | None, feedback_code: str | None,
                    episode_status: str | None,
                    skill: str | None = None) -> str | None:
    """Canonical fine code for one step, fixed priority (R05):

    1. mapped private reason of the failing branch;
    2. terminal statuses (budget exhaustion, false completion);
    3. the successful report;
    4. feedback-class-derived codes (capability/unsafety).
    """
    if private_reason and private_reason in FINE_REASON_MAP:
        return FINE_REASON_MAP[private_reason]
    if episode_status == "FAIL_MAX_STEPS":
        return "MAX_STEPS"
    if episode_status == "FAIL_FALSE_COMPLETION":
        return "FALSE_COMPLETION"
    if skill == "REPORT_DONE" and feedback_code == "EXECUTED":
        return "TASK_COMPLETED"
    if feedback_code == "OUT_OF_CAPABILITY":
        return "OUT_OF_CAPABILITY"
    if feedback_code == "UNSAFE":
        return "ACTION_UNSAFE"
    return None


class SessionMode(str, Enum):
    AGENT = "agent"
    ORACLE = "oracle"
    DEBUG = "debug"


# ------------------------------------------------------------------- feedback

class PublicActionFeedback(str, Enum):
    """EXACTLY four public action-result classes. No other category may cross
    the agent boundary."""

    EXECUTED = "EXECUTED"
    INVALID_ACTION = "INVALID_ACTION"
    OUT_OF_CAPABILITY = "OUT_OF_CAPABILITY"
    UNSAFE = "UNSAFE"


FEEDBACK_MESSAGES = {
    PublicActionFeedback.EXECUTED: "Action executed.",
    PublicActionFeedback.INVALID_ACTION:
        "The selected action is not applicable to the current observation or state.",
    PublicActionFeedback.OUT_OF_CAPABILITY:
        "The selected action is outside the robot's capability range.",
    PublicActionFeedback.UNSAFE:
        "The selected action cannot be executed safely.",
}


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: PublicActionFeedback
    message: str


# ------------------------------------------------------------ public actions

class _PublicAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str


class MoveAction(_PublicAction):
    """MOVE: signed base translation along the CURRENT heading.

    distance_cm > 0 -> forward, < 0 -> backward. Zero is invalid.
    Legal magnitude 5..100 cm by default (agent_protocol.min/max_move_cm).
    """

    skill: Literal["MOVE"]
    distance_cm: float


class TurnAction(_PublicAction):
    """TURN: signed base yaw rotation. angle_deg > 0 -> left (CCW),
    < 0 -> right (CW). Zero is invalid. Legal magnitude 5..180 deg."""

    skill: Literal["TURN"]
    angle_deg: float


class DiscreteTurnAction(_PublicAction):
    """Rotate the body in place: left=+90, right=-90, back=+180 degrees."""

    skill: Literal["TURN"]
    direction: Literal["left", "right", "back"]


class ViewAction(_PublicAction):
    """Absolute panoramic viewport angles in degrees; positive yaw looks left,
    positive pitch looks up. Zoom multiplies focal length (1-4; default 1).
    Neither the robot nor the camera mount moves.
    """

    skill: Literal["VIEW"]
    yaw: float = Field(ge=-180, le=180, allow_inf_nan=False)
    pitch: float = Field(default=0, ge=-90, le=90, allow_inf_nan=False)
    zoom: float = Field(default=1, ge=1, le=4, allow_inf_nan=False)


class PointAction(_PublicAction):
    """Point-directed skills: OPEN/CLOSE/GRASP/PLACE select an object;
    MOVE_TO selects a fixed floor position, or a visible furniture/object.
    The point is normalized on the CURRENT RGB frame. Floor: MOVE_TO checks a
    2D A* route, then arrives directly at that point, preserving body heading.
    Furniture/object (2026-10-26, AsgardBench-style automatic positioning):
    the robot is placed at a reachable stand pose in front of it, nearest the
    clicked side, facing it. Walls/ceilings are invalid targets.

    Coordinate convention: origin top-left, x rightward, y downward, both in
    [0, 1]. frame_id MUST identify the current primary image or current detail view; stale or unknown
    frame ids are INVALID_ACTION.
    """

    skill: Literal["OPEN", "CLOSE", "GRASP", "PLACE", "MOVE_TO", "CHECK"]

    class Point(BaseModel):
        model_config = ConfigDict(extra="forbid")

        frame_id: str
        x: float = Field(ge=0.0, le=1.0)
        y: float = Field(ge=0.0, le=1.0)

    point: Point


class ReportDoneAction(_PublicAction):
    """REPORT_DONE: no parameters. Privately evaluates the scenario goal
    predicate; a false completion terminates the episode."""

    skill: Literal["REPORT_DONE"]


PublicAction = Annotated[
    Union[DiscreteTurnAction, ViewAction, PointAction, ReportDoneAction],
    Field(discriminator="skill"),
]


# ----------------------------------------------------------------- observation

class EmbodimentObservation(BaseModel):
    """Robot-local calibration and executor state; never scene/target metadata."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    version: Literal["embodiment-v1"] = "embodiment-v1"
    robot_model: str
    executor_footprint_half_extent_m: float
    hand_capacity: int
    end_effector_link: Optional[str] = None
    executor_holding: bool
    projection: str
    image_size: tuple[int, int]
    view_yaw_deg: Optional[float] = None
    view_pitch_deg: Optional[float] = None
    camera_origin_in_base_m: tuple[float, float, float]
    body_forward_in_camera: tuple[float, float, float]
    body_up_in_camera: tuple[float, float, float]
    perspective_hfov_deg: Optional[float] = None


class DetailView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    image_png_b64: str
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float = 90.0
    zoom: float = 1.0
    width: int | None = None
    height: int | None = None
    source: str = "panorama_reprojection"


class InspectionView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_png_b64: str
    width: int
    height: int
    source: str = "isolated_object_multiview"
    description: str = "Four external appearance views of the selected object, around its local vertical axis. This is not a scene camera; it may reveal previously hidden exterior surfaces. Other objects are omitted; do not infer contents or emptiness. No scene actions can target this image."


class PublicObservation(BaseModel):
    """The complete agent-facing observation. Nothing else may be added.

    The v4.2 boundary forbids entity/category names. The explicit opt-in
    reason-feedback-v1 variant may disclose a blocking obstacle's semantic
    category in feedback.message, after a rejected action only.
    Always forbidden: entity instance IDs, object
    states, robot world pose, held-entity id, scene graph, pair_id, placement
    regime, oracle depth, segmentation, depth, world camera extrinsics, candidate
    or admissible action sets, interaction targets.
    """

    model_config = ConfigDict(extra="forbid")

    instruction: str
    frame_id: str
    image_png_b64: str
    planning_step: int
    max_planning_steps: int
    skill_library: list[str] = Field(default_factory=lambda: list(PUBLIC_SKILLS))
    feedback: Optional[Feedback] = None  # null on the initial observation
    embodiment: Optional[EmbodimentObservation] = None
    detail_view: Optional[DetailView] = None
    inspection_view: Optional[InspectionView] = None


class PublicStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_status: Literal["RUNNING", "SUCCESS", "FAIL_MAX_STEPS", "FAIL_FALSE_COMPLETION"]
    planning_step: int
    feedback: Feedback
    observation: PublicObservation


class AgentProtocolConfig(BaseModel):
    """Numeric bounds of the public action space (scenario-overridable),
    plus the protocol/feedback versions this scenario runs under (R05)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_move_cm: float = Field(default=5.0, gt=0)
    max_move_cm: float = Field(default=100.0, gt=0)
    min_turn_deg: float = Field(default=5.0, gt=0)
    max_turn_deg: float = Field(default=180.0, gt=0)
    protocol_version: str = PROTOCOL_VERSION
    feedback_version: str = FEEDBACK_VERSION

    @model_validator(mode="after")
    def _ordered_bounds(self):
        if self.min_move_cm > self.max_move_cm or self.min_turn_deg > self.max_turn_deg:
            raise ValueError("minimum action magnitude cannot exceed maximum")
        return self
