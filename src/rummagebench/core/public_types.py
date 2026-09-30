"""Final visual interaction protocol — public types (AGENT mode).

The evaluated agent receives ONLY: visual observations, the task instruction,
a fixed eight-skill library, and four-class feedback from its own previous
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
    "MOVE",
    "TURN",
    "OPEN",
    "CLOSE",
    "GRASP",
    "PLACE",
    "OBSERVE",
    "REPORT_DONE",
]


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


class PointAction(_PublicAction):
    """Object-directed skills (OPEN/CLOSE/GRASP/PLACE/OBSERVE): the object is
    selected by a normalized 2D point on the CURRENT RGB frame.

    Coordinate convention: origin top-left, x rightward, y downward, both in
    [0, 1]. frame_id MUST equal the latest current frame; stale or unknown
    frame ids are INVALID_ACTION.
    """

    skill: Literal["OPEN", "CLOSE", "GRASP", "PLACE", "OBSERVE"]

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
    Union[MoveAction, TurnAction, PointAction, ReportDoneAction],
    Field(discriminator="skill"),
]


# ----------------------------------------------------------------- observation

class ObserveView(BaseModel):
    """One auxiliary OBSERVE view. Contains ONLY an index and an RGB image."""

    model_config = ConfigDict(extra="forbid")

    view_index: int
    image_png_b64: str


class PublicObservation(BaseModel):
    """The complete agent-facing observation. Nothing else may be added.

    Forbidden anywhere in the serialized form: entity/category names, object
    states, robot world pose, held-entity id, scene graph, pair_id, placement
    regime, oracle depth, segmentation, depth, camera extrinsics, candidate
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
    observe_views: list[ObserveView] = Field(default_factory=list)


class PublicStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_status: Literal["RUNNING", "SUCCESS", "FAIL_MAX_STEPS", "FAIL_FALSE_COMPLETION"]
    planning_step: int
    feedback: Feedback
    observation: PublicObservation


class AgentProtocolConfig(BaseModel):
    """Numeric bounds of the public action space (scenario-overridable)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    min_move_cm: float = Field(default=5.0, gt=0)
    max_move_cm: float = Field(default=100.0, gt=0)
    min_turn_deg: float = Field(default=5.0, gt=0)
    max_turn_deg: float = Field(default=180.0, gt=0)

    @model_validator(mode="after")
    def _ordered_bounds(self):
        if self.min_move_cm > self.max_move_cm or self.min_turn_deg > self.max_turn_deg:
            raise ValueError("minimum action magnitude cannot exceed maximum")
        return self
