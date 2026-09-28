"""Canonical benchmark data model.

Every adapter (Python API, MCP, HTTP, CLI) uses exactly these definitions.
No adapter may redefine them (Architecture Rule 7).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EpisodeStatus(str, Enum):
    """Terminal and non-terminal episode states. Never use free-form strings."""

    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAIL_MAX_STEPS = "FAIL_MAX_STEPS"
    FAIL_WRONG_TARGET = "FAIL_WRONG_TARGET"
    FAIL_UNSAFE_ACTION = "FAIL_UNSAFE_ACTION"


class FailureReason(str, Enum):
    """Why a single step or the episode failed. NONE means no failure.

    Paper-facing taxonomy (§4):

    INVALID_ACTION   malformed action / unknown skill / invalid target schema
                     / semantically nonexistent action
    INVALID_STATE    valid semantic action but wrong current state
    UNREACHABLE      semantic candidate exists but no valid IK / joint-limit
                     satisfying interaction configuration exists
    COLLISION        IK configuration exists but every interaction candidate
                     violates collision constraints
    UNSAFE_ACTION    task-level safety policy violation
    MAX_STEPS        planning horizon exceeded
    WRONG_TARGET     grasped a non-target object with fail_on_wrong_grasp

    The debug flag not_in_manipulation_space may accompany UNREACHABLE events
    but never replaces the primary structured reason.
    """

    NONE = "NONE"
    INVALID_ACTION = "INVALID_ACTION"
    MAX_STEPS = "MAX_STEPS"
    WRONG_TARGET = "WRONG_TARGET"
    UNSAFE_ACTION = "UNSAFE_ACTION"
    # embodiment feasibility failures (structured, non-terminal)
    UNREACHABLE = "UNREACHABLE"
    COLLISION = "COLLISION"
    INVALID_STATE = "INVALID_STATE"


class TargetKind(str, Enum):
    """How the agent refers to an action target.

    - place: a semantic navigation anchor defined by the scenario (NAV only)
    - entity: an oracle/debug entity reference (MVP grounding mode)
    - pixel: a normalized pixel on a named camera; the evaluator privately maps
      it to a simulator instance via ground-truth segmentation. Kept in the data
      model now so visual grounding never breaks the action schema.
    """

    PLACE = "place"
    ENTITY = "entity"
    PIXEL = "pixel"


class TargetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: TargetKind
    value: Optional[str] = None
    camera: Optional[str] = None
    x: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_fields(self) -> "TargetRef":
        if self.type in (TargetKind.PLACE, TargetKind.ENTITY):
            if not self.value:
                raise ValueError(f"target type {self.type!r} requires 'value'")
        elif self.type == TargetKind.PIXEL:
            if self.camera is None or self.x is None or self.y is None:
                raise ValueError("pixel target requires 'camera', 'x' and 'y'")
            if self.value is None:
                # a pixel target may still carry a free-text guess, but it is optional
                self.value = ""
        return self

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value}
        if self.value is not None:
            d["value"] = self.value
        if self.camera is not None:
            d["camera"] = self.camera
        if self.x is not None:
            d["x"] = self.x
        if self.y is not None:
            d["y"] = self.y
        return d


class Action(BaseModel):
    """One semantic action, e.g. NAV(kitchen) / OPEN(cabinet_B) / GRASP(knife)."""

    model_config = ConfigDict(extra="forbid")

    skill: str
    target: TargetRef

    def to_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "target": self.target.to_dict()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Action":
        return cls.model_validate(raw)


class Observation(BaseModel):
    """What the evaluated agent is allowed to see.

    Privileged information (target id, GT poses, segmentation, full scene graph)
    must never appear here; debug tooling may read it from the backend instead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instruction: str
    rgb: Optional[np.ndarray] = None
    planning_step: int
    max_planning_steps: int
    previous_action_result: Optional[dict[str, Any]] = None
    # dynamic action space: skills available to THIS robot at THIS world state
    # (embodiment-aware grounding output; empty when the scenario disables it)
    available_skills: list[str] = Field(default_factory=list)
    # action_interface.mode == "candidate": semantic+state-valid skills for
    # VISIBLE objects only — no IK/collision filtering, no feasibility
    # metadata (the oracle verdict arrives as structured step feedback)
    candidate_skills: list[str] = Field(default_factory=list)


class FeasibilityVerdict(BaseModel):
    """Result of the geometric/state feasibility check (can, not how)."""

    feasible: bool
    reason: str = "none"  # none | UNREACHABLE | COLLISION | INVALID_STATE
    details: dict[str, Any] = Field(default_factory=dict)


class WorldState(BaseModel):
    """Abstracted world state relevant to skill grounding."""

    held_count: int = 0


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    observation: Observation
    episode_status: EpisodeStatus
    planning_step: int
    action: dict[str, Any]
    executed: bool
    failure_reason: FailureReason = FailureReason.NONE
    events: list[dict[str, Any]] = Field(default_factory=list)


class SkillResult(BaseModel):
    """Structured result of executing one skill. Never a bare bool."""

    skill: str
    target: dict[str, Any]
    executed: bool
    postcondition_satisfied: bool
    events: list[dict[str, Any]] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SafetyVerdict(BaseModel):
    """Result of SafetyValidator.check()."""

    safe: bool
    reason: str = "none"
    details: dict[str, Any] = Field(default_factory=dict)


CameraName = Literal["head_rgb"]
