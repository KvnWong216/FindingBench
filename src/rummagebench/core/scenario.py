"""Scenario specification: validated, data-driven episode definitions.

Scenario content must NOT live in Python source. Every episode is described
by a YAML file validated into these models at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnchorSpec(BaseModel):
    """World-frame pose of a semantic navigation anchor."""

    model_config = ConfigDict(extra="forbid")

    position: list[float] = Field(min_length=3, max_length=3)
    orientation: list[float] = Field(min_length=4, max_length=4)  # xyzw quaternion


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "r1pro"
    name: str = "robot_0"
    init_anchor: str
    obs_modalities: list[str] = Field(default_factory=lambda: ["rgb"])
    image_width: int = 224
    image_height: int = 224
    grasping_mode: Literal["sticky", "assisted", "physical"] = "sticky"
    # embodiment capability parameters for feasibility grounding
    reach_radius: float = 1.0  # max arm reach from base origin (m)
    z_min: float = 0.0  # interaction height band (m, world frame)
    z_max: float = 1.6
    hand_capacity: int = 1


class ObjectSpec(BaseModel):
    """An extra dataset object spawned into the scene by the scenario builder."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    model: Optional[str] = None  # None -> first sorted dataset model (deterministic)
    fixed_base: bool = False
    scale: Optional[list[float]] = None


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    category: str


class PlacementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    relation: Literal["inside", "on_top"]
    receptacle: str  # name of an existing scene furniture object


class InitialStateSpec(BaseModel):
    """Container / object states forced at build time, keyed by entity name."""

    model_config = ConfigDict(extra="forbid")

    open: Optional[bool] = None


class TerminationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_planning_steps: int = Field(default=16, ge=1)
    fail_on_wrong_grasp: bool = True
    fail_on_unsafe_action: bool = True
    succeed_when_holding_target: bool = True


class SafetySpec(BaseModel):
    """Deterministic benchmark-level safety rules for the MVP.

    forbidden_categories: grasping an object of these categories is unsafe.
    grasping_fixed_base_unsafe: grasping fixed furniture/appliances is unsafe.
    """

    model_config = ConfigDict(extra="forbid")

    forbidden_categories: list[str] = Field(default_factory=list)
    grasping_fixed_base_unsafe: bool = True


class AgentSpec(BaseModel):
    """Scenario-provided oracle sequences for scripted test agents.

    Keeping them in data (not Python) respects Architecture Rule 1: scripted
    baselines are just sequences of canonical actions authored per scenario.
    """

    model_config = ConfigDict(extra="forbid")

    scripted_success: list[dict[str, Any]] = Field(default_factory=list)
    scripted_wrong_object: list[dict[str, Any]] = Field(default_factory=list)
    unsafe_sequence: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    scene: SceneSpec
    robot: RobotSpec
    anchors: dict[str, AnchorSpec]
    target: TargetSpec
    objects: list[ObjectSpec] = Field(default_factory=list)
    placements: list[PlacementSpec] = Field(default_factory=list)
    initial_states: dict[str, InitialStateSpec] = Field(default_factory=dict)
    termination: TerminationSpec = Field(default_factory=TerminationSpec)
    skills: list[str] = Field(default_factory=lambda: ["NAV", "OPEN", "GRASP"])
    safety: SafetySpec = Field(default_factory=SafetySpec)
    agent: AgentSpec = Field(default_factory=AgentSpec)
    # expose the dynamically grounded action space to the agent observation
    expose_available_skills: bool = True

    @model_validator(mode="after")
    def _cross_check(self) -> "ScenarioSpec":
        if self.robot.init_anchor not in self.anchors:
            raise ValueError(
                f"robot.init_anchor {self.robot.init_anchor!r} is not a defined anchor"
            )
        if self.target.entity not in {o.name for o in self.objects}:
            raise ValueError(
                f"target.entity {self.target.entity!r} must be one of the spawned objects"
            )
        spawned = {o.name for o in self.objects}
        for p in self.placements:
            if p.entity not in spawned:
                raise ValueError(f"placement entity {p.entity!r} is not a spawned object")
        return self

    def entity_by_name(self, name: str) -> ObjectSpec | None:
        for o in self.objects:
            if o.name == name:
                return o
        return None


def load_scenario(path: str | Path) -> ScenarioSpec:
    """Load and validate a scenario YAML. Raises ScenarioValidationError on any problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ScenarioValidationError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ScenarioValidationError(f"scenario file {path} must contain a mapping")
    try:
        return ScenarioSpec.model_validate(raw)
    except Exception as e:
        raise ScenarioValidationError(f"scenario {path} failed validation: {e}") from e
