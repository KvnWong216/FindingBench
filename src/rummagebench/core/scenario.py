"""Scenario specification: validated, data-driven episode definitions.

Scenario content must NOT live in Python source. Every episode is described
by a YAML file validated into these models at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rummagebench.core.errors import ScenarioValidationError
from rummagebench.core.public_types import AgentProtocolConfig


class AnchorSpec(BaseModel):
    """World-frame pose of a semantic navigation anchor."""

    model_config = ConfigDict(extra="forbid")

    position: list[float] = Field(min_length=3, max_length=3)
    orientation: list[float] = Field(min_length=4, max_length=4)  # xyzw quaternion


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str


class KinematicsSpec(BaseModel):
    """Robot kinematic model definition (real embodiment grounding).

    The URDF is the single source of robot geometry: link frames, joint tree,
    joint limits (joint_limits_source: urdf) and <collision> bodies. It must
    be derived from the robot asset (OmniGibson articulation export via
    scripts/export_robot_kinematics.py or a vendor URDF) — never hand-written.

    reach_radius / z_min / z_max stay on RobotSpec as a coarse prefilter for
    the proxy backend (tests only); they can never authorize an interaction in
    the pinocchio backend.
    """

    model_config = ConfigDict(extra="forbid")

    urdf_path: str
    base_link: str = "base_link"
    end_effector_link: str
    # None -> all movable joints; "auto" -> the base->EEF chain only
    controlled_joints: Optional[Union[list[str], Literal["auto"]]] = None
    joint_limits_source: Literal["urdf"] = "urdf"
    # links treated as gripper/finger class by the allowed-collision matrix;
    # None -> auto-derive (EEF link descendants + 'finger|gripper|hand' names)
    gripper_links: Optional[list[str]] = None
    # deterministic multi-seed IK restarts
    ik_seed: int = 0


class ActionInterfaceSpec(BaseModel):
    """Agent-facing action protocol.

    admissible: expose only skills that passed the physical feasibility
    filter (A_t^admissible) via Observation.available_skills.
    candidate: expose semantic+state-valid skills for VISIBLE objects
    (A_t^candidate) via Observation.candidate_skills — no IK/collision
    filtering and no feasibility metadata; attempted actions are still
    judged by the in-environment feasibility oracle and answered with
    structured step feedback (SUCCESS / UNREACHABLE / COLLISION /
    INVALID_STATE), never with scores.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["admissible", "candidate"] = "admissible"


class FeasibilitySpec(BaseModel):
    """Feasibility engine configuration.

    backend: production benchmark requires 'pinocchio' (real URDF kinematics
    + configuration-space collision). 'proxy' (reach-radius) exists for unit
    tests only; production sessions must never fall back to it silently.
    mode: 'endpoint' (v1) checks existence of one collision-free interaction
    configuration. 'canonical_corridor' (planned) additionally samples a short
    straight approach; neither mode plans trajectories.
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["pinocchio", "proxy"] = "pinocchio"
    mode: Literal["endpoint", "canonical_corridor"] = "endpoint"
    ik_pos_tol: float = 0.005  # m
    ik_rot_tol: float = 0.0873  # rad (~5 deg)
    ik_max_iters: int = Field(default=200, ge=1)
    ik_restarts: int = Field(default=25, ge=0)
    collision_padding: float = 0.0  # m, safety margin subtracted from clearances


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "r1pro"
    name: str = "robot_0"
    init_anchor: str
    # §7: AGENT-mode scenarios request the synchronized private bundle by
    # declaring obs_modalities: [rgb, depth, seg_instance] in the scenario
    # (seg_semantic is auto-enabled by the sensor). Legacy scenarios stay
    # rgb-only.
    obs_modalities: list[str] = Field(default_factory=lambda: ["rgb"])
    # per-sensor modality scoping: heavy annotators (depth/seg) applied to
    # every camera of a multi-camera robot crash the syntheticdata graph on
    # some Isaac builds — restrict to the interaction camera instead.
    include_sensor_names: Optional[list[str]] = None
    exclude_sensor_names: Optional[list[str]] = None
    image_width: int = 224
    image_height: int = 224
    # optional camera focal length (mm); None keeps the OG default (17 mm).
    # Shorter widens the field of view, e.g. for non-square play resolutions.
    focal_length_mm: Optional[float] = Field(default=None, gt=0)
    grasping_mode: Literal["sticky", "assisted", "physical"] = "sticky"
    # embodiment capability parameters: coarse prefilter for the proxy
    # backend (tests only); the pinocchio backend derives feasibility from
    # the URDF instead.
    reach_radius: float = 1.0  # max arm reach from base origin (m)
    z_min: float = 0.0  # interaction height band (m, world frame)
    z_max: float = 1.6
    hand_capacity: int = 1
    kinematics: Optional[KinematicsSpec] = None


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
    link: Optional[str] = None

    @model_validator(mode="after")
    def _link_only_inside(self) -> "PlacementSpec":
        if self.link is not None and self.relation != "inside":
            raise ValueError("placement link is only valid for relation 'inside'")
        return self


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
    feasibility: FeasibilitySpec = Field(default_factory=FeasibilitySpec)
    action_interface: ActionInterfaceSpec = Field(default_factory=ActionInterfaceSpec)
    agent_protocol: AgentProtocolConfig = Field(default_factory=AgentProtocolConfig)
    # exact oracle semantic depth d* — computed by the oracle planner during
    # episode certification; NEVER hand-authored (None until certified)
    oracle_min_steps: Optional[int] = None
    # §6 placement regime + paired-episode id (natural/counterfactual pairs)
    placement_regime: Optional[str] = None
    pair_id: Optional[str] = None
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
