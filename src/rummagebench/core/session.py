"""BenchmarkSession: the canonical, transport-independent benchmark interface.

Pipeline per action:

    Agent action -> parser -> semantic validator -> target validator ->
    safety validator -> feasibility validator -> skill executor ->
    postcondition validator -> observation (regenerated skills)

All checks live here, not in agent adapters. Skills never decide episode
outcome; this class applies termination rules and the success definition.
The available action space is dynamic:

    A_t = Ground(RobotGeometry, ObjectInterface, WorldGeometry, WorldState_t)

— regenerated after every world update and exposed via the observation.
Physical feasibility (real IK + configuration-space collision in production)
gates every interaction; execution itself stays symbolic/instant.
"""

from __future__ import annotations

import logging
from typing import Any

from rummagebench.core.errors import FeasibilityBackendError, UnresolvableTargetError
from rummagebench.core.events import EpisodeLogWriter, make_event
from rummagebench.core.scenario import ScenarioSpec
from rummagebench.core.skill_grounder import SkillGrounder
from rummagebench.core.types import (
    Action,
    EpisodeStatus,
    FailureReason,
    FeasibilityVerdict,
    Observation,
    StepResult,
)
from rummagebench.grounding.entity import OracleEntityGrounding
from rummagebench.robots.model_loader import load_kinematics_backend
from rummagebench.robots.robot import RobotEmbodiment
from rummagebench.sim.base import SimBackend
from rummagebench.skills.registry import SkillRegistry, default_registry
from rummagebench.state.benchmark_state import BenchmarkWorldState
from rummagebench.validation.feasibility import FeasibilityValidator
from rummagebench.validation.safety import SafetyValidator
from rummagebench.validation.semantic import SemanticValidator
from rummagebench.validation.target import TargetValidator

logger = logging.getLogger(__name__)


def _build_feasibility(backend: SimBackend, scenario: ScenarioSpec) -> FeasibilityValidator:
    """Select the feasibility engine from the scenario config.

    backend=pinocchio (production): URDF kinematics + q-based collision; any
    failure to initialize raises FeasibilityBackendError (no silent fallback
    to the reach-radius proxy).
    backend=proxy: reach-radius + point-vs-AABB, unit-test mode only.
    """
    spec = scenario.feasibility
    kinematics = load_kinematics_backend(scenario.robot, spec)
    if spec.backend == "pinocchio":
        from rummagebench.feasibility.hpp_fcl_checker import PinocchioCollisionChecker

        collision = PinocchioCollisionChecker(
            kinematics, compute_min_distance=False
        )
        return FeasibilityValidator(
            backend,
            kinematics=kinematics,
            config_collision=collision,
            mode=spec.mode,
        )
    from rummagebench.feasibility.collision import CollisionChecker

    return FeasibilityValidator(
        backend,
        ik_solver=kinematics,  # ReachabilityIKSolver
        collision_checker=CollisionChecker(backend),
        mode=spec.mode,
    )


class BenchmarkSession:
    def __init__(
        self,
        backend: SimBackend,
        scenario: ScenarioSpec,
        log_path: str | None = None,
        registry: SkillRegistry | None = None,
        feasibility: FeasibilityValidator | None = None,
    ):
        self._backend = backend
        self._scenario = scenario
        self._registry = registry or default_registry()
        self._semantic = SemanticValidator(scenario, self._registry)
        self._target = TargetValidator()
        self._safety = SafetyValidator(scenario)
        self._robot_emb = RobotEmbodiment.from_spec(scenario.robot)
        self._feasibility = feasibility or _build_feasibility(backend, scenario)
        self._grounder = SkillGrounder(backend, scenario, feasibility=self._feasibility)
        self._grounding = OracleEntityGrounding(backend, scenario)

        # benchmark-owned semantic state (holding truth lives HERE, never in
        # the simulator's assisted-grasp internals)
        self._world_state = BenchmarkWorldState()

        self._status = EpisodeStatus.RUNNING
        self._planning_step = 0
        self._safe_history = True
        self._previous_result: dict[str, Any] | None = None
        self._instruction = scenario.instruction

        self._log = EpisodeLogWriter(log_path) if log_path else None

    # --------------------------------------------------------------- contract

    @property
    def episode_id(self) -> str:
        return self._scenario.id

    @property
    def scenario(self) -> ScenarioSpec:
        return self._scenario

    @property
    def robot(self) -> RobotEmbodiment:
        return self._robot_emb

    @property
    def world_state(self) -> BenchmarkWorldState:
        return self._world_state

    def available_skills(self) -> list[str]:
        """The dynamically grounded action space A_t (may be empty on error).

        A FeasibilityBackendError (misconfigured physical grounding) is NEVER
        swallowed: it propagates loudly instead of degrading the observation.
        """
        try:
            return self._grounder.ground(self._robot_emb, self._world_state)
        except FeasibilityBackendError:
            raise
        except Exception as e:
            logger.warning("skill grounding failed: %s", e)
            return []

    def reset(self) -> Observation:
        """Restore the deterministic initial snapshot and clear episode state."""
        self._backend.reset()
        self._world_state.clear()
        self._status = EpisodeStatus.RUNNING
        self._planning_step = 0
        self._safe_history = True
        self._previous_result = None
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            instruction=self._instruction,
            rgb=self._backend.get_observation(),
            planning_step=self._planning_step,
            max_planning_steps=self._scenario.termination.max_planning_steps,
            previous_action_result=self._previous_result,
            available_skills=(
                self.available_skills() if self._scenario.expose_available_skills else []
            ),
        )

    def status(self) -> EpisodeStatus:
        return self._status

    def act(self, action: Action | dict) -> StepResult:
        if isinstance(action, dict):
            action = Action.from_dict(action)
        return self._act(action)

    # ---------------------------------------------------------------- pipeline

    def _act(self, action: Action) -> StepResult:
        if self._status != EpisodeStatus.RUNNING:
            raise RuntimeError("episode is not running; call reset() first")

        # every submitted action consumes one semantic planning step
        self._planning_step += 1
        step = self._planning_step

        # horizon rule: H > H_max terminates as FAIL_MAX_STEPS
        if step > self._scenario.termination.max_planning_steps:
            self._status = EpisodeStatus.FAIL_MAX_STEPS
            return self._finish_step(
                action, step,
                validation={"semantic_valid": False, "target_valid": False, "safe": False},
                executed=False, postcondition=None,
                failure_reason=FailureReason.MAX_STEPS,
                events=[{"event": "planning_horizon_exceeded",
                         "max_planning_steps": self._scenario.termination.max_planning_steps}],
                resolved=None,
            )

        validation = {"semantic_valid": False, "target_valid": False, "safe": False,
                      "feasible": False}
        executed = False
        failure_reason = FailureReason.NONE
        events: list[dict[str, Any]] = []
        resolved = None

        # 1. semantic validation
        semantic = self._semantic.check(action)
        validation["semantic_valid"] = semantic.valid
        if not semantic.valid:
            return self._finish_step(
                action, step, validation, executed=False, postcondition=None,
                failure_reason=FailureReason.INVALID_ACTION, events=[
                    {"event": "invalid_action", "reason": semantic.reason}
                ],
                resolved=None,
            )

        # 2. target grounding + validation
        try:
            resolved = self._grounding.resolve(action.target)
            target_verdict = self._target.check(action.skill, resolved)
            validation["target_valid"] = target_verdict.valid
        except UnresolvableTargetError as e:
            resolved = None
            target_verdict = None
            events.append({"event": "unresolvable_target", "reason": str(e)})

        if not validation["target_valid"]:
            reason = (
                events[-1]["reason"]
                if events
                else (target_verdict.reason if target_verdict else "unknown target")
            )
            return self._finish_step(
                action, step, validation, executed=False, postcondition=None,
                failure_reason=FailureReason.INVALID_ACTION, events=events or [
                    {"event": "invalid_target", "reason": reason}
                ],
                resolved=resolved,
            )

        # 3. safety validation (task/policy-level rules; before any state transition)
        safety = self._safety.check(action.skill, resolved)
        validation["safe"] = safety.safe
        if not safety.safe:
            self._safe_history = False
            if self._scenario.termination.fail_on_unsafe_action:
                self._status = EpisodeStatus.FAIL_UNSAFE_ACTION
            return self._finish_step(
                action, step, validation, executed=False, postcondition=None,
                failure_reason=FailureReason.UNSAFE_ACTION,
                events=[{"event": "unsafe_action", "reason": safety.reason,
                         "safety_violation": True, "physical_failure": False,
                         "details": safety.details}],
                resolved=resolved,
            )

        # 4. physical feasibility validation (kinematics + collision + state;
        # non-terminal structured failure)
        feasibility = self._feasibility.check(
            action.skill, resolved, self._robot_emb, self._world_state
        )
        validation["feasible"] = feasibility.feasible
        if not feasibility.feasible:
            reason = FailureReason[feasibility.reason]  # UNREACHABLE/COLLISION/INVALID_STATE
            physical = reason in (FailureReason.UNREACHABLE, FailureReason.COLLISION)
            return self._finish_step(
                action, step, validation, executed=False, postcondition=None,
                failure_reason=reason,
                events=[{"event": "infeasible_action", "reason": feasibility.reason,
                         "safety_violation": False, "physical_failure": physical,
                         "details": feasibility.details}],
                resolved=resolved,
            )

        # 5. execute skill (instant symbolic transition; no hidden navigation)
        skill = self._registry.get(action.skill)
        assert skill is not None  # semantic validation guarantees this
        skill_result = skill.execute(self._backend, resolved, self._world_state)
        executed = skill_result.executed
        events.extend(skill_result.events)
        validation["postcondition_satisfied"] = skill_result.postcondition_satisfied

        # 6. termination rules
        self._apply_post_execution_rules(action, skill_result, resolved)
        if self._status == EpisodeStatus.FAIL_WRONG_TARGET:
            failure_reason = FailureReason.WRONG_TARGET

        return self._finish_step(
            action, step, validation,
            executed=executed,
            postcondition=skill_result.postcondition_satisfied,
            failure_reason=failure_reason,
            events=events,
            resolved=resolved,
            extra={"skill_details": skill_result.details},
        )

    def _apply_post_execution_rules(
        self, action: Action, skill_result, resolved
    ) -> None:
        """Success / wrong-target policy. Skills never decide these (Rule 4).

        Task success is evaluated against the BENCHMARK-owned holding state,
        never against the simulator's assisted-grasp internals.
        """
        term = self._scenario.termination

        if action.skill == "GRASP" and skill_result.executed and skill_result.postcondition_satisfied:
            held_entity = self._world_state.held_object
            if held_entity == self._scenario.target.entity:
                if term.succeed_when_holding_target and self._safe_history:
                    self._status = EpisodeStatus.SUCCESS
            else:
                if term.fail_on_wrong_grasp:
                    self._status = EpisodeStatus.FAIL_WRONG_TARGET

    def _finish_step(
        self,
        action: Action,
        step: int,
        validation: dict[str, Any],
        executed: bool,
        postcondition: bool | None,
        failure_reason: FailureReason,
        events: list[dict[str, Any]],
        resolved,
        extra: dict[str, Any] | None = None,
    ) -> StepResult:
        observation = Observation(
            instruction=self._instruction,
            rgb=self._backend.get_observation(),
            planning_step=step,
            max_planning_steps=self._scenario.termination.max_planning_steps,
            previous_action_result=None,  # filled below
            available_skills=(
                self.available_skills() if self._scenario.expose_available_skills else []
            ),
        )

        action_result = {
            "executed": executed,
            "postcondition_satisfied": postcondition,
            "failure_reason": failure_reason.value,
            **(extra or {}),
        }
        observation.previous_action_result = action_result

        step_result = StepResult(
            observation=observation,
            episode_status=self._status,
            planning_step=step,
            action=action.to_dict(),
            executed=executed,
            failure_reason=failure_reason,
            events=events,
        )

        self._previous_result = action_result

        if self._log is not None:
            self._log.append(
                make_event(
                    episode_id=self._scenario.id,
                    step=step,
                    action=action.to_dict(),
                    validation=validation,
                    execution=action_result,
                    status=self._status.value,
                    resolved_target=(
                        {"kind": resolved.kind.value, "entity": resolved.entity,
                         "place": resolved.place}
                        if resolved is not None
                        else None
                    ),
                )
            )
        return step_result
