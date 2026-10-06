"""AGENT-mode orchestrator: the final visual interaction protocol.

Wraps the frozen ORACLE core (BenchmarkSession: semantic/state validation,
pinocchio IK Stage A, coal collision Stage B, symbolic execution) behind the
public eight-skill protocol. The agent sees ONLY PublicObservation; every
grounding detail is evaluator-private (§19, §20).

Step accounting: each submission in a running, valid episode consumes one
public planning step. The public counter is authoritative; the internal
legacy executor counter advances only for dispatched object interactions.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from rummagebench.core.public_types import (
    PUBLIC_SKILLS,
    AgentProtocolConfig,
    Feedback,
    MoveAction,
    ObserveView,
    PointAction,
    PublicAction,
    PublicActionFeedback,
    PublicObservation,
    PublicStepResult,
    ReportDoneAction,
    TurnAction,
)
from rummagebench.core.types import Action, EpisodeStatus, FailureReason, TargetKind, TargetRef
from rummagebench.perception.frame_store import FrameStore
from rummagebench.perception.visual_bridge import (
    GROUNDING_VERSION,
    VisualBridge,
    VisualGroundingError,
)
from rummagebench.skills.move import execute_move, execute_turn
from rummagebench.skills.observe import ObserveConfig, run_observe
from rummagebench.skills.report_done import goal_satisfied

logger = logging.getLogger(__name__)

_PUBLIC_RUNNING = "RUNNING"


def _png_b64(rgb) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VisualProtocolSession:
    """The agent-facing protocol session (SESSION MODE: AGENT)."""

    def __init__(
        self,
        backend,
        scenario,
        protocol: AgentProtocolConfig | None = None,
        base_half_extent: float = 0.40,
        observe_cfg: ObserveConfig | None = None,
    ):
        # Session-specific policy must not mutate the caller's scenario.
        scenario = scenario.model_copy(deep=True)
        # AGENT mode: wrong grasps recoverable, holding the target does NOT
        # auto-succeed, unsafe attempts map to UNSAFE feedback (all
        # non-terminal). The public counter enforces the AGENT horizon.
        scenario.termination.fail_on_wrong_grasp = False
        scenario.termination.succeed_when_holding_target = False
        scenario.termination.fail_on_unsafe_action = False

        from rummagebench.core.session import BenchmarkSession

        self._backend = backend
        self._scenario = scenario
        self._session = BenchmarkSession(backend, scenario)
        # OPEN's base-intrusion check uses the same footprint as MOVE / TURN
        feas = getattr(self._session, "_feasibility", None)
        if feas is not None and hasattr(feas, "base_half_extent"):
            feas.base_half_extent = base_half_extent
        self._protocol = protocol or getattr(scenario, "agent_protocol", AgentProtocolConfig())
        self._observe_cfg = observe_cfg or ObserveConfig()
        self._base_half_extent = base_half_extent
        self._store = FrameStore()
        self._bridge = VisualBridge()
        self._public_step = 0
        self._status = _PUBLIC_RUNNING
        self._last_feedback: Feedback | None = None
        self._observe_views: list[ObserveView] = []
        self._trace: list[dict[str, Any]] = []
        self._trace_path: Path | None = None
        from pydantic import TypeAdapter

        self._action_adapter = TypeAdapter(PublicAction)

    # ------------------------------------------------------------- plumbing

    @property
    def episode_id(self) -> str:
        return self._session.episode_id

    @property
    def world_state(self):
        return self._session.world_state

    def status(self) -> str:
        return self._status

    def set_trace(self, path: str | Path | None) -> None:
        self._trace_path = Path(path) if path else None
        if self._trace_path:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, record: dict[str, Any]) -> None:
        self._trace.append(record)
        if self._trace_path:
            with self._trace_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _capture_current(self):
        frame = self._backend.capture_visual_frame()
        frame.frame_id = self._store.new_frame_id()
        self._store.register(frame)
        return frame

    def _feedback(self, code: PublicActionFeedback) -> Feedback:
        from rummagebench.core.public_types import FEEDBACK_MESSAGES

        return Feedback(code=code, message=FEEDBACK_MESSAGES[code])

    def _observation(self, frame) -> PublicObservation:
        return PublicObservation(
            instruction=self._scenario.instruction,
            frame_id=frame.frame_id,
            image_png_b64=_png_b64(frame.rgb),
            planning_step=self._public_step,
            max_planning_steps=self._scenario.termination.max_planning_steps,
            skill_library=list(PUBLIC_SKILLS),
            feedback=self._last_feedback,
            observe_views=list(self._observe_views),
        )

    def observe(self) -> PublicObservation:
        """Read the current public snapshot without advancing or recapturing."""
        if self._status == "ENGINE_ERROR":
            raise RuntimeError("episode invalidated by simulator failure; reset required")
        frame = self._store.get(self._store.current_id)
        if frame is None:
            raise RuntimeError("call reset() before observe()")
        return self._observation(frame)

    # --------------------------------------------------------------- reset

    def reset(self) -> PublicObservation:
        self._status = "ENGINE_ERROR"  # usable only after the entire reset succeeds
        self._session.reset()
        # the AGENT episode starts at the scenario spawn anchor: the visual
        # protocol has no NAV anchors, so the spawn is applied here
        from rummagebench.core.scenario import AnchorSpec

        init = self._scenario.anchors[self._scenario.robot.init_anchor]
        self._backend.teleport_robot(
            AnchorSpec(position=list(init.position), orientation=list(init.orientation))
        )
        self._backend.settle(5)
        self._store.reset()
        self._public_step = 0
        self._last_feedback = None
        self._observe_views = []
        self._trace.clear()
        frame = self._capture_current()
        if frame.meta.get("visual_grounding_supported") is False:
            raise RuntimeError(
                "AGENT visual grounding requires synchronized renderer instance "
                "segmentation, depth and camera geometry; RGB-only raycasts are "
                "diagnostic-only and do not establish RGB visibility. "
                f"Missing: {frame.meta.get('unsupported_reason', 'private modalities')}"
            )
        self._log({"event": "reset", "frame_id": frame.frame_id})
        observation = self._observation(frame)
        self._status = _PUBLIC_RUNNING
        return observation

    # ---------------------------------------------------------------- step

    def step(self, raw: dict[str, Any]) -> PublicStepResult:
        if self._status == "ENGINE_ERROR":
            raise RuntimeError("episode invalidated by simulator failure; reset required")
        try:
            return self._step_impl(raw)
        except Exception:
            # Covers capture/serialization/pose/log failures outside dispatch,
            # too. The original detail is only sent to the evaluator logger.
            logger.exception("visual protocol infrastructure failed")
            self._status = "ENGINE_ERROR"
            raise RuntimeError("episode invalidated by simulator failure; reset required") from None

    def _step_impl(self, raw: dict[str, Any]) -> PublicStepResult:
        if self._status == "ENGINE_ERROR":
            raise RuntimeError("episode invalidated by simulator failure; reset required")
        if self._status != _PUBLIC_RUNNING:
            frame = self._store.get(self._store.current_id)
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            return PublicStepResult(
                episode_status=self._status,
                planning_step=self._public_step,
                feedback=self._feedback(PublicActionFeedback.INVALID_ACTION),
                observation=self._observation(frame),
            )

        record: dict[str, Any] = {"raw_action": raw, "robot_pose_before": self._backend.robot_pose()[0]}
        try:
            action = self._action_adapter.validate_python(raw)
        except ValidationError:
            action = None

        self._observe_views = []
        frame = self._store.get(self._store.current_id)

        try:
            if action is None:
                self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
                record["feedback"] = "INVALID_ACTION"
                record["private_reason"] = "SCHEMA"
            elif isinstance(action, MoveAction):
                self._step_move(action, record)
            elif isinstance(action, TurnAction):
                self._step_turn(action, record)
            elif isinstance(action, ReportDoneAction):
                self._step_report_done(record)
            else:
                frame = self._step_point(action, frame, record) or frame
        except Exception as e:
            # Infrastructure failures are not agent mistakes. The simulator may
            # already have changed; do not return a stale snapshot or score this
            # as INVALID_ACTION. Invalidate this run until an explicit reset.
            logger.exception("public step failed; run invalidated")
            self._status = "ENGINE_ERROR"
            record["private_reason"] = f"ENGINE:{type(e).__name__}"
            record["run_invalidated"] = True
            self._log(record)
            raise RuntimeError("episode invalidated by simulator failure; reset required") from None

        # horizon: every submitted action consumes exactly one step (§16)
        self._public_step += 1
        if (
            self._status == _PUBLIC_RUNNING
            and self._public_step >= self._scenario.termination.max_planning_steps
        ):
            self._status = "FAIL_MAX_STEPS"

        frame = self._store.get(self._store.current_id)
        record["planning_step"] = self._public_step
        record["episode_status"] = self._status
        record["robot_pose_after"] = self._backend.robot_pose()[0]
        self._log(record)

        return PublicStepResult(
            episode_status=self._status,
            planning_step=self._public_step,
            feedback=self._last_feedback or self._feedback(PublicActionFeedback.INVALID_ACTION),
            observation=self._observation(frame),
        )

    # ------------------------------------------------------ action handlers

    def _step_move(self, action, record) -> None:
        d = action.distance_cm
        if d == 0 or not (self._protocol.min_move_cm <= abs(d) <= self._protocol.max_move_cm):
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "MOVE_PARAMETER_RANGE"
            return
        pose = self._backend.robot_pose()
        safe, final = execute_move(
            self._backend, pose, d / 100.0, self._base_half_extent,
            sample_m=0.05, margin=0.03,
        )
        if not safe:
            self._last_feedback = self._feedback(PublicActionFeedback.UNSAFE)
            record["feedback"] = "UNSAFE"
            record["private_reason"] = "MOVE_BLOCKED"
            return
        self._backend.settle(5)
        self._capture_current()
        self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
        record["feedback"] = "EXECUTED"

    def _step_turn(self, action, record) -> None:
        a = action.angle_deg
        if a == 0 or not (self._protocol.min_turn_deg <= abs(a) <= self._protocol.max_turn_deg):
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "TURN_PARAMETER_RANGE"
            return
        pose = self._backend.robot_pose()
        safe, _ = execute_turn(
            self._backend, pose, a, self._base_half_extent,
            sample_deg=5.0, margin=0.03,
        )
        if not safe:
            self._last_feedback = self._feedback(PublicActionFeedback.UNSAFE)
            record["feedback"] = "UNSAFE"
            record["private_reason"] = "TURN_BLOCKED"
            return
        self._backend.settle(5)
        self._capture_current()
        self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
        record["feedback"] = "EXECUTED"

    def _step_report_done(self, record) -> None:
        if goal_satisfied(self._session.world_state, self._scenario):
            self._status = "SUCCESS"
            self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
            record["feedback"] = "EXECUTED"
        else:
            self._status = "FAIL_FALSE_COMPLETION"
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "GOAL_UNSATISFIED"

    def _step_point(self, action: "PointAction", frame, record):
        if not self._store.is_current(action.point.frame_id):
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "STALE_FRAME"
            return None
        record["input_frame_id"] = action.point.frame_id
        record["point"] = [action.point.x, action.point.y]

        if action.skill == "OBSERVE":
            return self._step_observe(action, frame, record)

        resolved = self._resolve_point(frame, action.point.x, action.point.y, record)
        if resolved is None:
            return None
        entity, surface = resolved
        return self._dispatch_entity_skill(action.skill, entity, record)

    def _resolve_point(self, frame, x: float, y: float, record):
        """Point -> (entity, private surface point). Segmentation bridge when
        the host provides it (§8); PhysX raytest fallback on rgb-only hosts
        (§8.7 approximation, recorded per frame)."""
        import numpy as np

        seg_has_data = bool(np.asarray(frame.instance_segmentation).any())
        if not seg_has_data or frame.meta.get("visual_grounding_supported") is False:
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record.update(feedback="INVALID_ACTION", private_reason="VISIBILITY_UNAVAILABLE")
            return None
        try:
            entity, surface, detail = self._bridge.resolve(
                frame, x, y,
                lambda instance: self._backend.instance_to_entity(instance, frame)
                if "instance_labels" in frame.meta else self._backend.instance_to_entity(instance),
            )
        except VisualGroundingError as e:
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record.update({"feedback": "INVALID_ACTION", "private_reason": e.subreason})
            return None
        record.update(
            selected_entity=entity,
            grounding_version=GROUNDING_VERSION,
            selected_surface_point_world=[round(float(v), 4) for v in surface],
            **{f"bridge_{k}": v for k, v in detail.items()},
        )
        return entity, surface

    def _skill_applicable(self, skill: str, entity: str) -> bool:
        """Adapter-level applicability (§3.2): OPEN on a non-openable object,
        GRASP on fixed furniture, CLOSE on a closed drawer — all semantic
        non-applicabilities, mapped to INVALID_ACTION before any kinematics
        run. State-dependent backstops stay in the legacy validators."""
        from rummagebench.object_interface.base import build_object_interface

        info = self._backend.describe_entity(entity)
        if info is None:
            return False
        try:
            obj = build_object_interface(info, self._backend)
            return any(
                c.skill == skill
                for c in obj.available_skills(self._session.robot,
                                              self._session.world_state)
            )
        except Exception:
            # Unknown adapter failures must not bypass applicability checks.
            raise

    def _dispatch_entity_skill(self, skill: str, entity: str, record):
        if not self._skill_applicable(skill, entity):
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "SKILL_NOT_APPLICABLE_TO_ENTITY"
            return None
        legacy_action = Action(
            skill=skill, target=TargetRef(type=TargetKind.ENTITY, value=entity)
        )
        result = self._session.act(legacy_action)
        record["legacy_failure_reason"] = result.failure_reason.value
        record["legacy_executed"] = result.executed
        record["feedback"] = (self._map_legacy_feedback(result)).code.value
        self._last_feedback = self._map_legacy_feedback(result)
        if result.executed:
            self._backend.settle(5)
            self._backend.flush_render()
            return self._capture_current()
        return None  # world unchanged: keep the current frame

    def _map_legacy_feedback(self, result) -> Feedback:
        if result.executed and result.failure_reason == FailureReason.NONE:
            return self._feedback(PublicActionFeedback.EXECUTED)
        reason = result.failure_reason
        if reason == FailureReason.UNREACHABLE:
            return self._feedback(PublicActionFeedback.OUT_OF_CAPABILITY)
        if reason in (FailureReason.COLLISION, FailureReason.UNSAFE_ACTION):
            return self._feedback(PublicActionFeedback.UNSAFE)
        return self._feedback(PublicActionFeedback.INVALID_ACTION)

    def _step_observe(self, action, frame, record):
        resolved = self._resolve_point(frame, action.point.x, action.point.y, record)
        if resolved is None:
            return None
        entity, _surface = resolved
        record.update(selected_entity=entity, observe_target=True)

        accepted = run_observe(
            self._backend, entity, self._observe_cfg,
            base_half_extent=self._base_half_extent, collision_margin=0.03,
            render_private_frame=lambda: self._backend.capture_visual_frame(),
            visible_pixel_count=lambda fr, ent: self._backend.entity_visible_pixels(fr, ent),
            settle=lambda n: self._backend.settle(n),
            log=lambda r: record.setdefault("observe_viewpoints", []).append(
                {k: v for k, v in r.items() if k != "rgb"}
            ),
        )
        self._observe_views = [
            ObserveView(view_index=i, image_png_b64=_png_b64(v["rgb"]))
            for i, v in enumerate(accepted)
        ]
        if not self._observe_views:
            self._last_feedback = self._feedback(PublicActionFeedback.UNSAFE)
            record["feedback"] = "UNSAFE"
            record["private_reason"] = "NO_VALID_OBSERVE_VIEWPOINT"
            return None
        self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
        record["feedback"] = "EXECUTED"
        record["accepted_observe_views"] = len(accepted)
        return self._capture_current()  # fresh CURRENT frame after restore
