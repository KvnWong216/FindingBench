"""AGENT-mode orchestrator: the final visual interaction protocol.

Wraps the frozen ORACLE core (BenchmarkSession: semantic/state validation,
pinocchio IK Stage A, coal collision Stage B, symbolic execution) behind the
public visual skill protocol. The agent sees ONLY PublicObservation; every
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
    DiscreteTurnAction,
    PointAction,
    PublicAction,
    PublicActionFeedback,
    PublicObservation,
    EmbodimentObservation,
    DetailView,
    PublicStepResult,
    ReportDoneAction,
    REPORT_COMPLETE_ALIAS,
    ViewAction,
    fine_reason_for,
)
from rummagebench.core.events import (
    AGENT_ACTION_PROTOCOL,
    AGENT_FEEDBACK_PROTOCOL,
    EVALUATION_EVENTS_SCHEMA_VERSION,
    EVENT_TYPE_ACTION,
    EVENT_TYPE_INFRASTRUCTURE_ERROR,
    EVENT_TYPE_RESET,
    EVENT_TYPE_TERMINAL,
    SESSION_MODE_AGENT,
    run_metadata_event,
)
from rummagebench.core.types import Action, EpisodeStatus, FailureReason, TargetKind, TargetRef
from rummagebench.perception.frame_store import FrameStore
from rummagebench.perception.visual_bridge import (
    GROUNDING_VERSION,
    VisualBridge,
    VisualGroundingError,
)
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
        base_half_extent: float | None = None,
    ):
        if base_half_extent is None:
            base_half_extent = scenario.robot.base_half_extent
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
        self._base_half_extent = base_half_extent
        # Only the current panorama and latest detail are actionable. Public
        # history is recorded separately; do not retain 64 high-resolution
        # private RGB/depth/segmentation/ray bundles in memory.
        self._store = FrameStore(max_frames=2)
        self._bridge = VisualBridge()
        self._public_step = 0
        self._inspection_view = None
        self._view_angles = None
        self._panorama_frame = None
        self._status = _PUBLIC_RUNNING
        self._last_feedback: Feedback | None = None
        self._trace: list[dict[str, Any]] = []
        self._trace_path: Path | None = None
        # optional model identity recorded in the run metadata event (set by
        # the batch harness before reset; null when the harness omits it)
        self.model_version: str | None = None
        self.expose_embodiment = False
        # committed skill events of THIS episode (R04 temporal predicates:
        # only committed skill results may generate events)
        self._skill_events: list[dict[str, Any]] = []
        from pydantic import TypeAdapter

        self._action_adapter = TypeAdapter(PublicAction)

    # ------------------------------------------------------------- plumbing

    def set_planning_budget(self, max_steps: int) -> None:
        """Configure the actual visual session before reset, not its source copy."""
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if self._public_step:
            raise RuntimeError('cannot change planning budget during an episode')
        self._scenario.termination.max_planning_steps = max_steps
        self._session._scenario.termination.max_planning_steps = max_steps

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

    def _protocol_labels(self) -> dict[str, str]:
        return {
            "schema_version": EVALUATION_EVENTS_SCHEMA_VERSION,
            "session_mode": SESSION_MODE_AGENT,
            "action_protocol": AGENT_ACTION_PROTOCOL,
            "feedback_protocol": AGENT_FEEDBACK_PROTOCOL,
        }

    def _state_fingerprint(self) -> dict[str, Any]:
        """Evaluator-private pre/post state fingerprint (R03): robot base
        pose, benchmark holding state and the open-entity set."""
        ws = self._session.world_state
        opens = []
        try:
            opens = sorted(
                e for e in self._backend.entity_names() if self._backend.is_open(e)
            )
        except Exception:
            opens = None  # fingerprint best-effort; never breaks a step
        return {
            "robot_pose": [float(v) for v in self._backend.robot_pose()[0]],
            "held_object": ws.held_object,
            "open_entities": opens,
        }

    def entity_pixel(self, entity: str) -> tuple[float, float] | None:
        """Evaluator-private convenience for scripted DRIVERS ONLY: centre
        pixel of ``entity`` in the current frame, normalized. Never part of
        the public observation and never sent to a model."""
        frame = self._store.get(self._store.current_id)
        if frame is None:
            return None
        seg = np.asarray(frame.instance_segmentation)
        for instance in np.unique(seg):
            inst = int(instance)
            if inst == 0:
                continue
            label = (
                self._backend.instance_to_entity(inst, frame)
                if "instance_labels" in frame.meta
                else self._backend.instance_to_entity(inst)
            )
            if label != entity:
                continue
            ys, xs = np.nonzero(seg == instance)
            if xs.size == 0:
                continue
            return (
                float(xs.mean()) / max(frame.image_width - 1, 1),
                float(ys.mean()) / max(frame.image_height - 1, 1),
            )
        return None

    def _capture_current(self):
        if getattr(self._scenario.robot, "panorama", None) is not None:
            frame = self._backend.capture_panorama_frame()
            self._register_frame(frame)
            self._panorama_frame = frame
            # A fresh state capture provides only the primary panorama.
            # A pinhole supplement is generated solely by an explicit VIEW.
            self._inspection_view = None
            self._view_angles = None
        else:
            frame = self._backend.capture_visual_frame()
        if frame is not self._panorama_frame:
            self._register_frame(frame)
        return frame

    def _register_frame(self, frame):
        frame.frame_id = self._store.new_frame_id()
        self._store.register(frame)
        return frame

    def _feedback(self, code: PublicActionFeedback) -> Feedback:
        from rummagebench.core.public_types import FEEDBACK_MESSAGES

        return Feedback(code=code, message=FEEDBACK_MESSAGES[code])

    def _embodiment_observation(self, frame):
        if not self.expose_embodiment:
            return None
        from scipy.spatial.transform import Rotation
        position, quaternion = self._backend.robot_pose()
        base_rotation = Rotation.from_quat(quaternion).as_matrix()
        # Perspective frames retain the panorama reference extrinsics. Compose
        # the viewport rotation before expressing body axes in optical space.
        camera = np.asarray(frame.camera_extrinsics, dtype=float)
        optical_rotation = camera[:3, :3]
        if self._view_angles is not None and frame.meta.get('projection') == 'perspective' and not frame.meta.get('native_view_extrinsics'):
            yaw, pitch = self._view_angles
            optical_rotation = optical_rotation @ (
                Rotation.from_euler('y', -yaw, degrees=True).as_matrix() @
                Rotation.from_euler('x', pitch, degrees=True).as_matrix())
        spec = self._scenario.robot
        kin = spec.kinematics
        return EmbodimentObservation(
            robot_model=spec.model,
            executor_footprint_half_extent_m=self._base_half_extent,
            hand_capacity=spec.hand_capacity,
            end_effector_link=kin.end_effector_link if kin else None,
            executor_holding=self._session.world_state.held_object is not None,
            projection=frame.meta.get('projection', 'perspective'),
            image_size=(frame.image_width, frame.image_height),
            view_yaw_deg=self._view_angles[0] if self._view_angles else None,
            view_pitch_deg=self._view_angles[1] if self._view_angles else None,
            camera_origin_in_base_m=tuple(base_rotation.T @ (camera[:3, 3] - np.asarray(position))),
            body_forward_in_camera=tuple(optical_rotation.T @ base_rotation[:, 0]),
            body_up_in_camera=tuple(optical_rotation.T @ base_rotation[:, 2]),
            perspective_hfov_deg=frame.meta.get("hfov_deg", 90.0) if self._view_angles and frame.meta.get("projection") == "perspective" else None,
        )

    def _observation(self, frame) -> PublicObservation:
        primary = self._panorama_frame if self._scenario.robot.panorama is not None else frame
        detail = None
        if primary is not frame and self._view_angles is not None:
            detail = DetailView(frame_id=frame.frame_id, image_png_b64=_png_b64(frame.rgb),
                                yaw_deg=self._view_angles[0], pitch_deg=self._view_angles[1],
                                width=frame.image_width, height=frame.image_height,
                                hfov_deg=frame.meta.get("hfov_deg", 90.0),
                                zoom=frame.meta.get("view_zoom", 1.0),
                                source=frame.meta.get("view_source", "panorama_reprojection"))
        return PublicObservation(
            instruction=self._scenario.instruction,
            frame_id=primary.frame_id,
            image_png_b64=_png_b64(primary.rgb),
            planning_step=self._public_step,
            max_planning_steps=self._scenario.termination.max_planning_steps,
            skill_library=list(PUBLIC_SKILLS),
            feedback=self._last_feedback,
            embodiment=self._embodiment_observation(primary),
            detail_view=detail,
            inspection_view=self._inspection_view,
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
        self._inspection_view = None
        self._view_angles = None
        # the AGENT episode starts at the scenario spawn anchor: the visual
        # protocol has no NAV anchors, so the spawn is applied here
        from rummagebench.core.scenario import AnchorSpec

        if getattr(self._scenario, "behavior_task", None) is None:
            init = self._scenario.anchors[self._scenario.robot.init_anchor]
            self._backend.teleport_robot(
                AnchorSpec(position=list(init.position), orientation=list(init.orientation))
            )
            self._backend.settle(5)
        self._store.reset()
        self._panorama_frame = None
        self._public_step = 0
        self._last_feedback = None
        self._skill_events = []
        self._trace.clear()
        frame = self._capture_current()
        if frame.meta.get("visual_grounding_supported") is False:
            raise RuntimeError(
                "AGENT visual grounding requires synchronized renderer instance "
                "segmentation, depth and camera geometry; RGB-only raycasts are "
                "diagnostic-only and do not establish RGB visibility. "
                f"Missing: {frame.meta.get('unsupported_reason', 'private modalities')}"
            )
        # R03: run metadata first, then the versioned reset event. Model
        # identity/config hash are null unless the harness supplies them.
        self._log(run_metadata_event(
            self._session.episode_id, self._scenario,
            session_mode=SESSION_MODE_AGENT,
            action_protocol=AGENT_ACTION_PROTOCOL,
            feedback_protocol=AGENT_FEEDBACK_PROTOCOL,
            model_version=getattr(self, "model_version", None),
            config_hash=None,
            seed=None,
        ))
        self._log({
            "event_type": EVENT_TYPE_RESET,
            **self._protocol_labels(),
            "episode_id": self._session.episode_id,
            "frame_id": frame.frame_id,
            "state_fingerprint": self._state_fingerprint(),
        })
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

        self._inspection_view = None
        record: dict[str, Any] = {
            "event_type": EVENT_TYPE_ACTION,
            **self._protocol_labels(),
            "episode_id": self._session.episode_id,
            "raw_action": raw,
            "robot_pose_before": self._backend.robot_pose()[0],
            "state_fingerprint_before": self._state_fingerprint(),
        }
        # R05: the optional public report alias is normalized at the parse
        # boundary; the raw action is preserved verbatim in the trace
        if isinstance(raw, dict) and raw.get("skill") == REPORT_COMPLETE_ALIAS:
            raw = {**raw, "skill": "REPORT_DONE"}
            record["alias_normalized"] = f"{REPORT_COMPLETE_ALIAS}->REPORT_DONE"
        try:
            action = self._action_adapter.validate_python(raw)
        except ValidationError as error:
            record["schema_errors_private"] = [
                {"type": item["type"], "loc": list(item["loc"])}
                for item in error.errors(include_input=False, include_url=False)
            ]
            action = None
        # R03: parsed result of the raw output (None when the public schema
        # rejected it), so evaluation never has to re-parse raw output
        record["parsed_action"] = (
            action.model_dump(mode="json") if action is not None else None
        )

        frame = self._store.get(self._store.current_id)
        record["current_frame_id"] = frame.frame_id if frame is not None else None

        try:
            if action is None:
                self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
                record["feedback"] = "INVALID_ACTION"
                record["private_reason"] = "SCHEMA"
            elif isinstance(action, DiscreteTurnAction):
                from rummagebench.skills.move import execute_turn
                from rummagebench.core.failure_explanations import explain
                angle = {"left": 90, "right": -90, "back": 180}[action.direction]
                safe, _ = execute_turn(self._backend, self._backend.robot_pose(), angle,
                                       self._base_half_extent, sample_deg=5.0, margin=0.03)
                record.update(feedback="EXECUTED" if safe else "UNSAFE",
                              private_reason=None if safe else "TURN_BLOCKED",
                              turn_angle_deg=angle)
                if safe:
                    self._backend.settle(5)
                    self._capture_current()
                self._last_feedback = Feedback(code=PublicActionFeedback(record["feedback"]), message=explain(record))
            elif isinstance(action, ViewAction):
                if getattr(self._scenario.robot, "panorama", None) is None:
                    self._last_feedback = self._feedback(PublicActionFeedback.OUT_OF_CAPABILITY)
                    record["feedback"] = "OUT_OF_CAPABILITY"
                    record["private_reason"] = "VIEW_UNAVAILABLE"
                else:
                    self._view_angles = (action.yaw, action.pitch)
                    from rummagebench.perception.panorama import perspective_view
                    capture_view = getattr(self._backend, 'capture_view_frame', None)
                    config = self._scenario.robot.panorama
                    hfov = float(np.degrees(2 * np.arctan(1 / action.zoom)))
                    detail = (capture_view(action.yaw, action.pitch,
                              config.detail_width, config.detail_height, zoom=action.zoom)
                              if capture_view else perspective_view(
                                  self._panorama_frame, action.yaw, action.pitch,
                                  width=config.detail_width, height=config.detail_height, hfov=hfov))
                    detail.meta.update(view_zoom=action.zoom, hfov_deg=hfov)
                    self._register_frame(detail)
                    self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
                    record["feedback"] = "EXECUTED"
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
            record["event_type"] = EVENT_TYPE_INFRASTRUCTURE_ERROR
            record["private_reason"] = f"ENGINE:{type(e).__name__}"
            record["run_invalidated"] = True
            record["state_fingerprint_after"] = self._state_fingerprint()
            self._log(record)
            raise RuntimeError("episode invalidated by simulator failure; reset required") from None

        # The supplement belongs only to the response to a successful VIEW.
        # Consume it AFTER dispatch so the next point action may reference it.
        # Even rejected/non-parsed actions return to panorama-only observation.
        if not (isinstance(action, ViewAction) and record.get("feedback") == "EXECUTED"):
            self._view_angles = None
            if self._panorama_frame is not None:
                self._store.register(self._panorama_frame)

        # horizon: every submitted action consumes exactly one step (§16)
        self._public_step += 1
        if (
            self._status == _PUBLIC_RUNNING
            and self._public_step >= self._scenario.termination.max_planning_steps
        ):
            self._status = "FAIL_MAX_STEPS"

        frame = self._store.get(self._store.current_id)
        # R03 execution fields: whether the step took effect and whether its
        # postcondition landed (entity skills report the legacy verdict;
        # other skills default to the public EXECUTED verdict)
        record["executed"] = record.get("feedback") == "EXECUTED"
        record["postcondition_satisfied"] = record.get(
            "postcondition_satisfied", record["executed"]
        )
        # R05: canonical fine code for this branch (evaluator-private; the
        # public boundary stays the four feedback classes)
        record["fine_reason"] = fine_reason_for(
            record.get("private_reason"),
            record.get("feedback"),
            self._status,
            skill=(record.get("parsed_action") or {}).get("skill"),
        )
        record["binding"] = {
            "frame_id": record.get("input_frame_id") or record.get("current_frame_id"),
            "point": record.get("point"),
            "selected_entity": record.get("selected_entity"),
            "grounding_version": record.get("grounding_version"),
        }
        record["skill_events"] = list(record.get("skill_events", []))
        record["planning_step"] = self._public_step
        record["episode_status"] = self._status
        record["robot_pose_after"] = self._backend.robot_pose()[0]
        record["state_fingerprint_after"] = self._state_fingerprint()
        record["frame_id_after"] = frame.frame_id if frame is not None else None
        self._log(record)
        if self._status != _PUBLIC_RUNNING:
            # R03: terminal transitions are explicit events, never action rows
            self._log({
                "event_type": EVENT_TYPE_TERMINAL,
                **self._protocol_labels(),
                "episode_id": self._session.episode_id,
                "episode_status": self._status,
                "planning_step": self._public_step,
                "trigger_action": raw,
            })

        return PublicStepResult(
            episode_status=self._status,
            planning_step=self._public_step,
            feedback=self._last_feedback or self._feedback(PublicActionFeedback.INVALID_ACTION),
            observation=self._observation(frame),
        )

    # ------------------------------------------------------ action handlers

    def _step_report_done(self, record) -> None:
        if goal_satisfied(
            self._session.world_state, self._scenario,
            backend=self._backend, events=self._skill_events,
        ):
            self._status = "SUCCESS"
            self._last_feedback = self._feedback(PublicActionFeedback.EXECUTED)
            record["feedback"] = "EXECUTED"
        else:
            self._status = "FAIL_FALSE_COMPLETION"
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "GOAL_UNSATISFIED"

    def _step_point(self, action: "PointAction", frame, record):
        valid_ids = {self._store.current_id}
        if self._scenario.robot.panorama is not None and self._panorama_frame is not None:
            valid_ids.add(self._panorama_frame.frame_id)
        if action.point.frame_id not in valid_ids:
            self._last_feedback = self._feedback(PublicActionFeedback.INVALID_ACTION)
            record["feedback"] = "INVALID_ACTION"
            record["private_reason"] = "STALE_FRAME"
            return None
        frame = self._panorama_frame if self._panorama_frame is not None and action.point.frame_id == self._panorama_frame.frame_id else self._store.get(action.point.frame_id)
        if frame is None:
            raise RuntimeError("current image missing from frame store")
        record["input_frame_id"] = action.point.frame_id
        record["point"] = [action.point.x, action.point.y]

        resolved = self._resolve_point(frame, action.point.x, action.point.y, record)
        if resolved is None:
            return None
        entity, surface = resolved
        if action.skill == "CHECK":
            from rummagebench.core.public_types import InspectionView
            info = self._backend.describe_entity(entity)
            capture = getattr(self._backend, 'capture_check_image', None)
            supported = getattr(self._backend, 'check_appearance_supported', None)
            if (info is None or not info.graspable or info.openable or capture is None
                    or (supported is not None and not supported(entity))):
                record.update(feedback="OUT_OF_CAPABILITY", private_reason="CHECK_UNSUPPORTED")
                self._last_feedback = Feedback(code=PublicActionFeedback.OUT_OF_CAPABILITY,
                    message="Check not executed: exterior inspection supports movable, non-articulated objects only.")
                return None
            rgb = capture(entity)
            self._inspection_view = InspectionView(image_png_b64=_png_b64(rgb),
                width=rgb.shape[1], height=rgb.shape[0])
            record.update(feedback="EXECUTED", inspection_source="isolated_object_multiview")
            self._last_feedback = Feedback(code=PublicActionFeedback.EXECUTED,
                message="Check executed: four exterior appearance views are shown. The object and robot have not moved. This inspection image cannot be used for scene point actions.")
            return frame
        if action.skill == "MOVE_TO":
            from rummagebench.skills.move import execute_move_to, is_navigation_floor
            from rummagebench.core.failure_explanations import explain
            from rummagebench.skills.move import NOT_STAND_TARGETS, execute_move_to_entity
            info = self._backend.describe_entity(entity)
            is_floor = is_navigation_floor(self._backend, entity, info)
            if info is None or (not is_floor and info.category in NOT_STAND_TARGETS):
                record.update(feedback="INVALID_ACTION", private_reason="NAV_NOT_TARGETABLE")
            else:
                if is_floor:
                    success, reason, evidence = execute_move_to(self._backend, surface, self._base_half_extent)
                else:
                    # visible furniture / object: automatic stand pose facing it
                    success, reason, evidence = execute_move_to_entity(
                        self._backend, entity, surface, self._base_half_extent)
                record["navigation_private"] = evidence
                record.update(feedback="EXECUTED" if success else "OUT_OF_CAPABILITY", private_reason=reason)
                if success:
                    self._backend.flush_render()
                    frame = self._capture_current()
            self._last_feedback = Feedback(code=PublicActionFeedback(record["feedback"]), message=explain(record))
            return frame
        if action.skill == 'PLACE' and self._session.world_state.held_object is not None:
            support = getattr(self._backend, 'clicked_place_region', None)
            selected = support(entity, surface) if callable(support) else None
            if selected is None:
                record.update(feedback='OUT_OF_CAPABILITY', private_reason='PLACE_UNSUPPORTED_CLICK')
                self._last_feedback = Feedback(code=PublicActionFeedback.OUT_OF_CAPABILITY,
                    message='The clicked surface does not provide enough level support for the held object.')
                return None
            previous = getattr(self._backend, '_clicked_place', None)
            self._backend._clicked_place = (entity, selected[0], selected[1])
            try:
                return self._dispatch_entity_skill(action.skill, entity, record)
            finally:
                self._backend._clicked_place = previous
        if action.skill in ("OPEN", "CLOSE"):
            previous = getattr(self._backend, "_clicked_articulation", None)
            self._backend._clicked_articulation = (entity, list(surface))
            try:
                return self._dispatch_entity_skill(action.skill, entity, record)
            finally:
                self._backend._clicked_articulation = previous
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
        if not result.executed:
            # Keep the solver / collision evidence on the evaluator side.
            # Never serialize this candidate-level evidence in PublicObservation.
            record["manipulation_diagnostic_private"] = list(result.events)
        postcondition_ok = bool(
            (result.observation.previous_action_result or {}).get(
                "postcondition_satisfied")
        )
        record["postcondition_satisfied"] = postcondition_ok
        # committed skill events feed the temporal predicates (R04); only
        # committed executions may generate events
        if result.executed:
            record["skill_events"] = list(result.events)
            self._skill_events.extend(result.events)
        if result.executed and not postcondition_ok:
            # commit discipline (skills/realization.py) makes this
            # unreachable; if it ever happens the world may be half-committed
            # and the step must NOT surface as a model-facing verdict
            raise RuntimeError(
                f"{skill} reported EXECUTED with an unsatisfied postcondition"
            )
        record["feedback"] = (self._map_legacy_feedback(result)).code.value
        self._last_feedback = self._map_legacy_feedback(result)
        if result.executed:
            self._backend.settle(5)
            self._backend.flush_render()
            frame = self._capture_current()
            # R09: delivered-frame observation evidence for the search-state
            # tracker — visible entities (pixel counts) in THIS post-action
            # frame. Evaluator-private; unavailable evidence stays null and
            # the tracker keeps the unknown state.
            try:
                record["observed_evidence"] = {
                    name: self._backend.entity_visible_pixels(frame, name)
                    for name in self._backend.entity_names()
                    if self._backend.entity_visible_pixels(frame, name) > 0
                }
            except Exception:
                record["observed_evidence"] = None
            return frame
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
