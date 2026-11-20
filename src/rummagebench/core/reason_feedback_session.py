"""Opt-in experimental session: post-action plain-language reasons.

Used by the v4.3 humanclaw policy. Not imported by the historical v4.2. Object categories here intentionally cross a NEW information
boundary, named reason-feedback-v1. No recommendation, precheck, action alteration,
extra inference, or automatic retry. Only the failed action's evidence is exposed.
"""
from __future__ import annotations

from rummagebench.core.visual_session import VisualProtocolSession
from rummagebench.core.public_types import Feedback, PublicActionFeedback
from rummagebench.skills.move import navigation_aabbs, yaw_from_quat
from rummagebench.core.failure_explanations import explain, sweep

PROTOCOL = "reason-feedback-v1"


class ReasonFeedbackSession(VisualProtocolSession):
    def _protocol_labels(self):
        return {**super()._protocol_labels(), "feedback_protocol": PROTOCOL}

    def _log(self, record):
        if record.get("event_type") == "action":
            diagnostic = None
            selected_object = None
            if (record.get("private_reason") or record.get("legacy_failure_reason")) in ("SKILL_NOT_APPLICABLE_TO_ENTITY", "INVALID_STATE"):
                try:
                    # Explain only the object already selected by this failed action.
                    info = self._backend.describe_entity(record.get("selected_entity"))
                    if info is not None:
                        category = getattr(info, "category", None)
                        category = str(category).replace("_", " ") if category else None
                        if category and any(c in category for c in ("{", "}", "\n")):
                            category = None
                        selected_object = dict(
                            category=category,
                            cannot_pick_up=bool(info.fixed_base or not info.graspable),
                            openable=bool(info.fixed_base and info.openable),
                            place_supported=bool(
                                (info.fixed_base and info.openable and info.is_receptacle)
                                or (info.is_receptacle and not info.graspable)),
                            holding_empty=self._session.world_state.held_count == 0,
                            holding_full=self._session.world_state.held_count >= self._session.robot.hand_capacity)
                        if selected_object["openable"]:
                            selected_object["is_open"] = bool(self._backend.is_open(record["selected_entity"]))
                except Exception:
                    # Missing explanatory evidence never changes the verdict.
                    selected_object = None
            if record.get("private_reason") in ("MOVE_BLOCKED", "TURN_BLOCKED"):
                try:
                    # The rejected motion did not change the world. Read current
                    # geometry AFTER the executor rejected it, not before deciding.
                    backend = self._backend
                    controlled = getattr(backend, "robot_entity_names", lambda: set())()
                    obstacles = []
                    for name in backend.entity_names():
                        info = backend.describe_entity(name)
                        if name in controlled or info is None or getattr(backend, "is_holding", lambda _: False)(name):
                            continue
                        category = getattr(info, "category", None)
                        # Semantic category only; never raw instance IDs or coordinates.
                        category = str(category).replace("_", " ") if category else None
                        if category and "cabinet" in category:
                            category = "cabinet"
                        if category and any(c in category for c in ("{", "}", "\n")):
                            category = None
                        for lo, hi in navigation_aabbs(backend, name):
                            obstacles.append(dict(entity=name, category=category, lo=list(lo), hi=list(hi)))
                    yaw = yaw_from_quat(backend.robot_pose()[1])
                    diagnostic = sweep(record, yaw, obstacles, half_extent=self._base_half_extent)
                    diagnostic["verified_live"] = not diagnostic["predicted_safe"]
                    diagnostic["evidence"] = "post_rejection_live_executor_geometry"
                except Exception as error:
                    # Explanations must never change execution or score.
                    diagnostic = dict(verified_live=False, diagnostic_error=type(error).__name__)
            message = explain(record, (diagnostic or {}).get("blockers", []),
                              verified=(diagnostic or {}).get("verified_live", False),
                              selected_object=selected_object)
            self._last_feedback = Feedback(code=PublicActionFeedback(record["feedback"]), message=message)
            record["public_feedback_message"] = message
            record["failure_diagnostic_private"] = diagnostic
            if selected_object is not None:
                record["selected_object_feedback_evidence"] = selected_object
        super()._log(record)
