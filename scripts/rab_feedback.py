"""Single source of truth for FindingBench RAB public proprioceptive feedback
and task-stage mapping (protocol findingbench_rab_pinhole_v1).

Two feedback modes are supported via ``FINDINGBENCH_RAB_FEEDBACK_MODE``:
* ``four_class_v1`` (default): legacy 4-class ``{code, message}``.
* ``proprioceptive_v2``: ``{code, detail_code, message}`` with strictly
  declarative, zero-directive, zero-extra-information-gain physical facts
  (no numeric measurements, no entity names/IDs, no action advice, no
  REPORT_DONE ground-truth leakage).
"""
from __future__ import annotations

import os
from typing import Any

FEEDBACK_MODES = ("four_class_v1", "proprioceptive_v2")

# (private_error_head, optional_skill) -> (detail_code, neutral_factual_message)
# Every message MUST be a pure physical/state fact: no directives, no numbers,
# no entity names or scene-semantic pruning hints.
PROPRIOCEPTIVE_BY_ERROR: dict[str, tuple[str, str]] = {
    "NAV_NO_CLEAR_ROUTE": ("NAV_BLOCKED", "Base motion blocked along path."),
    "NAV_SEGMENT_CLEARANCE_BLOCKED": ("NAV_BLOCKED", "Base motion blocked along path."),
    "BASE_MOTION_SHORTFALL": ("NAV_BLOCKED", "Base motion blocked along path."),
    "TURN_RADIAL_CLEARANCE_BLOCKED": ("TURN_BLOCKED", "Base rotation blocked by nearby clearance."),
    "CLICK_NON_ARTICULATED": ("NO_ARTICULATION_HIT", "No operable joint at the selected point."),
    "ARTICULATION_TOO_FAR": ("OUT_OF_REACH", "Target point is out of reach."),
    "PRECONDITION_ALREADY_OPEN": ("JOINT_AT_LIMIT", "Joint is already at the target state."),
    "PRECONDITION_ALREADY_CLOSED": ("JOINT_AT_LIMIT", "Joint is already at the target state."),
    "CLICK_NON_INTERACTABLE": ("NO_TARGET_AT_POINT", "No interactable target at the selected point."),
    "PRECONDITION_HAND_OCCUPIED": ("GRIPPER_OCCUPIED", "Gripper is currently holding an object."),
    "PRECONDITION_HAND_EMPTY": ("GRIPPER_EMPTY", "Gripper is currently empty."),
    "GRASP_TOO_FAR": ("OUT_OF_REACH", "Target point is out of reach."),
    "PLACE_TOO_FAR": ("OUT_OF_REACH", "Target point is out of reach."),
    "ARM_IK_TARGET_FAILED": ("IK_INFEASIBLE", "No feasible arm configuration for the selected point."),
    "ARM_IK_PLACE_FAILED": ("IK_INFEASIBLE", "No feasible arm configuration for the selected point."),
    "ARM_REACH_FAILED": ("IK_INFEASIBLE", "No feasible arm configuration for the selected point."),
    "HAND_PATH_BLOCKED": ("ARM_PATH_OBSTRUCTED", "Arm path to the selected point is obstructed."),
    "ARTICULATION_BLOCKED": ("ARTICULATION_OBSTRUCTED", "Articulated motion blocked or disturbed nearby objects."),
    "UNSAFE_OVERSIZED_GRASP": ("OBJECT_EXCEEDS_GRIPPER", "Object size exceeds gripper capacity."),
    "CLICK_NON_HORIZONTAL_SURFACE": ("SURFACE_NOT_LEVEL", "Surface at the selected point is not horizontal."),
    "PLACE_ON_SELF": ("INVALID_SUPPORT", "Selected point cannot serve as a placement support."),
    "UNSAFE_PLACEMENT_UNSTABLE": ("PLACEMENT_UNSTABLE", "Object did not settle stably at the selected point."),
    "STALE_FRAME_ID": ("STALE_FRAME", "Frame identifier does not match current observation."),
    "SCHEMA_INVALID": ("PROTOCOL_ERROR", "Action format or protocol state is invalid."),
    "UNSUPPORTED_SKILL": ("PROTOCOL_ERROR", "Action format or protocol state is invalid."),
    "EPISODE_ALREADY_TERMINATED": ("PROTOCOL_ERROR", "Action format or protocol state is invalid."),
    "INFRA_EXCEPTION": ("RUNTIME_ERROR", "Action could not be completed due to a runtime error."),
}

# Skill-specific wording overrides that keep the same detail_code without adding hints.
SKILL_MESSAGE_OVERRIDES: dict[tuple[str, str], str] = {
    ("CLICK_NON_INTERACTABLE", "CHECK"): "No inspectable target at the selected point.",
    ("CLICK_NON_INTERACTABLE", "GRASP"): "No graspable target at the selected point.",
}

# Six task-centric execution stages for Task-Stage Funnel analysis.
TASK_STAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("stage1_navigation", "Stage 1: Navigation & Approach", ("MOVE_TO", "TURN")),
    ("stage2_articulation", "Stage 2: Container Articulation", ("OPEN", "CLOSE")),
    ("stage3_inspection", "Stage 3: Candidate Inspection", ("CHECK",)),
    ("stage4_grasp", "Stage 4: Target Grasp", ("GRASP",)),
    ("stage5_place", "Stage 5: Target Placement", ("PLACE",)),
    ("stage6_completion", "Stage 6: Completion Report", ("REPORT_DONE",)),
)
SKILL_TO_STAGE: dict[str, str] = {
    skill: stage_key for stage_key, _, skills in TASK_STAGES for skill in skills
}


def error_head(private_error: str | None) -> str | None:
    if not private_error:
        return None
    return str(private_error).split(":", 1)[0].strip()


def detail_for(code: str, private_error: str | None, skill: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(detail_code, factual_message)`` for one action outcome."""
    if code == "EXECUTED":
        return None, None
    head = error_head(private_error)
    if head and head in PROPRIOCEPTIVE_BY_ERROR:
        detail_code, msg = PROPRIOCEPTIVE_BY_ERROR[head]
        if skill and (head, skill) in SKILL_MESSAGE_OVERRIDES:
            msg = SKILL_MESSAGE_OVERRIDES[(head, skill)]
        return detail_code, msg
    return f"UNSPECIFIED_{code}", None


def feedback_mode(override: str | None = None) -> str:
    mode = (override or os.environ.get("FINDINGBENCH_RAB_FEEDBACK_MODE", "four_class_v1")).strip().lower()
    if mode not in FEEDBACK_MODES:
        return "four_class_v1"
    return mode


def format_public_feedback(
    code: str,
    private_error: str | None,
    *,
    skill: str | None = None,
    default_message: str,
    mode: str | None = None,
) -> dict[str, Any]:
    """Build the public ``feedback`` payload according to the active feedback mode."""
    active = feedback_mode(mode)
    if active == "four_class_v1":
        return dict(code=code, message=default_message)
    detail_code, msg = detail_for(code, private_error, skill=skill)
    return dict(code=code, detail_code=detail_code, message=msg or default_message)
