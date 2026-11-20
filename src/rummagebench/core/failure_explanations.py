"""Evidence-only explanations. No suggested actions or model-output repair."""
from __future__ import annotations

import math


def explain(event, blockers=(), *, verified=False, selected_object=None):
    """Return plain language; categories may be disclosed only if verified live."""
    action = event.get("parsed_action") or event.get("raw_action") or {}
    action = action if isinstance(action, dict) else {}
    skill = action.get("skill", "Action")
    skill = skill if isinstance(skill, str) else "Action"
    reason = event.get("private_reason") or event.get("legacy_failure_reason")
    navigation_reasons = {
        "NAV_NOT_FLOOR": "Move-to not executed: the selected pixel is not on a fixed floor surface.",
        "NAV_NOT_TARGETABLE": "Move-to not executed: the selected pixel is on a wall or ceiling; select floor, furniture or an object.",
        "NAV_TARGET_GEOMETRY_UNAVAILABLE": "Move-to not executed: the selected object's extent is unavailable.",
        "NAV_NO_STAND_POSE": "Move-to not executed: no reachable position in front of the selected object was found. The robot did not move.",
        "NAV_DIFFERENT_LEVEL": "Move-to not executed: the selected floor height differs from the robot base height by 0.30 m or more, or its height is invalid.",
        "NAV_SUPPORT_UNAVAILABLE": "Move-to not executed: same-level floor support could not be established.",
        "NAV_START_BLOCKED": "Move-to not executed: the current base footprint is blocked in the navigation map.",
        "NAV_GOAL_BLOCKED": "Move-to not executed: the robot's footprint does not fit at the selected floor position.",
        "NAV_NO_PATH": "Move-to not executed: no collision-free route was found on the 2D navigation grid. The robot did not move.",
        "NAV_SEARCH_LIMIT": "Move-to not executed: the navigation search reached its computation limit. Reachability is undetermined; the robot did not move.",
    }
    if reason in navigation_reasons:
        return navigation_reasons[reason]
    if event.get("feedback") == "EXECUTED":
        if skill == "MOVE_TO":
            if (event.get("navigation_private") or {}).get("planner") == "stand_pose_v1":
                return "Move-to executed: the robot now stands in front of the selected object, facing it."
            return "Move-to executed: the robot reached the selected floor position. Body heading is unchanged."
        if skill == "VIEW":
            return "View selected: a supplementary picture was returned. The robot's position and body heading did not change."
        if skill == "CHECK":
            return "Check executed: four exterior appearance views are shown. The object and robot have not moved. This inspection image cannot be used for scene point actions."
        if skill == "MOVE":
            return f"Movement executed: {abs(action['distance_cm']):g} cm {'forward' if action['distance_cm'] > 0 else 'backward'} along the body heading."
        if skill == "TURN":
            if "direction" in action:
                direction = action["direction"]
                return f"Turn executed: the body turned {direction} by {180 if direction == 'back' else 90} degrees in place."
            return f"Turn executed: the body turned {'left' if action['angle_deg'] > 0 else 'right'} by {abs(action['angle_deg']):g} degrees in place."
        if skill in ("GRASP", "PLACE", "OPEN", "CLOSE"):
            return {"GRASP": "Grasp executed: the selected object is now held.",
                    "PLACE": "Place executed: the held object was placed at the selected destination and released.",
                    "OPEN": "Open executed: the selected object is now open.",
                    "CLOSE": "Close executed: the selected object is now closed."}[skill]
        if skill == "REPORT_DONE":
            return "Completion accepted: the task goal is satisfied and the episode has ended."
        return "Action executed."
    if reason == "SCHEMA":
        errors = event.get("schema_errors_private") or []
        if errors:
            error = errors[0]
            # Only public schema field names; never echo arbitrary submitted keys.
            fields = [str(x) for x in error.get("loc", []) if x in
                      ("skill", "direction", "distance_cm", "angle_deg", "yaw", "pitch", "point", "frame_id", "x", "y")]
            field = ".".join(fields) or "action"
            kind = error.get("type")
            if kind == "missing":
                return f"Action not executed: required field {field} is missing."
            if kind == "union_tag_not_found":
                return "Action not executed: the action object has no skill field."
            if kind == "union_tag_invalid":
                return "Action not executed: the submitted skill name is not recognized."
            if kind == "extra_forbidden":
                return "Action not executed: an unsupported extra parameter was supplied."
            if kind in ("greater_than_equal", "less_than_equal", "finite_number"):
                return f"Action not executed: {field} is outside its allowed numeric range."
            return f"Action not executed: {field} has an invalid type or value."
    if reason in ("MOVE_BLOCKED", "TURN_BLOCKED"):
        verb = "Movement" if reason == "MOVE_BLOCKED" else "Turn"
        categories = sorted({b.get("category") for b in blockers if b.get("category")})
        obstacle = "an obstacle"
        if verified and categories:
            obstacle = "the " + " or ".join(categories[:2])
        return f"{verb} failed: the robot's base clearance check intersects {obstacle}. The robot did not move."
    if reason == "UNREACHABLE":
        if any((e.get("details") or {}).get("note") == "no interaction interface"
               for e in event.get("manipulation_diagnostic_private", [])):
            return f"{skill.title()} not executed: no interaction poses are defined for this operation on the selected object. Reachability was not established."
        if skill == "GRASP":
            return "Grasp failed: the arm could not reach a valid grasp pose for the selected object from its current position."
        return f"{skill.title()} failed: the arm could not reach a valid interaction pose for the selected object."
    if reason == "COLLISION":
        if skill == "OPEN" and any(
            (e.get("details") or {}).get("note") == "opened links intrude the robot base footprint"
            for e in event.get("manipulation_diagnostic_private", [])):
            return "Open rejected as unsafe: the opened object would overlap the robot's base footprint. Opening was not executed."
        if skill == "GRASP":
            return "Grasp rejected as unsafe: the collision check found no collision-free grasp pose among the tested candidates."
        return f"{skill.title()} rejected as unsafe: the collision check rejected the tested interaction configurations. The operation was not executed."
    if reason == "UNSAFE_ACTION":
        # Policy prohibition is not evidence of a predicted physical collision.
        return f"{skill.title()} not executed: a task safety rule prohibits this operation on the selected object."
    if reason in ("SKILL_NOT_APPLICABLE_TO_ENTITY", "INVALID_STATE"):
        evidence = selected_object or {}
        category = evidence.get("category")
        target = f"the {category}" if category else "the selected object"
        if skill == "GRASP":
            if evidence.get("cannot_pick_up"):
                return f"Grasp not attempted: you selected {target}; this object cannot be picked up."
            if evidence.get("holding_full"):
                return "Grasp not attempted: the robot's holding capacity is already full."
            return f"Grasp not attempted: {target} cannot be grasped in the current object or holding state."
        if skill in ("OPEN", "CLOSE"):
            if evidence.get("openable") is False:
                return f"{skill.title()} not attempted: {target} does not support opening or closing."
            if "is_open" in evidence and evidence["is_open"] == (skill == "OPEN"):
                return f"{skill.title()} not attempted: {target} is already {'open' if skill == 'OPEN' else 'closed'}."
        if skill == "PLACE":
            if evidence.get("place_supported") is False:
                return f"Place not attempted: {target} does not support placement."
            if evidence.get("holding_empty"):
                return "Place not attempted: the robot is not holding an object."
    if reason == "STALE_FRAME":
        point = action.get("point") or {}
        if isinstance(point, dict) and point.get("frame_id") == "CURRENT_FRAME":
            return 'Action failed: "CURRENT_FRAME" is a placeholder, not an image ID. No manipulation was attempted.'
        return "Action failed: the submitted frame_id does not match any image available in the current observation. No manipulation was attempted."
    messages = {
        "SCHEMA": "Action failed: its format or parameters do not match the skill definition.",
        "MOVE_PARAMETER_RANGE": "Movement failed: the requested distance is outside the allowed range.",
        "TURN_PARAMETER_RANGE": "Turn failed: the requested angle is outside the allowed range.",
        "SKILL_NOT_APPLICABLE_TO_ENTITY": "Action failed: this skill does not apply to the selected object in its current state.",
        "GOAL_UNSATISFIED": "Completion rejected: the task goal is not satisfied. The episode has ended.",
        "VISIBILITY_UNAVAILABLE": "Action failed: the selected image cannot be used to identify an object.",
        "NO_ENTITY_AT_PIXEL": "Action failed: no selectable object was identified at that image point.",
        "NO_VISUAL_TARGET": "Action not executed: no object was identified at the selected image point.",
        "NO_OWNING_ENTITY": "Action not executed: the selected image point could not be associated with an interactive object.",
        "NO_VALID_DEPTH": "Action not executed: the selected image point has no valid depth measurement.",
        "INVALID_DEPTH_GEOMETRY": "Action not executed: the selected image point could not be converted to a valid spatial location.",
        "VIEW_UNAVAILABLE": "View not executed: this environment has no panoramic camera for selecting a supplementary view.",
        "CHECK_UNSUPPORTED": "Check not executed: exterior inspection supports movable, non-articulated objects only.",
        "INVALID_STATE": "Action failed: the current object or holding state does not permit this interaction.",
    }
    if reason in messages:
        return messages[reason]
    return {
        "INVALID_ACTION": "Action failed: the selection or parameters are not valid in the current state.",
        "OUT_OF_CAPABILITY": "Action failed: the requested interaction is outside the robot's supported capability.",
        "UNSAFE": "Action failed: the executor's safety check rejected it.",
    }.get(event.get("feedback"), "No execution result is available.")


def sweep(event, yaw, obstacles, half_extent=.4, margin=.03):
    """Replay the same base-footprint test, without moving a simulator."""
    from rummagebench.skills.move import footprint_corners, _aabb_overlaps_footprint
    action = event.get("parsed_action") or event.get("raw_action") or {}
    skill = action.get("skill")
    if skill not in ("MOVE", "TURN"):
        return None
    x, y, _ = event["robot_pose_before"]
    value = float(action["distance_cm"] if skill == "MOVE" else
                  {"left":90,"right":-90,"back":180}[action["direction"]]
                  if "direction" in action else action["angle_deg"])
    if not math.isfinite(value):
        return None
    n = max(1, int(abs(value / 100 if skill == "MOVE" else value) /
                   (.05 if skill == "MOVE" else 5)) + 1)
    for i in range(1, n + 1):
        f = i / n
        xx = x + value / 100 * math.cos(yaw) * f if skill == "MOVE" else x
        yy = y + value / 100 * math.sin(yaw) * f if skill == "MOVE" else y
        angle = yaw + math.radians(value) * f if skill == "TURN" else yaw
        footprint = footprint_corners(xx, yy, angle, half_extent)
        hits = [o for o in obstacles if _aabb_overlaps_footprint((o["lo"], o["hi"]), footprint, margin)]
        if hits:
            return dict(predicted_safe=False, blockers=hits, sample=i, samples=n,
                        sample_pose=[xx, yy, angle], footprint=footprint)
    return dict(predicted_safe=True, blockers=[], samples=n)
