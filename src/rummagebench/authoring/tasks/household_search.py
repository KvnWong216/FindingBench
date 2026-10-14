"""A target-blind reference search, measured through the existing public skills.

This is a reference cost, not a real-scene optimum or the known-target oracle.
All submitted MOVE/TURN/OBSERVE/interaction/report calls cost one step,
including rejected calls. CPU summaries cannot manufacture simulator evidence.
"""
from __future__ import annotations

import math
from pydantic import TypeAdapter

from rummagebench.authoring.tasks.difficulty import interval_label
from rummagebench.core.public_types import PUBLIC_SKILLS, PublicAction

POLICY_VERSION = "nearest-uninspected-anchor-v1"


def reference_order(start_xy, anchors: dict[str, list[float]]) -> list[str]:
    """Only start and complete candidate geometry are inputs; no hidden target."""
    if not anchors or len(start_xy) != 2:
        raise ValueError("REFERENCE_COMPLETE_CANDIDATES_REQUIRED")
    current = tuple(start_xy)
    remaining = dict(anchors)
    if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in [start_xy, *remaining.values()]):
        raise ValueError("REFERENCE_INVALID_ANCHOR")
    result = []
    while remaining:
        sid = min(remaining, key=lambda k: (math.dist(current, remaining[k]), k))
        result.append(sid)
        current = remaining.pop(sid)
    return result


def count_public_trace(trace: list[dict]) -> dict:
    counts = {skill: 0 for skill in PUBLIC_SKILLS}
    for i, row in enumerate(trace, 1):
        TypeAdapter(PublicAction).validate_python(row.get("action", {}))
        skill = row.get("action", {}).get("skill")
        if skill not in counts or row.get("step") != i:
            raise ValueError("REFERENCE_TRACE_SKILL_OR_STEP_INVALID")
        if not isinstance(row.get("feedback"), str):
            raise ValueError("REFERENCE_TRACE_FEEDBACK_MISSING")
        counts[skill] += 1
    return {"steps": len(trace), "spatial": counts["MOVE"] + counts["TURN"],
            "operations": sum(v for k, v in counts.items() if k not in ("MOVE", "TURN")),
            "skill_calls": counts}


def summarize_reference_search(runs: dict[str, dict], priors: dict[str, float], intervals: dict) -> dict:
    if (set(runs) != set(priors) or not priors or any(p <= 0 or not math.isfinite(p) for p in priors.values())
            or abs(sum(priors.values()) - 1.) > 1e-8):
        raise ValueError("REFERENCE_WHOLE_FAMILY_AND_PRIOR_REQUIRED")
    orders = [r.get("candidate_order") for r in runs.values()]
    if (any(o != orders[0] for o in orders) or set(orders[0] or []) != set(priors)
            or len(orders[0]) != len(priors)):
        raise ValueError("REFERENCE_ORDER_NOT_COMPLETE_OR_TARGET_BLIND")
    outcomes, rejections = {}, []
    for location, run in sorted(runs.items()):
        metrics = count_public_trace(run["trace"])
        if (run.get("gpu_executed") is not True or run.get("policy") != POLICY_VERSION
                or run.get("status") != "SUCCESS" or not run["trace"]
                or run["trace"][-1]["action"] != {"skill": "REPORT_DONE"}
                or run["trace"][-1]["feedback"] != "EXECUTED"
                or not run.get("visited_slots")
                or len(run["visited_slots"]) > len(orders[0])
                or not any(row["action"]["skill"] == "GRASP" and row["feedback"] == "EXECUTED" for row in run["trace"])
                or run.get("visited_slots") != orders[0][:len(run.get("visited_slots", []))]):
            rejections.append(f"REFERENCE_SEARCH_FAILED:{location}")
        if run.get("closed_observe_bypass"):
            rejections.append(f"CLOSED_OBSERVE_BYPASS:{location}")
        outcomes[location] = {"prior": priors[location], **metrics, "status": run["status"]}
    expected = {axis: sum(priors[k] * outcomes[k][axis] for k in priors)
                for axis in ("spatial", "operations")}
    bins = {axis: interval_label(value, intervals[axis]) for axis, value in expected.items()}
    if None in bins.values():
        rejections.append("REFERENCE_COST_OUTSIDE_CONFIGURED_INTERVALS")
    return {"status": "measured_reference" if not rejections else "rejected",
            "scope": "real_public_skill_reference_cost", "policy": POLICY_VERSION,
            "candidate_order": orders[0], "expected": expected,
            "worst_case": {axis: max(o[axis] for o in outcomes.values()) for axis in expected},
            "per_target": outcomes, "reference_difficulty": bins, "rejections": rejections,
            "real_scene_optimality_proved": False,
            "cost_contract": "one per submitted public call; MOVE and TURN spatial; all other calls operational",
            "assumptions": ["complete_positive_prior", "fixed_target_blind_candidate_order",
                            "evaluator_renderer_grounding_for_reference_policy",
                            "actual_dynamic_routes_and_visibility; no symmetric-distance assumption"]}


def run_reference_search(backend, scenario, plan, slots, overlay, room_map) -> dict:
    """GPU ONLY; uses existing route planner, click grounding and public session."""
    from rummagebench.authoring.tasks.certify import TaskCertifier, EpisodeCertification, click_point, BASE_HALF_EXTENT
    from rummagebench.core.visual_session import VisualProtocolSession
    from rummagebench.sim.omnigibson.visibility import measure_visibility

    candidates = sorted(plan.candidate_slots)
    order = reference_order(plan.start["position"][:2],
                            {sid: list(overlay.slots[sid].navigation_anchor.xy) for sid in candidates})
    certifier = TaskCertifier(slots, overlay, room_map)
    out = EpisodeCertification()
    vs = VisualProtocolSession(backend, scenario, base_half_extent=BASE_HALF_EXTENT)
    result = {"policy": POLICY_VERSION, "gpu_executed": True, "candidate_order": order,
              "visited_slots": [], "trace": [], "closed_observe_bypass": False, "checks": []}
    vs.reset()

    def step(action):
        response = vs.step(action)
        result["trace"].append({"action": action, "feedback": response.feedback.code.value,
                                "status": response.episode_status, "step": response.planning_step})
        return response.feedback.code.value

    def click(skill, entity):
        frame = vs._store.get(vs._store.current_id)
        point = click_point(frame, backend, entity)
        if point is None:
            return False
        return step({"skill": skill, "point": {"frame_id": frame.frame_id,
                     "x": point[0], "y": point[1]}}) == "EXECUTED"

    def observe(entity, closed):
        original = backend.entity_visible_pixels
        observed = []
        def measured(frame, name):
            pixels = original(frame, name)
            if name == entity and pixels >= vs._observe_cfg.min_visible_pixels:
                observed.append(int(original(frame, scenario.target.entity)))
            return pixels
        # Instrument accepted OBSERVE images, using the very same production
        # call. No counterfactual extra views are treated as public steps.
        backend.entity_visible_pixels = measured
        try:
            ok = click("OBSERVE", entity)
        finally:
            backend.entity_visible_pixels = original
        result["checks"].append({"slot_entity": entity, "closed": closed,
                                  "accepted_view_target_pixels": observed})
        if closed and any(observed):
            result["closed_observe_bypass"] = True
        return ok

    by_id = slots.by_id()
    try:
        for sid in order:
            result["visited_slots"].append(sid)
            slot = by_id[sid]
            interaction = overlay.slots[sid]
            anchor = interaction.navigation_anchor
            if not certifier._drive_pose(vs, backend, plan, (anchor.xy[0], anchor.xy[1], anchor.yaw),
                                         sid, step, out):
                break
            if not observe(slot.parent_entity, not backend.is_open(slot.parent_entity)):
                break
            if result["closed_observe_bypass"] or not click("OPEN", slot.parent_entity):
                break
            anchor = interaction.reveal_anchor or interaction.navigation_anchor
            if not certifier._drive_pose(vs, backend, plan, (anchor.xy[0], anchor.xy[1], anchor.yaw),
                                         sid + "_reveal", step, out):
                break
            if not observe(slot.parent_entity, False):
                break
            frame = vs._store.get(vs._store.current_id)
            # The true target is consulted only when visible in the CURRENT
            # image. It never supplies an order, a route, or a new viewpoint.
            if backend.entity_visible_pixels(frame, scenario.target.entity) > 0:
                visibility = measure_visibility(backend, scenario.target.entity).to_dict()
                result["detection_visibility"] = visibility
                if visibility.get("ratio") is None or visibility["ratio"] < certifier.reveal_min:
                    break
                if click("GRASP", scenario.target.entity):
                    step({"skill": "REPORT_DONE"})
                break
        result["status"] = vs.status()
        result["route_details"] = out.details
    except Exception as exc:
        result.update(status="ENGINE_ERROR", error=f"{type(exc).__name__}: {exc}")
    return result
