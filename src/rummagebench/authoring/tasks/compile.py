"""Compiler: TaskPlan -> canonical ScenarioSpec.

Uses only fields the frozen schema already has (objects, placements inside /
on_top + link, initial_states, anchors, robot, target, instruction). Tier,
seed and generator metadata stay in the TaskPlan / certificate — never in
the runtime scenario. Navigation anchors are evaluator-private build/oracle
infrastructure; the agent-facing protocol stays the frozen 8 skills.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
from rummagebench.authoring.tasks.plan import TaskPlan, canonical_json, sha256_bytes
from rummagebench.authoring.tasks.slots import SceneSlots
from rummagebench.authoring.tasks.tiers.base import TierSpec
from rummagebench.core.scenario import ScenarioSpec

START_ANCHOR = "start"
# internal oracle / scripted-agent skills (same convention as every existing
# scenario); the evaluated agent always sees the frozen 8-skill protocol
ORACLE_SKILLS = ["NAV", "OPEN", "CLOSE", "GRASP", "PLACE"]


# certifier-added anchor (certify.py v3): the witness's viewpoint of the real
# target when no compiled anchor sees + grasps it; written into the candidate
VIEW_ANCHOR_SUFFIX = "__view"


def reveal_anchor_name(entity: str) -> str:
    return f"{entity}__reveal"


def compile_plan(plan: TaskPlan, scene_slots: SceneSlots,
                 overlay: SlotEmbodimentOverlay, tier: TierSpec) -> dict[str, Any]:
    """Return the scenario as a plain dict (validated by ScenarioSpec)."""
    slots = scene_slots.by_id()
    furniture = scene_slots.furniture_by_entity()
    tslot = slots[plan.target.slot_id]

    anchors: dict[str, Any] = {START_ANCHOR: deepcopy(plan.start)}
    # one anchor per furniture entity of the searched rooms that the overlay
    # can serve (legacy convention: anchor name == furniture entity name)
    for sid, inter in sorted(overlay.slots.items()):
        s = slots.get(sid)
        if s is None or s.room not in plan.search_rooms:
            continue
        anchors.setdefault(s.parent_entity, inter.navigation_anchor.to_anchor())

    objects, placements = [], []
    stack_top = plan.target.entity
    for o in plan.objects:
        s = slots[o.slot_id]
        objects.append({"name": o.entity, "category": o.category, "model": o.model})
        if o.role == "cover":
            # medium "buried": each cover rests ON the previous stack item; the
            # builder samples OnTop and verifies it after settling (resampling
            # otherwise). Dropping covers into the same drawer link did not
            # stack them: 0/12 covered targets in medium pilot 1 (A.10)
            placements.append({"entity": o.entity, "relation": "on_top",
                               "receptacle": stack_top})
            stack_top = o.entity
            continue
        p = {"entity": o.entity, "relation": s.relation, "receptacle": s.parent_entity}
        if s.link is not None:
            p["link"] = s.link
        placements.append(p)

    # every openable furniture of the searched rooms starts CLOSED: the target
    # container must be closed (start visibility 0) and the others must not
    # leak which one is "special"
    initial_states = {f.entity: {"open": False}
                      for f in sorted(furniture.values(), key=lambda f: f.entity)
                      if f.openable and f.room in plan.search_rooms}

    success = [{"skill": "NAV", "target": {"type": "place", "value": tslot.parent_entity}}]
    if tslot.requires_open:
        success.append({"skill": "OPEN",
                        "target": {"type": "entity", "value": tslot.parent_entity}})
    # the target slot's REVEAL pose (probe): where the revealed target is
    # visible and graspable; the agent reaches it with MOVE/TURN, the oracle
    # with NAV over this evaluator-private anchor
    tinter = overlay.slots.get(tslot.slot_id)
    if (tinter is not None and tinter.reveal_anchor is not None
            and tinter.reveal_anchor != tinter.navigation_anchor):
        reveal_name = reveal_anchor_name(tslot.parent_entity)
        anchors[reveal_name] = tinter.reveal_anchor.to_anchor()
        success.append({"skill": "NAV", "target": {"type": "place", "value": reveal_name}})
    grasp_target = {"skill": "GRASP",
                    "target": {"type": "entity", "value": plan.target.entity}}
    wrong = None
    same_slot = [o for o in plan.distractors if o.slot_id == tslot.slot_id]
    if same_slot:
        wrong = success + [{"skill": "GRASP",
                            "target": {"type": "entity", "value": same_slot[0].entity}}]

    doc: dict[str, Any] = {
        "id": plan.task_id,
        "instruction": plan.instruction,
        "scene": {"model": plan.scene},
        "robot": {**deepcopy(overlay.robot_config), "init_anchor": START_ANCHOR},
        "feasibility": deepcopy(overlay.feasibility) or {"backend": "pinocchio",
                                                         "mode": "endpoint"},
        "anchors": anchors,
        "target": {"entity": plan.target.entity, "category": plan.target.category},
        "objects": objects,
        "placements": placements,
        "initial_states": initial_states,
        "termination": {"max_planning_steps": tier.budget.max_planning_steps,
                        "fail_on_wrong_grasp": tier.fail_on_wrong_grasp,
                        "fail_on_unsafe_action": True,
                        "succeed_when_holding_target": True},
        "skills": list(ORACLE_SKILLS),
        "safety": {"forbidden_categories": [], "grasping_fixed_base_unsafe": True},
        "agent": {
            "scripted_success": success + [grasp_target],
            "scripted_wrong_object": wrong or [],
            "unsafe_sequence": [
                {"skill": "NAV", "target": {"type": "place", "value": tslot.parent_entity}},
                {"skill": "GRASP", "target": {"type": "entity",
                                              "value": tslot.parent_entity}}],
        },
    }
    ScenarioSpec.model_validate(doc)  # strict schema gate
    return doc


def with_view_anchors(doc: dict[str, Any], anchors: dict[str, Any]) -> dict[str, Any]:
    """Add certifier view anchors (name -> {position, orientation}) to a
    compiled scenario; the scripted success walks there before the GRASP.
    Idempotent; recompiling a candidate re-applies the anchors of its
    previous file (compile_tasks.py) so a certified file is never silently
    replaced by one without them."""
    doc = deepcopy(doc)
    succ = doc["agent"]["scripted_success"]
    for name, a in sorted(anchors.items()):
        doc["anchors"][name] = {"position": list(a["position"]),
                                "orientation": list(a["orientation"])}
        nav = {"skill": "NAV", "target": {"type": "place", "value": name}}
        if nav not in succ:
            succ.insert(len(succ) - 1, nav)  # before the final GRASP
    ScenarioSpec.model_validate(doc)
    return doc


def view_anchors_of(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (doc.get("anchors") or {}).items()
            if k.endswith(VIEW_ANCHOR_SUFFIX)}


def scenario_hash(doc: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(doc))
