"""Build-plan compiler: expands a ScenarioSpec into an ordered, static build plan.

Pure data transformation (no simulator). The OmniGibson backend executes the
plan; this module exists so the plan can be reviewed, diffed and dry-run
validated before touching the simulator.
"""

from __future__ import annotations

from typing import Any

from rummagebench.core.scenario import ScenarioSpec


def expand_build_plan(scenario: ScenarioSpec) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "scenario_id": scenario.id,
        "scene": scenario.scene.model,
        "robot": {
            "model": scenario.robot.model,
            "init_anchor": scenario.robot.init_anchor,
        },
        "steps": [],
    }
    plan["steps"].append({"op": "load_scene", "scene_model": scenario.scene.model})
    plan["steps"].append(
        {"op": "add_robot", "model": scenario.robot.model, "name": scenario.robot.name}
    )
    for spec in scenario.objects:
        plan["steps"].append(
            {
                "op": "spawn_object",
                "name": spec.name,
                "category": spec.category,
                "model": spec.model,
                "fixed_base": spec.fixed_base,
            }
        )
    for p in scenario.placements:
        plan["steps"].append(
            {
                "op": "place",
                "entity": p.entity,
                "relation": p.relation,
                "receptacle": p.receptacle,
            }
        )
    for entity, st in scenario.initial_states.items():
        if st.open is not None:
            plan["steps"].append({"op": "set_open", "entity": entity, "open": st.open})
    plan["steps"].append({"op": "settle"})
    for name in scenario.anchors:
        plan["steps"].append({"op": "verify_anchor", "anchor": name})
    plan["steps"].append({"op": "teleport_robot", "anchor": scenario.robot.init_anchor})
    plan["steps"].append({"op": "settle"})
    plan["steps"].append({"op": "capture_snapshot"})
    return plan
