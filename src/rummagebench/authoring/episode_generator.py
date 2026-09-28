"""Knife-search episode generator (§11-16): template + seed -> deterministic
resolved ScenarioSpec variants.

One template (scenarios/knife_search_001/scenario.yaml), one task family
(knife search in Beechwood_0_int). Variables per episode:

    target container   cabinet_A | drawer_A | cabinet_B (from template
                       initial_states; the knife is re-placed inside it)
    distractor spread  seeded permutation of the distractors over the
                       NON-target containers (never the target's, so the
                       distractors can never pre-solve the search)
    search order       seeded permutation of the container anchors, target
                       container LAST (the scripted oracle ends at the right
                       anchor before GRASP)
    instruction        seeded paraphrase
    robot variant      template embodiment or a restricted-URDF morphology
                       variant (real kinematic difference — reach_radius /
                       z_max are never touched)

Clairvoyant oracle minimum: NAV(target anchor) + OPEN + GRASP = 3 planning
steps (recorded as oracle_min_steps for Search Efficiency; labeled
clairvoyant — agents are NOT told the target location).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from rummagebench.core.scenario import (
    ActionInterfaceSpec,
    AnchorSpec,
    ObjectSpec,
    PlacementSpec,
    ScenarioSpec,
    load_scenario,
)

DEFAULT_INSTRUCTIONS = [
    "Help me cut some fruit. My knife may be in the kitchen.",
    "I need a knife to slice fruit. Please look around the kitchen for it.",
    "Could you find my kitchen knife? I think I left it in one of the cabinets.",
    "Grab the knife so we can cut the fruit — check the kitchen storage.",
    "Where did I put my knife? Please search the kitchen and bring it.",
    "Find the table knife in the kitchen and pick it up for me.",
]

# robot variants: REAL kinematic differences only (URDF path / controlled
# chain). Capability scalars (reach_radius, z_min/z_max) are never modified.
ROBOT_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "r1pro_restricted": {
        "kinematics.urdf_path": "build/robots/r1pro_restricted.urdf",
        "kinematics.controlled_joints": "auto",
    },
}


def generate_episode(
    template: str | Path | ScenarioSpec,
    seed: int,
    target_container: str,
    robot_variant: str = "default",
) -> ScenarioSpec:
    """Build one deterministic knife-search episode from the template."""
    if isinstance(template, (str, Path)):
        scenario = load_scenario(template)
    else:
        scenario = template.model_copy(deep=True)
    scenario = scenario.model_copy(deep=True)
    rng = random.Random(seed)

    containers = sorted(scenario.initial_states.keys())
    if not containers:
        raise ValueError("template has no closed containers to search")
    if target_container not in containers:
        raise ValueError(
            f"target_container {target_container!r} not among {containers}"
        )

    # --- re-place the target and spread the distractors -------------------
    inside = {
        p.entity: p.receptacle for p in scenario.placements if p.relation == "inside"
    }
    target_entity = scenario.target.entity
    if target_entity not in inside:
        raise ValueError(
            f"target entity {target_entity!r} is not placed inside a container "
            "in the template"
        )
    distractors = sorted(e for e in inside if e != target_entity)
    others = [c for c in containers if c != target_container]

    new_inside: dict[str, str] = {target_entity: target_container}
    for entity in distractors:
        new_inside[entity] = rng.choice(others)

    placements = []
    for p in scenario.placements:
        if p.relation == "inside":
            placements.append(PlacementSpec(
                entity=p.entity, relation="inside", receptacle=new_inside[p.entity]
            ))
        else:
            placements.append(p)
    scenario.placements = placements

    # --- scripted sequences (data, not Python: generator-authored oracles) -
    search_order = [c for c in containers]
    rng.shuffle(search_order)
    search_order.remove(target_container)
    search_order.append(target_container)  # oracle ends at the right anchor

    scripted_success: list[dict[str, Any]] = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}
    ]
    for container in search_order:
        scripted_success += [
            {"skill": "NAV", "target": {"type": "place", "value": container}},
            {"skill": "OPEN", "target": {"type": "entity", "value": container}},
        ]
    scripted_success.append(
        {"skill": "GRASP", "target": {"type": "entity", "value": target_entity}}
    )

    # wrong-object baseline: grasp a distractor from the first searched
    # container that actually holds one (guaranteed FAIL_WRONG_TARGET)
    scripted_wrong: list[dict[str, Any]] = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}
    ]
    wrong_target = None
    for container in search_order:
        scripted_wrong += [
            {"skill": "NAV", "target": {"type": "place", "value": container}},
            {"skill": "OPEN", "target": {"type": "entity", "value": container}},
        ]
        holder = next(
            (e for e, c in new_inside.items() if c == container), None
        )
        if holder is not None and wrong_target is None:
            wrong_target = holder
            break
    if wrong_target is None:
        raise RuntimeError("no distractor available for the wrong-object baseline")
    scripted_wrong.append(
        {"skill": "GRASP", "target": {"type": "entity", "value": wrong_target}}
    )

    scenario.agent.scripted_success = scripted_success
    scenario.agent.scripted_wrong_object = scripted_wrong
    scenario.instruction = rng.choice(DEFAULT_INSTRUCTIONS)
    scenario.id = f"knife_search_s{seed:03d}"
    scenario.oracle_min_steps = 3  # clairvoyant: NAV + OPEN + GRASP

    # --- robot variant (real URDF morphology only) -------------------------
    variant_overrides = ROBOT_VARIANTS.get(robot_variant)
    if variant_overrides is None:
        raise ValueError(
            f"unknown robot_variant {robot_variant!r}; "
            f"expected one of {sorted(ROBOT_VARIANTS)}"
        )
    for dotted, value in variant_overrides.items():
        section, field = dotted.split(".")
        kin = scenario.robot.kinematics
        if kin is None:
            raise ValueError("robot variant requires a template kinematics block")
        if section != "kinematics":
            raise ValueError(f"unsupported variant override {dotted!r}")
        setattr(kin, field, value)

    return scenario


def generate_episodes(
    template: str | Path,
    seeds: list[int],
    out_dir: str | Path = "build/generated_scenarios",
    robot_variant: str = "default",
    target_containers: list[str] | None = None,
) -> list[Path]:
    """Generate one resolved YAML per seed; targets cycle deterministically
    across the template containers (count balanced when divisible)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = load_scenario(template)
    containers = sorted(probe.initial_states.keys())
    if target_containers is None:
        target_containers = containers

    written: list[Path] = []
    for i, seed in enumerate(seeds):
        target = target_containers[i % len(target_containers)]
        scenario = generate_episode(template, seed, target, robot_variant)
        path = out_dir / f"{scenario.id}.yaml"
        path.write_text(
            yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        written.append(path)
    return written


def write_split(
    split_name: str,
    episode_paths: list[Path],
    robots: list[str],
    out_path: str | Path = "benchmark_splits",
) -> Path:
    """§16: a benchmark split = episode distribution x robot morphologies."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"{split_name}.yaml"
    path.write_text(
        yaml.safe_dump({
            "name": split_name,
            "episodes": [str(p) for p in episode_paths],
            "robots": robots,
        }, sort_keys=False),
        encoding="utf-8",
    )
    return path
