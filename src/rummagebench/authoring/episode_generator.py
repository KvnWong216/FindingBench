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

# §6: placement regimes come from an explicit task-family prior file
# (configs/task_priors/knife_search.yaml) — never vague randomness. Each
# regime lists (relation, containers). Counterfactual placements remain
# semantically executable, physically legal, and oracle-solvable; they
# violate common household location priors instead.
DEFAULT_TASK_PRIORS = {
    "target_category": "table_knife",
    "natural": {
        "relation": "inside",
        "containers": [
            "bottom_cabinet_no_top_qohxjq_0",
            "bottom_cabinet_no_top_spojpj_0",
            "bottom_cabinet_rvpunw_0",
        ],
    },
    "counterfactual": {
        "relation": "on_top",
        "containers": [
            "breakfast_table_uhrsex_0",
        ],
    },
}


def load_task_priors(path: str | Path | None = None) -> dict:
    """Task-family location priors for placement regimes."""
    if path is None:
        candidate = Path(__file__).resolve().parents[3] / (
            "configs/task_priors/knife_search.yaml"
        )
        if candidate.is_file():
            import yaml

            return yaml.safe_load(candidate.read_text(encoding="utf-8"))
        return dict(DEFAULT_TASK_PRIORS)
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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
    placement_regime: str = "natural",
    pair_id: str | None = None,
    task_priors: dict | None = None,
) -> ScenarioSpec:
    """Build one deterministic knife-search episode from the template."""
    if isinstance(template, (str, Path)):
        scenario = load_scenario(template)
    else:
        scenario = template.model_copy(deep=True)
    scenario = scenario.model_copy(deep=True)
    rng = random.Random(seed)
    priors = task_priors or load_task_priors()

    containers = sorted(scenario.initial_states.keys())
    if not containers:
        raise ValueError("template has no closed containers to search")
    if target_container not in containers:
        raise ValueError(
            f"target_container {target_container!r} not among {containers}"
        )

    # --- re-place the target per the REGIME and spread the distractors ----
    regime = priors.get(placement_regime)
    if regime is None:
        raise ValueError(
            f"placement_regime {placement_regime!r} not in task priors "
            f"(have: {sorted(priors)})"
        )
    target_entity = scenario.target.entity
    inside = {
        p.entity: p.receptacle for p in scenario.placements if p.relation == "inside"
    }
    distractors = sorted(e for e in inside if e != target_entity)

    target_relation = regime.get("relation", "inside")
    regime_containers = regime.get("containers", [])
    if target_container not in regime_containers:
        raise ValueError(
            f"target_container {target_container!r} is not a {placement_regime!r} "
            f"location in the task priors {regime_containers}"
        )

    # distractors spread round-robin over ALL containers — independent of
    # the target container, so natural/counterfactual pair arms share the
    # exact distractor placement (paired statistics require this)
    others = list(containers)

    new_placements: list[PlacementSpec] = []
    for p in scenario.placements:
        if p.entity == target_entity:
            new_placements.append(PlacementSpec(
                entity=p.entity, relation=target_relation,
                receptacle=target_container,
            ))
        elif p.relation == "inside":
            new_placements.append(PlacementSpec(
                entity=p.entity, relation="inside", receptacle=rng.choice(others),
            ))
        else:
            new_placements.append(p)
    # keep the target container's initial state forced closed so the episode
    # remains a search task when the regime is "inside"; for on_top regimes
    # the container state is irrelevant but harmless
    scenario.placements = new_placements

    # --- scripted sequences (data, not Python: generator-authored oracles) -
    # search order over the OPENABLE containers; if the target location is
    # itself openable it is searched LAST (the oracle ends at the right
    # anchor); otherwise (e.g. on_top of a table) all containers are searched
    # and the sequence ends with NAV to the target anchor + GRASP.
    search_order = [c for c in containers if c != target_container]
    rng.shuffle(search_order)
    target_openable = target_container in containers
    if target_openable:
        search_order.append(target_container)

    scripted_success: list[dict[str, Any]] = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}
    ]
    for container in search_order:
        scripted_success += [
            {"skill": "NAV", "target": {"type": "place", "value": container}},
            {"skill": "OPEN", "target": {"type": "entity", "value": container}},
        ]
    if not target_openable:
        scripted_success.append(
            {"skill": "NAV", "target": {"type": "place", "value": target_container}}
        )
    scripted_success.append(
        {"skill": "GRASP", "target": {"type": "entity", "value": target_entity}}
    )

    # wrong-object baseline: grasp a distractor from the first searched
    # container that actually holds one (guaranteed FAIL_WRONG_TARGET)
    scripted_wrong: list[dict[str, Any]] = [
        {"skill": "NAV", "target": {"type": "place", "value": "kitchen"}}
    ]
    wrong_target = None
    distractor_locations: dict[str, str] = {}
    for p in scenario.placements:
        if p.entity != target_entity and p.relation == "inside":
            distractor_locations[p.entity] = p.receptacle
        elif p.entity != target_entity:
            distractor_locations[p.entity] = p.receptacle
    for container in search_order:
        scripted_wrong += [
            {"skill": "NAV", "target": {"type": "place", "value": container}},
            {"skill": "OPEN", "target": {"type": "entity", "value": container}},
        ]
        holder = next(
            (e for e, c in distractor_locations.items() if c == container), None
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
    scenario.placement_regime = placement_regime
    scenario.pair_id = pair_id
    # oracle_min_steps is NOT set here: it is computed by the oracle semantic
    # planner during episode certification (evaluation/certification.py) and
    # written back post-certification. Hand-authored difficulty is removed.
    scenario.oracle_min_steps = None

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
    task_priors: dict | None = None,
    placement_regime: str = "natural",
) -> list[Path]:
    """Generate one resolved YAML per seed; targets cycle deterministically
    across the regime's containers (count balanced when divisible)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = load_scenario(template)
    priors = task_priors or load_task_priors()
    regime_containers = priors.get(placement_regime, {}).get(
        "containers",
        sorted(probe.initial_states.keys()),
    )
    if target_containers is None:
        target_containers = regime_containers

    written: list[Path] = []
    for i, seed in enumerate(seeds):
        target = target_containers[i % len(target_containers)]
        scenario = generate_episode(
            template, seed, target, robot_variant,
            placement_regime=placement_regime, task_priors=priors,
        )
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
    certificates: list[str] | None = None,
    depth_stats: dict | None = None,
    placement_regime: str | None = None,
) -> Path:
    """§16/§10: a benchmark split = certified episode distribution x robot
    morphologies. Split entries reference episode CERTIFICATES (not only
    YAML paths); depth statistics are recorded when certificates are given."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "name": split_name,
        "episodes": [str(p) for p in episode_paths],
        "robots": robots,
    }
    if certificates is not None:
        payload["certificates"] = certificates
    if depth_stats is not None:
        payload["depth_histogram"] = depth_stats.get("depth_histogram", {})
        payload["min_depth"] = depth_stats.get("min_depth")
        payload["max_depth"] = depth_stats.get("max_depth")
        payload["mean_depth"] = depth_stats.get("mean_depth")
    if placement_regime is not None:
        payload["placement_regime"] = placement_regime
    path = out_path / f"{split_name}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path
