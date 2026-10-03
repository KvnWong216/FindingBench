"""§4/rev§1: Level-1 scene grammar — three paradigms as data.

A ScenePlan is the discrete task structure BEFORE any occupancy/geometry
work: which objects, their roles, counts (from the dataset_v1 distributions)
and relations (inside-container, cover). Deterministic given
(seed, sampler state, support/container choices).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rummagebench.authoring.level1.config import DatasetConfig
from rummagebench.authoring.level1.coverage_sampler import CoverageSampler

PARADIGMS = ("container_rummage", "tabletop_clutter", "deliberate_cover")


@dataclass
class ObjectPlan:
    name: str
    category: str
    model: str
    role: str  # "target" | "inside" | "outside" | "cover" | "distractor"


@dataclass
class ScenePlan:
    environment_seed: int
    paradigm: str
    support_model: str
    container_model: str | None
    objects: list[ObjectPlan] = field(default_factory=list)
    cover_relation: str | None = None      # cavity_cover | footprint_cover
    counts: dict[str, int] = field(default_factory=dict)
    visibility_band: tuple[float, float] = (0.0, 1.0)

    def target(self) -> ObjectPlan:
        for o in self.objects:
            if o.role == "target":
                return o
        raise ValueError(f"scene plan {self.environment_seed} has no target")


def build_scene_plan(environment_seed: int, paradigm: str,
                     sampler: CoverageSampler, dataset_cfg: DatasetConfig,
                     support_model: str, container_model: str | None,
                     cover_model: tuple[str, str] | None = None,
                     exclude_pairs: set[tuple[str, str]] | None = None,
                     ) -> ScenePlan:
    """Draw object counts from the frozen distributions and assign roles.

    - container_rummage: N_inside (target included) + N_outside; container
      required.
    - tabletop_clutter: N_total dense clutter, no container.
    - deliberate_cover: N_total incl. target + a movable cover with a
      constructed cover relation.
    """
    if paradigm not in PARADIGMS:
        raise ValueError(f"unknown paradigm {paradigm!r}")
    plan = ScenePlan(environment_seed=environment_seed, paradigm=paradigm,
                     support_model=support_model, container_model=container_model,
                     visibility_band=dataset_cfg.visibility_bands[paradigm])
    rng = sampler  # only for .sample() ordering; counts use their own stream
    target_category, target_model = sampler.sample()
    target = ObjectPlan(name="target_object", category=target_category,
                        model=target_model, role="target")
    plan.objects.append(target)

    used_pairs = {("bowl", "adciys")} if (
        container_model == "adciys") else set()
    if cover_model:
        used_pairs.add(cover_model)
    if exclude_pairs:
        used_pairs |= set(exclude_pairs)

    def add_distractor(role: str) -> None:
        # one instance per (category, model) within a scene: duplicate USD
        # instantiation segfaults Kit on this host (2026-10-03 pilot finding)
        for _ in range(40):
            category, model = sampler.sample()
            if (category, model) not in used_pairs:
                used_pairs.add((category, model))
                plan.objects.append(ObjectPlan(
                    name=f"{role}_{len(plan.objects)}",
                    category=category, model=model, role=role))
                return
        raise ValueError("sampling exhausted: no unique (category, model) left")

    if paradigm == "container_rummage":
        if not container_model:
            raise ValueError("container_rummage requires a container model")
        n_inside = dataset_cfg.sample_count(
            dataset_cfg.rummage_inside_distribution, _count_rng(environment_seed))
        n_outside = dataset_cfg.sample_count(
            dataset_cfg.rummage_outside_distribution, _count_rng(environment_seed))
        # 2026-10-03 pilot finding: the 11th DatasetObject import segfaults
        # Kit on this host regardless of asset; cap total imports at 10 until
        # the host limit is understood (documented in CHANGELOG)
        # total imports = target + (n_inside-1 inside) + n_outside + container
        while n_inside + n_outside + 2 > 10:
            if n_inside > 6:
                n_inside -= 1
            elif n_outside > 0:
                n_outside -= 1
            else:
                break
        # target counts as one of N_inside
        for _ in range(max(0, n_inside - 1)):
            add_distractor("inside")
        for _ in range(n_outside):
            add_distractor("outside")
        plan.counts = {"n_inside": n_inside, "n_outside": n_outside,
                       "n_total": n_inside + n_outside}
    else:
        n_total = dataset_cfg.sample_count(
            dataset_cfg.object_count_distribution, _count_rng(environment_seed))
        for _ in range(max(0, n_total - 1)):
            add_distractor("distractor")
        plan.counts = {"n_total": n_total}

    if paradigm == "deliberate_cover":
        if not cover_model:
            raise ValueError("deliberate_cover requires (category, model) cover")
        cover_category, cover_model_id = cover_model
        plan.objects.append(ObjectPlan(name="cover_object",
                                       category=cover_category,
                                       model=cover_model_id, role="cover"))

    # canonical object naming: stable, role-prefixed, robot-independent
    for i, obj in enumerate(plan.objects):
        obj.name = f"{obj.role}_{i:02d}_{obj.category}"
    return plan


def _count_rng(environment_seed: int):
    """Count draws use the geometry stream so object-category sampling never
    influences how many objects a scene contains."""
    from rummagebench.authoring.level1.rng import derive_streams
    return derive_streams(environment_seed)["geometry"]
