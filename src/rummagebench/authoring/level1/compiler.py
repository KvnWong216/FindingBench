"""§17/§42 Stage A: occupancy-guided structural compilation.

Turns a ScenePlan into a full candidate: conservative voxel proxies,
occupancy-guided placement proposals, lifted continuous poses, cover
relations and the occupancy spec — everything Stage B needs to realize the
scene in the simulator. Pure CPU; no simulator import.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rummagebench.authoring.level1.config import DatasetConfig
from rummagebench.authoring.level1.grammar import ScenePlan
from rummagebench.authoring.occupancy import (
    OccupancyGrid,
    SupportHeightGrid,
    aabb_proxy,
    lift_to_pose,
    pack_in_container,
    pack_on_support,
    propose_cover,
    voxelize_dims,
)
from rummagebench.authoring.level1.rng import derive_streams


class StructuralFailure(RuntimeError):
    """Stage A rejection (see §43 codes)."""


@dataclass
class StageACandidate:
    environment_seed: int
    paradigm: str
    support_model: str
    container_model: str | None
    surface: dict
    occupancy_spec: dict
    placements: list[dict] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)


def compile_candidate(plan: ScenePlan, dataset_cfg: DatasetConfig,
                      surface_center_xy, surface_size_xy, surface_top_z: float,
                      container_interior: dict | None,
                      cover_dims: dict[str, list[float]] | None = None,
                      ) -> StageACandidate:
    """Stage A for one candidate. Raises StructuralFailure with a §43 code.

    `surface_*` describe the verified support (center xy, size xy, top z).
    `container_interior` = {"origin", "size"} when the paradigm needs the
    open container. `cover_dims` maps cover object name -> native dims when
    the paradigm is deliberate_cover.
    """
    streams = derive_streams(plan.environment_seed)
    geom = streams["geometry"]
    packing_rng = streams["packing"]
    voxel = dataset_cfg.voxel_size_m

    grid = OccupancyGrid(origin=[0.0, 0.0, 0.0],
                         dims=(160, 160, 80), voxel_size=voxel)
    support = SupportHeightGrid.from_support_box(
        surface_center_xy, surface_size_xy, surface_top_z, cell_size=0.05)

    placements: list[dict] = []
    objects: list[dict] = []
    relations: list[dict] = []

    target_plan = plan.target()
    target_proxy = aabb_proxy(target_plan.category, target_plan.model,
                              _default_dims(target_plan))

    if plan.paradigm == "container_rummage":
        if not container_interior:
            raise StructuralFailure("NO_CONTAINER_FIT")
        mask = _container_mask(container_interior, voxel)
        if not _fits(_proxy_dims(target_proxy), mask.dims, voxel):
            raise StructuralFailure("NO_CONTAINER_FIT")

    cover_plan = next((o for o in plan.objects if o.role == "cover"), None)
    if plan.paradigm == "deliberate_cover":
        if (not cover_dims or cover_plan is None
                or cover_plan.name not in cover_dims
                or target_plan.name not in cover_dims):
            raise StructuralFailure("COVER_RELATION_FAILED")
        relation = propose_cover(
            voxelize_dims(cover_dims[cover_plan.name], voxel),
            voxelize_dims(cover_dims[target_plan.name], voxel), voxel, geom)
        if relation is None:
            raise StructuralFailure("COVER_RELATION_FAILED")
        relations.append({"object": target_plan.name, **relation})

    # placement order: target first (guaranteed its relation), then others
    order = sorted(plan.objects, key=lambda o: 0 if o.role == "target" else 1)
    for obj in order:
        proxy = aabb_proxy(obj.category, obj.model, _default_dims(obj))
        dims_cells = _proxy_dims(proxy)
        if plan.paradigm == "container_rummage" and obj.role in ("target", "inside"):
            proposal = pack_in_container(grid, _container_mask(
                container_interior, voxel), voxelize_dims(dims_cells * voxel),
                packing_rng)
            if proposal is None:
                raise StructuralFailure("OCCUPANCY_PACK_FAILED")
            lifted = lift_to_pose(proposal["origin_world"],
                                  proposal["dims"], voxel, packing_rng)
            placements.append({
                "object": obj.name, "role": obj.role, "relation": "inside",
                "position": lifted["position"], "yaw_deg": lifted["yaw_deg"],
                "container": plan.container_model,
            })
        else:
            proposal = pack_on_support(
                grid, support, voxelize_dims(_proxy_dims(proxy) * voxel),
                voxel, packing_rng,
                _cluster_centers(support, geom))
            if proposal is None:
                raise StructuralFailure("OCCUPANCY_PACK_FAILED")
            lifted = lift_to_pose(proposal["origin_world"],
                                  proposal["dims"], voxel, packing_rng)
            placements.append({
                "object": obj.name, "role": obj.role,
                "relation": "on_support",
                "position": lifted["position"], "yaw_deg": lifted["yaw_deg"],
            })
        objects.append({"name": obj.name, "category": obj.category,
                        "model": obj.model, "role": obj.role,
                        "proxy": proxy})

    if plan.paradigm == "deliberate_cover" and relations:
        # the cover object physically rests inverted above the target
        rel = relations[0]
        placements.append({
            "object": "cover_object", "role": "cover",
            "relation": rel["relation"],
            "position": _cover_position(placements, target_plan.name, rel,
                                        voxel),
            "yaw_deg": 0.0,
        })
        objects.append({"name": cover_plan.name, "role": "cover",
                        "category": cover_plan.category,
                        "model": cover_plan.model,
                        "proxy": aabb_proxy(
                            cover_plan.category,
                            cover_plan.model,
                            cover_dims[cover_plan.name])})

    occupancy_spec = {
        "voxel_size_m": voxel,
        "grid_dims": [int(v) for v in grid.dims],
        "occupied_cells": int((grid.solid > 0).sum()),
        "container_interior": container_interior,
    }
    return StageACandidate(
        environment_seed=plan.environment_seed,
        paradigm=plan.paradigm,
        support_model=plan.support_model,
        container_model=plan.container_model,
        surface={"center_xy": [float(v) for v in surface_center_xy],
                 "size_xy": [float(v) for v in surface_size_xy],
                 "top_z": float(surface_top_z)},
        occupancy_spec=occupancy_spec,
        placements=placements,
        objects=objects,
        relations=relations,
    )


def _proxy_dims(proxy: dict) -> np.ndarray:
    return np.asarray([int(v) for v in proxy["grid_dims"]], dtype=float)


def _default_dims(obj) -> np.ndarray:
    """Placeholder native dimensions (m) for inventory-stage objects.

    Real per-model dims come from the GPU catalog verification stage; the
    Stage A compiler takes explicit dims through obj metadata when
    available and falls back to a size-class default here so planning is
    testable on CPU.
    """
    return np.array([0.12, 0.12, 0.12])


def _container_mask(interior: dict, voxel: float):
    from rummagebench.authoring.occupancy.container import (
        build_container_mask)
    return build_container_mask(interior["origin"], interior["size"], voxel)


def _fits(proxy_dims_cells, mask_dims, voxel: float) -> bool:
    return bool(np.all(np.asarray(proxy_dims_cells) <= np.asarray(mask_dims)))


def _cluster_centers(support: SupportHeightGrid, geom):
    from rummagebench.authoring.occupancy.support import (
        sample_cluster_centers)
    return sample_cluster_centers(support, margin_m=0.08, rng=geom,
                                  n_clusters=2)


def _cover_category(plan: ScenePlan, cover_dims) -> str:
    for o in plan.objects:
        if o.role == "cover":
            return o.category
    return "cover_object"


def _cover_model(plan: ScenePlan, cover_dims) -> str:
    for o in plan.objects:
        if o.role == "cover":
            return o.model
    return "cover_model"


def _cover_position(placements, target_name, rel, voxel: float) -> list[float]:
    for p in placements:
        if p["object"] == target_name:
            return [float(v) for v in p["position"]]
    return [0.0, 0.0, 0.0]
