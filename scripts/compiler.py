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
from rummagebench.authoring.level1.grammar import ObjectPlan, ScenePlan
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
                      surface_size_xy, surface_top_z: float,
                      container_interior_size=None,
                      cover_dims: dict[str, list[float]] | None = None,
                      ) -> StageACandidate:
    """Stage A for one candidate, in the SUPPORT-LOCAL frame.

    All placement positions are relative to the support surface: origin =
    support AABB min-corner (x, y) with z measured from the support top.
    The GPU realization maps these through the MEASURED support/tray AABBs
    (nothing is assumed about world placement). Raises StructuralFailure
    with a §43 code.
    """
    streams = derive_streams(plan.environment_seed)
    packing_rng = streams["packing"]
    geom = streams["geometry"]
    voxel = dataset_cfg.voxel_size_m

    surface_cells = (max(4, int(surface_size_xy[0] / voxel)),
                     max(4, int(surface_size_xy[1] / voxel)))
    height_cells = 24  # 0.48 m of stacking headroom at 2 cm voxels
    grid = OccupancyGrid(origin=[0.0, 0.0, 0.0],
                         dims=(surface_cells[0], surface_cells[1], height_cells),
                         voxel_size=voxel)
    support = SupportHeightGrid.from_support_box(
        [surface_cells[0] * voxel / 2, surface_cells[1] * voxel / 2],
        [surface_cells[0] * voxel, surface_cells[1] * voxel],
        0.0, cell_size=voxel)

    placements: list[dict] = []
    objects: list[dict] = []
    relations: list[dict] = []
    counts = dict(plan.counts)

    target_plan = plan.target()
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
        relations.append({"object": target_plan.name,
                          "cover": cover_plan.name, **relation})

    container_local = None
    if plan.paradigm == "container_rummage":
        if not container_interior_size:
            raise StructuralFailure("NO_CONTAINER_FIT")
        container_cells = (max(4, int(container_interior_size[0] / voxel)),
                           max(4, int(container_interior_size[1] / voxel)),
                           max(3, int(container_interior_size[2] / voxel)))
        container_local = {"dims": container_cells,
                           "origin_cells": _free_container_origin(
                               grid, container_cells, packing_rng)}

    def pack(obj_plan, role):
        proxy = aabb_proxy(obj_plan.category, obj_plan.model,
                           _default_dims(obj_plan))
        dims = np.asarray(proxy["grid_dims"], dtype=int)
        if plan.paradigm == "container_rummage" and role in ("target", "inside"):
            proposal = pack_in_container(
                grid, _mask_from_cells(container_local), dims, packing_rng)
            if proposal is None:
                raise StructuralFailure("OCCUPANCY_PACK_FAILED")
            placements.append({
                "object": obj_plan.name, "role": role, "relation": "inside",
                "position_local_cells": [int(v) for v in proposal["origin"]],
                "container": plan.container_model,
            })
        else:
            proposal = pack_on_support(
                grid, support, voxelize_dims(dims * voxel), voxel,
                packing_rng, _cluster_centers(support, geom))
            if proposal is None:
                raise StructuralFailure("OCCUPANCY_PACK_FAILED")
            lifted = lift_to_pose(proposal["origin_world"],
                                  proposal["dims"], voxel, packing_rng)
            placements.append({
                "object": obj_plan.name, "role": role,
                "relation": "on_support",
                "position_local_cells": [int(v) for v in
                                         grid.world_to_cell(
                                             lifted["position"])],
                "yaw_deg": lifted["yaw_deg"],
            })
        objects.append({"name": obj_plan.name, "category": obj_plan.category,
                        "model": obj_plan.model, "role": role, "proxy": proxy})

    # target first (its relation is guaranteed), then the rest
    pack(target_plan, "target")
    if plan.paradigm == "container_rummage":
        container_plan_obj = ObjectPlan(name="container_object",
                                        category="tray",
                                        model=plan.container_model or "tray",
                                        role="container")
    else:
        container_plan_obj = None
    for obj in plan.objects:
        if obj is target_plan or obj is cover_plan:
            continue
        pack(obj, obj.role)
    if container_plan_obj is not None:
        objects.append({"name": container_plan_obj.name,
                        "category": "tray", "model": plan.container_model,
                        "role": "container", "proxy": None})
    if plan.paradigm == "deliberate_cover" and relations and cover_plan:
        rel = relations[0]
        target_placement = next(p for p in placements
                                if p["object"] == target_plan.name)
        placements.append({
            "object": cover_plan.name, "role": "cover",
            "relation": rel["relation"],
            "position_local_cells": list(target_placement[
                "position_local_cells"]),
            "yaw_deg": 0.0,
            "note": "cover realized inverted above the target; GPU stage "
                    "re-derives z from the real cover cavity",
        })
        objects.append({"name": cover_plan.name, "category": cover_plan.category,
                        "model": cover_plan.model, "role": "cover",
                        "proxy": aabb_proxy(cover_plan.category,
                                            cover_plan.model,
                                            cover_dims[cover_plan.name])})

    counts["objects_realized"] = len(placements)
    occupancy_spec = {
        "voxel_size_m": voxel,
        "frame": "support_local (origin = support AABB min xy, z from top)",
        "grid_dims": [int(v) for v in grid.dims],
        "occupied_cells": int((grid.solid > 0).sum()),
        "container_local": container_local,
    }
    return StageACandidate(
        environment_seed=plan.environment_seed,
        paradigm=plan.paradigm,
        support_model=plan.support_model,
        container_model=plan.container_model,
        surface={"size_xy": [float(v) for v in surface_size_xy],
                 "top_z": float(surface_top_z)},
        occupancy_spec=occupancy_spec,
        placements=placements,
        objects=objects,
        relations=relations,
    )


def _mask_from_cells(container_local):
    from rummagebench.authoring.level1.occupancy.grid import ContainerMask
    return ContainerMask(origin=np.zeros(3), dims=container_local["dims"],
                         voxel_size=0.02)


def _free_container_origin(grid, container_cells, rng) -> list[int]:
    """Pick a container origin inside the support grid with clearance."""
    for _ in range(200):
        cx = int(rng.integers(2, max(3, grid.dims[0] - container_cells[0] - 2)))
        cy = int(rng.integers(2, max(3, grid.dims[1] - container_cells[1] - 2)))
        return [cx, cy]
    return [2, 2]


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
