"""Layout spec: declarative config models (§27) and generated placements."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LayoutConfig(BaseModel):
    """All generator constants are YAML-configurable — no Python magic
    numbers (§27)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    base_scene: str  # e.g. Beechwood_0_int (architectural shell, reused §25)
    storage_units: int = 3
    work_surfaces: int = 1
    clutter_per_surface: int = 3
    min_furniture_clearance_m: float = Field(default=0.10, ge=0.0)
    min_robot_clearance_m: float = Field(default=0.55, ge=0.0)
    max_sampling_attempts: int = 500
    # placement regions (world-frame AABBs inside the shell)
    placement_regions: list[list[float]] = Field(
        default_factory=list,
        description="each region: [min_x, min_y, max_x, max_y]",
    )
    robot_spawn: list[float] = Field(default_factory=lambda: [0.0, 0.0])


class Placement(BaseModel):
    """One generated object: exact model id + world transform (§30)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: str
    model_id: str
    position: list[float]
    orientation_deg: float  # yaw only: upright alignment (§28.3)
    aabb_size: list[float]  # [sx, sy, sz] from the asset catalog
    role: str  # storage | surface | clutter
