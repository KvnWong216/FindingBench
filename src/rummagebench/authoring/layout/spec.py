"""Strict candidate-layout schema. Transforms use AABB-center coordinates."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]

class LayoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    base_scene: str
    storage_units: int = Field(default=3, ge=0)
    work_surfaces: int = Field(default=1, ge=0)
    clutter_per_surface: int = Field(default=3, ge=0)
    min_furniture_clearance_m: float = Field(default=0.10, ge=0, allow_inf_nan=False)
    min_robot_clearance_m: float = Field(default=0.55, ge=0, allow_inf_nan=False)
    front_strip_extra_m: float = Field(default=0.30, ge=0, allow_inf_nan=False)
    max_sampling_attempts: int = Field(default=500, ge=1)
    placement_regions: list[list[Finite]] = Field(min_length=1)
    robot_spawn: list[Finite] = Field(default_factory=lambda: [0., 0.], min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_regions(self):
        for region in self.placement_regions:
            if len(region) != 4 or region[0] >= region[2] or region[1] >= region[3]:
                raise ValueError("placement_regions require [xmin, ymin, xmax, ymax] with positive area")
        return self

class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    category: str
    model_id: str
    position: list[Finite] = Field(min_length=3, max_length=3)
    orientation_deg: Finite
    aabb_size: list[Positive] = Field(min_length=3, max_length=3)
    role: Literal["storage", "surface", "clutter"]
    support_name: str | None = None
