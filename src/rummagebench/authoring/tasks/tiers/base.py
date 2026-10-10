"""TierSpec: the data contract of one tier (assets/tasks/tiers/<tier>_v<n>.yaml).

Bounds are inclusive integer/float ranges. A YAML bound may be written as a
scalar (exact), a two-item list [lo, hi], or a mapping {min: lo, max: hi}
with either side optional. The Python source never duplicates tier numbers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Bound(BaseModel):
    """Inclusive range; None on a side means unbounded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, raw: Any) -> Any:
        if isinstance(raw, bool):
            raise ValueError("a bound cannot be a boolean")
        if isinstance(raw, (int, float)):
            return {"min": raw, "max": raw}
        if isinstance(raw, (list, tuple)):
            if len(raw) != 2:
                raise ValueError(f"range list must be [lo, hi], got {raw!r}")
            return {"min": raw[0], "max": raw[1]}
        return raw

    @model_validator(mode="after")
    def _ordered(self) -> "Bound":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"bound min {self.min} > max {self.max}")
        return self

    def contains(self, value: float) -> bool:
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True

    def describe(self) -> str:
        lo = "-inf" if self.min is None else f"{self.min:g}"
        hi = "inf" if self.max is None else f"{self.max:g}"
        return f"[{lo}, {hi}]"


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # frozen protocol horizon: 16 for every tier
    max_planning_steps: int = Field(default=16, ge=1)
    # steps held back for partial-observation search on top of the
    # full-information lower bound; 0 = gate on the bare lower bound
    search_reserve_steps: int = Field(default=0, ge=0)


class SearchSpec(BaseModel):
    """Search-structure bounds checked on the certificate."""

    model_config = ConfigDict(extra="forbid")

    room_count: Bound
    candidate_slot_count: Bound
    containment_depth: Bound = Bound(min=0, max=1)
    rearrangement_depth: Bound
    lookalike_count: Bound


class SlotFillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extra_items: Bound = Bound(min=0, max=0)
    max_fill_ratio: float = Field(default=0.30, gt=0, le=1)


class OtherContainersSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    occupied_fraction: Bound = Bound(min=0.0, max=0.0)
    items: Bound = Bound(min=0, max=0)
    max_fill_ratio: float = Field(default=0.5, gt=0, le=1)


class OtherSurfacesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: Bound = Bound(min=0, max=0)
    # summed object footprints / surface footprint
    max_area_ratio: float = Field(default=0.5, gt=0, le=1)


class ModeGateSpec(BaseModel):
    """Simulator-measured gates of one mode."""

    model_config = ConfigDict(extra="forbid")

    open_required: bool = False
    # target must NOT be graspable before the tier's required reveal
    pre_reveal_graspable: bool = False
    # benchmark threshold (what the tier promises)
    revealed_visibility_min: float = Field(default=0.60, ge=0, le=1)
    # stricter generator acceptance boundary (RTX is not pixel-deterministic)
    generator_visibility_margin: float = Field(default=0.65, ge=0, le=1)

    @model_validator(mode="after")
    def _margin_not_looser(self) -> "ModeGateSpec":
        if self.generator_visibility_margin < self.revealed_visibility_min:
            raise ValueError("generator_visibility_margin must be >= "
                             "revealed_visibility_min")
        return self


class GatesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_visibility_max: float = Field(default=0.0, ge=0, le=1)
    modes: dict[str, ModeGateSpec]


class InstructionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # str.format templates; fields: {object}, {room}
    templates: list[str] = Field(min_length=1)
    # used when the search is scoped to a landmark region; fields: {object},
    # {room}, {landmark}
    region_templates: list[str] = Field(default_factory=list)


class TierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: str
    version: int = Field(ge=1)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    # mode -> sampling weight; must match gates.modes keys
    modes: dict[str, float]
    search: SearchSpec
    target_slot: SlotFillSpec = Field(default_factory=SlotFillSpec)
    # medium: total container population independent of rearrangement depth
    target_container_total_items: Optional[Bound] = None
    # medium: flat cover objects stacked on the target (count)
    covers: Optional[Bound] = None
    # curated perceptual lookalike groups for this tier, on top of the
    # eligibility taxonomy keys (categories a camera confuses at a glance:
    # spoon / fork / knife). Tier-scoped so easy's inputs never change
    lookalike_groups: list[list[str]] = Field(default_factory=list)
    # scenario termination: a non-target GRASP ends the episode. Rearrangement
    # tiers need it False (the visual agent protocol already makes wrong
    # grasps recoverable; this lets the full-information oracle move covers)
    fail_on_wrong_grasp: bool = True
    other_containers: OtherContainersSpec = Field(default_factory=OtherContainersSpec)
    other_surfaces: OtherSurfacesSpec = Field(default_factory=OtherSurfacesSpec)
    # builder-cost cap on non-target objects per episode (placement + settle
    # time grows with every spawned object); None = uncapped
    max_distractors: Optional[int] = Field(default=None, ge=0)
    gates: GatesSpec
    instructions: Optional[InstructionSpec] = None
    # False disables planning for this tier; its structural contract is
    # still enforced by the certificate gate
    planner_enabled: bool = True

    @model_validator(mode="after")
    def _modes_consistent(self) -> "TierSpec":
        if not self.modes:
            raise ValueError("a tier needs at least one mode")
        if any(w <= 0 for w in self.modes.values()):
            raise ValueError("mode weights must be positive")
        missing = set(self.modes) - set(self.gates.modes)
        if missing:
            raise ValueError(f"modes without gates: {sorted(missing)}")
        return self

    @property
    def name(self) -> str:
        return f"{self.tier}_v{self.version}"


def load_tier_spec(path: str | Path) -> TierSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"tier spec {path} must be a mapping")
    return TierSpec.model_validate(raw)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
