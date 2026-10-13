"""Versioned household proposals over the existing priors, assets and protocol.

Only kitchen_cabinet_search is implemented. Specs are evaluator-private;
natural language proposes data, never executable Python or hidden positions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rummagebench.authoring.level1.appearance import sample_appearance
from rummagebench.authoring.level1.rng import stage_seed
from rummagebench.authoring.tasks.eligibility import ObjectEligibility
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path
from rummagebench.authoring.tasks.priors import PlacementPriors

UPSTREAM_COMMIT = "b11700ebac77695d8f53538705cfdc0cd890c54d"


class StrictData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SearchLocation(StrictData):
    location_id: str
    room_type: Literal["kitchen"] = "kitchen"
    slot_type: Literal["cabinet_interior", "drawer"] = "cabinet_interior"
    relation: Literal["inside"] = "inside"
    prior: float = Field(gt=0, le=1)
    initial_open: Literal[False] = False
    source_entity: str | None = None
    assembly_entity: str | None = None


class TargetAsset(StrictData):
    category: str
    model: str
    internal_entity: str = "search_target"
    bbox_m: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def positive_geometry(self):
        if min(self.bbox_m) <= 0:
            raise ValueError("target dimensions must be positive")
        return self


class HouseholdTaskSpec(StrictData):
    schema_version: Literal[1] = 1
    generator_version: Literal["household-infinigen-1"] = "household-infinigen-1"
    task_id: str
    family_id: str
    base_layout_id: str
    seed: int = Field(ge=0)
    index: int = Field(ge=0)
    scene_seed: int = Field(ge=0)
    family: Literal["kitchen_cabinet_search"] = "kitchen_cabinet_search"
    room_type: Literal["kitchen"] = "kitchen"
    instruction: str
    target: TargetAsset
    locations: list[SearchLocation] = Field(min_length=2, max_length=8)
    target_location: str
    background_policy: Literal["preserve_upstream_household_semantics_all_surfaces"]
    appearance: dict
    appearance_config: dict = Field(default_factory=dict)
    robot_config_ref: str
    robot_id: Literal["r1pro"] = "r1pro"
    required_skills: list[str]
    max_planning_steps: int = Field(ge=1)
    success_rule: Literal["REPORT_DONE_while_holding_target"] = "REPORT_DONE_while_holding_target"
    wrong_grasp_terminal: Literal[False] = False
    initial_visibility_max: float = Field(ge=0, le=1)
    requested_difficulty: dict[str, Literal["low", "high"]]
    proxy_metrics: dict = Field(default_factory=dict)
    proxy_difficulty: dict = Field(default_factory=dict)
    measured_metrics: dict = Field(default_factory=dict)
    final_difficulty: dict | None = None
    status: Literal["candidate", "pending_simulation", "validated", "rejected"] = "candidate"
    reasons: list[str] = Field(default_factory=list)
    upstream_commit: Literal[UPSTREAM_COMMIT] = UPSTREAM_COMMIT
    input_hashes: dict[str, str]

    @model_validator(mode="after")
    def semantic_contract(self):
        from rummagebench.core.public_types import PUBLIC_SKILLS
        ids = [s.location_id for s in self.locations]
        if len(ids) != len(set(ids)) or self.target_location not in ids:
            raise ValueError("duplicate or unknown candidate location")
        if abs(sum(s.prior for s in self.locations) - 1) > 1e-8:
            raise ValueError("target prior must sum to one over the complete candidate set")
        if set(self.requested_difficulty) != {"spatial", "operations"}:
            raise ValueError("both requested cost dimensions are required")
        if set(self.required_skills) - set(PUBLIC_SKILLS) or not {"OPEN", "GRASP", "REPORT_DONE"} <= set(self.required_skills):
            raise ValueError("invalid public skills for cabinet search")
        if self.initial_visibility_max != 0:
            raise ValueError("cabinet-search target must initially be invisible")
        if self.status == "validated" or self.final_difficulty is not None:
            # CPU authoring never grants simulator acceptance. A separate
            # receipt verifier must authorize release after real measurement.
            raise ValueError("proposal schema cannot grant physical acceptance or final difficulty")
        if not self.input_hashes or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                                        for v in self.input_hashes.values()):
            raise ValueError("all source/config hashes must be SHA256 digests")
        canonical_json(self.model_dump(mode="json"))  # also rejects NaN nested in arbitrary metric/config dictionaries
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def public_task(self) -> dict:
        return {"instruction": self.instruction, "max_planning_steps": self.max_planning_steps}


def sample_spec(policy_path: Path, repo: Path, seed: int, index: int,
                *, appearance_config: dict | None = None) -> HouseholdTaskSpec:
    policy = yaml.safe_load(Path(policy_path).read_text())
    if (policy["implemented_family"] != "kitchen_cabinet_search"
            or policy["upstream"]["commit"] != UPSTREAM_COMMIT or policy["target_prior"] != "uniform"):
        raise ValueError("UNSUPPORTED_TASK_FAMILY")
    pri_path, el_path = repo / policy["placement_priors"], repo / policy["eligibility"]
    priors, eligibility = PlacementPriors.load(pri_path), ObjectEligibility.load(el_path)
    eligible = [c for c in policy["target_categories"] if c in eligibility.categories and eligibility.categories[c].models
                and priors.is_approved(c, "cabinet_interior", "kitchen")]
    if not eligible:
        raise ValueError("NO_APPROVED_TARGET_ASSET")
    rng = np.random.default_rng(stage_seed(seed, index, "household_target_asset"))
    category = sorted(eligible)[int(rng.integers(len(eligible)))]
    models = sorted(eligibility.categories[category].models, key=lambda m: m.model_id)
    model = models[int(rng.integers(len(models)))]
    rng = np.random.default_rng(stage_seed(seed, index, "household_candidates"))
    lo, hi = policy["candidate_count"]
    n = int(rng.integers(lo, hi + 1))
    locations = [SearchLocation(location_id=f"storage_{i}", prior=1 / n) for i in range(n)]
    choice = np.random.default_rng(stage_seed(seed, index, "household_target_location"))
    target_location = locations[int(choice.integers(n))].location_id
    scene_seed = stage_seed(seed, index, "household_background")
    layout = "ig_" + sha256_bytes(canonical_json({"commit": UPSTREAM_COMMIT, "scene_seed": scene_seed,
                                                "profile": "kitchen_complete"}))[:20]
    family_id = "family_" + sha256_bytes(canonical_json({"layout": layout, "category": category,
                                                        "model": model.model_id, "candidates": n}))[:20]
    appearance_path = repo / policy["appearance_config"]
    appearance_cfg = appearance_config or yaml.safe_load(appearance_path.read_text())
    appearance = sample_appearance(np.random.default_rng(stage_seed(seed, index, "household_appearance")), appearance_cfg)
    level_rng = np.random.default_rng(stage_seed(seed, index, "household_requested_difficulty"))
    levels = {k: ("high" if level_rng.integers(2) else "low") for k in ("spatial", "operations")}
    hashes = {"policy": sha256_path(policy_path), "priors": sha256_path(pri_path),
              "eligibility": sha256_path(el_path), "robot_config": sha256_path(repo / policy["robot_config"]),
              "appearance_config": sha256_bytes(canonical_json(appearance_cfg)),
              "appearance_source_file": sha256_path(appearance_path)}
    identity = sha256_bytes(canonical_json({"family": family_id, "target_location": target_location,
                                           "seed": seed, "index": index, "inputs": hashes}))[:20]
    return HouseholdTaskSpec(task_id="task_" + identity, family_id=family_id, base_layout_id=layout,
        seed=seed, index=index, scene_seed=scene_seed,
        instruction=f"Find the {category.replace('_', ' ')} in the kitchen cabinets and pick it up.",
        target=TargetAsset(category=category, model=model.model_id, bbox_m=list(model.bbox)),
        locations=locations, target_location=target_location, background_policy=policy["background_policy"],
        appearance=appearance, appearance_config=appearance_cfg, robot_config_ref=policy["robot_config"], required_skills=policy["required_skills"],
        max_planning_steps=policy["max_planning_steps"], initial_visibility_max=policy["initial_visibility_max"],
        requested_difficulty=levels, input_hashes=hashes)


def grouped_split(specs: list[HouseholdTaskSpec], *, test_fraction: float = .2,
                  split_seed: int = 0) -> dict[str, str]:
    if not 0 <= test_fraction <= 1:
        raise ValueError("invalid split fraction")
    result = {}
    for spec in specs:
        digest = sha256_bytes(canonical_json({"base_layout": spec.base_layout_id, "seed": split_seed}))
        fraction = int(digest[:16], 16) / 2**64
        result[spec.task_id] = "test" if fraction < test_fraction else "dev"
    return result


def validate_spec_assets(spec: HouseholdTaskSpec, repo: Path, policy_path: Path) -> None:
    """Mandatory semantic gate for template or externally proposed JSON specs."""
    policy = yaml.safe_load(Path(policy_path).read_text())
    pri_path, el_path = repo / policy["placement_priors"], repo / policy["eligibility"]
    priors, eligibility = PlacementPriors.load(pri_path), ObjectEligibility.load(el_path)
    if (spec.robot_config_ref != policy["robot_config"] or spec.background_policy != policy["background_policy"]
            or spec.family != policy["implemented_family"] or spec.max_planning_steps != policy["max_planning_steps"]
            or not policy["candidate_count"][0] <= len(spec.locations) <= policy["candidate_count"][1]
            or any(s.slot_type != "cabinet_interior" or abs(s.prior - 1/len(spec.locations)) > 1e-8 for s in spec.locations)):
        raise ValueError("UNSUPPORTED_OR_CHANGED_FAMILY_POLICY")
    if spec.target.category not in policy["target_categories"] or not eligibility.is_eligible(spec.target.category, spec.target.model):
        raise ValueError("INVALID_TARGET_CATEGORY_OR_MODEL")
    box = eligibility.categories[spec.target.category].model(spec.target.model).bbox
    if list(box) != spec.target.bbox_m:
        raise ValueError("TARGET_GEOMETRY_DIFFERS_FROM_WHITELIST")
    if sha256_bytes(canonical_json(spec.appearance_config)) != spec.input_hashes.get("appearance_config"):
        raise ValueError("SPEC_INPUT_CHANGED:appearance_config")
    for location in spec.locations:
        if not priors.is_approved(spec.target.category, location.slot_type, location.room_type):
            raise ValueError("UNAPPROVED_TARGET_LOCATION_PRIOR")
    for key, path in (("policy", policy_path), ("priors", pri_path), ("eligibility", el_path),
                      ("robot_config", repo / policy["robot_config"]),
                      ("appearance_source_file", repo / policy["appearance_config"])):
        if spec.input_hashes.get(key) != sha256_path(path):
            raise ValueError(f"SPEC_INPUT_CHANGED:{key}")


def family_members(spec: HouseholdTaskSpec) -> list[HouseholdTaskSpec]:
    """Complete target-position family; never prune locations after generation."""
    result = []
    for location in spec.locations:
        doc = spec.model_dump(mode="json")
        doc["target_location"] = location.location_id
        doc["task_id"] = "task_" + sha256_bytes(canonical_json({"family": spec.family_id,
            "target_location": location.location_id, "seed": spec.seed, "index": spec.index,
            "inputs": spec.input_hashes}))[:20]
        result.append(HouseholdTaskSpec.model_validate(doc))
    return result

