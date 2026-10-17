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



SPLIT_STRATEGY = "base-layout-sha256-v1"


def layout_split_label(layout: str, seed: int, test_fraction: float) -> str:
    if seed < 0 or not 0 <= test_fraction <= 1:
        raise ValueError("INVALID_SPLIT_PARAMETERS")
    digest = sha256_bytes(canonical_json({"base_layout": layout, "seed": seed}))
    return "test" if int(digest[:16], 16) / 2**64 < test_fraction else "dev"


class SplitAssignment(StrictData):
    strategy_version: Literal["base-layout-sha256-v1"] = SPLIT_STRATEGY
    split_seed: int = Field(ge=0)
    test_fraction: float = Field(ge=0, le=1)
    base_layout_id: str
    source_layout_group: str | None = None
    label: Literal["dev", "test"]

    @model_validator(mode="after")
    def check_assignment(self):
        if self.label != layout_split_label(self.source_layout_group or self.base_layout_id, self.split_seed, self.test_fraction):
            raise ValueError("SPLIT_LABEL_DIFFERS_FROM_PLANNED_GROUP")
        return self


def source_layout_group_id(upstream_commit: str, scene_seed: int) -> str:
    return "ig_source_" + sha256_bytes(canonical_json({"upstream":upstream_commit,"scene_seed":scene_seed}))[:20]

class HouseholdTaskSpec(StrictData):
    schema_version: Literal[1] = 1
    generator_version: Literal["household-infinigen-1", "household-infinigen-2", "household-infinigen-native-3"] = "household-infinigen-2"
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
    locations: list[SearchLocation] = Field(min_length=2, max_length=128)
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
    split_assignment: SplitAssignment | None = None
    requested_difficulty: dict[str, Literal["low", "high"]]
    generation_constraints: dict = Field(default_factory=dict)
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
        from rummagebench.core.legacy_public_types import PUBLIC_SKILLS
        ids = [s.location_id for s in self.locations]
        if self.generator_version!='household-infinigen-native-3' and len(ids)>8:
            raise ValueError('SOURCE_PROPOSAL_LOCATION_LIMIT_EXCEEDED')
        if self.generator_version=='household-infinigen-native-3' and not self.generation_constraints.get('native_region_lineage'):
            raise ValueError('NATIVE_REGION_SPEC_REQUIRES_PROTECTED_SOURCE_LINEAGE')
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
        if self.split_assignment is not None and self.split_assignment.base_layout_id != self.base_layout_id:
            raise ValueError("SPLIT_BASE_LAYOUT_MISMATCH")
        canonical_json(self.model_dump(mode="json"))  # also rejects NaN nested in arbitrary metric/config dictionaries
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def public_task(self) -> dict:
        return {"instruction": self.instruction, "max_planning_steps": self.max_planning_steps}


def sample_spec(policy_path: Path, repo: Path, seed: int, index: int,
                *, appearance_config: dict | None = None, split_seed: int = 0,
                test_fraction: float = .2, difficulty_profile: str | None = None) -> HouseholdTaskSpec:
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
    names = policy["sampling_profiles"]
    profile_name = difficulty_profile or names[index % len(names)]
    profile = policy["request_profiles"].get(profile_name)
    if profile is None:
        raise ValueError("UNSUPPORTED_DIFFICULTY_PROFILE")
    supported = profile_candidate_counts(policy, profile_name)
    n = int(supported[int(rng.integers(len(supported)))])
    requirements = generation_requirements(policy, profile_name, n)
    locations = [SearchLocation(location_id=f"storage_{i}", prior=1 / n) for i in range(n)]
    choice = np.random.default_rng(stage_seed(seed, index, "household_target_location"))
    target_location = locations[int(choice.integers(n))].location_id
    scene_seed = stage_seed(seed, index, "household_background")
    layout = "ig_" + sha256_bytes(canonical_json({"commit": UPSTREAM_COMMIT, "scene_seed": scene_seed,
                                                "profile": "kitchen_complete", "requirements": requirements}))[:20]
    family_id = "family_" + sha256_bytes(canonical_json({"layout": layout, "category": category,
                                                        "model": model.model_id, "candidates": n}))[:20]
    appearance_path = repo / policy["appearance_config"]
    appearance_cfg = appearance_config or yaml.safe_load(appearance_path.read_text())
    appearance = sample_appearance(np.random.default_rng(stage_seed(seed, index, "household_appearance")), appearance_cfg)
    levels = {k: profile[k] for k in ("spatial", "operations")}
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
        split_assignment=SplitAssignment(split_seed=split_seed, test_fraction=test_fraction, base_layout_id=layout,
            source_layout_group=source_layout_group_id(UPSTREAM_COMMIT,scene_seed),
            label=layout_split_label(source_layout_group_id(UPSTREAM_COMMIT,scene_seed),split_seed,test_fraction)),
        requested_difficulty=levels, generation_constraints=requirements, input_hashes=hashes)


def grouped_split(specs: list[HouseholdTaskSpec], *, test_fraction: float = .2,
                  split_seed: int = 0) -> dict[str, str]:
    if not 0 <= test_fraction <= 1:
        raise ValueError("invalid split fraction")
    result = {}
    for spec in specs:
        if spec.task_id in result:
            raise ValueError("DUPLICATE_SPECIFICATION_ID")
        assignment = spec.split_assignment
        if assignment is not None:
            if assignment.split_seed != split_seed or assignment.test_fraction != test_fraction:
                raise ValueError("SPLIT_PARAMETERS_DIFFER_FROM_PLANNING")
            result[spec.task_id] = assignment.label
        else:
            result[spec.task_id] = layout_split_label(spec.base_layout_id, split_seed, test_fraction)
    return result


def validate_spec_assets(spec: HouseholdTaskSpec, repo: Path, policy_path: Path) -> None:
    """Mandatory semantic gate for template or externally proposed JSON specs."""
    if spec.generator_version=='household-infinigen-native-3':
        from rummagebench.authoring.tasks.infinigen_component_binding import validate_native_family_spec
        validate_native_family_spec(spec,repo)
        return
    policy = yaml.safe_load(Path(policy_path).read_text())
    pri_path, el_path = repo / policy["placement_priors"], repo / policy["eligibility"]
    priors, eligibility = PlacementPriors.load(pri_path), ObjectEligibility.load(el_path)
    if (spec.robot_config_ref != policy["robot_config"] or spec.background_policy != policy["background_policy"]
            or spec.family != policy["implemented_family"] or spec.max_planning_steps != policy["max_planning_steps"]
            or not policy["candidate_count"][0] <= len(spec.locations) <= policy["candidate_count"][1]
            or any(s.slot_type != "cabinet_interior" or abs(s.prior - 1/len(spec.locations)) > 1e-8 for s in spec.locations)):
        raise ValueError("UNSUPPORTED_OR_CHANGED_FAMILY_POLICY")
    requirements = spec.generation_constraints
    if not requirements or requirements != generation_requirements(policy, requirements.get("profile"), len(spec.locations)):
        raise ValueError("GENERATION_REQUIREMENTS_DIFFER_FROM_POLICY")
    if spec.requested_difficulty != {k: requirements[k+"_label"] for k in ("spatial", "operations")}:
        raise ValueError("REQUESTED_DIFFICULTY_DIFFERS_FROM_CONSTRAINTS")
    if spec.target.category not in policy["target_categories"] or not eligibility.is_eligible(spec.target.category, spec.target.model):
        raise ValueError("INVALID_TARGET_CATEGORY_OR_MODEL")
    box = eligibility.categories[spec.target.category].model(spec.target.model).bbox
    if list(box) != spec.target.bbox_m:
        raise ValueError("TARGET_GEOMETRY_DIFFERS_FROM_WHITELIST")
    if sha256_bytes(canonical_json(spec.appearance_config)) != spec.input_hashes.get("appearance_config"):
        raise ValueError("SPEC_INPUT_CHANGED:appearance_config")
    try:
        expected_appearance=sample_appearance(np.random.default_rng(stage_seed(spec.seed,spec.index,'household_appearance')),
                                             spec.appearance_config)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('APPEARANCE_CONFIGURATION_INVALID') from exc
    if spec.appearance != expected_appearance:
        raise ValueError('APPEARANCE_REQUEST_DIFFERS_FROM_SEEDED_CONFIGURATION')
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


def native_direct_tier(candidate_count: int, max_planning_steps: int):
    """Instantiate the independent direct-family contract over every region.

    The complete native candidate count is an EXACT structural requirement.
    This contract grants no difficulty label and changes no physical gate.
    """
    from rummagebench.authoring.tasks.tiers import load_tier
    from rummagebench.authoring.tasks.tiers.base import TierSpec
    contract=load_tier('household_direct_v1')
    if not contract.search.candidate_slot_count.contains(candidate_count):
        raise ValueError('NATIVE_DIRECT_CANDIDATE_COUNT_INVALID')
    doc=contract.model_dump(mode='json')
    doc['search']['candidate_slot_count']={'min':candidate_count,'max':candidate_count}
    doc['budget']['max_planning_steps']=max_planning_steps
    return TierSpec.model_validate(doc)


def execution_tier(spec: HouseholdTaskSpec):
    from rummagebench.authoring.tasks.tiers import load_tier
    if spec.generator_version=='household-infinigen-native-3':
        return native_direct_tier(len(spec.locations),spec.max_planning_steps)
    tier=load_tier('easy_v1')
    return tier.model_copy(update={'budget':tier.budget.model_copy(update={'max_planning_steps':spec.max_planning_steps})})


def profile_candidate_counts(policy: dict, profile_name: str) -> list[int]:
    from rummagebench.authoring.tasks.difficulty import separable_search_cost, interval_label
    from rummagebench.authoring.tasks.household_search import validate_search_model
    validate_search_model(policy["search_model"])
    profile = policy["request_profiles"].get(profile_name)
    if profile is None:
        raise ValueError("UNSUPPORTED_DIFFICULTY_PROFILE")
    lo, hi = policy["candidate_count"]
    supported = []
    for n in profile["candidate_counts"]:
        if not isinstance(n, int) or not max(lo, 2) <= n <= min(hi, 8):
            raise ValueError("PROFILE_CANDIDATE_COUNT_OUTSIDE_SUPPORTED_RANGE")
        model = policy["search_model"]
        # The last member pays for every inspection. Its movement is at
        # least the requested expected spatial lower bound. If even that
        # bound exceeds the public horizon, this complete cell cannot run.
        spatial_lower = policy["difficulty_intervals"]["spatial"][profile["spatial"]][0]
        if spatial_lower + n * model["inspection_operations"] + model["finish_operations"] > policy["max_planning_steps"]:
            raise ValueError("REQUESTED_COST_CELL_EXCEEDS_PUBLIC_HORIZON")
        costs = separable_search_cost([0.]*n, [0.]*n, [1/n]*n,
                                     [model["inspection_operations"]]*n, finish_cost=model["finish_operations"])
        if interval_label(costs["expected_operations"], policy["difficulty_intervals"]["operations"]) == profile["operations"]:
            supported.append(n)
    if not supported or len(supported) != len(profile["candidate_counts"]):
        raise ValueError("OPERATION_DIFFICULTY_UNREACHABLE_FOR_PROFILE")
    return sorted(set(supported))


def generation_requirements(policy: dict, profile_name: str, n: int) -> dict:
    if n not in profile_candidate_counts(policy, profile_name):
        raise ValueError("CANDIDATE_COUNT_NOT_SUPPORTED_FOR_REQUEST")
    profile = policy["request_profiles"][profile_name]
    result={"version": 1, "profile": profile_name, "candidate_count": n,
            "storage_min_distance_m": float(profile["storage_min_distance_m"]),
            "spatial_label": profile["spatial"], "operations_label": profile["operations"],
            "spatial_interval": policy["difficulty_intervals"]["spatial"][profile["spatial"]],
            "operations_interval": policy["difficulty_intervals"]["operations"][profile["operations"]],
            "difficulty_intervals": {k: policy["difficulty_intervals"][k] for k in ("spatial", "operations")},
            "search_model": policy["search_model"],
            "postfilter": "exact_candidate_count_then_geometric_model_cost_and_bins; no physical acceptance"}
    if 'source_storage_dimensions_m' in policy:
        result['source_storage_dimensions_m']=policy['source_storage_dimensions_m']
    for key in ('source_sink_parameter_ranges_m','source_sink_front_margin_m','source_kitchen_cabinets_closed_only'):
        if key in policy:result[key]=policy[key]
    return result
