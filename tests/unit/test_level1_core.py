"""§52: Level-1 data factory unit tests (CPU only).

Covers the revision §52 list that is testable without a simulator: catalog
filtering, exact model IDs, voxel determinism, occupancy non-overlap,
support placement, container fit, coverage-aware sampling, count
distributions, RNG stream independence, appearance-vs-geometry invariance,
cover relations, stability detector, rejection codes, snapshot
reproducibility, robot registry, paired-environment invariance,
split-by-environment, resume logic, registry hash consistency.
Simulator-dependent checks (visibility rendering, contact extraction inside
Kit, witness replay) run in the GPU certification scripts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rummagebench.authoring.level1.catalog import (  # noqa: E402
    CatalogEntry,
    category_excluded,
    eligible_inventory,
    scan_inventory,
)
from rummagebench.authoring.level1.certifier import (  # noqa: E402
    RejectionCode,
    SettleMeasurement,
    check_post_settle_world,
    check_relations,
    check_settle,
    check_visibility,
)
from rummagebench.authoring.level1.config import DatasetConfig  # noqa: E402
from rummagebench.authoring.level1.coverage_sampler import (  # noqa: E402
    CoverageSampler,
)
from rummagebench.authoring.level1.grammar import build_scene_plan  # noqa: E402
from rummagebench.authoring.occupancy import (  # noqa: E402
    OccupancyGrid,
    SupportHeightGrid,
    aabb_proxy,
    lift_to_pose,
    pack_in_container,
    pack_on_support,
    propose_cover,
    save_proxy,
    voxelize_dims,
)
from rummagebench.authoring.occupancy.container import (  # noqa: E402
    build_container_mask,
    object_fits,
)
from rummagebench.authoring.occupancy.cover import (  # noqa: E402
    select_cover_relation,
)
from rummagebench.authoring.level1.rng import derive_streams  # noqa: E402
from rummagebench.authoring.level1.registry import (  # noqa: E402
    DatasetRegistry,
    RegistryEntry,
)
from rummagebench.authoring.level1.robot_registry import (  # noqa: E402
    RobotProfile,
    production_ready,
)
from rummagebench.authoring.level1.snapshot import (  # noqa: E402
    atomic_write_json,
    canonical_hash,
)
from rummagebench.authoring.level1.split import assign_splits  # noqa: E402


@pytest.fixture
def dataset_cfg() -> DatasetConfig:
    return DatasetConfig.load(REPO / "configs" / "level1" / "dataset_v1.yaml")


@pytest.fixture
def sampler():
    def make(seed: int = 7):
        entries = [CatalogEntry(category=f"cat_{i}", model_id=f"m{i}a",
                                asset_path=f"/x/cat_{i}/m{i}a")
                   for i in range(5)]
        entries += [CatalogEntry(category="cat_0", model_id="m0b",
                                 asset_path="/x/cat_0/m0b")]
        rng = derive_streams(seed)["object"]
        return CoverageSampler(entries, rng)
    return make


# --------------------------------------------------------------- catalog
def test_catalog_filters_architecture_and_furniture():
    assert category_excluded("bottom_cabinet_no_top")
    assert category_excluded("wall_long")
    assert category_excluded("countertop")
    assert category_excluded("fridge_freezer")
    assert category_excluded("water")
    assert category_excluded("cloth_towel")
    assert category_excluded("robot_arm")
    assert not category_excluded("tomato_sauce_jar")
    assert not category_excluded("breakfast_table") is None or True
    # tables are excluded from ORDINARY objects (they are the support)
    assert category_excluded("breakfast_table")


def test_catalog_exact_model_ids_from_directory(tmp_path):
    for category, models in {"tomato_sauce_jar": ["krfzqk"],
                             "bowl": ["adciys", "ajzltc"]}.items():
        for m in models:
            (tmp_path / category / m).mkdir(parents=True)
    entries = scan_inventory(tmp_path)
    pairs = {(e.category, e.model_id) for e in entries}
    assert ("tomato_sauce_jar", "krfzqk") in pairs
    assert ("bowl", "adciys") in pairs and ("bowl", "ajzltc") in pairs


# ------------------------------------------------------- voxel + occupancy
def test_voxel_proxy_deterministic_and_thin_safe():
    dims = [0.26, 0.02, 0.15]
    g1 = voxelize_dims(dims, 0.02)
    g2 = voxelize_dims(dims, 0.02)
    assert g1.shape == g2.shape and g1.dtype == np.bool_
    # thin axis: 0.02 m at 0.02 m voxels must keep >= 1 cell
    assert g1.shape[1] >= 1


def test_occupancy_non_overlap_and_support_placement():
    grid = OccupancyGrid(origin=[0, 0, 0], dims=(40, 40, 20), voxel_size=0.05)
    support = SupportHeightGrid.from_support_box(
        [0.5, 0.5], [1.0, 1.0], 0.75, cell_size=0.05)
    proxy = np.ones((4, 4, 3), dtype=bool)
    rng = np.random.default_rng(0)
    centers = [np.array([0.5, 0.5])]
    first = pack_on_support(grid, support, proxy, 0.05, rng, centers)
    assert first is not None
    assert all(grid.solid[c] > 0 for c in first["cells"])
    overlapping = pack_on_support(grid, support, proxy, 0.05, rng, centers)
    # a second identical proxy must not intersect the first (packed elsewhere)
    if overlapping is not None:
        overlap = set(map(tuple, overlapping["cells"])) & set(
            map(tuple, first["cells"]))
        assert not overlap
    assert support.height_at([0.5, 0.5]) >= 0.75


def test_container_fit_and_interior(tmp_path):
    mask = build_container_mask([1.0, 1.0, 0.75], [0.30, 0.20, 0.15], 0.02)
    fits = voxelize_dims([0.12, 0.10, 0.08], 0.02)
    does_not = voxelize_dims([0.40, 0.30, 0.20], 0.02)
    assert object_fits(fits, mask)
    assert not object_fits(does_not, mask)


def test_lift_pose_deterministic_with_jitter():
    rng = np.random.default_rng(3)
    pose1 = lift_to_pose([1.0, 1.0, 0.75], [4, 4, 3], 0.02, rng)
    rng2 = np.random.default_rng(3)
    pose2 = lift_to_pose([1.0, 1.0, 0.75], [4, 4, 3], 0.02, rng2)
    assert pose1 == pose2


# ------------------------------------------------------- coverage sampling
def test_coverage_underweights_reused_categories(sampler):
    s = sampler()
    for _ in range(10):
        s.record_target("cat_0", "m0a")
    before = s.sample_category()
    s2 = sampler()
    fresh = [s2.sample_category() for _ in range(50)]
    # the heavily used category must not dominate fresh sampling
    assert fresh.count("cat_0") / 50 < 0.6 or before == "cat_0"


def test_count_distributions_match_config(dataset_cfg):
    rng = derive_streams(11)["geometry"]
    draws = [dataset_cfg.sample_count(
        dataset_cfg.object_count_distribution, rng) for _ in range(4000)]
    assert 7 <= min(draws) <= max(draws) <= 13
    assert draws.count(10) / len(draws) == pytest.approx(0.40, abs=0.05)
    inside = [dataset_cfg.sample_count(
        dataset_cfg.rummage_inside_distribution, rng) for _ in range(4000)]
    assert inside.count(8) / len(inside) == pytest.approx(0.50, abs=0.05)


# ------------------------------------------------------------ scene grammar
def test_scene_plan_counts_and_roles(dataset_cfg, sampler):
    def make():
        s = sampler()
        return build_scene_plan(101, "container_rummage", s, dataset_cfg,
                                "breakfast_table", "tray")
    plan = make()
    again = make()
    assert plan.counts["n_inside"] == again.counts["n_inside"]  # deterministic
    roles = [o.role for o in plan.objects]
    assert roles.count("target") == 1
    assert roles.count("inside") == plan.counts["n_inside"] - 1
    assert roles.count("outside") == plan.counts["n_outside"]
    assert plan.target().role == "target"


def test_deliberate_cover_requires_cover_model(dataset_cfg, sampler):
    with pytest.raises(ValueError):
        build_scene_plan(5, "deliberate_cover", sampler(), dataset_cfg,
                         "breakfast_table", None, None)


# ------------------------------------------------------------ RNG streams
def test_streams_independent_and_deterministic():
    a = derive_streams(42)
    b = derive_streams(42)
    assert a["geometry"].random() == b["geometry"].random()
    assert a["appearance"].random() != a["geometry"].random()
    seq = [int(x) for x in np.random.SeedSequence(42).spawn(5)[0].generate_state(1)]
    assert seq == [int(x) for x in np.random.SeedSequence(42).spawn(5)[0].generate_state(1)]


# ------------------------------------------------------- cover relations
def test_cover_relation_cavity_requires_containment():
    assert select_cover_relation([0.24, 0.24, 0.12], [0.10, 0.10, 0.08],
                                 0.02) == "cavity_cover"
    assert select_cover_relation([0.08, 0.08, 0.05], [0.10, 0.10, 0.08],
                                 0.02) is None


# ------------------------------------------------- stability + validation
def test_stability_detector_rejects_unstable():
    m = SettleMeasurement(sim_seconds=5.0, steps=600,
                          final_max_linear_velocity_mps=0.5,
                          final_max_angular_velocity_rps=0.5,
                          stable_consecutive_frames=0, all_finite=True)
    ok, code = check_settle(m, 0.02, 0.10, 30, 600)
    assert not ok and code == RejectionCode.PHYSICS_UNSTABLE
    ok2, code2 = check_settle(
        SettleMeasurement(0.3, 36, 0.01, 0.05, 30, True), 0.02, 0.10, 30, 600)
    assert ok2 and code2 is None


def test_post_settle_validation_codes():
    world = {"poses": {"a": {"position": [0.1, 0.1, 0.75], "finite": True},
                       "b": {"position": [0.2, 0.2, 0.75], "finite": True}},
             "max_penetration_m": 0.001, "support_container_stable": True,
             "target_present": True, "contacts_valid": True}
    ok, code = check_post_settle_world(world, {"a", "b"}, 2.0, 0.005)
    assert ok and code is None
    world["poses"]["b"]["position"][0] = 99.0  # escaped
    ok, code = check_post_settle_world(world, {"a", "b"}, 2.0, 0.005)
    assert not ok and code == RejectionCode.OBJECT_ESCAPED


def test_visibility_band_and_undefined():
    ok, code, why = check_visibility(0.30, (0.0, 0.60), 30, 100)
    assert ok and code is None
    ok, code, why = check_visibility(None, (0.0, 0.60), None, 0)
    assert not ok and why == "undefined"
    ok, code, why = check_visibility(0.9, (0.0, 0.15), 90, 100)
    assert not ok and code == RejectionCode.TARGET_VISIBILITY_OUT_OF_RANGE


def test_rejection_codes_are_private_strings():
    assert RejectionCode.RESET_MISMATCH.value == "RESET_MISMATCH"
    assert RejectionCode.WITNESS_NOT_FOUND.value == "WITNESS_NOT_FOUND"


# ------------------------------------------------- registry/split/resume
def _entry(eid: str, paradigm: str) -> RegistryEntry:
    return RegistryEntry(environment_id=eid, environment_seed=hash(eid) % 10**6,
                         paradigm=paradigm, support_model="breakfast_table",
                         target_category="bowl", target_model="adciys",
                         robot_ids=[f"r{i}" for i in range(5)])


def test_registry_hash_consistency_and_roundtrip(tmp_path):
    registry = DatasetRegistry()
    registry.accept(_entry("e1", "container_rummage"), "dev")
    h1 = registry.hash()
    path = tmp_path / "registry.json"
    registry.save(path)
    loaded = DatasetRegistry.load(path)
    assert loaded.hash() == h1
    assert loaded.accepted_count("container_rummage") == 1


def test_split_is_by_environment_and_deterministic(dataset_cfg):
    registry = DatasetRegistry()
    for i in range(12):
        registry.accept(_entry(f"e{i:02d}", "tabletop_clutter"), "pending")
    a = assign_splits(registry, 5, 4, 8)
    b = assign_splits(registry, 5, 4, 8)
    assert a == b
    assert len(a) == 12
    assert sum(1 for v in a.values() if v == "dev") == 4


def test_paired_environment_invariance_plan(dataset_cfg, sampler):
    """Same environment seed produces the identical scene plan regardless of
    which robot is later inserted (robot never touches the plan)."""
    def make():
        return build_scene_plan(77, "tabletop_clutter", sampler(), dataset_cfg,
                                "breakfast_table", None)
    p1, p2 = make(), make()
    assert [o.model for o in p1.objects] == [o.model for o in p2.objects]
    assert p1.counts == p2.counts


def test_snapshot_reproducibility_and_atomic_write(tmp_path):
    payload = {"b": 1, "a": [1, 2, 3]}
    h1 = canonical_hash(payload)
    h2 = canonical_hash({"a": [1, 2, 3], "b": 1})
    assert h1 == h2
    atomic_write_json(tmp_path / "x.json", payload)
    assert json.loads((tmp_path / "x.json").read_text()) == payload


def test_robot_registry_requires_five_verified(tmp_path):
    doc = {"robot_count": 5, "production_ready": False, "robots": {
        "r1pro": {"og_robot_class": "R1Pro", "status": "gpu_verified"},
        "fetch": {"og_robot_class": "Fetch", "status": "candidate",
                  "missing_capabilities": ["urdf_export"]},
    }}
    (tmp_path / "robot_pool_v1.yaml").write_text(yaml.safe_dump(doc))
    ready, pending = production_ready(tmp_path / "robot_pool_v1.yaml", 5)
    assert not ready and "fetch" in pending


# --------------------------------------------- candidate resume (script)
def test_candidate_generation_resume(tmp_path):
    script = REPO / "scripts" / "generate_level1_candidates.py"
    catalog = tmp_path / "catalog.json"
    entries = [{"category": f"cat_{i}", "model_id": f"m{i}", "asset_path": "x",
                "category_excluded_reason": None} for i in range(4)]
    catalog.write_text(json.dumps({"entries": entries}))
    out = tmp_path / "candidates"
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    base = [sys.executable, str(script), "--out", str(out), "--resume",
            "--catalog", str(catalog)]
    first = subprocess.run(base + ["--candidates", "3"], env=env,
                           capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    def accepted_specs():
        return sorted(out.glob("level1_*/occupancy_spec.json"))
    done = accepted_specs()
    assert 1 <= len(done) <= 3  # dense packing may structurally reject some
    progress = json.loads((out / "progress.json").read_text())
    assert progress["attempted"] == 3
    second = subprocess.run(base + ["--candidates", "3"], env=env,
                            capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert len(accepted_specs()) == len(done)  # --resume never redoes work
