"""§39 layout generator: reproducibility, rejection, validation rules."""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rummagebench.authoring.layout.asset_catalog import AssetCatalog  # noqa: E402
from rummagebench.authoring.layout.generator import LayoutGenerator, _rects_overlap, footprint  # noqa: E402
from rummagebench.authoring.layout.manifest import build_manifest  # noqa: E402
from rummagebench.authoring.layout.spec import LayoutConfig  # noqa: E402
from rummagebench.authoring.layout.validator import LayoutValidator  # noqa: E402

CONFIG = Path(__file__).parent / "fixtures" / "layout_test.yaml"
CATALOG = REPO / "configs" / "layouts" / "asset_pool_v1.yaml"


def _config(**overrides) -> LayoutConfig:
    raw = yaml.safe_load(CONFIG.read_text())["layout"]
    raw.update(overrides)
    return LayoutConfig.model_validate(raw)


def _catalog():
    return AssetCatalog.from_yaml(CATALOG)


def test_same_layout_seed_identical_manifest():
    cfg = _config()
    gen = LayoutGenerator(_catalog(), cfg)
    l1 = gen.generate(7)
    l2 = gen.generate(7)
    m1 = build_manifest(l1, {"valid": True})
    m2 = build_manifest(l2, {"valid": True})
    assert m1["manifest_hash"] == m2["manifest_hash"]
    assert [p.model_dump() for p in l1.placements] == [p.model_dump() for p in l2.placements]


def test_different_layout_seed_different_layout():
    cfg = _config()
    gen = LayoutGenerator(_catalog(), cfg)
    a = [(p.name, p.model_id) for p in gen.generate(1).placements]
    b = [(p.name, p.model_id) for p in gen.generate(2).placements]
    assert a != b


def test_no_furniture_overlap_in_generated_layout():
    cfg = _config()
    layout = LayoutGenerator(_catalog(), cfg).generate(0)
    furniture = [p for p in layout.placements if p.role != "clutter"]
    rects = [footprint(p) for p in furniture]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not _rects_overlap(a, b, cfg.min_furniture_clearance_m)


def test_impossible_region_rejects_instead_of_degrading():
    cfg = _config(storage_units=50, placement_regions=[[-1.0, -1.0, 1.0, 1.0]])
    layout = LayoutGenerator(_catalog(), cfg).generate(0)
    assert layout.rejected, "expected NO_VALID_PLACEMENT rejections"
    assert all(r["reason"] == "NO_VALID_PLACEMENT_AFTER_MAX_ATTEMPTS"
               for r in layout.rejected)


def test_validation_flags_missing_storage():
    cfg = _config(storage_units=99)
    layout = LayoutGenerator(_catalog(), cfg).generate(0)
    results = LayoutValidator(cfg).validate(layout, _catalog())
    assert not results["required_storage_units"]
    assert not results["valid"]


def test_model_ids_must_come_from_catalog():
    catalog = _catalog()
    cfg = _config()
    layout = LayoutGenerator(catalog, cfg).generate(0)
    results = LayoutValidator(cfg).validate(layout, catalog)
    assert results["model_ids_in_catalog"]


def test_catalog_missing_model_fails_loudly():
    catalog = _catalog()
    import pytest

    with pytest.raises(ValueError):
        catalog.get("storage", "totally_unknown_model")
    with pytest.raises(ValueError):
        catalog.models_for("nonexistent_role")



# Regression tests for the 88f099e acceptance gaps.
import pytest
from collections import Counter
from rummagebench.authoring.layout.generator import GeneratedLayout, placements_overlap
from rummagebench.authoring.layout.spec import Placement


def _generated(seed=0, **overrides):
    cfg = _config(**overrides)
    catalog = _catalog()
    return cfg, catalog, LayoutGenerator(catalog, cfg).generate(seed)


@pytest.mark.parametrize('seed', range(10))
def test_generated_clutter_has_support_and_requested_counts(seed):
    cfg, catalog, layout = _generated(seed)
    result = LayoutValidator(cfg).validate(layout, catalog)
    assert result['structural_valid'], result
    assert Counter(p.role for p in layout.placements) == {
        'storage': cfg.storage_units, 'surface': cfg.work_surfaces,
        'clutter': cfg.work_surfaces * cfg.clutter_per_surface,
    }
    assert not layout.rejected
    for clutter in [p for p in layout.placements if p.role == 'clutter']:
        support = next(p for p in layout.placements if p.name == clutter.support_name)
        assert not placements_overlap(clutter, support, cfg.min_furniture_clearance_m)
        assert clutter.category == catalog.get('clutter', clutter.model_id).category


def test_missing_sim_checks_are_not_passes():
    cfg, catalog, layout = _generated()
    result = LayoutValidator(cfg).validate(layout, catalog)
    assert result['structural_valid']
    assert not result['valid'] and not result['sim_accepted']
    for key in ('support', 'asset_available', 'scene_collision', 'search_connectivity',
                'observe_viewpoint', 'stage_a_interaction'):
        assert result['checks'][key]['status'] == 'not_run'


def test_support_callback_is_executed_and_failure_blocks_acceptance():
    cfg, catalog, layout = _generated()
    called = []
    def support(p):
        called.append(p.name)
        return False
    result = LayoutValidator(cfg, {'support': support}).validate(layout, catalog)
    assert len(called) == len(layout.placements)
    assert result['checks']['support']['status'] == 'fail'
    assert not result['valid']


def test_callback_exception_fails_closed():
    cfg, catalog, layout = _generated()
    def broken(p):
        raise RuntimeError('asset unresolved')
    result = LayoutValidator(cfg, {'asset_available': broken}).validate(layout, catalog)
    assert result['checks']['asset_available'] == {'status': 'fail', 'reason': 'asset unresolved'}
    assert not result['sim_accepted']


def test_missing_surface_or_clutter_is_invalid():
    for role in ('surface', 'clutter'):
        cfg, catalog, layout = _generated()
        layout.placements.remove(next(p for p in layout.placements if p.role == role))
        assert not LayoutValidator(cfg).validate(layout, catalog)['structural_valid']


def test_clutter_overlap_or_missing_support_is_invalid():
    cfg, catalog, layout = _generated()
    clutter = [p for p in layout.placements if p.role == 'clutter']
    clutter[1].position = list(clutter[0].position)
    assert not LayoutValidator(cfg).validate(layout, catalog)['no_object_overlap']
    clutter[0].support_name = 'missing'
    assert not LayoutValidator(cfg).validate(layout, catalog)['clutter_supported']


def test_undersized_region_rejects_without_numpy_exception():
    cfg, catalog, layout = _generated(placement_regions=[[0, 0, .1, .1]])
    assert layout.rejected and not layout.placements
    assert not LayoutValidator(cfg).validate(layout, catalog)['structural_valid']


def test_small_region_is_skipped_if_another_fits():
    cfg, catalog, layout = _generated(placement_regions=[[0, 0, .1, .1], [-10, -10, 10, 10]])
    assert LayoutValidator(cfg).validate(layout, catalog)['structural_valid']


def test_clutter_exhaustion_is_recorded():
    cfg, catalog, layout = _generated(clutter_per_surface=100, max_sampling_attempts=3)
    assert any(r['role'] == 'clutter' for r in layout.rejected)
    assert not LayoutValidator(cfg).validate(layout, catalog)['structural_valid']


def test_later_surface_cannot_block_previous_cabinet_front():
    cfg = _config(storage_units=1, work_surfaces=1, clutter_per_surface=0, robot_spawn=[10,10])
    storage = Placement(name='s', category='bottom_cabinet', model_id='bottom_cabinet_no_top_qudfwe_0',
                        position=[0,0,.45], aabb_size=[1,.65,.9], orientation_deg=0, role='storage')
    blocker = Placement(name='b', category='countertop', model_id='countertop_tpuwys_0',
                        position=[0,-.65,.45], aabb_size=[1,.35,.9], orientation_deg=0, role='surface')
    layout = GeneratedLayout(config=cfg, layout_seed=0, placements=[storage])
    assert not LayoutGenerator(_catalog(),cfg)._admissible(layout, blocker)
    layout.placements.append(blocker)
    assert not LayoutValidator(cfg).validate(layout,_catalog())['storage_access_strips_free']
    layout.placements.reverse()
    assert not LayoutValidator(cfg).validate(layout,_catalog())['storage_access_strips_free']


@pytest.mark.parametrize('kwargs', [
    {'storage_units': -1}, {'max_sampling_attempts': 0}, {'robot_spawn': [0]},
    {'placement_regions': [[0,0,0,1]]}, {'placement_regions': [[0,0,1]]},
    {'placement_regions': [[0,0,float('inf'),1]]},
])
def test_invalid_config_fails_at_schema(kwargs):
    with pytest.raises(ValueError):
        _config(**kwargs)


def test_catalog_preserves_category_and_rejects_nonpositive_geometry():
    from rummagebench.authoring.layout.asset_catalog import AssetEntry
    catalog = _catalog()
    assert catalog.get('surface','countertop_tpuwys_0').category == 'countertop'
    with pytest.raises(ValueError):
        AssetEntry(model_id='bad',aabb_size=[1,0,1])


def test_manifest_labels_unapplied_candidate():
    cfg,catalog,layout = _generated()
    manifest = build_manifest(layout, LayoutValidator(cfg).validate(layout,catalog))
    assert manifest['artifact_kind'] == 'candidate_layout'
    assert manifest['simulator_application'] == 'not_implemented'
    assert manifest['transform_convention'] == 'aabb_center_world'
    assert 'storage' not in manifest['asset_categories']


def test_acceptance_requires_loaded_layout_and_every_sim_check():
    cfg, catalog, layout = _generated()
    names = ('support', 'asset_available', 'scene_collision', 'search_connectivity',
             'observe_viewpoint', 'stage_a_interaction')
    checks = {name: lambda p: True for name in names}
    assert not LayoutValidator(cfg, checks).validate(layout, catalog)['sim_accepted']
    checks['layout_applied'] = lambda loaded: loaded is layout
    assert LayoutValidator(cfg, checks).validate(layout, catalog)['sim_accepted']
    checks['asset_available'] = lambda p: 'unresolved'
    result = LayoutValidator(cfg, checks).validate(layout, catalog)
    assert not result['sim_accepted']
    assert result['checks']['asset_available']['status'] == 'fail'


def test_cli_marks_candidate_and_can_require_real_acceptance(tmp_path):
    import subprocess
    args = [sys.executable, str(REPO/'scripts/generate_layout.py'),
            '--config', str(CONFIG), '--catalog', str(CATALOG), '--out', str(tmp_path)]
    candidate = subprocess.run(args, capture_output=True, text=True)
    assert candidate.returncode == 0, candidate.stderr
    import json
    data = json.loads(candidate.stdout)
    assert data['structural_valid'] is True
    assert data['valid'] is False and data['sim_accepted'] is False
    accepted = subprocess.run(args + ['--require-sim-accepted'], capture_output=True, text=True)
    assert accepted.returncode == 1


def test_zero_clearance_still_rejects_spawn_inside_furniture():
    from rummagebench.authoring.layout.generator import _point_near_rect
    assert _point_near_rect(0, 0, ((-1, -1), (1, 1)), 0)
