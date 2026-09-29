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
