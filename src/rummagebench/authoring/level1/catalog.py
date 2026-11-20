"""§7/§8: BEHAVIOR ordinary-object catalog — inventory stage (CPU only).

Scans the installed BEHAVIOR object dataset into an inventory of exact
(category, model_id, asset_path) records. Geometry/physics metadata
(AABB, volume, mass, simulator verification) is intentionally None here:
this host has no CPU USD reader, so per-model geometry is filled lazily by
the GPU verification stage (revision §15: staged preprocessing). Categories
are excluded only through the documented denylist below — never by hand.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# §7: documented simulator reasons for category exclusion. Matched against
# the category name; anything not excluded is inventoried as an ordinary
# candidate and the GPU verification stage decides physical eligibility.
CATEGORY_DENYLIST = [
    # architecture / structural
    r"^wall", r"^floor", r"^ceiling", r"^door", r"^window", r"^stairs",
    r"^roof", r"^ceiling_", r"_door$", r"^doorframe", r"^threshold",
    # furniture / fixtures (not ordinary movable objects)
    r"^bottom_cabinet", r"^metal_bottom_cabinet", r"^cabinet", r"^countertop",
    r"^shelf", r"^shelving", r"table$", r"^desk$", r"^chair",
    r"^sofa", r"^couch", r"^bed", r"^bench", r"^stool", r"^stand",
    r"^fridge", r"^refrigerator", r"^oven", r"^stove", r"^range_",
    r"^dishwasher", r"^washing_machine", r"^dryer", r"^microwave",
    r"^toaster_oven", r"^sink", r"^bathtub", r"^toilet", r"^fireplace",
    r"^heater", r"^radiator", r"^furniture", r"^wardrobe", r"^dresser",
    r"^drawers$", r"^bunk", r"^crib", r"^swivel", r"^rocking",
    # robots
    r"^robot",
    # fluids / particles / systems (not rigid objects)
    r"water$", r"^water", r"juice$", r"milk$", r"oil$", r"sauce$",
    r"^dust", r"particle", r"^smoke", r"^steam", r"^dirt", r"^sand",
    r"^gravel", r"^salt$", r"^sugar$", r"^flour$", r"^pepper$", r"spice$",
    r"powder$", r"chemicals?", r"^detergent_$", r"shampoo$",
    # cloth / deformables
    r"cloth", r"towel", r"curtain", r"carpet", r"rug$", r"blanket",
    r"sheet$", r"pillow", r"cushion", r"bag$", r"backpack", r"plastic_bag",
    r"string$", r"rope$", r"wire$", r"cable$", r"chain$", r"duct_tape",
    r"tape$", r"bandage", r"scarf", r"necklace", r"bracelet", r"strap",
    # foodstuffs that are deformable/particle-like or clearly non-rigid props
    r"^meat", r"^bacon", r"^chicken", r"^beef", r"^fish$", r"bread$",
    r"lettuce$", r"spinach$", r"cabbage$", r"broccoli$", r"cauliflower$",
    r"cheese$", r"butter$", r"dough", r"cake$", r"pie$", r"pizza$",
    r"sandwich$", r"wrap$", r"salad$", r"fruit$", r"vegetable$",
]

_DENY_RE = [re.compile(p) for p in CATEGORY_DENYLIST]

# categories that are known-small rigid household objects are provisionally
# target/distractor eligible at inventory stage; cover eligibility additionally
# requires cavity/footprint shape checks at GPU stage.
_COVER_HINT = re.compile(r"bowl|tray|plate|bin|basket|box$|pot$|pan$|lid|dish")


def category_excluded(category: str) -> Optional[str]:
    """Return the denylist reason for an excluded category, else None."""
    for pattern in _DENY_RE:
        if pattern.search(category):
            return pattern.pattern
    return None


@dataclass
class CatalogEntry:
    category: str
    model_id: str
    asset_path: str
    # --- GPU verification stage fills these (None = pending) ---
    native_aabb_xyz: Optional[list[float]] = None
    collision_aabb_xyz: Optional[list[float]] = None
    approximate_volume_m3: Optional[float] = None
    mass_kg: Optional[float] = None
    size_class: Optional[str] = None
    flatness: Optional[float] = None
    elongation: Optional[float] = None
    compactness: Optional[float] = None
    simulator_load_pass: Optional[bool] = None
    collision_valid: Optional[bool] = None
    settle_pass: Optional[bool] = None
    target_eligible: Optional[bool] = None
    distractor_eligible: Optional[bool] = None
    cover_eligible: Optional[bool] = None
    voxel_proxy_path: Optional[str] = None
    # inventory-stage provisional flags (geometry-independent)
    category_excluded_reason: Optional[str] = None
    verified: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_category_stats(objects_root: Path) -> dict[str, dict]:
    """Dataset-provided per-category averages (mass kg, volume m3, density).
    Located at <objects_root>/../metadata/avg_category_specs.json."""
    path = Path(objects_root).parent / "metadata" / "avg_category_specs.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def size_gate(category: str, stats: dict, max_mass_kg: float,
              max_volume_m3: float) -> Optional[str]:
    """Data-driven size gating (§7): a category is an ordinary Level-1 object
    only when the DATASET's own averages fit a tabletop workspace. Documented
    thresholds — never per-category hand curation."""
    stats_entry = stats.get(category)
    if stats_entry is None:
        return None
    mass = stats_entry.get("mass")
    volume = stats_entry.get("volume")
    if mass is not None and float(mass) > max_mass_kg:
        return "category_mass_exceeds_limit"
    if volume is not None and float(volume) > max_volume_m3:
        return "category_volume_exceeds_limit"
    return None


def scan_inventory(objects_root: Path, max_mass_kg: float = 3.0,
                   max_volume_m3: float = 0.01) -> list[CatalogEntry]:
    """Exact dataset scan: every (category, model) directory pair, sorted.
    Model IDs come from the installed dataset directory names — never guessed
    from scene-instance names (spec §8)."""
    objects_root = Path(objects_root)
    if not objects_root.is_dir():
        raise FileNotFoundError(f"BEHAVIOR objects root missing: {objects_root}")
    stats = load_category_stats(objects_root)
    entries: list[CatalogEntry] = []
    for category_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        category = category_dir.name
        reason = category_excluded(category) or size_gate(
            category, stats, max_mass_kg, max_volume_m3)
        for model_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            entries.append(CatalogEntry(
                category=category,
                model_id=model_dir.name,
                asset_path=str(model_dir),
                category_excluded_reason=reason,
            ))
    return entries


def eligible_inventory(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    """Inventory-stage eligibility: not denylisted and within the ordinary
    size gate. Geometry verification remains pending (revision §15)."""
    return [e for e in entries if e.category_excluded_reason is None]


def save(entries: list[CatalogEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"entries": [e.to_dict() for e in entries]}
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load(path: Path) -> list[CatalogEntry]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CatalogEntry(**d) for d in doc["entries"]]
