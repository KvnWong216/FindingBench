"""vla_difficulty_v2 object-pool prefilter (CPU). Evaluator-only.

For candidate graspable categories, read every BEHAVIOR model's bbox size
(misc/metadata.json) and base-link diffuse texture, and keep models that are
plausibly graspable AND visually salient against beige / brown / grey
furniture. This is only a prefilter: grasp / place and the in-scene RGB
contrast gate are certified in simulation later.

Saliency = Hasler-Suesstrunk colourfulness of the opaque texels plus mean HSV
saturation; the dominant hue gives a colour word for instructions.

    python scripts/v2/score_object_pool.py
Writes build/vla_difficulty_v2/object_pool_candidates.json
"""
from __future__ import annotations

import colorsys
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = Path(os.environ["OMNIGIBSON_DATA_PATH"]) / "behavior-1k-assets/objects"
OUT = ROOT / "build/vla_difficulty_v2/object_pool_candidates.json"

# everyday tabletop items; long thin cutlery / tools excluded (v1: a silver knife
# on a beige counter was invisible in RGB, 2026-10-25)
CATEGORIES = [
    "apple", "orange", "lemon", "lime", "pear", "banana", "bell_pepper", "mango", "peach", "avocado",
    "mug", "bowl", "teacup", "coffee_cup", "can_of_soda", "water_bottle", "bottle_of_water",
    "bottle_of_shampoo", "spray_bottle", "box_of_cereal", "carton_of_milk", "box_of_crackers",
    "teddy_bear", "toy_car", "toy_figure", "tennis_ball", "baseball", "softball",
    "alarm_clock", "sponge", "pencil_case", "lunch_box",
]
# size window (m): smallest side >= 4 cm (not thin), largest side 6-26 cm, height <= 30 cm
MIN_SIDE, MAX_SIDE_LO, MAX_SIDE_HI, MAX_HEIGHT = 0.04, 0.06, 0.26, 0.30

HUES = [(15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "teal"),
        (260, "blue"), (300, "purple"), (345, "pink"), (360, "red")]


def colour_word(rgb: np.ndarray) -> str:
    h, s, v = colorsys.rgb_to_hsv(*(rgb / 255.0))
    if v < 0.18:
        return "black"
    if s < 0.18:
        return "white" if v > 0.75 else "grey"
    if 0.03 < h < 0.12 and v < 0.55:
        return "brown"
    deg = h * 360
    return next(name for limit, name in HUES if deg <= limit)


def texture_stats(path: Path) -> dict | None:
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:  # noqa: BLE001
        return None
    im.thumbnail((256, 256))
    a = np.asarray(im, dtype=float)
    px = a[a[..., 3] > 128][:, :3]
    if len(px) < 50:
        return None
    rg, yb = px[:, 0] - px[:, 1], 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    colourful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    mx, mn = px.max(1), px.min(1)
    sat = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0).mean())
    # dominant colour = median of the most saturated half (labels / bodies)
    order = np.argsort(-(mx - mn))
    dom = np.median(px[order[: max(1, len(px) // 2)]], axis=0)
    return {"colourfulness": round(colourful, 1), "saturation": round(sat, 3),
            "dominant_rgb": [int(v) for v in dom], "colour": colour_word(dom)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true",
                    help="score every object category of build/vla_difficulty_v2/official_pairs.json "
                         "(looser saliency; the in-scene S1 RGB gate decides)")
    args = ap.parse_args()
    global OUT
    categories = CATEGORIES
    min_colour, min_sat = 35, 0.30
    if args.official:
        pairs = json.loads((ROOT / "build/vla_difficulty_v2/official_pairs.json").read_text())["pairs"]
        categories = sorted({c for r in pairs for c in r["object_categories"]})
        OUT = ROOT / "build/vla_difficulty_v2/object_pool_official.json"
        min_colour, min_sat = 25, 0.22
    rows = []
    for cat in categories:
        cdir = OBJECTS / cat
        if not cdir.is_dir():
            continue
        for mdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            meta = mdir / "misc/metadata.json"
            if not meta.exists():
                continue
            size = json.loads(meta.read_text()).get("bbox_size")
            if not size:
                continue
            sx, sy, sz = map(float, size)
            row = {"category": cat, "model": mdir.name, "bbox_size": [round(sx, 3), round(sy, 3), round(sz, 3)]}
            reasons = []
            if min(sx, sy, sz) < MIN_SIDE:
                reasons.append("too thin")
            if not MAX_SIDE_LO <= max(sx, sy, sz) <= MAX_SIDE_HI:
                reasons.append("size out of range")
            if sz > MAX_HEIGHT:
                reasons.append("too tall")
            tex = sorted((mdir / "material").glob("*base_link*diffuse*.png")) or \
                sorted((mdir / "material").glob("*diffuse*.png"))
            stats = texture_stats(tex[0]) if tex else None
            if stats is None:
                reasons.append("no diffuse texture")
            else:
                row.update(stats)
                if stats["colourfulness"] < min_colour and stats["saturation"] < min_sat:
                    reasons.append("low colour saliency")
                if stats["colour"] in ("brown", "grey", "white") and stats["saturation"] < min_sat:
                    reasons.append("furniture-like colour")
            row["accepted"] = not reasons
            row["reasons"] = reasons
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    acc = [r for r in rows if r["accepted"]]
    OUT.write_text(json.dumps({"criteria": {"min_side": MIN_SIDE, "max_side": [MAX_SIDE_LO, MAX_SIDE_HI],
                                            "max_height": MAX_HEIGHT},
                               "models": rows}, indent=1) + "\n")
    by_cat: dict[str, list[str]] = {}
    for r in acc:
        by_cat.setdefault(r["category"], []).append(f"{r['model']}:{r['colour']}")
    print(f"{len(acc)}/{len(rows)} models accepted in {len(by_cat)} categories")
    for c, ms in by_cat.items():
        print(f"  {c:18s} {len(ms):2d}  {' '.join(ms[:8])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
