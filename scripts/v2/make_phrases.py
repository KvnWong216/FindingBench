"""Draft assets/tasks/vla_difficulty_v2/phrases.yaml from the scene surveys (CPU).

Instructions must name furniture unambiguously:
  * room nouns from the room type (bedrooms numbered by the survey are
    disambiguated by size: "large / small bedroom", else excluded)
  * a furniture noun from its category; if the noun is not unique in its
    room, "the <noun> next to the <landmark>" with the nearest fixed object
    whose category IS unique in that room (<= 1.5 m, from LANDMARKS);
    otherwise the furniture is marked unusable (left out of the file)
  * drawers named by height rank among the cabinet's drawer interiors;
    drawers sharing a height (side-by-side columns) are left out

The draft is reviewed by hand before generate_tasks.py uses it.

    python scripts/v2/make_phrases.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SURVEY = ROOT / "runs/vla_difficulty_v2/_survey"
OUT = ROOT / "assets/tasks/vla_difficulty_v2/phrases.yaml"
SCENES = sorted(p.name.split(".")[0] for p in SURVEY.glob("*.candidates.json"))
ROOMS = {"kitchen": "kitchen", "living_room": "living room", "bathroom": "bathroom", "corridor": "hallway",
         "entryway": "entryway", "garage": "garage", "private_office": "office", "dining_room": "dining room",
         "bedroom": "bedroom", "utility_room": "utility room", "storage_room": "storage room"}
NOUNS = {"countertop": "counter", "bottom_cabinet_no_top": "cabinet", "bottom_cabinet": "cabinet",
         "bookcase": "bookcase", "desk": "desk", "nightstand": "nightstand", "shelf": "shelf",
         "coffee_table": "coffee table", "breakfast_table": "table", "bar": "bar counter",
         "dresser": "dresser", "cedar_chest": "chest", "console_table": "console table", "dining_table": "dining table"}
LANDMARKS = {"sofa": "sofa", "standing_tv": "TV", "television_receiver": "TV", "piano": "piano",
             "fridge": "fridge", "refrigerator": "fridge", "oven": "oven", "bed": "bed", "desk": "desk",
             "armchair": "armchair", "fireplace": "fireplace", "wood_fireplace": "fireplace",
             "drop_in_sink": "sink", "furniture_sink": "sink", "toilet": "toilet", "bathtub": "bathtub",
             "washer": "washing machine", "dishwasher": "dishwasher", "microwave": "microwave",
             "coffee_table": "coffee table", "floor_lamp": "floor lamp", "mirror": "mirror",
             "standing_mirror": "mirror", "loudspeaker": "speaker", "pot_plant": "plant"}
# hand-checked names for rooms the room type alone cannot tell apart (2026-10-25,
# from the survey's furniture lists: each feature occurs in that room only)
ROOM_OVERRIDES = {"house_double_floor_upper": {
    "bedroom_0": "bedroom with the maps on the wall",
    "bedroom_1": "bedroom with the wall shelves",
    "bedroom_2": "bedroom with the room divider"}}
# hand-checked furniture names where the automatic landmark rule fails (survey AABBs,
# 2026-10-26): the clearance patch made the Beechwood worktop + two cabinets a
# freestanding island; the cabinets are 1.0 m and 0.5 m wide
FURNITURE_OVERRIDES = {"Beechwood_0_int": {
    "countertop_tpuwys_0": "island counter",
    "bottom_cabinet_no_top_pluwfl_0": "wide cabinet under the island counter",
    "bottom_cabinet_no_top_qohxjq_0": "narrow cabinet under the island counter"},
    # two wall shelves one above the other (tops at 1.47 m and 0.97 m)
    "house_double_floor_upper": {"shelf_sapyzl_0": "upper wall shelf", "shelf_sapyzl_1": "lower wall shelf"},
    # hotel: a two-level wall unit beside the TV, three nightstands along the beds,
    # a long low shelf beside the sink and a four-level shelf rack
    "hotel_suite_large": {"shelf_aqtmhm_0": "lower shelf next to the TV",
                          "shelf_nbmidu_0": "upper shelf next to the TV",
                          "nightstand_maeowc_2": "nightstand farthest from the desk",
                          "shelf_rglkkf_0": "long low shelf next to the sink",
                          "shelf_xogqqm_0": "top shelf of the four-level shelf rack",
                          "shelf_xogqqm_2": "third shelf from the top of the four-level shelf rack"},
    # two pedestal sinks: one cabinet stands between them, the other in the corner
    "Pomaria_2_int": {"bottom_cabinet_no_top_qohxjq_1": "cabinet between the two sinks",
                      "bottom_cabinet_no_top_qohxjq_0": "corner cabinet next to the sinks"}}
SAME_LEVEL_M = 0.05   # drawers whose rims differ by less than this sit side by side
ORDINAL = ["top drawer", "second drawer from the top", "third drawer from the top", "fourth drawer from the top"]


def noun_of(category: str) -> str | None:
    for k in sorted(NOUNS, key=len, reverse=True):
        if category == k or category.startswith(k):
            return NOUNS[k]
    return None


def landmark_of(category: str) -> str | None:
    for k in sorted(LANDMARKS, key=len, reverse=True):
        if category == k or category.startswith(k):
            return LANDMARKS[k]
    return None


def centre(b):
    return ((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2)


def gap(a, b) -> float:
    dx = max(a[0][0] - b[1][0], b[0][0] - a[1][0], 0)
    dy = max(a[0][1] - b[1][1], b[0][1] - a[1][1], 0)
    return (dx * dx + dy * dy) ** 0.5


def distinguish(name, peers, entities, room_of):
    """Evidence-backed, viewpoint-independent comparative descriptions.

    Distances compare every same-noun peer; dimensions require a conspicuous
    unique extreme. No texture colors are guessed from category/model names.
    """
    obj=entities[name];noun=noun_of(obj['category']);room=room_of[name]
    for landmark, other in sorted(entities.items()):
        lm=landmark_of(other['category'])
        if landmark==name or room_of[landmark]!=room or lm is None or lm==noun:
            continue
        if sum(room_of[k]==room and landmark_of(v['category'])==lm for k,v in entities.items())!=1:
            continue
        distances={k:gap(entities[k]['aabb'],other['aabb']) for k in peers}
        own=distances[name];rest=[d for k,d in distances.items() if k!=name]
        if rest and own+.3 < min(rest):
            return {'noun':f'{noun} closest to the {lm}', 'description_evidence':
                    {'kind':'nearest_landmark','landmark':landmark,'distances_m':distances}}
        if rest and own > max(rest)+.3:
            return {'noun':f'{noun} farthest from the {lm}', 'description_evidence':
                    {'kind':'farthest_landmark','landmark':landmark,'distances_m':distances}}
    for adjective,axis in [('tallest',2),('widest',None)]:
        sizes={k: (entities[k]['aabb'][1][axis]-entities[k]['aabb'][0][axis] if axis is not None else
                    max(entities[k]['aabb'][1][i]-entities[k]['aabb'][0][i] for i in (0,1))) for k in peers}
        other=max(v for k,v in sizes.items() if k!=name)
        if sizes[name] >= other*1.3 and sizes[name]-other>=.2:
            return {'noun':f'{adjective} {noun}', 'description_evidence':{'kind':adjective,'dimensions_m':sizes}}
    return None


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--scenes', nargs='+', help='Merge only selected scenes; preserve other reviewed descriptions')
    args=parser.parse_args()
    doc=yaml.safe_load(OUT.read_text()) if args.scenes and OUT.exists() else {}
    for scene in args.scenes or SCENES:
        survey = json.loads((SURVEY / f"{scene}.json").read_text())
        ents = {n: e for n, e in survey["entities"].items() if e.get("aabb")}
        room_of = {n: e.get("room_at_center") or (e.get("in_rooms") or [None])[0] for n, e in ents.items()}
        rooms = sorted({r for r in room_of.values() if r})
        cands = json.loads((SURVEY / f"{scene}.candidates.json").read_text())
        reachable = {s["room"] for s in cands["starts"]} | {c["room"] for c in cands["candidates"]}
        names = {}
        for r in rooms:
            kind = r.rsplit("_", 1)[0]
            if r in ROOM_OVERRIDES.get(scene, {}):
                names[r] = ROOM_OVERRIDES[scene][r]
            elif kind in ROOMS and r in reachable:
                names[r] = ROOMS[kind]
        # rooms of the same type: "<type> with the <landmark>", the landmark
        # category occurring in exactly one of those rooms; else excluded
        dup = {v for v in names.values() if list(names.values()).count(v) > 1}
        for v in dup:
            same = [r for r, n in names.items() if n == v]
            cats = {r: {landmark_of(e["category"]) for m, e in ents.items() if room_of[m] == r
                        and landmark_of(e["category"])} for r in same}
            for r in same:
                own = sorted(c for c in cats[r] if all(c not in cats[o] for o in same if o != r))
                names[r] = f"{v} with the {own[0]}" if own else None
        names = {r: v for r, v in names.items() if v}
        furniture = {}
        for n, e in ents.items():
            noun = noun_of(e["category"])
            r = room_of[n]
            if noun is None or r is None or r not in names:
                continue
            same = [m for m, f in ents.items() if room_of[m] == r and noun_of(f["category"]) == noun]
            row = {"noun": noun, "room": r}
            override = FURNITURE_OVERRIDES.get(scene, {}).get(n)
            if override:
                row["noun"] = override
            elif len(same) > 1:
                description = distinguish(n, same, ents, room_of)
                if description is None:
                    continue
                row.update(description)
            inter = sorted(e.get("interiors") or [], key=lambda it: -it["hi"][2])
            drawers = {}
            zs = [it["hi"][2] for it in inter]
            for k, it in enumerate(inter):
                if sum(abs(z - zs[k]) < SAME_LEVEL_M for z in zs) > 1 or k >= len(ORDINAL):
                    continue
                drawers[it["link"]] = ORDINAL[k]
            if drawers:
                row["drawers"] = drawers
            furniture[n] = row
        # Preserve previously reviewed descriptions; add only new coverage.
        previous=doc.get(scene,{})
        names.update(previous.get('rooms',{}))
        furniture.update(previous.get('furniture',{}))
        doc[scene] = {"rooms": names, "furniture": dict(sorted(furniture.items()))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# DRAFT from scripts/v2/make_phrases.py -- review by hand (room names, landmarks).\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    for s, d in doc.items():
        print(s, d["rooms"])
        for n, r in d["furniture"].items():
            print("   ", n, "->", r["noun"], "|", r["room"], "|", r.get("drawers", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
