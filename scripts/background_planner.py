"""Conservative finite-space proposals for sparse, contextual native props.

Room semantics choose a curated asset family, never a goal. All task entities,
source/delivery furniture and declared articulated states are excluded. This
module proposes geometry; native contact and paired dynamics still decide.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'sparse_native_background_v3_light'
SUPPORTS = frozenset(('countertop', 'coffee_table', 'breakfast_table', 'desk',
    'console_table', 'dining_table', 'pedestal_table', 'reception_desk', 'checkout_counter'))
STRUCTURAL = frozenset(('walls', 'floors', 'ceilings'))
ROOM_CATEGORIES = {
    'kitchen': ('mug', 'water_glass', 'bowl', 'plate', 'apple', 'banana'),
    'dining_room': ('mug', 'water_glass', 'apple', 'banana', 'bowl', 'plate'),
    'living_room': ('hardback', 'box_of_tissues', 'mug', 'pot_plant'),
    'bedroom': ('hardback', 'box_of_tissues', 'notebook', 'alarm_clock'),
    'office': ('notebook', 'hardback', 'pencil_holder', 'box_of_tissues', 'mug'),
    'bathroom': ('box_of_tissues', 'soap_dispenser', 'pot_plant'),
    'corridor': ('pot_plant', 'vase', 'hardback', 'box_of_tissues'),
    'garage': ('notebook', 'box_of_tissues', 'mug'),
    'utility_room': ('box_of_tissues', 'notebook', 'mug'),
    'grocery_store': ('notebook', 'pencil_holder', 'box_of_tissues', 'mug'),
    'balcony': ('pot_plant', 'mug', 'water_glass'),
}
FAMILIES = {
    'fruit': ('apple', 'banana', 'orange', 'pear', 'fruit', 'peach', 'lemon', 'grape', 'produce'),
    'paper': ('hardback', 'paperback', 'book', 'notebook', 'notepad', 'paper', 'magazine', 'receipt', 'folder'),
    'tableware': ('mug', 'cup', 'glass', 'plate', 'bowl', 'tableware', 'utensil', 'spoon', 'fork', 'cutlery'),
    'toiletries': ('soap', 'dispenser', 'shampoo', 'conditioner', 'toiletr', 'lotion', 'perfume', 'cleanser'),
    'tissues': ('tissue', 'napkin', 'towel', 'toilet_paper'),
    'plants': ('plant', 'flower', 'vase', 'potting'),
    'stationery': ('pencil', 'pen', 'stationery', 'marker', 'crayon'),
    'clocks': ('clock', 'alarm', 'watch'),
}


def rank(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def room_kind(room):
    value = re.sub(r'_\d+$', '', room)
    if value in ROOM_CATEGORIES:
        return value
    return {'study': 'office', 'childs_room': 'office', 'childrens_room': 'office',
            'entryway': 'corridor', 'hallway': 'corridor', 'storage_room': 'utility_room',
            'pantry': 'kitchen', 'garden': 'balcony', 'terrace': 'balcony',
            'laundry_room': 'utility_room'}.get(value)


def family_hits(text):
    text = text.lower().replace(' ', '_')
    return {family for family, words in FAMILIES.items() if any(w in text for w in words)}


def excluded_categories(task_document):
    categories = {obj['category'] for obj in task_document['objects']}
    instruction = task_document['instruction'].lower()
    # Inventory includes distractors and comparison samples too. Avoid adding
    # a new member of any searched/compared semantic family, not just exact IDs.
    families = family_hits(instruction) | set().union(*(family_hits(c) for c in categories))
    excluded = set(categories)
    for choices in ROOM_CATEGORIES.values():
        for category in choices:
            if family_hits(category) & families or category.replace('_', ' ') in instruction:
                excluded.add(category)
    if re.search(r'\b(food|ingredient|grocer|snack|fruit|produce)\w*\b', instruction):
        excluded.update(('apple', 'banana'))
    return excluded, sorted(families)


def build_policy(assets_root):
    """Read real native metadata. No resizing, invented models or fluids/cloth."""
    assets_root = Path(assets_root)
    pool = {}
    for category in sorted({c for values in ROOM_CATEGORIES.values() for c in values}):
        rows = []
        for path in sorted((assets_root/'objects'/category).glob('*/misc/metadata.json')):
            content = path.read_bytes()
            size = json.loads(content).get('bbox_size', [])
            if (len(size) != 3 or any(not math.isfinite(v) or v <= 0 for v in size)
                    or max(size) > .30 or min(size[:2]) < .035 or size[2] < .006
                    or size[2]/max(size[:2]) > 2.5):
                continue
            rows.append(dict(category=category, model=path.parents[1].name, size=size,
                metadata_path=str(path.resolve()), metadata_sha256=hashlib.sha256(content).hexdigest()))
        # Broad model/color coverage, stable across machines and queue order.
        pool[category] = sorted(rows, key=lambda r: rank(category, r['model']))[:16]
    return dict(version=VERSION, room_categories=ROOM_CATEGORIES, supports=sorted(SUPPORTS),
        max_props=1, max_coverage_fraction=.08, minimum_task_gap_m=1., edge_clearance_m=.08,
        max_prop_height_m=.30, support_height_band_m=[.55, 1.30], assets=pool)


def valid_box(box):
    return (box is not None and len(box) == 2 and all(len(row) == 3 for row in box)
            and all(math.isfinite(v) for row in box for v in row)
            and all(box[1][i] > box[0][i] for i in range(3)))


def gap(a, b):
    return math.hypot(*(max(0., a[0][i]-b[1][i], b[0][i]-a[1][i]) for i in (0, 1)))


def finite_centres(support, size, obstacles, protected, seed):
    """Rectangle-arrangement vertices and cell centres, not blind sampling.

    Inflating obstacles by half the new object's size leaves finite free
    rectangles for its centre. Protected workspaces use a conservative square
    expansion of the 1-m gap. Native OnTop still tests real surface shape.
    """
    half = [v/2 for v in size]
    lo = [support[0][i]+.08+half[i] for i in (0, 1)]
    hi = [support[1][i]-.08-half[i] for i in (0, 1)]
    if any(b < a for a, b in zip(lo, hi)):
        return []
    rectangles = []
    z0, z1 = support[1][2]+.01, support[1][2]+.01+size[2]
    for box in obstacles:
        if box[1][2] <= support[1][2]+.001 or box[0][2] >= z1+.03:
            continue
        rectangles.append(([box[0][i]-half[i]-.03 for i in (0, 1)],
                           [box[1][i]+half[i]+.03 for i in (0, 1)]))
    rectangles.extend(([box[0][i]-half[i]-1. for i in (0, 1)],
                       [box[1][i]+half[i]+1. for i in (0, 1)]) for box in protected)
    rectangles = [(a, b) for a, b in rectangles if all(b[i] >= lo[i] and a[i] <= hi[i] for i in (0, 1))]
    axes = []
    for i in (0, 1):
        cuts = sorted({lo[i], hi[i], *[max(lo[i], min(hi[i], edge[i])) for r in rectangles for edge in r]})
        axes.append(sorted({*cuts, *[(a+b)/2 for a, b in zip(cuts, cuts[1:])]}))
    points = []
    for x in axes[0]:
        for y in axes[1]:
            point = [x, y]
            if any(all(a[i]-1e-8 <= point[i] <= b[i]+1e-8 for i in (0, 1)) for a, b in rectangles):
                continue
            points.append(point)
    centre = [(lo[i]+hi[i])/2 for i in (0, 1)]
    # Prefer usable interior positions, then seeded tie breaks. Keep a small,
    # finite fallback set for oval tabletops or native bbox/root offsets.
    points.sort(key=lambda p: (round(math.dist(p, centre), 2), rank(seed, p)))
    return points[:24]


def propose(task, records, protected_names, policy, variant=0, limit=24):
    desired_props = policy.get('max_props', 1)
    if desired_props not in (1, 2):
        raise ValueError('INVALID_BACKGROUND_OBJECT_BUDGET')
    excluded, families = excluded_categories(task)
    bad = [name for name, row in records.items() if not valid_box(row['aabb'])]
    if bad:
        raise ValueError('UNAVAILABLE_NATIVE_GEOMETRY:'+','.join(bad[:8]))
    protected = [records[name]['aabb'] for name in sorted(protected_names)]
    candidates = []
    for name, row in records.items():
        support = row['aabb']
        if (name in protected_names or row['category'] not in policy['supports'] or row['n_joints']
                or not .55 <= support[1][2] <= 1.30):
            continue
        neighbours = [r['aabb'] for n, r in records.items() if n != name and r['category'] not in STRUCTURAL]
        for room in row['rooms']:
            kind = room_kind(room)
            if kind is None:
                continue
            categories = [c for c in policy['room_categories'][kind] if c not in excluded]
            categories.sort(key=lambda c: rank(f'{task["id"]}:{variant}:{room}', c))
            assets = []
            for category in categories:
                choices = sorted(policy['assets'].get(category, []),
                                 key=lambda a: rank(f'{task["id"]}:{variant}', a['model']))
                assets.extend(choices[:3])
            spots = []
            area = math.prod(support[1][i]-support[0][i] for i in (0, 1))
            for asset in assets:
                if asset['size'][0]*asset['size'][1]/area > .08:
                    continue
                points = finite_centres(support, asset['size'], neighbours, protected,
                                        f'{task["id"]}:{variant}:{asset["model"]}')
                for point in points[:6]:
                    spots.append(dict(category=asset['category'], model=asset['model'], size=asset['size'],
                                      position_xy=point, metadata_sha256=asset['metadata_sha256']))
            # The light default adds one prop. A reviewed policy can allow a
            # second, but never at the expense of a safe single-object plan.
            for first in spots:
                pair = [first]
                for second in spots:
                    if desired_props == 1:
                        break
                    if second['category'] == first['category'] or math.dist(first['position_xy'], second['position_xy']) < .22:
                        continue
                    if sum(p['size'][0]*p['size'][1] for p in (first, second))/area > .08:
                        continue
                    a, b = first['position_xy'], second['position_xy']
                    if (abs(a[0]-b[0]) <= (first['size'][0]+second['size'][0])/2+.03 and
                            abs(a[1]-b[1]) <= (first['size'][1]+second['size'][1])/2+.03):
                        continue
                    pair.append(second); break
                proximity = min(gap(support, box) for box in protected)
                candidates.append(dict(parent_task_id=task.get('parent_task_id'), room=room, support=name,
                    semantic_review=f'{VERSION}: {kind} household props; excluded task inventory, text and semantic families {families}; untouched source/destination and articulated task furniture; native verification required.',
                    semantic_exclusions=sorted(excluded), props=pair, planner_version=VERSION,
                    _sort=(0 if len(pair) == desired_props else 1, proximity,
                           rank(f'{task["id"]}:{variant}', f'{name}:{first["model"]}:{first["position_xy"]}'))))
                if len([p for p in candidates if p['support'] == name and p['room'] == room]) >= 8:
                    break
    candidates.sort(key=lambda p: p.pop('_sort'))
    if not candidates:
        raise ValueError('NO_SAFE_CONTEXTUAL_BACKGROUND_SURFACE')
    # Interleave supports so native rejection of an oval tabletop does not
    # consume the entire retry budget at the same spot.
    selected, rest, seen = [], [], set()
    for plan in candidates:
        if plan['support'] in seen:
            rest.append(plan)
        else:
            selected.append(plan); seen.add(plan['support'])
    return (selected+rest)[:limit]
