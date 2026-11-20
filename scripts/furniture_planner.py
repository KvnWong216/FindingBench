"""Sparse household furnishing. Geometry proposals never certify physics."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from background_planner import room_kind as background_room_kind, family_hits, rank, valid_box

VERSION = 'all_furniture_sparse_v1'


def room_kind(room):
    """Map explicit native room labels; do not guess from unknown names."""
    import re
    value=re.sub(r'_\d+$','',room.lower())
    return {'private_office':'office','home_office':'office','pantry_room':'kitchen',
            'storage':'utility_room','laundry':'utility_room'}.get(value) or background_room_kind(value)


TABLES = frozenset(('countertop','coffee_table','breakfast_table','desk','console_table',
    'dining_table','pedestal_table','reception_desk','checkout_counter','commercial_kitchen_table',
    'conference_table','gaming_table','garden_coffee_table','lab_table','nightstand'))
CABINETS = frozenset(('bottom_cabinet','bottom_cabinet_no_top','top_cabinet','metal_bottom_cabinet',
    'cabinet_base','dresser','nightstand','filing_cabinet'))
# Interior and exposed surfaces have different human uses. No fluids, cloth,
# resizing, refrigerator props or invented asset names.
SEMANTICS = {
 'kitchen': {'surface': ('mug','bowl','box_of_tissues','coffee_cup','apple'),
             'inside': ('bowl','mug','plate','sponge','coffee_cup','tablefork','teaspoon','tablespoon','vegetable_peeler')},
 'dining_room': {'surface': ('mug','box_of_tissues','bowl','coffee_cup'),
                 'inside': ('plate','bowl','mug','box_of_tissues')},
 'living_room': {'surface': ('box_of_tissues','hardback','mug'),
                 'inside': ('hardback','notebook','box_of_tissues')},
 'bedroom': {'surface': ('hardback','box_of_tissues','notebook'),
             'inside': ('notebook','hardback','comb','box_of_tissues')},
 'office': {'surface': ('notebook','hardback','box_of_tissues','mug'),
            'inside': ('notebook','hardback','pencil','eraser','box_of_tissues')},
 'bathroom': {'surface': ('bar_soap','soap_dish','box_of_tissues'),
              'inside': ('bar_soap','comb','soap_dish','box_of_tissues')},
 'garage': {'surface': ('open_end_wrench','screwdriver','notebook'),
            'inside': ('open_end_wrench','screwdriver','notebook')},
 'utility_room': {'surface': ('sponge','bar_soap','box_of_tissues'),
                  'inside': ('sponge','bar_soap','box_of_tissues')},
 'corridor': {'surface': ('box_of_tissues','hardback'), 'inside': ('hardback','box_of_tissues')},
 'balcony': {'surface': ('mug','bowl'), 'inside': ('mug','bowl')},
 'grocery_store': {'surface': ('box_of_tissues','notebook'),
                   'inside': ('box_of_tissues','bowl','mug')},
}


def build_policy(root):
    assets = {}
    for cat in sorted({c for room in SEMANTICS.values() for choices in room.values() for c in choices}):
        rows=[]
        for p in sorted((Path(root)/'objects'/cat).glob('*/misc/metadata.json')):
            raw=p.read_bytes(); size=json.loads(raw).get('bbox_size', [])
            if (len(size)!=3 or any(not math.isfinite(v) or v<=0 for v in size)
                    or max(size)>.30 or min(size[:2])<.012 or size[2]<.004
                    or size[2]/min(size[:2])>1.35):
                continue
            rows.append(dict(category=cat,model=p.parents[1].name,size=size,
                metadata_sha256=hashlib.sha256(raw).hexdigest()))
        assets[cat]=sorted(rows,key=lambda a:rank(cat,a['model']))[:12]
    return dict(version=VERSION,semantics=SEMANTICS,assets=assets,desired_count=1,
        edge_clearance_m=.025,neighbour_clearance_m=.025,max_coverage_fraction=.12,
        max_tilt_rad=.08,stable_seconds=3.,original_position_tolerance_m=.001)


def eligible(category):
    return category in TABLES or category in CABINETS or ('cabinet' in category and 'door' not in category)


def frozen_placement(location):
    """Relations in the recipe; exact world coordinates live in the snapshot.

    Repeating explicit XY coordinates makes legacy clearance setup translate
    already relocated furniture placements again before its integrity check.
    """
    result=dict(entity=location['entity'],relation='inside' if location['mode']=='inside' else 'on_top',
                receptacle=location['support'])
    if location['mode']=='inside':result['link']=location['link']
    return result


def excluded(task):
    cats={o['category'] for o in task.get('objects',[])}
    text=task.get('instruction','').lower().replace(' ','_')
    # A spoon task must not ban unrelated mugs or plates throughout the home.
    # Family-wide exclusions are needed only when the instruction asks for an
    # entire semantic class; exact authored categories are always protected.
    broad={'fruit':('all_the_fruit','all_fruit','every_fruit'),
           'tableware':('all_the_dishes','all_the_tableware','all_the_utensils'),
           'paper':('all_the_books','all_books','all_the_notebooks'),
           'toiletries':('all_the_toiletries',)}
    families={family for family,words in broad.items() if any(w in text for w in words)}
    all_cats={c for room in SEMANTICS.values() for choices in room.values() for c in choices}
    blocked=cats | {c for c in all_cats if family_hits(c)&families or c in text}
    # Native categories are finer than ordinary household nouns: a second
    # coffee_cup can satisfy a user's "cup" request just as readily as a mug.
    if cats & {'mug','coffee_cup'}:blocked.update(('mug','coffee_cup'))
    return blocked


def candidates(location,policy,task,obstacles):
    """Finite free-rectangle centres, with exact 3D inflated obstacle tests."""
    lo,hi=location['bounds']; floor=location['floor_z']; margin=policy['edge_clearance_m']
    kind=location.get('room_kind'); mode=location['mode']; rejected=excluded(task)
    pool=[]
    for cat in policy['semantics'].get(kind,{}).get(mode,[]):
        if cat not in rejected:
            pool.extend(policy['assets'].get(cat,[]))
    # Low, wide models first; deterministic identity changes across scenes.
    pool.sort(key=lambda a:(round(a['size'][2]/min(a['size'][:2]),1),rank(task['id']+location['id'],a['model'])))
    area=(hi[0]-lo[0])*(hi[1]-lo[1]); plans=[]
    for asset in pool:
        sx,sy,sz=asset['size']
        if sx*sy>area*policy['max_coverage_fraction'] or floor+sz+.012>hi[2]:continue
        if mode=='inside' and floor+sz/2<lo[2]+.001:continue
        low=[lo[i]+asset['size'][i]/2+margin for i in (0,1)]
        high=[hi[i]-asset['size'][i]/2-margin for i in (0,1)]
        if any(a>b for a,b in zip(low,high)):continue
        z0,z1=floor+.004,floor+.004+sz
        rectangles=[]
        for box in obstacles:
            if box[1][2]<z0-.01 or box[0][2]>z1+.01:continue
            rectangles.append(([box[0][i]-asset['size'][i]/2-policy['neighbour_clearance_m'] for i in (0,1)],
                               [box[1][i]+asset['size'][i]/2+policy['neighbour_clearance_m'] for i in (0,1)]))
        axes=[]
        for i in (0,1):
            cuts=sorted({low[i],high[i],*[max(low[i],min(high[i],v[i])) for r in rectangles for v in r]})
            axes.append(sorted({*cuts,*[(a+b)/2 for a,b in zip(cuts,cuts[1:])]}))
        points=[[x,y] for x in axes[0] for y in axes[1]
            if not any(a[0]-1e-7<=x<=b[0]+1e-7 and a[1]-1e-7<=y<=b[1]+1e-7 for a,b in rectangles)]
        center=[(low[i]+high[i])/2 for i in (0,1)]
        points.sort(key=lambda p:math.dist(p,center))
        for point in points[:6]:plans.append(dict(**asset,xy=point,floor_z=floor))
        if len(plans)>=24:break
    return plans[:24]


def occupied(location,records):
    """Count small existing items, never count a cabinet's own shell or robot."""
    lo,hi=location['bounds']; floor=location['floor_z']; names=[]
    for name,row in records.items():
        if name==location['support'] or row.get('fixed_base') or row.get('robot') or row.get('n_joints'):continue
        box=row['aabb']
        if not valid_box(box):continue
        center=[(box[0][i]+box[1][i])/2 for i in range(3)]
        if (all(lo[i]<=center[i]<=hi[i] for i in (0,1)) and floor-.015<=box[0][2]<=floor+.06
                and (location['mode']=='surface' or center[2]<hi[2])):
            names.append(name)
    return sorted(names)


def preserves_navigation(new_boxes, original_boxes, band=(.30,.50)):
    """No new MOVE obstacle: every added base-band rectangle was blocked already."""
    for lo,hi in new_boxes:
        if hi[2]<band[0] or lo[2]>band[1]:continue
        if not any(b[2]>=band[0] and a[2]<=band[1]
                   and all(a[i]-.001<=lo[i] and hi[i]<=b[i]+.001 for i in (0,1))
                   for a,b in original_boxes):return False
    return True


def outside_navigation_band(world_box, floor_z, band=(.30,.50)):
    """A complete live body AABB outside the MOVE band cannot add an obstacle.

    Reject invalid geometry and keep a 1 mm guard; boundary cases must use
    the full native collider decomposition, as before.
    """
    if not valid_box(world_box) or not math.isfinite(floor_z):return False
    return world_box[0][2]-floor_z>band[1]+.001 or world_box[1][2]-floor_z<band[0]-.001


def original_control_poses(complete, original):
    """Same actor topology for physics; compare only the actual parent bodies."""
    if not set(original)<=set(complete):raise ValueError('ORIGINAL_OBJECT_REMOVED_FROM_CONTROL')
    return {name:complete[name] for name in original}


def passive_articulation_within_baseline(original,control,current):
    """Allow an untouched, unreferenced articulation's observed natural range.

    Caller must exclude task references and any new prop's support. Translation stays strict;
    joints must remain between original and unchanged-control positions.
    This does not excuse new prop motion or contact with a door.
    """
    if not (set(original['links'])==set(control['links'])==set(current['links'])):return False
    if not (len(original['joints'])==len(control['joints'])==len(current['joints'])):return False
    def angle(a,b):
        scale=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
        return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))/scale))
    for name,(p,q) in current['links'].items():
        pc,qc=control['links'][name];po,qo=original['links'][name]
        if math.dist(p,pc)>.001 or angle(qo,q)>angle(qo,qc)+.01:return False
    return all(min(a,b)-.005<=v<=max(a,b)+.005
        for a,b,v in zip(original['joints'],control['joints'],current['joints']))


passive_door_within_baseline = passive_articulation_within_baseline
