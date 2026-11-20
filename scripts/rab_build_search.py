"""Generate FindingBench search scene layers over verified RAB layouts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from rab_assets import LayerBundle
from rab_available_resources import digest, namespace, provision
from rab_migration_after_assets import CANDIDATES


# Phrase labels describe the actual assets. The category group also identifies
# semantically plausible distractors and valid household storage surfaces.
CATALOG = {
    'pantry': {
        'coke_c3': 'Coca-Cola can', 'sprite_c1': 'Sprite can',
        'grapefruit_can': 'grapefruit sparkling water', 'starbucks': 'Starbucks coffee can',
        'gatorade_m': 'orange Gatorade', 'gatorade_t': 'blue Gatorade',
        'good2grow': 'Good2Grow juice bottle', 'water_bottle': 'water bottle',
        'pring_bbq': 'barbecue Pringles', 'honeymaid': 'Honey Maid crackers',
        'oreo_gf': 'gluten-free Oreos', 'zbar_t': 'Zbar snack bar',
        'skit_littles': 'Skittles Littles', 'chomps_r': 'Chomps beef stick',
        'coffee_bag': 'coffee bag',
    },
    'tableware': {
        'coffee_cup_blue_compact': 'blue cup', 'coffee_cup_cream_classic': 'cream cup',
        'coffee_cup_green_tall': 'tall green cup', 'paper_cup': 'paper cup',
        'meal_fork': 'fork', 'meal_knife': 'table knife', 'paper_plate_task2': 'paper plate',
    },
    'personal': {
        'wallet': 'wallet', 'glasses_case': 'glasses case', 'remote_tv': 'TV remote',
        'notebook_green': 'green notebook', 'baseball_cap': 'baseball cap',
        'power_bank': 'power bank', 'book_red': 'red book',
        'cloth_clear_memo_pad': 'memo pad', 'cloth_clear_tape_measure': 'tape measure',
        'cloth_clear_eraser': 'eraser', 'cloth_clear_pouch': 'small pouch',
    },
    'toys': {
        'cloth_clear_bear': 'toy bear', 'cloth_clear_cat': 'toy cat',
        'cloth_clear_duck': 'toy duck', 'cloth_clear_car': 'toy car',
        'cloth_clear_house': 'toy house', 'cloth_clear_penguin': 'toy penguin',
        'cloth_clear_rabbit': 'toy rabbit', 'cloth_clear_robot': 'toy robot',
        'cloth_clear_train': 'toy train', 'cloth_clear_block': 'toy block',
    },
}
FIXTURES = {
    'coffee': ('/World/Assets/CoffeeTable/Mesh', 'coffee table', 'living room'),
    'side': ('/World/Assets/SmallCoffeeTable', 'small side table', 'living room'),
    'shelf': ('/World/Assets/ShelvingUnit', 'pantry shelf', 'pantry'),
    'counter': ('/World/Background/Shell/CounterTop_17', 'kitchen counter', 'kitchen'),
}
ALLOWED = {
    'pantry': ('shelf', 'counter', 'coffee'),
    'tableware': ('counter', 'coffee', 'side', 'shelf'),
    'personal': ('coffee', 'side', 'shelf'),
    'toys': ('coffee', 'side', 'shelf'),
}


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def bounds(cache, prim):
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    return np.array([box.GetMin(), box.GetMax()], dtype=float)


def triangles(mesh, transforms):
    vertices = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    matrix = np.asarray(transforms.GetLocalToWorldTransform(mesh.GetPrim()))
    vertices = vertices @ matrix[:3, :3] + matrix[3, :3]
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    faces = np.split(indices, np.cumsum(mesh.GetFaceVertexCountsAttr().Get())[:-1])
    return vertices[np.array([(face[0], face[i], face[i + 1])
                              for face in faces for i in range(1, len(face)-1)])]


def supports(stage):
    cache = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy', 'guide'])
    transforms = UsdGeom.XformCache()
    fixtures = {}
    for name, (path, _, _) in FIXTURES.items():
        root = stage.GetPrimAtPath(path)
        if not root or not root.IsActive():
            continue
        boards = []
        for prim in Usd.PrimRange(root):
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.GetAttribute('physics:collisionEnabled').Get() is False):
                continue
            if name == 'shelf' and 'shelf_boards' not in str(prim.GetPath()).lower():
                continue
            if prim.IsA(UsdGeom.Cube):
                box = bounds(cache, prim)
                boards.append((box, None, str(prim.GetPath())))
            elif prim.IsA(UsdGeom.Mesh):
                tris = triangles(UsdGeom.Mesh(prim), transforms)
                flat = tris[np.ptp(tris[:,:,2], axis=1) < .003]
                heights = np.rint(flat[:,:,2].mean(axis=1)*200)/200
                for height in sorted(set(heights)):
                    if not .23 <= height <= 1.4:
                        continue
                    plane = flat[heights==height]
                    box = np.array([plane.min(axis=(0,1)), plane.max(axis=(0,1))])
                    if min(box[1,:2]-box[0,:2]) < .13:
                        continue
                    xy = plane[:,:,:2]
                    delta1,delta2 = xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]
                    area = np.abs(delta1[:,0]*delta2[:,1]-delta1[:,1]*delta2[:,0]).sum()/2
                    if area < .15*np.prod(box[1,:2]-box[0,:2]):
                        continue
                    boards.append((box, xy, str(prim.GetPath())))
        boards.sort(key=lambda row: row[0][1,2], reverse=True)
        if boards:
            # The highest usable plane is the exterior top on tables. The
            # pantry can use several boards, with observed inter-board clearance.
            if name == 'shelf':
                top_boards = []
                for board in boards:
                    if not top_boards or top_boards[-1][0][1,2]-board[0][1,2]>.035:
                        top_boards.append(board)
                fixtures[name] = top_boards
            else:
                fixtures[name] = boards[:1]
    obstacles = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World')):
        if prim.IsA(UsdGeom.Boundable) and prim.HasAPI(UsdPhysics.CollisionAPI):
            if prim.GetAttribute('physics:collisionEnabled').Get() is False:
                continue
            box = bounds(cache, prim)
            if box is not None:
                obstacles.append((str(prim.GetPath()), box))
    return fixtures, obstacles


def enable_shelf_support(stage, overlay_path):
    """Use the released static shelf meshes without changing their geometry."""
    root = stage.GetPrimAtPath(FIXTURES['shelf'][0])
    if not root:
        return None
    changed = []
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in Usd.PrimRange(root):
        if ('shelf_boards' in str(prim.GetPath()).lower()
                and prim.HasAPI(UsdPhysics.CollisionAPI)
                and prim.GetAttribute('physics:collisionEnabled').Get() is False):
            prim.GetAttribute('physics:collisionEnabled').Set(True)
            changed.append(str(prim.GetPath()))
    if not changed:
        return None
    layer = Sdf.Layer.CreateAnonymous('shelf_support.usda')
    layer.TransferContent(stage.GetSessionLayer())
    layer.subLayerPaths = [stage.GetRootLayer().identifier]
    if not layer.Export(str(overlay_path)):
        raise RuntimeError('Failed to export shelf collider overlay')
    return overlay_path


def inside_mesh(corners, plane):
    if plane is None:
        return True
    a = plane[:,0]
    u = plane[:,1] - a
    v = plane[:,2] - a
    det = u[:,0]*v[:,1]-u[:,1]*v[:,0]
    valid = abs(det) > 1e-9
    uv = corners[:,None,:] - a
    alpha = uv[:,:,0]*v[:,1] - uv[:,:,1]*v[:,0]
    beta = u[:,0]*uv[:,:,1] - u[:,1]*uv[:,:,0]
    alpha = alpha[:,valid]/det[valid]
    beta = beta[:,valid]/det[valid]
    return bool(np.all(np.any((alpha >= -1e-5) & (beta >= -1e-5) &
                              (alpha+beta <= 1.00001), axis=1)))


def placement(fixtures, obstacles, name, asset_box, rng, occupied, clearance=0.,
              candidate_allowed=None, inset=None, separation=.012):
    size = asset_box[1]-asset_box[0]
    inset = inset if inset is not None else (.008 if name.startswith('drawer') else {'shelf':.045}.get(name,.025))
    for box, plane, path in rng.sample(fixtures[name], len(fixtures[name])):
        z = box[1,2]
        low, high = box[0,:2]+size[:2]/2+inset, box[1,:2]-size[:2]/2-inset
        if np.any(high <= low):
            continue
        candidates = [(x,y) for x in np.linspace(low[0],high[0],9)
                      for y in np.linspace(low[1],high[1],9)]
        rng.shuffle(candidates)
        blocking = [b for p,b in obstacles if p != path
                    and not (name == 'shelf' and p.startswith(FIXTURES['shelf'][0]+'/'))
                    and not p.startswith('/World/Robot/')
                    and b[1,2] > z+.008 and b[0,2] < z+size[2]+.01]
        if name == 'shelf' and any(z+.011 < other[0][1,2] < z+size[2]+.045
                                  for other in fixtures[name]):
            continue
        blocking.extend(b for b in occupied if b[1,2]>z and b[0,2]<z+size[2]+.01)
        for point in candidates:
            lo = np.array(point)-size[:2]/2
            hi = np.array(point)+size[:2]/2
            if any(np.all(hi+separation>b[0,:2]) and np.all(lo-separation<b[1,:2]) for b in blocking):
                continue
            corners = np.array([lo, hi, [lo[0],hi[1]], [hi[0],lo[1]], point])
            if not inside_mesh(corners, plane):
                continue
            result = np.array([[*lo,z+.003+clearance],[*hi,z+.003+clearance+size[2]]])
            if candidate_allowed is not None and not candidate_allowed(result):
                continue
            return result, path
    return None


def read_asset(path, bundle):
    layer = bundle.clone(path)
    stage = Usd.Stage.Open(str(layer))
    root = stage.GetDefaultPrim()
    if not root or UsdGeom.GetStageUpAxis(stage) != 'Z' or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        return None
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=1 or bodies[0].GetAttribute('physics:kinematicEnabled').Get() is True:
        return None
    body = bodies[0]
    if body.GetAttribute('physics:rigidBodyEnabled').Get() is False:
        return None
    if not any(p.HasAPI(UsdPhysics.CollisionAPI)
               and p.GetAttribute('physics:collisionEnabled').Get() is not False
               for p in Usd.PrimRange(body)):
        return None
    if any(UsdGeom.Xformable(p).GetResetXformStack()
           for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Xformable)):
        return None
    cache = UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
    box = bounds(cache,root)
    if box is None or not np.isfinite(box).all() or np.max(box[1]-box[0])>.48:
        return None
    return dict(layer=str(layer), box=box, sha256=digest(path),
                body=str(body.GetPath())[len(str(root.GetPath())):])


def calibrated_asset(asset, pose):
    if asset['sha256'] != pose['source_sha256']:
        raise ValueError('Pose does not match release asset')
    x,y,z,w = pose['rotation_xyzw']
    rotation = Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage,'/Calibration')
    parent.MakeMatrixXform().Set(rotation)
    node = stage.DefinePrim('/Calibration/Model')
    node.GetReferences().AddReference(asset['layer'])
    box = bounds(UsdGeom.BBoxCache(0,['default','render','proxy','guide']),node)
    if box is None:
        raise ValueError('No bounds for calibrated asset')
    return dict(asset,box=box,rotation=rotation,clearance=pose['floor_to_bounds_min_m'])


def build_episode(row, layout_layer, fixtures, obstacles, assets, used, usage, output):
    rng = random.Random(row['id']+str(row.get('placement_retry','')))
    short = row['requested_regime'] == 'short_range'
    old = [goal['category'] for goal in row['original_goals']]
    preferred = {item for category in old for item in CANDIDATES.get(category, [])}
    options = []
    for group, labels in CATALOG.items():
        for target in labels.keys() & assets.keys():
            if row.get('required_target') and target != row['required_target']:
                continue
            if target in row.get('excluded_targets', []):
                continue
            for source in ALLOWED[group]:
                for dest in ('coffee','side','counter'):
                    if source == dest or source not in fixtures or dest not in fixtures:
                        continue
                    family = (target,source,dest)
                    if family in used:
                        continue
                    is_short = FIXTURES[source][2] == FIXTURES[dest][2]
                    if short != is_short:
                        continue
                    options.append((group,target,source,dest))
    options.sort()
    rng.shuffle(options)
    options.sort(key=lambda o:(o[1] not in preferred, usage[o[1]]))
    for group,target,source,dest in options:
        labels = CATALOG[group]
        target_names = [target]
        if row.get('preserve_goal_count',False) and len(old)>1:
            remaining = sorted((labels.keys() & assets.keys())-
                               {target,*row.get('excluded_targets',[])})
            assigned = {old[0]:target}
            for category in old[1:]:
                if category not in assigned:
                    rng.shuffle(remaining)
                    remaining.sort(key=lambda key:key not in CANDIDATES.get(category,[]))
                    if not remaining:
                        break
                    assigned[category] = remaining.pop(0)
                target_names.append(assigned[category])
            if len(target_names)!=len(old):
                continue
        reservations = []
        for name in target_names:
            result = placement(fixtures,obstacles,dest,assets[name]['box'],rng,
                               [box for box,_ in reservations],assets[name].get('clearance',0.))
            if result is None:
                break
            reservations.append(result)
        if len(reservations)!=len(target_names):
            continue
        selected = []
        occupied = [box for box,_ in reservations]
        # Use the released physical objects from the same everyday category
        # as distractors. Reject scenes that cannot fit a meaningful search.
        distractors = sorted((labels.keys() & assets.keys()) -
                             {*target_names, *row.get('excluded_distractors', [])})
        rng.shuffle(distractors)
        for index,name in enumerate(target_names+distractors):
            is_target = index<len(target_names)
            result = placement(fixtures,obstacles,source,assets[name]['box'],rng,occupied,
                               assets[name].get('clearance',0.))
            if result is None:
                if is_target:
                    break
                continue
            box, support = result
            occupied.append(box)
            selected.append((name,box,support,is_target))
            if len(selected) >= max(6,len(target_names)+2):
                break
        if len(selected)<len(target_names)+2:
            continue
        rng.shuffle(selected)
        output.mkdir(parents=True,exist_ok=True)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage,1)
        UsdGeom.SetStageUpAxis(stage,'Z')
        placed = []
        for index,(name,box,support,is_target) in enumerate(selected):
            asset = assets[name]
            parent = UsdGeom.Xform.Define(stage,f'/World/FindingBench/Item_{index:02}')
            node = stage.DefinePrim(str(parent.GetPath())+'/Model')
            node.GetReferences().AddReference(asset['layer'])
            translation = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(box[0]-asset['box'][0])))
            parent.MakeMatrixXform().Set(asset.get('rotation',Gf.Matrix4d(1))*translation)
            placed.append(dict(body=str(node.GetPath())+asset['body'], asset=name,
                               source_sha256=asset['sha256'], support=support,
                               expected_bounds_m=box.tolist(),target=is_target))
        # The exact target remains in evaluator metadata. The prompt names
        # the general search region and destination furniture a person sees.
        region = ('on the pantry shelf' if source=='shelf'
                  else 'in the '+FIXTURES[source][2])
        destination = FIXTURES[dest][1]
        noun = f'the {labels[target]}'
        if len(target_names)>1:
            goal_labels = Counter(labels[key] for key in target_names)
            quantity = {2:'two',3:'three',4:'four'}
            words = [f'the {label}' if count==1 else
                     f'{quantity.get(count,str(count))} {label}' + ('' if label.endswith('s') else 's')
                     for label,count in goal_labels.items()]
            listed = ' and '.join(words) if len(words)<3 else ', '.join(words[:-1])+', and '+words[-1]
            noun = listed
        prompts = [
            f'Could you find {noun} {region} and bring {"them" if len(target_names)>1 else "it"} to the {destination}?',
            f'Please look for {noun} {region} and set {"them" if len(target_names)>1 else "it"} on the {destination}.',
        ]
        scene = output/'scene.usda'
        temp = output/'scene.tmp.usda'
        layer = Sdf.Layer.CreateAnonymous('search.usda')
        layer.TransferContent(stage.GetRootLayer())
        layer.subLayerPaths = [str(layout_layer)]
        if not layer.Export(str(temp)):
            raise RuntimeError('Could not write scene layer')
        temp.replace(scene)
        used.add((target,source,dest))
        usage[target] += 1
        return dict(id=row['id'],parent_task_id=row['parent_task_id'],
                    original_instruction=row['original_instruction'],instruction=rng.choice(prompts),
                    source_backup_sha256=row['source_sha256'],original_categories=old,
                    target_asset=target,search_fixture=source,destination_fixture=dest,
                    target_assets=target_names,
                    regime=row['requested_regime'],scene=str(scene),scene_sha256=digest(scene),
                    rewrite='RAB-specific item, search region and destination; original request preserved in ledger',
                    placed=placed,delivery=dict(support=reservations[0][1],relation='on_top',
                         clear_area_m=reservations[0][0].tolist(),
                         reserved_placements=[dict(asset=name,support=support,area_m=box.tolist())
                           for name,(box,support) in zip(target_names,reservations)]),
                    physical_validation='pending',status='usd_scene_generated')
    return None


def load_assets(mapping, repo, bundle, poses):
    assets = {}
    errors = {}
    for group in CATALOG.values():
        for key in group:
            path = mapping.get(f'libraries/current_assets/all_assets/{key}/asset.usda')
            dynamic = (repo / 'src/robotagentbench/backends/profiles/room'
                       / 'openend/tasks/cloth_clear/dynamic.f_hyv0g1' / (key+'.usda'))
            if path and dynamic.is_file():
                path = dynamic
            if not path:
                continue
            try:
                item = read_asset(Path(path),bundle)
                if item is not None and (poses is None or key in poses):
                    assets[key] = item if poses is None else calibrated_asset(item,poses[key])
                else:
                    errors[key] = 'No compatible dynamic source or calibrated pose'
            except (ValueError, FileNotFoundError) as error:
                errors[key] = str(error)
    return assets, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--calibrated',action='store_true')
    parser.add_argument('--output-name',default=None)
    args = parser.parse_args()
    run = args.run.resolve()
    queue = [json.loads(line) for line in (run/'migration_queue.jsonl').read_text().splitlines()]
    output = run/(args.output_name or ('search_build_v2' if args.calibrated else 'search_build_v1'))
    output.mkdir(exist_ok=True)
    resources = run/'available_resources'
    bundle = LayerBundle(namespace(resources),output/'layers')
    passed = {row['id']: row for receipt in output.glob('rab_*/episode.json')
              if (row := json.loads(receipt.read_text()))
              and digest(Path(row['scene'])) == row['scene_sha256']}
    start = time.time()
    errors = {}
    poses = (json.loads((run/'pose_calibration/poses.json').read_text())['assets']
             if args.calibrated else None)
    while True:
        mapping,count = provision(args.repo.resolve(),run,resources)
        assets, asset_errors = load_assets(mapping,args.repo.resolve(),bundle,poses)
        usage = Counter(r['target_asset'] for r in passed.values())
        for key,path in sorted(mapping.items()):
            prefix = 'libraries/original_absolute/mnt/agent-bench-data/lingjun/releases/living50-candidate-20260919/layouts/'
            match = re.fullmatch(prefix+r'ID_(\d+)_LIVING_STRAIGHT\.usdc',key)
            if match is None:
                continue
            layout = int(match.group(1))
            rows = [r for r in queue if r['requested_layout_id']==layout]
            if not rows or all(r['id'] in passed for r in rows):
                continue
            try:
                # Compose one background per layout, shared by all task layers.
                layer = bundle.clone(Path(path))
                stage = Usd.Stage.Open(str(layer))
                overlay = enable_shelf_support(stage,output/'layers'/f'layout_{layout:02d}_shelf.usda')
                fixtures,obstacles = supports(stage)
                used = {(r['target_asset'],r['search_fixture'],r['destination_fixture'])
                        for r in passed.values() if r['layout_id']==layout}
                for row in rows:
                    if row['id'] in passed:
                        continue
                    episode = build_episode(row,overlay or layer,fixtures,obstacles,assets,used,
                                            usage,output/row['id'])
                    if episode is not None:
                        episode.update(layout_id=layout,layout_sha256=digest(Path(path)))
                        save_json(output/row['id']/'episode.json',episode)
                        passed[row['id']]=episode
                errors.pop(layout,None)
            except (ValueError, FileNotFoundError) as error:
                errors[layout] = str(error)
            save_json(output/'status.json',dict(generated=len(passed),total=len(queue),native_validated=0,
                      layouts=len({r['layout_id'] for r in passed.values()}),assets=len(assets),archives=count,
                      asset_errors=asset_errors,layout_errors=errors,started_at=start,updated_at=time.time()))
        if len(passed)==len(queue) or count==33 or time.time()-start>23*3600:
            break
        time.sleep(30)
    save_json(output/'manifest.json',dict(generated=len(passed),total=len(queue),tasks=list(passed.values()),
              pending=[row['id'] for row in queue if row['id'] not in passed],physical_validation='pending'))


if __name__ == '__main__':
    main()
