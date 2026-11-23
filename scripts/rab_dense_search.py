"""Build and physically check dense search episodes from the archived recipes."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

DRAWER = '/World/Assets/CoffeeTable/black_box_1'
JOINT = DRAWER + '_slide'
DRAWERS = {'drawer':DRAWER, 'drawer_alt':'/World/Assets/CoffeeTable/black_box_3'}
FIXTURE_NAMES = {'coffee':'coffee table', 'side':'small side table', 'counter':'white kitchen counter'}
SNACK_DRAWER_GROUPS = (
    {'coke_c3','sprite_c1','grapefruit_can','starbucks','gatorade_m','gatorade_t','good2grow','water_bottle'},
    {'pring_bbq','honeymaid','oreo_gf','zbar_t','skit_littles','chomps_r','coffee_bag'},
)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(256*1024),b''):
            checksum.update(block)
        return checksum.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def recipe(row):
    import yaml
    source = Path(row['source_backup'])
    if digest(source) != row['source_sha256']:
        raise ValueError('Archived recipe changed')
    spec = yaml.safe_load(source.read_text())
    targets = {goal['entity'] for goal in row['original_goals']}
    relations = [p for p in spec['placements'] if p['entity'] in targets]
    if not relations:
        raise ValueError('Archived target placements are missing')
    origins = {p['receptacle'] for p in relations}
    decoys = Counter(p['receptacle'] for p in spec['placements'] if p['entity'] not in targets)
    return {
        'initial_relations':relations,
        'original_search_distractors':sum(decoys[p] for p in origins),
        'closed_container_required':any(p['relation']=='inside' for p in relations),
        'original_initial_states':spec.get('initial_states', {}),
        'target_count':len(targets),
    }


def compose_drawer(layer, output, reference, slide, *, scale_xy=1.):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from rab_build_search import bounds
    table = '/World/Assets/CoffeeTable'
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths = [str(slide), str(layer)]
    stage.SetEditTarget(stage.GetSessionLayer())
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(JOINT))
    drawer = stage.GetPrimAtPath(DRAWER)
    if not joint or not drawer.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('No original RAB drawer mechanism in this layout')
    if drawer.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Drawer is kinematic')
    transforms = UsdGeom.XformCache()
    table_prim = stage.GetPrimAtPath(table)
    current = transforms.GetLocalToWorldTransform(table_prim)
    delta = reference.GetInverse()*current
    if not all(.94 <= delta.GetRow3(i).GetLength() <= 1.06 for i in range(3)):
        raise ValueError('Drawer scaling is incompatible with released slide')
    if scale_xy != 1.:
        from rab_build_search import supports
        import numpy as np
        cache = UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
        before = bounds(cache,table_prim)
        _, obstacles = supports(stage)
        xf = UsdGeom.Xformable(table_prim)
        old = xf.GetLocalTransformation()
        reset = xf.GetResetXformStack()
        # Asset-local Z is vertical in the already relocated RAB room layer.
        if abs(current.GetRow3(2).GetNormalized()[2]) < .99:
            raise ValueError('Cannot widen a table with a tilted local up axis')
        wide = Gf.Matrix4d().SetScale(Gf.Vec3d(scale_xy,scale_xy,1.))*old
        op = xf.MakeMatrixXform()
        op.Set(wide)
        xf.SetResetXformStack(reset)
        cache.Clear()
        after = bounds(cache,table_prim)
        colliders = [(path,box) for path,box in obstacles
                     if not path.startswith(table+'/') and box[1,2] > before[0,2]+.04
                     and box[0,2] < before[1,2]
                     and not (np.all(before[1,:2] > box[0,:2]+.012)
                              and np.all(before[0,:2] < box[1,:2]-.012))]
        offsets = [(x,y) for x in np.arange(-.24,.241,.03) for y in np.arange(-.24,.241,.03)
                   if x*x+y*y <= .24**2+1e-9]
        for x,y in sorted(offsets,key=lambda xy:xy[0]**2+xy[1]**2):
            center = np.array([x,y])
            if not any(np.all(after[1,:2]+center > box[0,:2]+.012) and
                       np.all(after[0,:2]+center < box[1,:2]-.012) for _,box in colliders):
                parent = transforms.GetLocalToWorldTransform(table_prim.GetParent())
                offset = Gf.Vec3d(x,y,0.)
                if not reset:
                    offset = parent.GetInverse().TransformDir(offset)
                op.Set(wide*Gf.Matrix4d().SetTranslate(offset))
                break
        else:
            raise ValueError('No collision-free location within 24 cm for wider drawer table')
        transforms.Clear()
        current = transforms.GetLocalToWorldTransform(table_prim)
        delta = reference.GetInverse()*current
    rotation = delta.RemoveScaleShear().ExtractRotationQuat()
    p, q = joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(*delta.Transform(Gf.Vec3d(*p))))
    joint.GetLocalRot0Attr().Set(Gf.Quatf(rotation*Gf.Quatd(q.GetReal(),Gf.Vec3d(*q.GetImaginary()))))
    other_path = DRAWERS['drawer_alt']
    other = stage.GetPrimAtPath(other_path)
    if not other.HasAPI(UsdPhysics.RigidBodyAPI) or other.GetAttribute('physics:kinematicEnabled').Get() is True:
        raise ValueError('Second dynamic drawer unavailable')
    cache = UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
    first_box, other_box = bounds(cache,drawer), bounds(cache,other)
    if first_box is None or other_box is None or any(
            abs((first_box[1]-first_box[0])[i]-(other_box[1]-other_box[0])[i])>.015 for i in range(3)):
        raise ValueError('Second drawer geometry differs from the released slide body')
    transfer_drawer_collision(stage, slide, other_path)
    other_world = transforms.GetLocalToWorldTransform(other).RemoveScaleShear()
    first_world = transforms.GetLocalToWorldTransform(drawer).RemoveScaleShear()
    rel = joint.GetLocalRot1Attr().Get()
    ort = other_world.ExtractRotationQuat().GetInverse()*first_world.ExtractRotationQuat()
    second = UsdPhysics.PrismaticJoint.Define(stage,other_path+'_slide')
    second.CreateBody1Rel().SetTargets([other_path])
    second.CreateAxisAttr(joint.GetAxisAttr().Get())
    second.CreateLocalPos0Attr(Gf.Vec3f(*other_world.ExtractTranslation()))
    second.CreateLocalPos1Attr(joint.GetLocalPos1Attr().Get())
    second.CreateLocalRot0Attr(joint.GetLocalRot0Attr().Get())
    second.CreateLocalRot1Attr(Gf.Quatf(ort*Gf.Quatd(rel.GetReal(),Gf.Vec3d(*rel.GetImaginary()))))
    second.CreateLowerLimitAttr(0.)
    second.CreateUpperLimitAttr(joint.GetUpperLimitAttr().Get())
    UsdPhysics.FilteredPairsAPI.Apply(other).CreateFilteredPairsRel().AddTarget(table)
    combined = Sdf.Layer.CreateAnonymous('articulated_drawer.usda')
    combined.TransferContent(stage.GetSessionLayer())
    combined.subLayerPaths = [str(slide), str(layer)]
    if not combined.Export(str(output)):
        raise ValueError('Failed to export drawer layout')
    return output


def transfer_drawer_collision(stage, slide, destination):
    """Transfer the released physical overrides to an identical drawer shell."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source_path = Sdf.Path(DRAWER)
    target_path = Sdf.Path(destination)
    overlay = Sdf.Layer.FindOrOpen(str(slide))
    source = overlay.GetPrimAtPath(source_path)
    destination_layer = stage.GetSessionLayer()
    source_body = stage.GetPrimAtPath(source_path)
    overrides = []
    for prim in Usd.PrimRange(source_body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        patch = overlay.GetPrimAtPath(prim.GetPath())
        if patch is None:
            continue
        target = prim.GetPath().ReplacePrefix(source_path, target_path)
        other = stage.GetPrimAtPath(target)
        if not other.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh) or not other.IsA(UsdGeom.Mesh):
            raise ValueError('Second drawer collision topology differs from released drawer')
        mesh, replacement = UsdGeom.Mesh(prim), UsdGeom.Mesh(other)
        if any(a.Get() != b.Get() for a,b in (
            (mesh.GetPointsAttr(), replacement.GetPointsAttr()),
            (mesh.GetFaceVertexIndicesAttr(), replacement.GetFaceVertexIndicesAttr()),
            (mesh.GetFaceVertexCountsAttr(), replacement.GetFaceVertexCountsAttr()),
        )):
            raise ValueError('Second drawer collision mesh differs from released drawer')
        overrides.append((patch,target))
    if len(overrides)!=2:
        raise ValueError('Released drawer needs matching wall and bottom colliders')
    Sdf.CreatePrimInLayer(destination_layer,target_path)
    for material in ('DrawerFloorMat','DrawerWallMat'):
        address = source_path.AppendChild(material)
        if not Sdf.CopySpec(overlay,address,destination_layer,target_path.AppendChild(material)):
            raise ValueError('Could not transfer released drawer physics material')
    for patch,target in overrides:
        Sdf.CreatePrimInLayer(destination_layer,target)
        if not Sdf.CopySpec(overlay,patch.path,destination_layer,target):
            raise ValueError('Could not transfer released drawer collision overrides')
        material = stage.GetPrimAtPath(patch.path).GetRelationship('material:binding:physics')
        target_material = [path.ReplacePrefix(source_path,target_path) for path in material.GetTargets()]
        stage.GetPrimAtPath(target).GetRelationship('material:binding:physics').SetTargets(target_material)
    for name in ('physxRigidBody:linearDamping','physxRigidBody:angularDamping'):
        prop = source.attributes[name]
        if not Sdf.CopySpec(overlay,prop.path,destination_layer,target_path.AppendProperty(name)):
            raise ValueError('Could not transfer released drawer rigid-body damping')


def reference_table_transform(path):
    from pxr import Sdf, Usd, UsdGeom
    source = Sdf.Layer.FindOrOpen(str(path))
    destination = Sdf.Layer.CreateAnonymous('reference_transform.usda')
    table = Sdf.Path('/World/Assets/CoffeeTable')
    # Calibration needs only the authored transform; opening the original
    # room with its historical references can invoke the network resolver.
    for path in table.GetPrefixes():
        origin = source.GetPrimAtPath(path)
        prim = Sdf.CreatePrimInLayer(destination,path)
        prim.specifier = Sdf.SpecifierDef
        prim.typeName = 'Xform'
        if origin:
            for attr in origin.attributes:
                if attr.name=='xformOpOrder' or attr.name.startswith('xformOp:'):
                    Sdf.CopySpec(source,attr.path,destination,attr.path)
    stage = Usd.Stage.Open(destination)
    return UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(table))


def drawer_support(stage, path):
    from pxr import Usd, UsdGeom, UsdPhysics
    from rab_build_search import bounds
    parent = stage.GetPrimAtPath(path)
    floors = [prim for prim in Usd.PrimRange(parent)
              if '/Bottom_0/' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)]
    if len(floors)!=1:
        raise ValueError('Drawer floor unavailable')
    cache = UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
    box = bounds(cache,floors[0])
    if box is None or min(box[1,:2]-box[0,:2]) < .10:
        raise ValueError('Drawer interior has invalid dimensions')
    return [(box,None,str(floors[0].GetPath()))]


def drawer_headroom(stage, fixtures):
    """Conservative overhead clearance along the driven 15 cm sweep."""
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    from rab_build_search import triangles

    table = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Assets/CoffeeTable/Mesh'))
    faces = triangles(table, UsdGeom.XformCache())
    flat = faces[np.ptp(faces[:,:,2], axis=1) < .003]
    low, high = flat.min(axis=1), flat.max(axis=1)
    room = {}
    for zone, path in DRAWERS.items():
        floor = fixtures[zone][0][0]
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(path+'_slide'))
        q = joint.GetLocalRot0Attr().Get()
        axis = np.eye(3)['XYZ'.index(str(joint.GetAxisAttr().Get()))]
        direction = Gf.Rotation(Gf.Quatd(q.GetReal(),Gf.Vec3d(*q.GetImaginary())))
        delta = .15*np.asarray(direction.TransformDir(Gf.Vec3d(*axis)))
        sweep_low = floor[0,:2]+np.minimum(delta[:2],0)
        sweep_high = floor[1,:2]+np.maximum(delta[:2],0)
        above = ((low[:,2] > floor[1,2]+.015)
                 & np.all(high[:,:2] > sweep_low, axis=1)
                 & np.all(low[:,:2] < sweep_high, axis=1))
        if not above.any():
            raise ValueError('Drawer ceiling geometry unavailable')
        room[zone] = float(low[above,2].min()-floor[1,2]-max(delta[2],0)-.020)
    return room


def quarter_turn(asset):
    """Rotate the calibrated stable pose about world up, preserving its height."""
    import numpy as np
    from pxr import Gf

    lo, hi = asset['box']
    return dict(asset, box=np.array([[-hi[1],lo[0],lo[2]],[-lo[1],hi[0],hi[2]]]),
                rotation=asset['rotation']*Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),90)))


def side_pose(asset):
    """Candidate for horizontal storage of closed packets/bottles; needs PhysX admission."""
    import numpy as np
    from pxr import Gf

    lo, hi = asset['box']
    return dict(asset, box=np.array([[lo[2],lo[1],-hi[0]],[hi[2],hi[1],-lo[0]]]),
                rotation=asset['rotation']*Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,1,0),90)),
                clearance=0., pose_variant='horizontal_candidate')


def notebook_top(item):
    """Return the actual planar book or notebook cover in world coordinates."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from rab_build_search import triangles

    asset = item['source']
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage,'/Calibration')
    translate = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(item['box'][0]-asset['box'][0])))
    parent.MakeMatrixXform().Set(asset['rotation']*translate)
    model = stage.DefinePrim('/Calibration/Model')
    model.GetReferences().AddReference(asset['layer'])
    xf = UsdGeom.XformCache()
    top_z = item['box'][1,2]
    surface = []
    for prim in Usd.PrimRange(model):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            faces = triangles(UsdGeom.Mesh(prim),xf)
            surface.extend(faces[np.max(abs(faces[:,:,2]-top_z),axis=1) < .0015,:,:2])
    return np.asarray(surface) if surface else None


def mesh_footprint(asset):
    """Use the rendered, collidable mesh vertices instead of the USD hint bounds."""
    import numpy as np
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(asset['layer'])
    root=stage.GetDefaultPrim()
    xf=UsdGeom.XformCache()
    pieces=[]
    for prim in Usd.PrimRange(root):
        if (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdGeom.Imageable(prim).ComputeVisibility()!='invisible'):
            matrix=xf.GetLocalToWorldTransform(prim)*asset['rotation']
            pieces.extend(matrix.Transform(point) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    if not pieces:
        raise ValueError('No visible collision mesh for tighter footprint')
    points=np.asarray(pieces)
    box=np.array([points.min(axis=0),points.max(axis=0)])
    return dict(asset,box=box,clearance=float(asset['clearance']+box[0,2]-asset['box'][0,2]),
                pose_variant='calibrated_mesh_footprint')


def stacked_book(base, headroom, floor_z, occupied, attempt):
    """Small offset on a measured planar cover; native motion is still required."""
    import numpy as np
    from rab_build_search import inside_mesh
    surface=notebook_top(base)
    if surface is None:
        return None
    size=base['box'][1]-base['box'][0]
    z=base['box'][1,2]
    if z+size[2]+.002>floor_z+headroom:
        return None
    center=base['box'].mean(axis=0)[:2]
    cover=np.unique(surface.reshape(-1,2),axis=0)
    cover_mid=cover.mean(axis=0)
    for sign in (1,-1):
        offset=np.eye(2)[attempt%2]*sign*.012
        mid=center+offset
        contacts=np.vstack([cover_mid,cover_mid+(cover-cover_mid)*.65])+offset
        if not inside_mesh(contacts,surface):
            continue
        lo,hi=mid-size[:2]/2,mid+size[:2]/2
        if any(b[1,2]>z+.0005 and b[0,2]<z+size[2]+.01 and
               np.all(hi+.012>b[0,:2]) and np.all(lo-.012<b[1,:2]) for b in occupied):
            continue
        return np.array([[*lo,z+.002],[*hi,z+.002+size[2]]])
    return None


def intersects(origin, target, box):
    import numpy as np
    direction = target-origin
    near, far = 0., .98
    for index in range(3):
        if abs(direction[index])<1e-8:
            if not box[0,index]<=origin[index]<=box[1,index]:
                return False
            continue
        interval = sorted((box[:,index]-origin[index])/direction[index])
        near,far = max(near,interval[0]),min(far,interval[1])
        if far<near:
            return False
    return True


def viewpoints(center, obstacles, extra=False):
    import numpy as np
    for radius, height in ((.85,.28), (1.10,.18), (.70,.19)) if extra else ((.85,.28),):
        for angle in np.linspace(0,2*np.pi,24,endpoint=False):
            head = np.array([center[0]+radius*np.cos(angle),center[1]+radius*np.sin(angle),
                             min(1.55,max(1.,center[2]+height))])
            if not any(lo[2]<.52 and hi[2]>.24 and
                       np.all(head[:2]+.23>lo[:2]) and np.all(head[:2]-.23<hi[:2])
                       for _,(lo,hi) in obstacles):
                yield head

def bind_names(targets, labels):
    counts = Counter(labels[key] for key in targets)
    quantities = {2:'two',3:'three',4:'four'}
    phrases = [f'the {name}' if count==1 else
               f'{quantities.get(count,count)} {name}'+('' if name.endswith('s') else 's')
               for name,count in counts.items()]
    if len(phrases)==1:
        return phrases[0]
    return ', '.join(phrases[:-1])+' and '+phrases[-1]


def pack(row, contract, layer, fixtures, obstacles, assets, attempt, headroom=None,
         snack_drawers=False, stack_notebook=False, repair_occlusion=False, stack_targets=False):
    import numpy as np
    from rab_build_search import ALLOWED, CATALOG, placement
    rng = random.Random(f'{row["id"]}:{attempt}')
    targets = [r['asset'] for r in row['placed'] if r['target']]
    if len(targets)!=contract['target_count'] or not set(targets)<=assets.keys():
        raise ValueError('Missing admitted target models')
    group = next((key for key,labels in CATALOG.items() if set(targets)<=labels.keys()),None)
    if group is None:
        raise ValueError('No compatible household inventory group')
    pool = sorted((CATALOG[group].keys() & assets.keys())-set(targets)-{'baseball_cap'})
    if len(pool)<2:
        raise ValueError('Too few comparable distractors')
    closed = contract['closed_container_required']
    if closed and not DRAWERS.keys()<=fixtures.keys():
        raise ValueError('Two articulated search fixtures are required')
    # Refrigerated foods are a separate recipe family, and require a fridge.
    if closed and any('fridge' in p['receptacle'].lower() or 'refrigerator' in p['receptacle'].lower()
                      for p in contract['initial_relations']):
        raise ValueError('Refrigerated recipe needs an admitted fridge')
    snack_variant = closed and group=='pantry' and snack_drawers
    if closed and group=='pantry' and not snack_variant:
        raise ValueError('Pantry container recipe needs an admitted food cupboard')
    if snack_variant:
        inventory = set().union(*SNACK_DRAWER_GROUPS)
        if not set(targets)<=inventory:
            raise ValueError('Target is not packaged pantry inventory')
        pool = [name for name in pool if name in inventory]
    headroom = headroom or dict.fromkeys(DRAWERS,.105)
    variants = {name:(asset,quarter_turn(asset)) for name,asset in assets.items()}
    if snack_variant:
        for name in inventory & assets.keys():
            flat = side_pose(assets[name])
            variants[name] += (flat,quarter_turn(flat))

    def compatible(asset, zone):
        if zone not in DRAWERS:
            return True
        size = asset['box'][1]-asset['box'][0]
        footprint = fixtures[zone][0][0]
        return (size[2]+.003+max(0.,asset.get('clearance',0.)) <= headroom[zone]
                and np.all(size[:2]+.016 < footprint[1,:2]-footprint[0,:2]))

    if closed and any(not any(compatible(asset,zone) for asset in variants[name] for zone in DRAWERS)
                      for name in targets):
        raise ValueError('Calibrated target cannot fit either measured drawer')

    sink = row['destination_fixture']
    locs = [key for key in ALLOWED[group] if key in fixtures and key!=sink]
    rng.shuffle(locs)
    primary = rng.choice(sorted(DRAWERS)) if closed else (
        row['search_fixture'] if attempt==0 and row['search_fixture'] in locs else locs[0] if locs else None)
    if primary is None or sink not in fixtures:
        return None
    secondary = [key for key in locs if key!=primary]
    if closed:
        secondary = [key for key in DRAWERS if key!=primary] + [
            key for key in secondary if key in ('coffee','side')]
        if snack_variant and 'side' in fixtures and sink!='side':
            secondary.append('side')
    else:
        # An occupied delivery fixture is valid when the actual drop
        # footprints stay reserved and free of all distractor boxes.
        if sink in ALLOWED[group]:
            secondary.append(sink)
        rng.shuffle(secondary)
    if not secondary:
        return None
    needed = max(6,contract['original_search_distractors'])
    supplements = []
    if group=='tableware':
        # Snack packets can share storage with picnic utensils and paper cups.
        pantry = {'zbar_t','chomps_r','skit_littles','oreo_gf','honeymaid'} if closed else CATALOG['pantry'].keys()
        supplements = sorted((pantry & assets.keys())-set(targets))
    if 2*(len(pool)+len(supplements))<needed:
        raise ValueError('Insufficient comparable item models for the archived clutter count')
    occupied, records, delivery = [],[],[]
    similar = (set().union(*(members for members in SNACK_DRAWER_GROUPS if members.intersection(targets)))
               if snack_variant else None)

    def place(name, zone, *, allowed=None, target=False):
        # The single drawer's wall mesh covers its footprint as an AABB. Its
        # authored bottom defines the interior pad. Native motion validates it.
        blockers = ([r for r in obstacles if not r[0].startswith(DRAWERS[zone]+'/')
                     and r[0]!='/World/Assets/CoffeeTable/Mesh'] if zone in DRAWERS else obstacles)
        options = list(variants[name])
        if zone not in DRAWERS:
            options = options[:2]
        if attempt%3 == 1:
            options[:2] = reversed(options[:2])
        for asset in options:
            if not compatible(asset,zone):
                continue
            result = placement(fixtures,blockers,zone,asset['box'],rng,occupied,
                               asset.get('clearance',0.),candidate_allowed=allowed)
            if result is not None:
                box,path = result
                occupied.append(box)
                records.append(dict(asset=name,zone=zone,box=box,target=target,support=path,source=asset))
                return True
        return False

    for name in targets:
        if not place(name,sink,target=True):
            return None
        delivery.append(records.pop())
    for name in sorted(targets,key=lambda key:-np.prod(assets[key]['box'][1,:2]-assets[key]['box'][0,:2])):
        zones = [primary]
        if closed:
            zones.append(secondary[0])
            if len(targets)>1 and sum(r['zone']==primary for r in records)>=2:
                zones.reverse()
        if stack_targets and closed and name in ('book_red','notebook_green'):
            if place(name,primary,target=True):
                continue
            bases=[r for r in records if r['asset']==name and r['target'] and r['zone']==primary]
            if bases:
                base=max(bases,key=lambda r:r['box'][1,2])
                box=stacked_book(base,headroom[primary],fixtures[primary][0][0][1,2],occupied,attempt)
                if box is not None:
                    footprint=fixtures[primary][0][0]
                    if np.all(box[0,:2] >= footprint[0,:2]+.008) and np.all(box[1,:2] <= footprint[1,:2]-.008):
                        occupied.append(box)
                        records.append(dict(asset=name,zone=primary,box=box,target=True,
                                            source=base['source'],stacked_on=base))
                        continue
        if not any(place(name,zone,target=True) for zone in zones):
            return None
    if closed and not any(item['zone']==primary for item in records):
        primary,secondary[0] = secondary[0],primary

    shield = None
    if not closed:
        for target in list(records):
            center = target['box'].mean(axis=0)
            eyes = list(viewpoints(center,obstacles,repair_occlusion))
            rng.shuffle(eyes)
            # Taller comparable items are likely to interrupt line of sight.
            candidates = sorted(pool+(supplements if repair_occlusion else []),
                                key=lambda key:-(assets[key]['box'][1,2]-assets[key]['box'][0,2]))
            if repair_occlusion:
                candidates = candidates[attempt%min(7,len(candidates)):] + candidates[:attempt%min(7,len(candidates))]
            ends = [center] if not repair_occlusion else [
                target['box'][0]+(target['box'][1]-target['box'][0])*np.array(f)
                for f in ((.5,.5,.35),(.5,.5,.55),(.5,.5,.75),(.25,.5,.55),(.75,.5,.55))]
            def obscured(eye, box):
                return any(intersects(eye,end,box) for end in ends)
            for name in candidates[:(len(candidates) if repair_occlusion else 9)]:
                if place(name,primary,allowed=lambda box:any(obscured(e,box) for e in eyes)):
                    blocker = records[-1]
                    hidden = [eye.tolist() for eye in eyes if obscured(eye,blocker['box'])]
                    shield = (target,blocker,hidden)
                    break
            if shield:
                break
        if shield is None:
            return None
    else:
        candidates = sorted(pool+supplements,key=lambda key:(similar is not None and key not in similar,
                            np.prod(assets[key]['box'][1,:2]-assets[key]['box'][0,:2])))
        for name in candidates:
            if place(name,primary):
                break
        if stack_notebook and not any(not r['target'] and r['zone']==primary for r in records):
            base = next((r for r in records if r['zone']==primary and r['target']
                         and r['asset']=='notebook_green'),None)
            if base is not None:
                surface = notebook_top(base)
                if surface is not None:
                    cover = base['box'].copy()
                    cover[0,2] = cover[1,2]
                    floor_z = fixtures[primary][0][0][1,2]
                    stack_fixture = {primary:[(cover,surface,base['support'])]}
                    for name in ('cloth_clear_eraser','cloth_clear_memo_pad','power_bank'):
                        if name not in pool:
                            continue
                        for asset in variants[name]:
                            result = placement(stack_fixture,[],primary,asset['box'],rng,occupied,
                                asset.get('clearance',0.),
                                candidate_allowed=lambda box:box[1,2] <= floor_z+headroom[primary])
                            if result:
                                box,_ = result
                                occupied.append(box)
                                records.append(dict(asset=name,zone=primary,box=box,target=False,
                                                    stacked_on=base,source=asset))
                                break
                        if records[-1].get('stacked_on') is base:
                            break
        if not any(not item['target'] and item['zone']==primary for item in records):
            return None

    rng.shuffle(pool)
    # Put plausible decoys in another search location before filling the core.
    visits = [secondary[0], secondary[0], primary]
    visits += [key for _ in range(needed) for key in ([primary]+secondary)]
    for zone in visits:
        if sum(not r['target'] for r in records)>=needed:
            break
        if zone in DRAWERS and sum(r['zone']==zone for r in records)>=4:
            continue
        choices = sorted(pool,key=lambda name:(similar is not None and name not in similar,
                        sum(r['asset']==name for r in records)))+supplements
        for name in choices:
            if sum(r['asset']==name for r in records)>=2:
                continue
            if place(name,zone):
                break
    if (sum(not r['target'] for r in records)<needed
            or sum(not r['target'] and r['zone']==secondary[0] for r in records)<2
            or sum(not r['target'] and r['asset'] in (similar if snack_variant else pool)
                   for r in records)<2):
        return None
    rng.shuffle(records)
    for index,item in enumerate(records):
        item['id'] = f'Item_{index:02d}'
    return dict(layer=str(layer),records=records,delivery=delivery,shield=shield,group=group,
                search=primary,decoy=secondary[0],targets=targets,minimum_distractors=needed,
                measured_drawer_headroom_m=headroom if closed else {},snack_drawers=snack_variant)


def author(row, plan, contract, directory):
    from pxr import Gf, Sdf, Usd, UsdGeom
    from rab_build_search import CATALOG
    directory.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage,'Z')
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    placed = []
    for item in plan['records']:
        asset = item['source']
        root = UsdGeom.Xform.Define(stage,'/World/FindingBench/'+item['id'])
        model = stage.DefinePrim(str(root.GetPath())+'/Model')
        model.GetReferences().AddReference(asset['layer'])
        delta = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(item['box'][0]-asset['box'][0])))
        root.MakeMatrixXform().Set(asset['rotation']*delta)
        item['body']=str(model.GetPath())+asset['body']
        if 'stacked_on' in item:
            base = item['stacked_on']
            item['support'] = f"/World/FindingBench/{base['id']}/Model"+base['source']['body']
        placed.append(dict(body=item['body'],asset=item['asset'],target=item['target'],
                           support=item['support'],expected_bounds_m=item['box'].tolist(),
                           zone=item['zone'],source_sha256=asset['sha256'],
                           pose_variant=asset.get('pose_variant','calibrated')))
    layer = Sdf.Layer.CreateAnonymous('dense_search.usda')
    layer.TransferContent(stage.GetRootLayer())
    layer.subLayerPaths = [plan['layer']]
    scene = directory/'scene.usda'
    if not layer.Export(str(scene)):
        raise ValueError('USD export failed')
    shield = None
    if plan['shield']:
        target,blocker,eyes = plan['shield']
        shield = dict(target=target['body'],blocker=blocker['body'],viewpoints=eyes)
    # The instruction gives the search area and unambiguous natural-language
    # object + destination phrases (with diverse phrasing across tasks),
    # without identifying the hiding container, blocker, skill, or restoration rules.
    from rab_instruction_diversity import compose_diverse_instruction
    scope = 'living_room' if contract['closed_container_required'] else 'kitchen_and_living'
    instruction = compose_diverse_instruction(
        seed_key=row['id'],
        target_assets=list(plan['targets']),
        delivery_fixture=row['destination_fixture'],
        search_scope=scope,
        layout_id=int(row['layout_id']),
    )
    migration = (dict(kind='living_room_snack_storage',
                     reason='Shelf-stable packaged inventory assigned to existing interactive snack drawers',
                     archived_sources=sorted({p['receptacle'] for p in contract['initial_relations']}))
                 if plan['snack_drawers'] else None)
    return dict(id=row['id'],parent_task_id=row['parent_task_id'],layout_id=row['layout_id'],
                regime=row['regime'],instruction=instruction,baseline_instruction=row['instruction'],
                original_instruction=row['original_instruction'],source_backup_sha256=row['source_backup_sha256'],
                baseline_scene_sha256=row['scene_sha256'],scene=str(scene),scene_sha256=digest(scene),
                placed=placed,original_recipe=contract,search_fixture=plan['search'],
                alternative_search_fixture=plan['decoy'],minimum_distractors=plan['minimum_distractors'],
                measured_drawer_headroom_m=plan['measured_drawer_headroom_m'],
                search_storage_migration=migration,
                delivery=dict(relation='on_top',fixture=row['destination_fixture'],
                    placements=[dict(asset=x['asset'],support=x['support'],bounds_m=x['box'].tolist())
                                for x in plan['delivery']]),
                arrangement=dict(kind='articulated_drawer' if contract['closed_container_required'] else 'occluded_open',
                    articulations={key:path+'_slide' for key,path in DRAWERS.items()}
                    if contract['closed_container_required'] else {},
                    shield=shield))


def archived_drawer_candidates(directory):
    """Prefer packs that were stable until the old drawer moved."""
    candidates = []
    for path in directory.glob('attempt_*/episode.json'):
        receipt = path.with_name('native_validation.json')
        if not receipt.is_file():
            continue
        validation = load(receipt)
        objects = validation.get('objects', [])
        unsettled = sum(not item['stable'] for item in objects)
        drift = max((item['drift_m'] for item in objects),default=float('inf'))
        candidates.append((not bool(validation.get('dense_checks')),unsettled,drift,path))
    return [row[-1] for row in sorted(candidates)[:4]]


def reuse_drawer_placement(path, layer, directory):
    from pxr import Sdf
    episode = load(path)
    if digest(episode['scene'])!=episode['scene_sha256']:
        raise ValueError('Archived drawer scene hash changed')
    original = Sdf.Layer.FindOrOpen(episode['scene'])
    amended = Sdf.Layer.CreateAnonymous('repaired_drawer_episode.usda')
    amended.TransferContent(original)
    amended.subLayerPaths = [str(layer)]
    directory.mkdir(parents=True,exist_ok=True)
    scene = directory/'scene.usda'
    if not amended.Export(str(scene)):
        raise ValueError('Could not export repaired drawer episode')
    return dict(episode,scene=str(scene),scene_sha256=digest(scene),
                previous_placement=str(path))


def native_check(episode, simulator, interface, scene_query):
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    from rab_build_search import bounds
    from rab_validate_search import validate
    mechanisms = episode['arrangement']['articulations']
    parameters = {}

    def before(stage):
        for zone,path in mechanisms.items():
            prim = stage.GetPrimAtPath(path)
            joint = UsdPhysics.PrismaticJoint(prim)
            if not joint:
                raise ValueError('Drawer joint unavailable')
            low,high = joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()
            if not math.isfinite(low) or not math.isfinite(high) or not low<=0<=high or high<.15:
                raise ValueError('Incompatible drawer limits')
            drive = UsdPhysics.DriveAPI.Apply(prim,'linear')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(4000.)
            drive.CreateDampingAttr(400.)
            drive.CreateMaxForceAttr(300.)
            drive.CreateTargetPositionAttr(0.)
            parameters[zone]=drive

    def position(path):
        pose = interface.get_rigidbody_transformation(path)
        if not pose.get('ret_val'):
            raise ValueError('Native rigid-body pose unavailable: '+path)
        return np.array([float(x) for x in pose['position']])

    def blocked(eyes, target, blocker, box):
        evidence = []
        for origin in eyes:
            samples = [(x,y,z) for x,y in ((.5,.5),(.25,.5),(.75,.5),(.5,.25),(.5,.75))
                       for z in (.35,.55,.75)]
            for fraction in samples:
                end = box[0]+(box[1]-box[0])*np.array(fraction)
                direction=end-np.array(origin)
                distance=np.linalg.norm(direction)
                if not distance>0:
                    continue
                hits=[]
                def record(hit):
                    hits.append((str(hit.rigid_body),str(hit.collision),float(hit.distance)))
                    return True
                scene_query.raycast_all(Gf.Vec3f(*origin),Gf.Vec3f(*(direction/distance)),
                                        distance+.035,record)
                hits.sort(key=lambda hit:hit[2])
                matches=lambda hit,body: any(path==body or path.startswith(body+'/') for path in hit[:2])
                if hits and matches(hits[0],blocker) and any(matches(x,target) for x in hits[1:]):
                    evidence.append(dict(eye=origin,target_point=end.tolist(),hits=hits[:5]))
        return evidence

    def after(stage, poses, initial):
        cache=UsdGeom.BBoxCache(0,['default','render','proxy','guide'])
        snapshots={}
        for index,row in enumerate(episode['placed']):
            snapshots[row['body']]=dict(pose=initial[index].tolist(),support=row['support'],
                                       zone=row['zone'],target=row['target'])
        restoration=dict(frame='world',bodies=snapshots,articulations_m={key:0. for key in mechanisms})
        if not mechanisms:
            shield=episode['arrangement']['shield']
            target=next(row for row in episode['placed'] if row['body']==shield['target'])
            root=target['body'].split('/Model',1)[0]
            box=bounds(cache,stage.GetPrimAtPath(root))
            evidence=blocked(shield['viewpoints'],shield['target'],shield['blocker'],box)
            return dict(passed=bool(evidence),occlusion_rays=evidence,
                        scope='PhysX collider line of sight with movable distractor first and target behind',
                        restoration_reference=restoration)
        clock=75/60
        def sweep(zone,a,b):
            nonlocal clock
            for step in range(240):
                fraction=min(1.,(step+1)/180)
                fraction=fraction*fraction*(3-2*fraction)
                parameters[zone].GetTargetPositionAttr().Set(a+(b-a)*fraction)
                simulator.simulate(1/60,clock)
                simulator.fetch_results()
                clock+=1/60
            return position(DRAWERS[zone]),poses()
        measurements=[]
        drawer_starts={}
        for zone in (episode['alternative_search_fixture'],episode['search_fixture']):
            start=position(DRAWERS[zone])
            drawer_starts[zone]=start
            masks=[i for i,row in enumerate(episode['placed']) if row['zone']==zone]
            opened,open_poses=sweep(zone,0.,.15)
            closed,close_poses=sweep(zone,.15,0.)
            travel=float(np.linalg.norm(opened-start))
            close_error=float(np.linalg.norm(closed-start))
            open_drift=np.linalg.norm((open_poses[masks,:3]-opened)-(initial[masks,:3]-start),axis=1)
            closed_drift=np.linalg.norm((close_poses[masks,:3]-closed)-(initial[masks,:3]-start),axis=1)
            outside=[i for i in range(len(initial)) if i not in masks]
            external_drift=np.linalg.norm(close_poses[outside,:3]-initial[outside,:3],axis=1)
            for point in snapshots.values():
                if point['zone']==zone:
                    point['drawer_relative_xyz_m']=(np.array(point['pose'][:3])-start).tolist()
            passed=(travel>=.10 and close_error<=.015 and np.max(open_drift)<.04
                    and np.max(closed_drift)<.04 and (not len(outside) or np.max(external_drift)<.04))
            measurements.append(dict(zone=zone,passed=bool(passed),open_travel_m=travel,
                close_error_m=close_error,drawer_item_open_drift_m=open_drift.tolist(),
                drawer_item_close_drift_m=closed_drift.tolist(),exterior_item_drift_m=external_drift.tolist()))
        # Direct pose measurement from the controlled open/close sequence.
        # The observed support labels here come from containment verification.
        final_bodies={}
        for index,row in enumerate(episode['placed']):
            point=dict(pose=close_poses[index].tolist(),support=row['support'])
            if row['zone'] in drawer_starts:
                point['drawer_relative_xyz_m']=(close_poses[index,:3]-position(DRAWERS[row['zone']])).tolist()
            final_bodies[row['body']]=point
        from rab_restore import score_restoration
        restore_control=score_restoration(restoration,dict(bodies=final_bodies,
            articulations_m={key:float(np.linalg.norm(position(DRAWERS[key])-start))
                             for key,start in drawer_starts.items()}))
        return dict(passed=all(row['passed'] for row in measurements),drawers=measurements,native_steps=1035,
                    scope='Temporary diagnostic drive with contained items; robot manipulation not exercised',
                    restore_under_joint_control=restore_control,
                    restoration_reference=restoration)
    return validate(episode,simulator,interface,before_attach=before,after_settle=after)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lane',choices=('drawer','open'),required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--attempts',type=int,default=8)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--prior-run',type=Path,help='Reuse prior admissions and retry drawer placements')
    parser.add_argument('--admitted-from',type=Path,help='Carry forward matching admissions from a prior run')
    parser.add_argument('--attempt-offset',type=int,default=0,help='Skip previously used packing seeds')
    parser.add_argument('--snack-drawers',action='store_true',
                        help='Use existing closed drawers for shelf-stable packaged inventory only')
    parser.add_argument('--stack-notebook',action='store_true',
                        help='Use a measured notebook cover when no decoy fits on the drawer floor')
    parser.add_argument('--tight-remote',action='store_true')
    parser.add_argument('--stack-targets',action='store_true')
    parser.add_argument('--repair-occlusion',action='store_true')
    parser.add_argument('--wide-drawer-scale',type=float,default=1.)
    parser.add_argument('--task-id',action='append',help='Restrict to specific task IDs for a pilot')
    args=parser.parse_args()
    if not 1. <= args.wide_drawer_scale <= 1.5:
        parser.error('--wide-drawer-scale must be between 1 and 1.5')
    run,repo,output=(getattr(args,key).resolve() for key in ('run','repo','output'))
    lane=output/args.lane
    lane.mkdir(parents=True,exist_ok=True)
    lock=(lane/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='starting',started_at=time.time(),lane=args.lane)
    def progress(**updates):
        state.update(updates,updated_at=time.time())
        save(lane/'status.json',state)
    progress()
    baseline=run/'search_build_v3/release/physics_manifest.json'
    sources={row['id']:row for row in map(json.loads,(run/'migration_queue.jsonl').read_text().splitlines())}
    tasks=[]
    for row in load(baseline)['tasks']:
        if args.task_id and row['id'] not in args.task_id:
            continue
        contract=recipe(sources[row['id']])
        if contract['closed_container_required']==(args.lane=='drawer'):
            tasks.append((row,contract))
    tasks.sort(key=lambda item:(item[0]['layout_id'],item[0]['id']))
    if args.limit:
        # Pilot the smallest compatible target instances before broad packing.
        if args.lane=='drawer':
            tasks.sort(key=lambda item:(not any(p['target'] and p['asset']=='glasses_case' for p in item[0]['placed']),
                                        item[1]['target_count'],item[0]['layout_id']))
        tasks=tasks[:args.limit]
    save(lane/'queue.json',dict(baseline_sha256=digest(baseline),
         tasks=[dict(id=row['id'],layout_id=row['layout_id'],contract=contract) for row,contract in tasks]))
    progress(status='loading_usd' if args.prepare_only else 'starting_physx',total=len(tasks))
    if not args.prepare_only:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=64,height=64,multi_gpu=False,
                               limit_cpu_threads=8,extra_args=['--/log/file='+str(lane/'kit.log')]))
        import carb
        import omni.physx
        settings=carb.settings.get_settings()
        settings.set_bool('/physics/suppressReadback',False)
        settings.set_bool('/physics/updateToUsd',True)
        sim=omni.physx.get_physx_simulation_interface()
        interface=omni.physx.get_physx_interface()
        scene_query=omni.physx.get_physx_scene_query_interface()
    from pxr import Sdf, Usd
    from rab_assets import AssetNamespace, LayerBundle
    from rab_available_resources import namespace,provision
    from rab_build_search import load_assets,supports
    resources=run/'available_resources'
    mapping,_=provision(repo,run,resources)
    bundle=LayerBundle(namespace(resources),lane/'asset_layers')
    assets,errors=load_assets(mapping,repo,bundle,load(run/'pose_calibration/poses.json')['assets'])
    if args.tight_remote and 'remote_tv' in assets:
        assets['remote_tv']=mesh_footprint(assets['remote_tv'])
    save(lane/'asset_errors.json',errors)
    drawer_inputs=None
    if args.lane=='drawer':
        sys.path.insert(0,str(repo/'src'))
        ns=AssetNamespace.from_runtime(run/'Runtime-room-0.1.7')
        logical='/home/lingjun/robot_agent_arena/api-protocol/assets/house_scene_l1/demo_sim/'
        reference=reference_table_transform(ns.resolve(logical+'scene_room_baseline.usda'))
        slide=LayerBundle(ns,lane/'asset_layers').clone(ns.resolve(logical+'livingroom_drawer_slide.usda'))
        drawer_inputs=(reference,slide)
    counts=Counter()
    accepted=[]
    processed=0
    layout_cache={}
    for row,contract in tasks:
        ident=row['id']
        directory=lane/'cases'/ident
        final=directory/'result.json'
        previous=load(final) if final.exists() else {}
        prior_folder = args.prior_run.resolve()/args.lane/'cases'/ident if args.prior_run else None
        admitted = (args.admitted_from.resolve()/args.lane/'cases'/ident if args.admitted_from
                    else prior_folder if args.lane=='open' else None)
        if not previous and admitted and (admitted/'result.json').is_file():
            prior=load(admitted/'result.json')
            if prior['status']=='accepted_dense_initialization':
                ep=load(prior['episode'])
                receipt=load(prior['physics_receipt'])
                if (ep['id']==ident and ep['baseline_scene_sha256']==row['scene_sha256']
                        and ep['source_backup_sha256']==row['source_backup_sha256']
                        and digest(ep['scene'])==ep['scene_sha256']==receipt['scene_sha256']
                        and receipt['status']=='physx_initialization_pass'
                        and receipt.get('dense_checks',{}).get('passed')):
                    previous=prior
                    save(final,previous)
        if previous.get('status')=='accepted_dense_initialization':
            result=load(final)
        else:
            result=dict(id=ident,layout_id=row['layout_id'],status='pending_repair',attempts=[])
            progress(status='building',processed=processed,counts=dict(counts),current=ident)
            try:
                key=row['layout_id']
                if key not in layout_cache:
                    source=Sdf.Layer.FindOrOpen(row['scene'])
                    if len(source.subLayerPaths)!=1:
                        raise ValueError('Baseline scene does not have one shared room layer')
                    layout=Path(source.subLayerPaths[0])
                    if drawer_inputs:
                        path=lane/'asset_layers'/f'layout_{key:02d}_drawer.usda'
                        layout=compose_drawer(layout,path,*drawer_inputs,scale_xy=args.wide_drawer_scale)
                    stage=Usd.Stage.Open(str(layout))
                    fixtures,obstacles=supports(stage)
                    if drawer_inputs:
                        fixtures.update({zone:drawer_support(stage,path) for zone,path in DRAWERS.items()})
                    clearance=drawer_headroom(stage,fixtures) if drawer_inputs else None
                    layout_cache[key]=(layout,fixtures,obstacles,clearance)
                layout,fixtures,obstacles,clearance=layout_cache[key]
                reuse = archived_drawer_candidates(prior_folder) if prior_folder and args.lane=='drawer' else []
                for attempt in range(args.attempts+len(reuse)):
                    dest=directory/f'attempt_{attempt+1:02d}'
                    if (dest/'episode.json').exists():
                        episode=load(dest/'episode.json')
                        if digest(episode['scene'])!=episode['scene_sha256']:
                            raise ValueError('Prepared scene hash changed')
                    elif attempt<len(reuse):
                        episode=reuse_drawer_placement(reuse[attempt],layout,dest)
                        save(dest/'episode.json',episode)
                    else:
                        plan=pack(row,contract,layout,fixtures,obstacles,assets,
                                  attempt-len(reuse)+args.attempt_offset,clearance,args.snack_drawers,
                                  args.stack_notebook,args.repair_occlusion,args.stack_targets)
                        if plan is None:
                            result['attempts'].append(dict(attempt=attempt,status='no_dense_fit'))
                            continue
                        episode=author(row,plan,contract,dest)
                        if args.wide_drawer_scale != 1.:
                            episode['drawer_layout_variant'] = dict(table_scale_xy=args.wide_drawer_scale)
                        save(dest/'episode.json',episode)
                    if args.prepare_only:
                        result.update(status='prepared_for_native',episode=str(dest/'episode.json'))
                        break
                    progress(status='validating_physics',processed=processed,counts=dict(counts),
                             current=ident,attempt=attempt)
                    report=native_check(episode,sim,interface,scene_query)
                    report.update(scene_sha256=episode['scene_sha256'])
                    save(dest/'native_validation.json',report)
                    result['attempts'].append(dict(attempt=attempt,status=report['status']))
                    if report['status']=='physx_initialization_pass':
                        result.update(status='accepted_dense_initialization',episode=str(dest/'episode.json'),
                                      physics_receipt=str(dest/'native_validation.json'))
                        break
            except Exception as exc:
                result.update(error=str(exc),traceback=traceback.format_exc())
            save(final,result)
        counts[result['status']]+=1
        if result['status']=='accepted_dense_initialization':
            accepted.append(result)
        processed+=1
        progress(status='building',processed=processed,counts=dict(counts),current=ident)
        save(lane/'accepted.json',dict(tasks=accepted,scope='Dense initialization with occlusion or articulated drawer motion'))
    progress(status='finished',processed=processed,counts=dict(counts),current=None,finished_at=time.time())
    os._exit(0)


if __name__=='__main__':
    main()
