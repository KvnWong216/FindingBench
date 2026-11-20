"""Add native-sized storage to surveyed free floor; retain route clearance checks.

This is a construction proposal, not a physics or interaction certificate.
"""
import copy
import json
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, label
import generate_dense_pilot as g

MODELS = {
    'cabinet': ('bottom_cabinet', 'jrhgeu'),
    'drawer': ('bottom_cabinet_no_top', 'qohxjq'),
    'dresser': ('nightstand', 'uudzva'),
    'fridge': ('fridge', 'hivvdf'),
    'shelf': ('shelf', 'vanltx'),
    'desk': ('desk', 'tmluxa'),
    'table': ('breakfast_table', 'iaritq'),
    'counter': ('bottom_cabinet', 'jrhgeu'),
    'basket': ('breakfast_table', 'iaritq'),
    'bowl': ('breakfast_table', 'iaritq'),
}
_PROPOSALS = {}
_REPLAY_LAYOUTS = {}
_FOOTPRINT_MASKS = {}


def storage_floor_domain(layout, room):
    """Furniture centres use semantic floor cells, not robot standing cells."""
    floors = [(name, row) for name, row in layout.entities.items()
              if row['category'] == 'floors' and row.get('aabb')
              and abs(row['aabb'][1][2]) <= .3 + 1e-6]
    ids = tuple(i for i, name in layout.map.names.items() if name == room)
    cache = getattr(layout.map, '_storage_floor_points', None)
    if cache is None:
        cache = {}
        layout.map._storage_floor_points = cache
    if ids not in cache:
        cache[ids] = np.argwhere(np.isin(layout.map.room_ins, ids))
    return floors, cache[ids]


def screen_furniture_footprints(layout,points_xy,offsets,margin=.1):
    """Batch exact rectangle/AABB tests against measured scene obstacles.

    A long narrow shelf need not fit inside a wide empty circle. This remains
    conservative collision-box authoring, not a native physics certificate.
    """
    import hashlib
    points=np.asarray(points_xy,float).reshape(-1,2);offsets=np.asarray(offsets,float)
    boxes=list(layout.doc.get('footprint_boxes',[]))
    boxes.extend([f['aabb'][0][0],f['aabb'][0][1],f['aabb'][1][0],f['aabb'][1][1]]
                 for f in getattr(layout,'inserted_furniture',[]))
    boxes=np.asarray(boxes,float).reshape(-1,4)
    if offsets.shape!=(2,2) or np.any(offsets[1]<=offsets[0]) or not np.isfinite(offsets).all():raise ValueError('INVALID_FURNITURE_FOOTPRINT')
    if not np.isfinite(points).all() or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<boxes[:,:2]):raise ValueError('INVALID_AUTHORING_OBSTACLE_BOXES')
    key=hashlib.sha256(points.tobytes()+offsets.tobytes()+boxes.tobytes()+str(float(margin)).encode()).digest()
    if key in _FOOTPRINT_MASKS:return _FOOTPRINT_MASKS[key].copy()
    import torch
    if torch.cuda.is_available():
        centers=torch.as_tensor(points,device='cuda',dtype=torch.float64)
        bounds=torch.as_tensor(boxes,device='cuda',dtype=torch.float64)
        delta=torch.as_tensor(offsets,device='cuda',dtype=torch.float64)
        lower=centers+delta[0]-margin;upper=centers+delta[1]+margin
        valid=torch.ones(len(points),dtype=torch.bool,device='cuda')
        # Bound transient GPU memory even for large-house collision inventories.
        for start in range(0,len(boxes),1024):
            chunk=bounds[start:start+1024]
            overlap=(upper[:,None,:]>chunk[None,:,:2]).all(-1)&(lower[:,None,:]<chunk[None,:,2:]).all(-1)
            valid &= ~overlap.any(-1)
        result=valid.cpu().numpy()
    else:
        lower=points+offsets[0]-margin;upper=points+offsets[1]+margin;result=np.ones(len(points),bool)
        for start in range(0,len(boxes),1024):
            chunk=boxes[start:start+1024]
            result &= ~(((upper[:,None,:]>chunk[None,:,:2]).all(-1))&((lower[:,None,:]<chunk[None,:,2:]).all(-1))).any(-1)
    if len(_FOOTPRINT_MASKS)>=128:_FOOTPRINT_MASKS.pop(next(iter(_FOOTPRINT_MASKS)))
    _FOOTPRINT_MASKS[key]=result.copy()
    return result


def enrich_constructed_interiors(entities):
    """Attach size-only hints for native-size furniture, never world geometry."""
    from audit_dense_layouts import enrich_offline_interiors,load_geometry_index
    path=g.ROOT/'build/native_fillable_geometry/index.json'
    if not path.exists():return
    index=load_geometry_index(path)
    scene=dict(objects_info=dict(init_info={n:dict(args=dict(category=r['category'],
        model=r['model'],scale=[1,1,1])) for n,r in entities.items()}),
        state=dict(registry=dict(object_registry={n:dict(root_link=dict(ori=[0,0,0,1])) for n in entities})))
    enrich_offline_interiors(entities,scene,index,g.ASSETS/'objects')


def model_openable(category,model):
    """Match native Open compatibility: tags or verified non-fixed joints."""
    path=g.ASSETS/'objects'/category/model/'misc/metadata.json'
    if not path.exists():return False
    meta=json.loads(path.read_text())
    if any('openable' in tags for tags in meta.get('link_tags',{}).values()):return True
    index=g.ROOT/'build/native_fillable_geometry/index.json'
    if not index.exists():return False
    from audit_dense_layouts import load_geometry_index
    geometry=load_geometry_index(index).get('assets',{}).get(f'{category}/{model}')
    if not geometry:return False
    from audit_dense_layouts import asset_digest
    source=g.ASSETS/'objects'/category/model/'usd'/f'{model}.encrypted.usd'
    if (not source.exists() or asset_digest(source)!=geometry.get('encrypted_asset_sha256')
            or asset_digest(path)!=geometry.get('metadata_sha256')):return False
    return any(j['type'] in ('PhysicsPrismaticJoint','PhysicsRevoluteJoint') for j in geometry.get('joints',[]))


def replay_layout(layout, furniture):
    """Rebuild the exact obstacle geometry of a previously authored layout."""
    key=(layout.scene,getattr(layout,'_authoring_geometry_identity',None),json.dumps(furniture,sort_keys=True))
    if key in _REPLAY_LAYOUTS:return _REPLAY_LAYOUTS[key]
    result=copy.copy(layout);result.entities=dict(layout.entities);result.free=layout.free.copy()
    result.inserted_furniture=copy.deepcopy(furniture)
    n=layout.map.size
    for f in furniture:
        lo,hi=np.asarray(f['aabb'][0]),np.asarray(f['aabb'][1])
        result.entities[f['name']]=dict(category=f['category'],model=f['model'],aabb=f['aabb'],
            in_rooms=[f['room']],openable=model_openable(f['category'],f['model']),interiors=[])
        enrich_constructed_interiors({f['name']:result.entities[f['name']]})
        r0,c0=layout.map.to_map(lo[0]-layout.radius-.1,lo[1]-layout.radius-.1)
        r1,c1=layout.map.to_map(hi[0]+layout.radius+.1,hi[1]+layout.radius+.1)
        r0,c0=max(0,r0),max(0,c0);r1,c1=min(n,r1+1),min(n,c1+1)
        yy,xx=np.mgrid[r0:r1,c0:c1];xx=(xx-n/2)*.1;yy=(yy-n/2)*.1
        distance=np.hypot(np.maximum(np.maximum(lo[0]-xx,0),xx-hi[0]),np.maximum(np.maximum(lo[1]-yy,0),yy-hi[1]))
        result.free[r0:r1,c0:c1] &= distance>layout.radius
    result.labels,result.nlabels=label(result.free)
    result.room_clear=distance_transform_edt(result.free)*.1
    result.points=np.argwhere(result.free&(result.room_clear>=.2));result.cache={};result._constructed_route_cache={}
    _REPLAY_LAYOUTS[key]=result
    return result


def reuse_proposals(layouts, rows, bindings, roomtypes, kinds, targetkey, size):
    """Reuse measured furniture coordinates, not a task's answer or object layout."""
    import yaml
    scenes={x.scene:x for x in layouts};options=[]
    for row in rows:
        b=bindings.get(row['parent_task_id'],{})
        if (g.roomkind(row['source_room']),g.roomkind(row['destination_room']))!=roomtypes:continue
        if (b.get('source_kind'),b.get('destination_kind'))!=kinds:continue
        if row['scene'] not in scenes:continue
        spec=yaml.safe_load((g.ROOT/row['scenario']).read_text())
        layout=replay_layout(scenes[row['scene']],row.get('inserted_furniture',[]))
        placement=next(p for p in spec['placements'] if p['entity']==row['target_entities'][0])
        source=placement['receptacle'];relation=placement['relation'];link=placement.get('link')
        if kinds[0] in ('basket','bowl'):
            source=next(p['receptacle'] for p in spec['placements'] if p['entity']==source);relation='on_top';link=None
        dest=row['destination_receptacle']
        if kinds[1]=='basket':dest=next(p['receptacle'] for p in spec['placements'] if p['entity']==dest)
        if source not in layout.entities or dest not in layout.entities:continue
        entity=layout.entities[source];extent=np.subtract(entity['aabb'][1],entity['aabb'][0])
        if kinds[0] in ('cabinet','drawer','fridge') and not entity.get('openable'):continue
        if relation=='inside' and not g.fits_known_interior(entity,size):continue
        if np.any(np.sort(extent[:2])<np.sort(size[:2])+.005):continue
        src=dict(entity=source,relation=relation,extent=extent.tolist())
        if link:src['link']=link
        route=row['route'];valid=True
        for x,y in route['path_xy']:
            # to_world/to_map use float arithmetic; nearest grid index avoids
            # accidental truncation to the preceding cell for an exact point.
            rr=int(round(y/layout.map.resolution+layout.map.size/2));cc=int(round(x/layout.map.resolution+layout.map.size/2))
            if not (0<=rr<layout.map.size and 0<=cc<layout.map.size and layout.free[rr,cc]):valid=False;break
        if not valid:continue
        layout._constructed_route_cache[(source,row['source_room'],dest,row['destination_room'])]=route
        options.append((layout,row['source_room'],targetkey,src,row['destination_room'],dest,
                        float(np.linalg.norm(g.boxcenter(entity)[:2]-g.boxcenter(layout.entities[dest])[:2]))))
        if len(options)>=2:break
    return options


def model_for(kind, required_size=None, *, require_interior=False):
    if kind not in MODELS:return None
    cat,model=MODELS[kind]
    size=np.asarray(json.loads((g.ASSETS/'objects'/cat/model/'misc/metadata.json').read_text())['bbox_size'])
    needs_open=kind in ('cabinet','drawer','fridge')

    def admissible(candidate_model, extent):
        if needs_open and not model_openable(cat,candidate_model):return False
        if required_size is not None and np.any(np.sort(extent[:2])<np.sort(required_size[:2])+.06):return False
        if require_interior:
            if required_size is None:raise ValueError('Interior screening requires item dimensions')
            entity=dict(category=cat,model=candidate_model,interiors=[])
            enrich_constructed_interiors({'candidate':entity})
            if not g.fits_known_interior(entity,required_size):return False
        return True

    if not admissible(model,size):
        choices=[]
        for path in (g.ASSETS/'objects'/cat).glob('*/misc/metadata.json'):
            candidate_model=path.parent.parent.name
            candidate=np.asarray(json.loads(path.read_text())['bbox_size'])
            if (.35<candidate[2]<1.25 and max(candidate[:2])<2.0 and
                admissible(candidate_model,candidate)):
                choices.append((float(np.prod(candidate[:2])),candidate_model,candidate))
        if not choices:return None
        _,model,size=min(choices,key=lambda x:(x[0],x[1]))
    return cat,model,size


def footprint_fits_floor(xy, size, floor, opening_clearance_m=0., footprint_offsets_m=None):
    """Include an authoring opening sweep, not only the closed footprint."""
    lo,hi=np.asarray(floor['aabb'][0]),np.asarray(floor['aabb'][1])
    half=np.asarray(size[:2])/2+opening_clearance_m
    offsets=np.asarray(footprint_offsets_m) if footprint_offsets_m is not None else np.array([-half,half])
    xy=np.asarray(xy)
    return bool(np.all(xy+offsets[0]>lo[:2]) and np.all(xy+offsets[1]<hi[:2]))


def observed_opening_offsets(category,model):
    """Read hash-bound native opening geometry, never a feasibility label."""
    from audit_dense_layouts import asset_digest
    path=g.ROOT/'build/native_support_geometry/index.json'
    if not path.exists():return None
    record=json.loads(path.read_text()).get('assets',{}).get(category+'/'+model,{})
    bounds=record.get('opened_footprint_offsets_m')
    if bounds is None:return None
    asset=g.ASSETS/'objects'/category/model
    if (record.get('metadata_sha256')!=asset_digest(asset/'misc/metadata.json') or
            record.get('encrypted_asset_sha256')!=asset_digest(asset/'usd'/f'{model}.encrypted.usd')):return None
    bounds=np.asarray(bounds,float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[0]>=bounds[1]):return None
    return bounds


def augment(layout, room, kind, name, variant=0, required_size=None, *, require_interior=False,
            opening_clearance_m=0., footprint_offsets_m=None,yaw_degrees=0):
    """Return an independent layout and receptacle proposal, or None."""
    selected=model_for(kind,required_size,require_interior=require_interior)
    if selected is None:return None
    cat,model,size=selected
    if yaw_degrees not in (0,90):raise ValueError('Furniture authoring yaw must be 0 or 90 degrees')
    if yaw_degrees:
        # Only closed, rigid work surfaces are covered by this bounded pose
        # adapter. Moving-link interiors/opening sweeps need their own mapping.
        if (kind not in ('table','desk') or model_openable(cat,model) or opening_clearance_m
                or footprint_offsets_m is not None or observed_opening_offsets(cat,model) is not None):return None
        size=np.asarray(size)[[1,0,2]].copy()
    candidates = []
    # Furniture does not need a robot-sized free patch UNDER its centre.
    # Sample the semantic room's floor, then run the unchanged physical
    # footprint/opening tests. Robot approach and connectivity checks run
    # after insertion. This admits normal cabinet sites near a wall.
    floors, points = storage_floor_domain(layout, room)
    half=np.asarray(size[:2])/2+opening_clearance_m
    offsets=np.asarray(footprint_offsets_m) if footprint_offsets_m is not None else np.array([-half,half])
    measured=observed_opening_offsets(cat,model)
    if measured is not None:
        # The caller's zero-clearance fallback must not discard an opening
        # envelope already observed in native physics. Keep a small margin.
        offsets=np.array([np.minimum(offsets[0],measured[0]-.03),np.maximum(offsets[1],measured[1]+.03)])
    world=(points[:,[1,0]].astype(float)-layout.map.size/2)*layout.map.resolution
    admissible=screen_furniture_footprints(layout,world,offsets)
    for point_index,(rr, cc) in enumerate(points):
        if not admissible[point_index]:continue
        x,y = (float(value) for value in world[point_index])
        for floor, row in floors:
            lo,hi=np.array(row['aabb'][0]),np.array(row['aabb'][1])
            if footprint_fits_floor([x,y],size,row,opening_clearance_m,offsets):
                candidates.append((-layout.room_clear[rr,cc], int(rr),int(cc),x,y,floor,float(hi[2])))
                break
    if not candidates:
        return None
    candidates.sort()
    # Try spatially separated sites, not adjacent grid cells.
    sites=[]
    for point in candidates:
        if all(math.hypot(point[3]-p[3],point[4]-p[4])>.65 for p in sites):
            sites.append(point)
        if len(sites)>=8:break
    if variant>=len(sites):return None
    _,rr,cc,x,y,floor,z=sites[variant]
    result=copy.copy(layout)
    result.entities=dict(layout.entities)
    lo=[x-size[0]/2,y-size[1]/2,z];hi=[x+size[0]/2,y+size[1]/2,z+size[2]]
    result.entities[name]=dict(category=cat,model=model,aabb=[lo,hi],in_rooms=[room],
                               openable=model_openable(cat,model),interiors=[])
    enrich_constructed_interiors({name:result.entities[name]})
    result.inserted_furniture=list(getattr(layout,'inserted_furniture',[]))+[
        dict(name=name,category=cat,model=model,floor=floor,position_xy=[x,y],aabb=[lo,hi],room=room)]
    if yaw_degrees:
        result.inserted_furniture[-1]['orientation_hint_xyzw']=[0.,0.,math.sqrt(.5),math.sqrt(.5)]
    result.free=layout.free.copy()
    n=result.map.size
    r0,c0=result.map.to_map(lo[0]-result.radius-.1,lo[1]-result.radius-.1)
    r1,c1=result.map.to_map(hi[0]+result.radius+.1,hi[1]+result.radius+.1)
    r0,c0=max(0,r0),max(0,c0);r1,c1=min(n,r1+1),min(n,c1+1)
    yy,xx=np.mgrid[r0:r1,c0:c1];xx=(xx-n/2)*.1;yy=(yy-n/2)*.1
    distance=np.hypot(np.maximum(np.maximum(lo[0]-xx,0),xx-hi[0]),
                      np.maximum(np.maximum(lo[1]-yy,0),yy-hi[1]))
    result.free[r0:r1,c0:c1] &= distance>result.radius
    result.labels,result.nlabels=label(result.free)
    result.room_clear=distance_transform_edt(result.free)*.1
    result.points=np.argwhere(result.free & (result.room_clear>=.2))
    result.cache={}
    result._constructed_route_cache={}
    relation='inside' if kind in ('cabinet','drawer','fridge') else 'on_top'
    return result,dict(entity=name,relation=relation,extent=size.tolist())


def add_to_recipe(spec, row, layout):
    for furniture in getattr(layout,'inserted_furniture',[]):
        name=furniture['name']
        spec['objects'].append(dict(name=name,category=furniture['category'],model=furniture['model'],fixed_base=True))
        spec['placements'].append(dict(entity=name,relation='on_top',receptacle=furniture['floor'],position_xy=furniture['position_xy']))
        if furniture.get('orientation_hint_xyzw') is not None:
            spec['placements'][-1]['orientation_hint_xyzw']=list(furniture['orientation_hint_xyzw'])
    row['inserted_furniture']=getattr(layout,'inserted_furniture',[])
    row['object_count']=len(spec['objects'])


def propose(layouts, source_room_type, destination_room_type, source_kind, destination_kind,
            targetkey, size, destination_objects, scene_counts):
    """Try native furniture additions in the actual requested semantic rooms."""
    inside_source=source_kind in ('cabinet','drawer','fridge')
    sm=model_for(source_kind,size,require_interior=inside_source);dm=model_for(destination_kind,size)
    key=(source_room_type,destination_room_type,source_kind,destination_kind,
         sm[1] if sm else None,dm[1] if dm else None,tuple(x.scene for x in layouts))
    if key in _PROPOSALS:
        return [(o[0],o[1],targetkey,*o[3:]) for o in _PROPOSALS[key]]
    found=[]
    for layout in sorted(layouts,key=lambda x:(scene_counts[x.scene],x.scene)):
        rooms=sorted({r for o in layout.entities.values() for r in o.get('in_rooms',[])})
        sr=[r for r in rooms if g.roomkind(r)==source_room_type]
        dr=[r for r in rooms if g.roomkind(r)==destination_room_type]
        for room in sr:
            for droom in dr:
                if source_room_type==destination_room_type and room!=droom:continue
                for variant in range(4):
                    source_result=augment(layout,room,source_kind,'search_storage',variant,required_size=size,
                                          require_interior=inside_source)
                    if source_result is None:continue
                    with_source,src=source_result
                    # Gross external fit is only a necessary condition.
                    # Native interior geometry is checked during construction.
                    if np.any(np.sort(src['extent'][:2])<np.sort(size[:2])+.01):continue
                    destinations=destination_objects(with_source,droom,destination_kind)
                    for dest in destinations:
                        if dest==src['entity']:continue
                        route=with_source.route(src['entity'],room,dest,droom)
                        if route:
                            distance=float(np.linalg.norm(g.boxcenter(with_source.entities[src['entity']])[:2]-g.boxcenter(with_source.entities[dest])[:2]))
                            found.append((with_source,room,targetkey,src,droom,dest,distance))
                            break
                    if found and found[-1][0].scene==layout.scene:break
                    # Destination cabinet means its top, not another inside goal.
                    for dv in range(3):
                        dest_result=augment(with_source,droom,destination_kind,'delivery_surface',dv,required_size=size)
                        if dest_result is None:continue
                        both,dst=dest_result
                        route=both.route(src['entity'],room,dst['entity'],droom)
                        if route:
                            distance=float(np.linalg.norm(g.boxcenter(both.entities[src['entity']])[:2]-g.boxcenter(both.entities[dst['entity']])[:2]))
                            found.append((both,room,targetkey,src,droom,dst['entity'],distance))
                            break
                    if found and found[-1][0].scene==layout.scene:break
                if found and found[-1][0].scene==layout.scene:break
            if found and found[-1][0].scene==layout.scene:break
        if len(found)>=1:break
    _PROPOSALS[key]=found
    return found
