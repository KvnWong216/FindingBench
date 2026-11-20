"""Author a reviewable 100-episode dense-search catalogue from installed assets.

No generated recipe is certified playable by this compiler. Static route
screening, native physics, CHECK recognition, interaction clearance and human
completion are separate gates. Public instructions contain no solver hints.
"""
from pathlib import Path
from collections import Counter, deque
import argparse
import csv
import hashlib
import html
import json
import math
import heapq
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt, label
from rummagebench.authoring.tasks.room_map import RoomMap
from rummagebench.core.scenario import load_scenario
from survey_scene import scene_block

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets')
OUT=ROOT/'scenarios/dense_search_pilot'
# Semantic nouns are authored, not inferred from arbitrary asset identifiers.
ITEMS={
 'soy':('soy_sauce_bottle','saujjl','bottle of soy sauce'),
 'salt':('salt_bottle','wdpcmk','salt shaker'),
 'sugar':('jar_of_sugar','bholak','jar of sugar'),
 'mustard':('bottle_of_mustard','qbbqat','bottle of mustard'),
 'cola':('cola_bottle','oyqdtz','bottle of cola'),
 'juice':('bottle_of_orange_juice','edltwh','bottle of orange juice'),
 'apple_juice':('bottle_of_apple_juice','xvrbdy','bottle of apple juice'),
 'water':('bottle_of_water','atvnqy','bottle of water'),
 'controller':('gaming_controller','yofqbq','game controller'),
 'phone':('cell_phone','dbhfuh','phone'),
 'flashlight':('flashlight','jtrbkc','flashlight'),
 'wallet':('wallet','lagjae','wallet'),
 'glasses':('eyeglasses','gphhuu','eyeglasses'),
 'keys':('key_chain','gqbwfv','key ring'),
 'shampoo':('bottle_of_shampoo','stjjjm','bottle of shampoo'),
 'soap':('bar_soap','lyigsj','bar of soap'),
 'toothpaste':('tube_of_toothpaste','vsclsj','tube of toothpaste'),
 'brush':('hairbrush','bdylnb','hairbrush'),
 'comb':('comb','nybyjz','comb'),
 'apple':('apple','agveuv','apples'),
 'orange':('orange','nlygru','oranges'),
 'chili':('chili','agecro','chili pepper'),
 'carrot':('carrot','nktmff','carrot'),
 'pork':('pork_chop','bdwvkg','pork chop'),
 'yogurt':('cup_of_yogurt','kihdsj','cup of yogurt'),
 'multi':('bottle_of_medicine','egondf',"bottle labelled Women's Multi"),
 'coq10':('bottle_of_medicine','fmfwng','bottle labelled Co-Q10'),
 'krill':('bottle_of_medicine','kasbsy','bottle labelled Krill Oil'),
 'stapler':('stapler','kgnsas','stapler'),
 'pencil':('pencil','almpve','pencil'),
 'book':('paperback_book','xxknda','paperback book'),
 'notebook':('notebook','svisnb','notebook'),
 'tissues':('box_of_tissues','uglbjc','box of tissues'),
 'cookies':('box_of_cookies','qkzrdd','box of cookies'),
 'tea':('bottle_of_tea','iladfg','bottle of tea'),
 'vinegar':('bottle_of_vinegar','snzyfk','bottle of vinegar'),
 'pepper':('bottle_of_black_pepper','honise','pepper shaker'),
 'banana':('banana','verqwv','banana'),
 'bell_pepper':('bell_pepper','ggurxn','bell pepper'),
 'milk':('carton_of_milk',None,'carton of milk'),
 'face_cream':('bottle_of_face_cream',None,'bottle of face cream'),
}
# Family, room prefixes, source type, target pool, goal type, quota.
FAMILIES=[
 ('kitchen_cupboards',('kitchen','pantry_room'),'cabinet',['soy','salt','sugar','mustard'],'surface',14),
 ('living_cabinet',('living_room','television_room'),'drawer',['controller','phone','flashlight'],'surface',10),
 ('bedroom_drawers',('bedroom',),'drawer',['wallet','glasses','keys'],'surface',6),
 ('bedroom_basket',('bedroom',),'basket',['wallet','glasses'],'surface',4),
 ('labelled_supplements',('bedroom','bathroom'),'drawer',['multi','coq10','krill'],'surface',10),
 ('bathroom_drawers',('bathroom',),'drawer',['shampoo','soap','toothpaste','comb','brush'],'surface',12),
 ('drinks_shopping',('grocery_store','pantry_room','kitchen'),'shelf',['cola','juice','apple_juice'],'basket',8),
 ('fruit_bowl',('living_room','kitchen'),'bowl',['apple','orange'],'sink',8),
 ('vegetable_basket',('garden','kitchen','pantry_room'),'basket',['chili','carrot'],'surface',6),
 ('fridge_food',('kitchen',),'fridge',['pork','yogurt'],'sink',8),
 ('bathroom_basket',('bathroom',),'basket',['soap','shampoo'],'surface',6),
 ('breakfast_basket',('dining_room','kitchen'),'basket',['yogurt','milk'],'surface',4),
 ('desk_search',('private_office','bedroom'),'drawer',['stapler','pencil'],'surface',4),
]
POOLS={
 'kitchen_cupboards':['salt','sugar','mustard','pepper','vinegar','tea','water','cookies'],
 'living_cabinet':['controller','phone','flashlight','keys','notebook','book','wallet','glasses'],
 'bedroom_basket':['wallet','glasses','keys','phone','comb','notebook','pencil','tissues'],
 'bedroom_drawers':['wallet','glasses','keys','phone','comb','notebook','pencil','tissues'],
 'labelled_supplements':['multi','coq10','krill','face_cream','soap','toothpaste','comb'],
 'bathroom_drawers':['shampoo','soap','toothpaste','comb','brush','face_cream','multi','tissues'],
 'drinks_shopping':['cola','juice','apple_juice','water','tea','milk','cookies'],
 'fruit_bowl':['apple','orange','banana','chili','bell_pepper','carrot'],
 'vegetable_basket':['chili','bell_pepper','carrot','apple','orange'],
 'fridge_food':['pork','yogurt','milk','apple','orange','carrot','bell_pepper'],
 'bathroom_basket':['soap','shampoo','toothpaste','face_cream','comb','cookies','water'],
 'breakfast_basket':['yogurt','milk','juice','water','apple','cookies'],
 'desk_search':['stapler','pencil','notebook','book','keys','phone','comb'],
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def roomkind(room):return room.rsplit('_',1)[0]
def boxcenter(row):return (np.array(row['aabb'][0])+np.array(row['aabb'][1]))/2

def asset(key):
    cat,model,noun=ITEMS[key]
    if model is None:model=sorted((ASSETS/'objects'/cat).glob('*/misc/metadata.json'))[0].parent.parent.name
    p=ASSETS/'objects'/cat/model/'misc/metadata.json'
    m=json.loads(p.read_text())
    return dict(category=cat,model=model,noun=noun,size=m['bbox_size'],metadata_sha256=digest(p))


class Layout:
    """Conservative offline navigation proposals, never a live feasibility claim."""
    def __init__(self,path):
        self.path=path;self.scene=path.stem;self.doc=json.loads(path.read_text());self.entities=self.doc['entities']
        geometry_index=ROOT/'build/native_fillable_geometry/index.json'
        native_scene=ASSETS/'scenes'/self.scene/'json'/f'{self.scene}_best.json'
        if geometry_index.exists() and native_scene.exists():
            from audit_dense_layouts import enrich_offline_interiors,load_geometry_index
            enrich_offline_interiors(self.entities,json.loads(native_scene.read_text()),
                load_geometry_index(geometry_index),ASSETS/'objects')
        self.map=RoomMap.load(ASSETS/'scenes'/self.scene,ASSETS/'metadata/room_categories.txt',resolution=.1)
        # Circumscribed configured footprint, plus the runtime margin.
        common=yaml.safe_load((ROOT/'assets/tasks/vla_difficulty_v2/common.yaml').read_text())
        self.radius=float(common['robot'].get('base_half_extent',.4))*math.sqrt(2)+.03
        free=distance_transform_edt(self.map.room_ins>0)*.1 > self.radius
        n=self.map.size
        for x0,y0,x1,y1 in self.doc.get('footprint_boxes',[]):
            lo=self.map.to_map(x0-self.radius-.1,y0-self.radius-.1)
            hi=self.map.to_map(x1+self.radius+.1,y1+self.radius+.1)
            r0,c0=max(0,lo[0]),max(0,lo[1]);r1,c1=min(n,hi[0]+1),min(n,hi[1]+1)
            if r0>=r1 or c0>=c1:continue
            yy,xx=np.mgrid[r0:r1,c0:c1];xx=(xx-n/2)*.1;yy=(yy-n/2)*.1
            dd=np.hypot(np.maximum(np.maximum(x0-xx,0),xx-x1),np.maximum(np.maximum(y0-yy,0),yy-y1))
            free[r0:r1,c0:c1] &= dd>self.radius
        self.free=free;self.labels,self.nlabels=label(free);self.room_clear=distance_transform_edt(free)*.1
        self.points=np.argwhere(free & (self.room_clear>=.2));self.cache={}
    def approaches(self,name,room):
        k=(name,room)
        if k in self.cache:return self.cache[k]
        row=self.entities[name];lo,hi=np.array(row['aabb'][0]),np.array(row['aabb'][1]);out=[]
        ids=[i for i,n in self.map.names.items() if n==room]
        points=self.points[np.isin(self.map.room_ins[self.points[:,0],self.points[:,1]],ids)]
        xx=(points[:,1]-self.map.size/2)*self.map.resolution
        yy=(points[:,0]-self.map.size/2)*self.map.resolution
        dd=np.hypot(np.maximum(np.maximum(lo[0]-xx,0),xx-hi[0]),np.maximum(np.maximum(lo[1]-yy,0),yy-hi[1]))
        for i in np.flatnonzero((dd>=.55)&(dd<=1.05)):
            r,c=points[i];x,y=float(xx[i]),float(yy[i]);d=float(dd[i])
            center=(lo+hi)/2;out.append((d,(int(r),int(c)),(x,y),round(math.atan2(center[1]-y,center[0]-x)/(math.pi/2))*(math.pi/2)))
        out.sort(key=lambda a:(a[0],a[1]));self.cache[k]=out[:64];return self.cache[k]
    def route(self,source,sroom,destination,droom):
        max_steps=120 if sroom==droom else 200
        starts=self.approaches(destination,droom);ends=self.approaches(source,sroom)
        for _,a,xy,yaw in starts[:16]:
            choices=[p for p in ends if self.labels[p[1]]==self.labels[a]]
            if not choices:continue
            goal=min(choices,key=lambda p:math.dist(a,p[1]))[1]
            heuristic=lambda p:abs(p[0]-goal[0])+abs(p[1]-goal[1])
            q=[(heuristic(a),0,a)];parents={a:None};costs={a:0}
            while q:
                priority,cost,(r,c)=heapq.heappop(q)
                if cost!=costs.get((r,c)):continue
                if (r,c)==goal:break
                if priority>max_steps:break
                for nr,nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                    nxt=(nr,nc)
                    newcost=cost+1
                    if (0<=nr<self.map.size and 0<=nc<self.map.size and self.free[nxt]
                        and newcost<costs.get(nxt,float('inf')) and newcost+heuristic(nxt)<=max_steps):
                        parents[nxt]=(r,c);costs[nxt]=newcost
                        heapq.heappush(q,(newcost+heuristic(nxt),newcost,nxt))
            if goal not in parents:continue
            path=[];cur=goal
            while cur is not None:path.append(self.map.to_world(*cur));cur=parents[cur]
            path.reverse()
            if len(path)-1>max_steps:continue
            return dict(start=[*xy,yaw],path_xy=path,length_m=round((len(path)-1)*.1,2),
                        grid_m=.1,base_radius_m=self.radius,endpoint_free_patch_radius_m=.2,
                        source_approach_count=len(ends),destination_approach_count=len(starts),
                        provenance='frozen native survey; all-yaw base disc; live replay and arm IK still required')
        return None


def exposed_surface_entity(entities,name):
    """Resolve a fixture top to a unique measured slab directly covering it.

    OnTop requires contact with the named body. A separate countertop above a
    cabinet is a different body, even though both describe the same location.
    Ambiguous partial slabs are left unresolved; this is no feasibility claim.
    """
    row=entities[name]
    if not row.get('aabb') or not any(k in row.get('category','') for k in ('cabinet','dresser','desk','table')):return name
    lo,hi=np.asarray(row['aabb'],float);area=float(np.prod((hi-lo)[:2]))
    if area<=0:return name
    candidates=[]
    for other,slab in entities.items():
        if slab.get('category')!='countertop' or not slab.get('aabb'):continue
        if not set(row.get('in_rooms',[]))&set(slab.get('in_rooms',[])):continue
        a,b=np.asarray(slab['aabb'],float)
        if not -.005<=a[2]-hi[2]<=.08 or not 0<b[2]-a[2]<=.15:continue
        coverage=float(np.prod(np.maximum(0,np.minimum(hi[:2],b[:2])-np.maximum(lo[:2],a[:2]))))/area
        if coverage>=.8:candidates.append(other)
    return candidates[0] if len(candidates)==1 else name


def surfaces(layout,room,kind='surface'):
    result=[]
    for name,row in layout.entities.items():
        if room not in row.get('in_rooms',[]) or not row.get('aabb'):continue
        c=row['category'];lo,hi=np.array(row['aabb'][0]),np.array(row['aabb'][1]);extent=hi-lo
        if kind=='sink':ok=c in ('sink','furniture_sink')
        else:ok=any(s in c for s in ('table','countertop','desk','bottom_cabinet','dresser'))
        if ok and .35<=hi[2]<=1.25 and min(extent[:2])>.25:
            result.append(exposed_surface_entity(layout.entities,name) if kind!='sink' else name)
    return list(dict.fromkeys(result))


def fits_known_interior(row, size):
    """Require measured internal clearance before proposing inside placement.

    Upright dimensions with horizontal axis swap are the authoring policy;
    tilted fits are not searched to rescue an oversized candidate.
    """
    from audit_dense_layouts import authoring_volumes
    volumes=authoring_volumes(row)
    return bool(volumes) and any(
        extent[2] >= size[2]+.005 and
        np.all(np.sort(extent[:2]) >= np.sort(size[:2])+.005)
        for extent in volumes.values())

def interior_fit_rank(row, size):
    from audit_dense_layouts import authoring_volumes,upright_fit
    volumes=authoring_volumes(row)
    if not volumes:return 1
    return 0 if any(upright_fit(size,extent) for extent in volumes.values()) else 2


def sources(layout,room,kind,size):
    result=[]
    for name,row in layout.entities.items():
        if room not in row.get('in_rooms',[]) or not row.get('aabb'):continue
        cat=row['category']
        surface_categories = {
            'counter': {'countertop', 'bottom_cabinet', 'bottom_cabinet_no_top'},
            'table': {'table', 'breakfast_table', 'coffee_table', 'console_table', 'dining_table', 'end_table', 'folding_table', 'side_table'},
            'desk': {'desk'},
            'dresser': {'dresser', 'nightstand'},
        }
        if kind in surface_categories:
            if exposed_surface_entity(layout.entities,name)!=name:continue
            extent=np.array(row['aabb'][1])-np.array(row['aabb'][0])
            if cat in surface_categories[kind] and .35<=row['aabb'][1][2]<=1.25 and np.all(np.sort(extent[:2])>=np.sort(size[:2])+.04):
                result.append(dict(entity=name,relation='on_top',extent=extent.tolist()))
        elif kind=='drawer':
            if not any(c in cat for c in ('cabinet','desk','dresser')):continue
            if not fits_known_interior(row,size):continue
            for v in row.get('interiors',[]):
                ext=np.array(v['hi'])-np.array(v['lo'])
                if ext[2]>size[2]+.025 and np.all(np.sort(ext[:2])>np.sort(size[:2])+.04) and v['lo'][2]<1.2:
                    result.append(dict(entity=name,link=v['link'],relation='inside',extent=ext.tolist()))
        elif kind=='cabinet' and row.get('openable') and 'cabinet' in cat:
            extent=np.array(row['aabb'][1])-np.array(row['aabb'][0])
            if extent[2]>.4 and .3<row['aabb'][1][2]<2.0 and fits_known_interior(row,size):
                result.append(dict(entity=name,relation='inside',extent=extent.tolist()))
        elif kind=='fridge' and cat=='fridge':
            result.append(dict(entity=name,relation='inside',extent=(np.array(row['aabb'][1])-row['aabb'][0]).tolist()))
        elif kind=='shelf' and any(s in cat for s in ('shelf','bookcase','countertop')) and row['aabb'][1][2]<1.6:
            result.append(dict(entity=name,relation='on_top',extent=(np.array(row['aabb'][1])-row['aabb'][0]).tolist()))
    if kind in ('basket','bowl'):
        result=[dict(entity=n,relation='on_top',extent=(np.array(layout.entities[n]['aabb'][1])-layout.entities[n]['aabb'][0]).tolist()) for n in surfaces(layout,room)]
    return result


def refresh_review():
    manifest=json.loads((OUT/'manifest.json').read_text())
    rows=manifest['tasks']
    specs={}
    for row in rows:
        path=ROOT/row['scenario']
        specs[row['id']]=yaml.safe_load(path.read_text())
        row['instruction']=specs[row['id']]['instruction']
        result=ROOT/'runs/dense_search_pilot'/row['id']/'validation.json'
        report=json.loads(result.read_text()) if result.exists() else {}
        current=report.get('scenario_sha256')==digest(path)
        current=current and all((ROOT/name).is_file() and digest(ROOT/name)==sha for name,sha in report.get('runtime_sources',{}).items())
        current=current and (bool(report.get('fresh_native_setup')) or report.get('status')!='initialization_validated')
        row['physics_status']=report.get('status','pending') if current else 'pending'
        row['physics_error']=report.get('error','') if current else ''
    manifest['physics_counts']=dict(Counter(r['physics_status'] for r in rows))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fields=['id','family','scene','instruction','target_count','object_count','distractor_count','physics_status']
    with (OUT/'review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in fields)+'</tr>' for r in rows)
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Dense search review</title><style>body{font:15px sans-serif;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:9px;text-align:left}th{position:sticky;top:0;background:#eee}</style><h1>Dense household search: 100 authoring recipes</h1><p>Static routes are proposals from frozen surveys. Physics status is shown separately. Initialization does not certify interaction feasibility, visual identifiability, or successful human completion.</p><table><tr>'+''.join('<th>'+k+'</th>' for k in fields)+'</tr>'+table+'</table>')
    # Audit actual runtime YAML, not cached prompts or nominal family quotas.
    scene_counts=Counter(specs[r['id']]['scene']['model'] for r in rows)
    categories=Counter();models=Counter();target_models=Counter();target_categories=Counter()
    for r in rows:
        for obj in specs[r['id']]['objects']:
            categories[obj['category']]+=1
            models[obj['category']+'/'+obj['model']]+=1
            if obj['name'] in r['target_entities']:
                target_models[obj['category']+'/'+obj['model']]+=1
                target_categories[obj['category']]+=1
    prompts=Counter(r['instruction'] for r in rows)
    audit=dict(source=manifest['source'],task_count=len(rows),
        scene_counts=dict(scene_counts),
        source_room_type_counts=dict(Counter(r['source_room'].rsplit('_',1)[0] for r in rows)),
        family_counts=dict(Counter(r['family'] for r in rows)),
        unique_prompts=len(prompts),duplicate_prompts={k:v for k,v in prompts.items() if v>1},
        cross_room_tasks=sum(r['source_room']!=r['destination_room'] for r in rows),
        target_category_count=len(target_categories),target_model_count=len(target_models),
        target_models=dict(target_models),inserted_category_count=len(categories),
        inserted_model_count=len(models),inserted_models=dict(models),
        object_count_distribution=dict(Counter(len(specs[r['id']]['objects']) for r in rows)),
        multi_target_tasks=sum(r['target_count']>1 for r in rows),
        physics_counts=manifest['physics_counts'],substitutions=manifest['substitutions'],
        limitations=['Scene IDs do not certify independent houses.',
            'Asset counts cover authored objects, not background scene furniture.',
            'Static diversity does not demonstrate model generalization.',
            'Initialization does not certify task solvability or CHECK identifiability.',
            'No held-out scene/asset split is established by this catalogue.'])
    (OUT/'coverage.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'prompts.txt').write_text('\n\n'.join(r['id']+'\n'+r['instruction'] for r in rows)+'\n')
    summary=('<h2>Source coverage audit</h2><p>'
        +f"{len(rows)} instances; {len(scene_counts)} scene IDs; {len(prompts)} distinct instruction strings; "
        +f"{len(target_categories)} target categories / {len(target_models)} target models; "
        +f"{audit['cross_room_tasks']} cross-room tasks."
        +'</p><p>BEHAVIOR assets and scenes, newly authored tasks and layouts. '
        'Suitable for a pilot; cross-house and cross-asset generalization is not established. '
        'Scene IDs are not necessarily independent houses. '
        'Prompts below are copied from runtime scenario YAML, with no added solver hints.</p>'
        +'<details><summary>Full coverage counts and limitations</summary><pre>'
        +html.escape(json.dumps(audit,indent=2))+'</pre></details>')
    page=OUT/'review.html';page.write_text(page.read_text().replace('<table>',summary+'<table>',1))
    return manifest


def write_recipe(tid, option, family, sourcekind, goalkind, data, common, route, variant_index=0, target_keys=None):
    layout,room,target,src,droom,dest,dist=option
    folder=OUT/tid;folder.mkdir(parents=True,exist_ok=True)
    objs=[];placements=[];states={};aux=[]
    def add(cat,model,placement):
        name=f'item_{len(objs):02d}';objs.append(dict(name=name,category=cat,model=model));placements.append(dict(entity=name,**placement));return name
    source=src['entity'];relation=src['relation'];link=src.get('link')
    if sourcekind in ('basket','bowl'):
        # Native container, ordinary surface support. No resizing.
        cat,model=('shopping_basket','vsxhsv') if sourcekind=='basket' and family!='bedroom_basket' else ('wicker_basket','dgkhyn')
        source=add(cat,model,dict(relation='on_top',receptacle=source));aux.append(source);relation='inside';link=None
    goalrec=dest;goalrelation='inside' if goalkind=='sink' else 'on_top'
    if goalkind=='basket':
        goalrec=add('shopping_basket','vsxhsv',dict(relation='on_top',receptacle=dest));aux.append(goalrec);goalrelation='inside'
    def placement(rec=source,rel=relation,selected=link):
        d=dict(relation=rel,receptacle=rec)
        if selected:d['link']=selected
        return d
    target_keys=target_keys or [target]*(2 if family=='fruit_bowl' else 1)
    targets_n=len(target_keys)
    target_names=[add(data[k]['category'],data[k]['model'],placement()) for k in target_keys]
    target_models={data[k]['model'] for k in target_keys}
    target_categories={data[k]['category'] for k in target_keys}
    pool=[k for k in POOLS[family] if data[k]['model'] not in target_models and
          (data[k]['category'] not in target_categories or data[target]['category']=='bottle_of_medicine')]
    # Fit candidates individually. Joint packing is a separate native gate.
    if link:
        ext=np.array(src['extent']);pool=[k for k in pool if data[k]['size'][2]<ext[2]-.015 and
              np.all(np.sort(data[k]['size'][:2])<np.sort(ext[:2])-.025)]
    if len(pool)<2:raise ValueError(f'Insufficient distinguishable distractors: {tid}')
    alternatives=[]
    if sourcekind=='cabinet':
        alternatives=[s for s in sources(layout,room,sourcekind,data[target]['size']) if s['entity']!=src['entity'] and
            np.linalg.norm(boxcenter(layout.entities[s['entity']])[:2]-boxcenter(layout.entities[src['entity']])[:2])<3]
    area_used=sum(float(np.prod(data[k]['size'][:2])) for k in target_keys)
    area_limit=.65*float(np.prod(src['extent'][:2])) if link else float('inf')
    for i in range(min(8,12-len(objs))):
        recipient=alternatives[0]['entity'] if alternatives and i>=5 else source
        fitting=[k for k in pool if area_used+float(np.prod(data[k]['size'][:2]))<=area_limit and
                 (relation!='inside' or recipient not in layout.entities or
                  fits_known_interior(layout.entities[recipient],data[k]['size']))]
        if not fitting:break
        if relation=='inside' and recipient in layout.entities:
            # Prefer individually fitting optional clutter when native geometry
            # is known. Unknown geometry remains eligible for native sampling.
            ranked=[k for k in fitting if interior_fit_rank(layout.entities[recipient],data[k]['size'])==0]
            if ranked:fitting=ranked
        k=fitting[(i+variant_index)%len(fitting)];item=data[k]
        area_used+=float(np.prod(item['size'][:2]))
        pl=placement()
        if alternatives and i>=5:
            alt=alternatives[0];pl=dict(relation='inside',receptacle=alt['entity']);states[alt['entity']]={'open':False}
        add(item['category'],item['model'],pl)
    if layout.entities[src['entity']].get('openable') and relation=='inside' and not aux:states[src['entity']]={'open':False}
    # A public, unique local destination is fixed by the initial facing
    # direction; source identities and evaluation hints stay private.
    destcat=layout.entities[dest]['category']
    noun=('counter' if 'counter' in destcat else 'desk' if 'desk' in destcat else 'cabinet' if 'cabinet' in destcat else 'table')
    where=roomkind(room).replace('_',' ')
    room_phrase=('this ' if room==droom else 'the ')+where
    if sourcekind=='drawer':search=f'the drawers in {room_phrase}'
    elif sourcekind=='cabinet':search=f'the cupboards in {room_phrase}'
    elif sourcekind=='fridge':search='the kitchen fridge'
    elif sourcekind=='bowl':search=f'the fruit basket in {room_phrase}'
    elif sourcekind=='basket':search=f'the basket in {room_phrase}' if family=='bedroom_basket' else f'the shopping basket in {room_phrase}'
    else:search=f'the shelves in {room_phrase}'
    if goalkind=='sink':destination='the kitchen sink in front of you'
    elif goalkind=='basket':destination=f'the shopping basket on the {noun} in front of you'
    elif goalkind=='bathroom':destination=f'the {noun} in front of you in the bathroom'
    else:destination=f'the {noun} in front of you'
    subject=('all the '+data[target]['noun']) if targets_n>1 else ('the '+data[target]['noun'])
    pronoun='them' if targets_n>1 or target in ('glasses','keys') else 'it'
    prep='in' if goalrelation=='inside' else 'on'
    instruction=f'Find {subject} in {search} and put {pronoun} {prep} {destination}.'
    x,y,yaw=route['start']
    spec=dict(id=tid,instruction=instruction,scene=scene_block(layout.scene),
        robot={**common['robot'],'init_anchor':'start'},anchors={'start':dict(position=[x,y,.0053],orientation=[0,0,math.sin(yaw/2),math.cos(yaw/2)])},
        target=dict(entity=target_names[0],category=data[target]['category']),objects=objs,placements=placements,initial_states=states,
        goal=[dict(kind=goalrelation,entity=t,receptacle=goalrec) for t in target_names],
        termination={**common['termination'],'max_planning_steps':200},feasibility=common['feasibility'],
        skills=['MOVE_TO','TURN','VIEW','CHECK','OPEN','CLOSE','GRASP','PLACE','REPORT_DONE'],expose_available_skills=False)
    path=folder/'scenario.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False));load_scenario(path)
    audit=dict(id=tid,family=family,scene=layout.scene,scenario=str(path.relative_to(ROOT)),scenario_sha256=digest(path),
        instruction=instruction,target_key=target,target_entities=target_names,target_count=targets_n,
        source_room=room,destination_room=droom,source_receptacle=source,destination_receptacle=goalrec,
        object_count=len(objs),distractor_count=len(objs)-len(aux)-targets_n,
        source_survey_sha256=digest(layout.path),route=route,
        status='static_screened_physics_pending',human_reference_steps=None,
        required_gates=['fresh_native_initialization','stable_dense_layout','target_CHECK_identifiability',
            'all_yaw_live_navigation_and_tolerant_endpoints','reachable_open_grasp_place','initial_goal_false','human_completion'],
        restoration_entities=[o['name'] for o in objs if o['name'] not in target_names]+list(states),
        restoration_scoring='Separate continuous diagnostic; no restoration instruction or hard success predicate')
    (folder/'authoring.private.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def review_semantic_catalogue():
    """Audit Astra-authored intents; asset matching is not scene instantiation."""
    import re
    from scipy.sparse import csr_matrix
    root=ROOT/'build/dense_search_semantic_1k'
    inventory={r['category']:r['models'] for r in json.loads((root/'available_assets.json').read_text())}
    rooms=json.loads((root/'available_rooms.json').read_text())
    aliases={'office':'private_office','study':'private_office','supermarket':'grocery_store',
             'living':'living_room','dining':'dining_room','utility':'utility_room','storage':'storage_room'}
    rows=[];errors=[]
    required={'id','tier','domain','need','instruction','source_room_type','destination_room_type',
              'target_description','candidate_asset_categories','search_challenge','semantic_signature','asset_gap',
              'intent_family_id','intent_family_description'}
    authored_paths=[root/name for name in ('astra_food.json','astra_personal.json','astra_utility.json') if (root/name).exists()]
    for path in authored_paths:
        authored=json.loads(path.read_text())
        if not isinstance(authored,list):raise ValueError(f'{path}: expected list')
        for source in authored:
            row=dict(source);missing=required-set(row)
            if missing:raise ValueError(f"{path}: {row.get('id')} missing {missing}")
            row['author_file']=path.name
            row['source_room_type']=aliases.get(row['source_room_type'],row['source_room_type'])
            row['destination_room_type']=aliases.get(row['destination_room_type'],row['destination_room_type'])
            same=row['source_room_type']==row['destination_room_type']
            # Type inequality establishes different room types, not a verified route.
            row['structural_issues']=[]
            if row['tier'] not in ('short','long'):row['structural_issues'].append('invalid_tier')
            if row['tier']=='short' and not same:row['structural_issues'].append('short_crosses_room_types')
            if row['tier']=='long' and same:row['structural_issues'].append('long_room_instances_need_disambiguation')
            cats=row['candidate_asset_categories']
            row['missing_categories']=[c for c in cats if c not in inventory]
            row['asset_candidates']={c:inventory[c] for c in cats if c in inventory}
            row['candidate_scenes']=[scene for scene,names in rooms.items()
                if {row['source_room_type'],row['destination_room_type']} <= {roomkind(n) for n in names}]
            row['grounding_status']='semantic_candidate_not_instantiated'
            row['difficulty_status']='ungraded_no_human_oracle'
            row['grounding_gates']=['formal_goal_translation_including_sets_alternatives_and_order','target_and_distractor_visual_evidence','unambiguous_source_and_destination',
                'stable_dense_placement','tolerant_connected_route','reachable_interactions','human_completion']
            if row['structural_issues']:errors.append({'id':row['id'],'issues':row['structural_issues']})
            rows.append(row)
    if not rows:raise ValueError('No Astra-authored task files yet')
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task IDs')
    normalize=lambda t:' '.join(re.findall(r'[a-z0-9]+',t.lower()))
    duplicate_groups=[]
    for field in ('instruction','semantic_signature'):
        groups={}
        for r in rows:groups.setdefault(normalize(r[field]),[]).append(r['id'])
        duplicate_groups.extend({'field':field,'ids':v} for v in groups.values() if len(v)>1)
    # Lexical retrieval only: flags candidate pairs for semantic editorial review.
    stop=set('the a an and or to in on from of for it them my me this that please put bring find get take with at into by is are'.split())
    docs=[];df=Counter()
    for r in rows:
        words=[w for w in normalize(r['instruction']+' '+r['search_challenge']).split() if w not in stop]
        counts=Counter(words+[' '.join(words[i:i+2]) for i in range(len(words)-1)])
        docs.append(counts);df.update(counts.keys())
    vocab={w:i for i,w in enumerate(sorted(df))};ii=[];jj=[];vv=[]
    for i,doc in enumerate(docs):
        weights={vocab[w]:(1+math.log(n))*math.log(1+len(rows)/df[w]) for w,n in doc.items()}
        norm=math.sqrt(sum(w*w for w in weights.values())) or 1
        for j,w in weights.items():ii.append(i);jj.append(j);vv.append(w/norm)
    mat=csr_matrix((vv,(ii,jj)),shape=(len(rows),len(vocab)));sim=(mat@mat.T).tocoo()
    pairs=[{'left':rows[i]['id'],'right':rows[j]['id'],'lexical_similarity':round(float(v),4)}
           for i,j,v in zip(sim.row,sim.col,sim.data) if i<j and v>=.30]
    pairs.sort(key=lambda p:-p['lexical_similarity'])
    family_counts=Counter(r['intent_family_id'] for r in rows)
    rejected=[]
    for path in sorted(root.glob('*_rejected_near_duplicates.json')):
        archived=json.loads(path.read_text())
        if isinstance(archived,dict):archived=archived.get('removed_rows',[])
        if isinstance(archived,list):rejected.extend(archived)
    excluded={}
    for r in rejected:
        task=r.get('task',r)
        excluded[task['id']]={'task':task,'reason':r.get('reason',r.get('rejection_reason',r.get('review_rejection_reason')))}
    for path in sorted(root.glob('*_editorial_resolution.json')):
        for decision in json.loads(path.read_text()).get('decisions',[]):
            if decision.get('excluded_record'):
                task=decision['excluded_record']
                excluded[task['id']]={'task':task,'reason':decision.get('reason')}
    if set(excluded)&set(ids):raise ValueError('Excluded task remains in retained catalogue')
    (root/'excluded_tasks.json').write_text(json.dumps(list(excluded.values()),indent=2)+'\n')
    prototype_audit_path=root/'prototype_semantic_audit.json'
    prototype_audit=json.loads(prototype_audit_path.read_text()) if prototype_audit_path.exists() else {}
    prototypes=[]
    for fid,count in sorted(family_counts.items()):
        members=[r for r in rows if r['intent_family_id']==fid]
        descriptions=sorted({r['intent_family_description'] for r in members})
        prototypes.append(dict(id=fid,descriptions=descriptions,task_count=count,
            task_ids=[r['id'] for r in members],review_status='semantic_review_pending'))
    (root/'prototypes.json').write_text(json.dumps(prototypes,indent=2)+'\n')
    with (root/'prototypes.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['prototype_id','description','task_count','task_ids','review_status'])
        for p in prototypes:
            writer.writerow([p['id'],' / '.join(p['descriptions']),p['task_count'],
                             ', '.join(p['task_ids']),p['review_status']])
    summary=dict(author_model='gpt-6-astra',
        author_files={p.name:digest(p) for p in authored_paths},
        prototype_count=len(family_counts),prototype_counts=dict(family_counts),
        excluded_candidates_count=len(excluded),
        global_prototype_threshold_status=prototype_audit.get('meets_200_prototypes','not_reviewed'),
        oversized_prototypes={k:v for k,v in family_counts.items() if v>5},
        prototype_review_status='Requires semantic review; different IDs are not proof of distinct intent.',
        requested=1000,initial_draft_count=1000,retained_candidates=len(rows),
        requested_tiers={'short':400,'long':600},authored=len(rows),
        tiers=dict(Counter(r['tier'] for r in rows)),domains=dict(Counter(r['domain'] for r in rows)),
        source_rooms=dict(Counter(r['source_room_type'] for r in rows)),
        destination_rooms=dict(Counter(r['destination_room_type'] for r in rows)),
        exact_duplicate_groups=duplicate_groups,structural_issues=errors,
        explicit_asset_gaps=sum(bool(r['asset_gap']) for r in rows),
        missing_category_tasks=sum(bool(r['missing_categories']) or not r['asset_candidates'] for r in rows),
        no_surveyed_room_pair=sum(not r['candidate_scenes'] for r in rows),
        lexical_review_pair_count=len(pairs),instantiated=0,physically_validated=0,
        caveats=['Language-authored candidates, not runnable simulation tasks.',
            'Lexical similarity is a review aid, not proof of semantic uniqueness.',
            'Native category existence does not verify color, label, function, size or recognizability.',
            'Scene room co-occurrence does not verify navigation or manipulation feasibility.',
            'Short/long is task structure; difficulty awaits human reference trajectories.'])
    (root/'catalogue.json').write_text(json.dumps({'summary':summary,'tasks':rows},indent=2)+'\n')
    (root/'similarity_review.json').write_text(json.dumps(pairs,indent=2)+'\n')
    fields=['id','tier','domain','intent_family_id','instruction','need','source_room_type','destination_room_type','target_description','search_challenge','semantic_signature','asset_gap']
    with (root/'review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (root/'prompts.txt').write_text('\n\n'.join(r['id']+' ['+r['tier']+']\n'+r['instruction'] for r in rows)+'\n')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k] or ''))+'</td>' for k in fields)+'</tr>' for r in rows)
    (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Astra household search candidates</title>'
        '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd;min-width:110px}th{position:sticky;top:0;background:#eee}td:nth-child(5){min-width:330px}</style>'
        '<h1>Astra-authored household search candidates</h1><p>Semantic draft. Not instantiated or validated in simulation. Only the instruction is public agent input; other columns are private authoring-review metadata.</p><details><summary>Audit counts and caveats</summary><pre>'
        +html.escape(json.dumps(summary,indent=2))+'</pre></details><table><tr>'+''.join('<th>'+k+'</th>' for k in fields)+'</tr>'+table+'</table>')
    print(json.dumps(summary,indent=2))


def inventory_native_assets():
    """Enumerate native model evidence, without declaring manipulation feasibility."""
    from bddl.object_taxonomy import ObjectTaxonomy
    taxonomy=ObjectTaxonomy();root=ROOT/'build/dense_search_semantic_1k'
    nonsample=set((ASSETS/'metadata/non_sampleable_categories.txt').read_text().splitlines())
    categories=[];models=[];errors=[];survey_usage={};native_scene_usage={};native_scene_count=0
    for path in sorted((ASSETS/'scenes').glob('*/json/*_best.json')):
        native_scene_count+=1
        for entity,row in json.loads(path.read_text()).get('objects_info',{}).get('init_info',{}).items():
            args=row.get('args',{});key=(args.get('category'),args.get('model'))
            native_scene_usage.setdefault(key,[]).append(dict(scene=path.parent.parent.name,
                entity=entity,rooms=args.get('in_rooms',[]),scale=args.get('scale'),source=str(path.relative_to(ASSETS))))
    for path in sorted((ROOT/'runs/vla_difficulty_v2/_survey').glob('*.json')):
        if '.candidates.' in path.name:continue
        for entity,row in json.loads(path.read_text()).get('entities',{}).items():
            key=(row.get('category'),row.get('model'))
            survey_usage.setdefault(key,[]).append({'scene':path.stem,'entity':entity,'rooms':row.get('in_rooms',[])})
    for catdir in sorted((ASSETS/'objects').iterdir()):
        if not catdir.is_dir():continue
        syn=taxonomy.get_synset_from_category(catdir.name)
        abilities=sorted(taxonomy.get_abilities(syn)) if syn else []
        lineage=([syn]+taxonomy.get_ancestors(syn)) if syn else []
        static_prior=sorted(set(lineage)&nonsample)
        names=[];extents=[]
        for meta in sorted(catdir.glob('*/misc/metadata.json')):
            try:
                doc=json.loads(meta.read_text());size=doc.get('bbox_size')
                valid_bbox=bool(size and len(size)==3 and all(math.isfinite(float(v)) and float(v)>0 for v in size))
                if not valid_bbox:errors.append({'metadata':str(meta),'error':'missing/nonpositive bbox; retained in inventory'})
                model=meta.parent.parent.name;names.append(model)
                if valid_bbox:extents.append(size)
                modeldir=meta.parent.parent
                textures=sorted(str(p.relative_to(ASSETS)) for p in (modeldir/'material').glob('*diffuse*'))
                usd=sorted(str(p.relative_to(ASSETS)) for p in (modeldir/'usd').glob('*.usd'))
                physics=('nonrigid' if any(a in abilities for a in ('cloth','softBody','deformable')) else
                         'rigid' if 'rigidBody' in abilities else 'unknown')
                models.append(dict(category=catdir.name,model=model,asset_id=catdir.name+'/'+model,
                    synset=syn,abilities=abilities,physics_prior=physics,
                    bbox_size_m=size,bbox_valid=valid_bbox,metadata_path=str(meta.relative_to(ASSETS)),metadata_sha256=digest(meta),
                    diffuse_textures=textures,usd_files=usd,native_non_sampleable_ancestors=static_prior,
                    observed_in_surveys=survey_usage.get((catdir.name,model),[]),
                    referenced_in_native_scenes=native_scene_usage.get((catdir.name,model),[]),
                    geometry_and_labels_verified=False,grasp_feasibility='unverified'))
            except Exception as e:errors.append({'metadata':str(meta),'error':str(e)})
        if names:
            categories.append(dict(category=catdir.name,synset=syn,abilities=abilities,models=names,
                model_count=len(names),bbox_min_m=np.min(extents,axis=0).tolist() if extents else None,
                bbox_max_m=np.max(extents,axis=0).tolist() if extents else None,native_non_sampleable_ancestors=static_prior))
    report=dict(asset_root=str(ASSETS),category_count=len(categories),model_count=len(models),native_scene_count=native_scene_count,
        physics_counts=dict(Counter(m['physics_prior'] for m in models)),errors=errors,
        note='Inventory evidence only. Native non-sampleable taxonomy is a prior, not a graspability verdict. Native JSON references and survey occurrence are not new-task deployment.')
    (root/'asset_inventory.json').write_text(json.dumps({'summary':report,'categories':categories,'models':models},indent=2)+'\n')
    (root/'asset_categories.json').write_text(json.dumps(categories,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def review_asset_coverage():
    """Allocate native model candidates within authored semantic bindings, not scenes."""
    root=ROOT/'build/dense_search_semantic_1k'
    inventory=json.loads((root/'asset_inventory.json').read_text())
    models={m['asset_id']:m for m in inventory['models']};bycategory={}
    for m in models.values():bycategory.setdefault(m['category'],[]).append(m)
    catalogue=json.loads((root/'catalogue.json').read_text())
    for name,sha in catalogue['summary'].get('author_files',{}).items():
        if digest(root/name)!=sha:raise ValueError('Catalogue stale; run --semantic-review before --asset-coverage')
    tasks=catalogue['tasks'];taskmap={r['id']:r for r in tasks}
    bindings={};alignment_sources={};errors=[];role_catalogue=[]
    for domain in ('food','personal','utility'):
        path=root/(domain+'_asset_alignment.json')
        if not path.exists():continue
        doc=json.loads(path.read_text());alignment_sources[path.name]=digest(path)
        current=root/('astra_'+domain+'.json')
        if doc.get('source_after_sha256')!=digest(current):
            errors.append({'file':path.name,'error':'source_hash_mismatch'})
            continue
        role_catalogue.extend(doc.get('extra_category_roles',[]))
        for b in doc.get('task_bindings',[]):
            if b['id'] in bindings:raise ValueError('Duplicate binding '+b['id'])
            bindings[b['id']]=b
    uses={key:[] for key in models};counts=Counter();categorycounts=Counter();plans=[]
    def choose(category,constraints,forbidden=(),preferences=None):
        if category not in bycategory:return None
        allowed=constraints.get(category)
        pool=[m for m in bycategory[category] if m['bbox_valid'] and m['usd_files']
              and m['asset_id'] not in forbidden and (allowed is None or m['model'] in allowed)]
        if not pool:return None
        # Preference only; this is not a claim that smaller means graspable.
        preferences=preferences or {}
        preferred=preferences.get('preferred_models_by_category',preferences).get(category,[])
        return min(pool,key=lambda m:(counts[m['asset_id']],0 if m['model'] in preferred else 1,max(m['bbox_size_m']),m['model']))
    for task in tasks:
        b=bindings.get(task['id'])
        if b is None:continue
        selections=[]
        for role,field,constraints in (
            ('target','target_categories',b.get('target_model_constraints',{})),
            ('reference','reference_categories',b.get('reference_model_constraints',{})),
            ('receptacle','receptacle_categories',b.get('receptacle_model_constraints',{})),
            ('distractor','distractor_categories',b.get('distractor_model_constraints',{}))):
            categories=list(dict.fromkeys(b.get(field,[])))
            if role=='reference':categories=list(dict.fromkeys(categories+list(constraints)))
            if role=='distractor':
                categories=sorted(categories,key=lambda c:(categorycounts[c],c))[:8]
            for category in categories:
                forbidden={s['asset_id'] for s in selections if s['role']=='target'} if role=='distractor' else set()
                m=choose(category,constraints,forbidden,b.get(role+'_model_preferences',{}))
                if m is None:
                    available={m['asset_id'] for m in bycategory.get(category,[]) if m['bbox_valid']}
                    reason=('only_target_identical_model_available' if available and available<=forbidden else 'no_model_candidate')
                    errors.append({'id':task['id'],'role':role,'category':category,'error':reason})
                    continue
                aid=m['asset_id'];counts[aid]+=1;categorycounts[category]+=1
                use={'task_id':task['id'],'role':role};uses[aid].append(use)
                selections.append(dict(asset_id=aid,role=role,bbox_size_m=m['bbox_size_m'],
                    metadata_sha256=m['metadata_sha256'],physics_prior=m['physics_prior'],
                    negative_example_validation_required=role=='distractor' and category in b.get('target_categories',[]),
                    representation_review_required=m['physics_prior']!='rigid',
                    large_model_grasp_review_required=role=='target' and max(m['bbox_size_m'])>.60,
                    native_non_sampleable_ancestors=m['native_non_sampleable_ancestors']))
        plans.append(dict(id=task['id'],instruction_sha256=hashlib.sha256(task['instruction'].encode()).hexdigest(),
            selections=selections,unresolved_requirements=b.get('unresolved_requirements',[]),
            criterion_evidence=b.get('criterion_evidence',[]),
            status='model_allocation_proposal_not_instantiated',
            note='Each entry is a category/model candidate, not an object count or a compiled goal. References, quantities, fit, visibility and placements still need validation.'))
    # Alternate asset instances reuse a reviewed task; they are NOT new prompts
    # or additional semantic tasks. Model-specific restrictions remain binding.
    variant_candidates={}
    for tid,b in bindings.items():
        if tid not in taskmap:continue
        for role in ('target','distractor','reference','receptacle'):
            constraints=b.get(role+'_model_constraints',{})
            cats=list(dict.fromkeys(b.get(role+'_categories',[])+list(constraints)))
            for cat in cats:
                if role=='distractor' and cat in b.get('target_categories',[]):continue
                for m in bycategory.get(cat,[]):
                    if (m['bbox_valid'] and m['usd_files'] and
                        (cat not in constraints or m['model'] in constraints[cat])):
                        variant_candidates.setdefault(m['asset_id'],[]).append((tid,role))
    variants=[];variant_uses={};variant_load=Counter()
    for aid in sorted(variant_candidates):
        if uses[aid]:continue
        tid,role=min(variant_candidates[aid],key=lambda v:(variant_load[v[0]],v))
        variant_load[tid]+=1
        entry=dict(parent_task_id=tid,asset_id=aid,role=role,
            status='alternate_asset_instance_proposal_not_instantiated',
            constraints=bindings[tid].get('unresolved_requirements',[]),
            note='Reuses the parent instruction. Not a new semantic task; visual criteria, references, fit and physics must be revalidated.')
        variants.append(entry);variant_uses[aid]=entry
    (root/'asset_variant_plan.json').write_text(json.dumps(variants,indent=2)+'\n')
    allowedroles={}
    for row in role_catalogue:
        allowedroles.setdefault(row['category'],[]).append(row)
    contexts=[];context_uses={};context_load=Counter();context_issues=[]
    for aid,m in sorted(models.items()):
        if uses[aid] or aid in variant_uses or not m['bbox_valid']:continue
        candidates=[]
        for r in allowedroles.get(m['category'],[]):
            if r['role'].startswith('excluded') or r.get('proposal_status')=='excluded_or_requires_parent_assembly':continue
            for tid in r.get('recommended_existing_task_ids',[]):
                t=taskmap.get(tid)
                if t is None:
                    context_issues.append({'category':m['category'],'task_id':tid,'error':'unknown_recommended_task'});continue
                rooms=set(r.get('appropriate_rooms',[]))&{t['source_room_type'],t['destination_room_type']}
                if not rooms:continue
                if m['category'] in bindings.get(tid,{}).get('target_categories',[]):continue
                candidates.append((tid,r,sorted(rooms)))
        if not candidates:continue
        tid,r,rooms=min(candidates,key=lambda c:(context_load[c[0]],c[0],c[1]['role']))
        context_load[tid]+=1
        entry=dict(asset_id=aid,parent_task_id=tid,role=r['role'],room_candidates=rooms,
            reason=r['reason'],status='context_variant_proposal_not_instantiated',
            note='An alternative context instance, not an instruction to insert all proposed models together. Native size, assembly, static/rigid representation, fit, visibility and nonblocking placement require validation.')
        contexts.append(entry);context_uses[aid]=entry
    (root/'asset_context_plan.json').write_text(json.dumps(contexts,indent=2)+'\n')
    ledger=[]
    for aid,m in models.items():
        if uses[aid]:status='allocated_proposal'
        elif aid in variant_uses:status='alternate_instance_proposal'
        elif aid in context_uses:status='context_instance_proposal'
        elif not m['bbox_valid']:status='geometry_metadata_review'
        elif m['observed_in_surveys']:status='survey_observed_not_allocated'
        elif m.get('referenced_in_native_scenes'):status='native_scene_reference_not_allocated'
        elif any(not r['role'].startswith('excluded') and r.get('proposal_status')!='excluded_or_requires_parent_assembly'
                 for r in allowedroles.get(m['category'],[])):status='semantic_role_proposed_not_allocated'
        elif m['category'] in allowedroles:status='excluded_or_parent_assembly_review'
        else:status='unassigned'
        ledger.append(dict(asset_id=aid,category=m['category'],model=m['model'],status=status,
            allocated_uses=uses[aid],alternate_instance=variant_uses.get(aid),observed_in_surveys=m['observed_in_surveys'],
            context_instance=context_uses.get(aid),
            referenced_in_native_scenes=m.get('referenced_in_native_scenes',[]),
            proposed_semantic_roles=allowedroles.get(m['category'],[]),physics_prior=m['physics_prior'],
            bbox_valid=m['bbox_valid'],instantiated_new_tasks=0))
    before={}
    for path in (root/'before_asset_alignment').glob('astra_*.json'):
        for task in json.loads(path.read_text()):before[task['id']]=task
    changes=[dict(id=t['id'],before=before[t['id']]['instruction'],after=t['instruction']) for t in tasks
             if t['id'] in before and t['instruction']!=before[t['id']]['instruction']]
    covered={aid for aid,v in uses.items() if v}
    coverage_by_role={role:dict(models=len({aid for aid,v in uses.items() if any(u['role']==role for u in v)}),
                              categories=len({models[aid]['category'] for aid,v in uses.items() if any(u['role']==role for u in v)}))
                      for role in ('target','distractor','reference','receptacle')}
    summary=dict(inventory_categories=len(bycategory),inventory_models=len(models),task_count=len(tasks),
        tasks_with_binding=len(plans),prompts_changed=len(changes),
        target_categories_before=len({c for t in before.values() for c in t['candidate_asset_categories']}),
        target_categories_after=len({c for t in tasks for c in t['candidate_asset_categories']}),
        allocated_models=len(covered),allocated_categories=len({models[a]['category'] for a in covered}),
        allocation_fraction=len(covered)/len(models),coverage_by_role=coverage_by_role,
        alternate_asset_instances=len(variants),
        models_with_primary_or_alternate_proposal=len(covered|set(variant_uses)),
        context_asset_instances=len(contexts),
        models_with_any_task_proposal=len(covered|set(variant_uses)|set(context_uses)),
        categories_with_any_task_proposal=len({models[aid]['category'] for aid in covered|set(variant_uses)|set(context_uses)}),
        tasks_without_target_category=sum(not b.get('target_categories') for b in bindings.values()),
        context_issues=context_issues,
        status_counts=dict(Counter(r['status'] for r in ledger)),alignment_sources=alignment_sources,
        issues=errors,instantiated_new_tasks=0,
        caveats=['Allocation is a private authoring proposal, not actual simulator use.',
            'All native models stay in the denominator, including geometry issues and nonrigid priors.',
            'Unused assets are explicit; no irrelevant objects or scale changes are inserted to inflate coverage.',
            'Metadata evidence does not certify readable labels, collision-free access, or two-finger grasp feasibility.'])
    (root/'asset_model_plan.json').write_text(json.dumps({'summary':summary,'tasks':plans},indent=2)+'\n')
    (root/'asset_coverage.json').write_text(json.dumps({'summary':summary,'assets':ledger},indent=2)+'\n')
    (root/'prompt_asset_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    fields=['asset_id','category','model','status','physics_prior','bbox_valid','task_roles','alternate_task_role','context_task_role','survey_scenes','native_scene_refs']
    with (root/'asset_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in ledger:
            writer.writerow({**{k:row[k] for k in fields[:6]},
                'task_roles':'; '.join(u['task_id']+':'+u['role'] for u in row['allocated_uses']),
                'alternate_task_role':(row['alternate_instance']['parent_task_id']+':'+row['alternate_instance']['role']) if row['alternate_instance'] else '',
                'context_task_role':(row['context_instance']['parent_task_id']+':'+row['context_instance']['role']) if row['context_instance'] else '',
                'native_scene_refs':'; '.join(sorted({u['scene'] for u in row['referenced_in_native_scenes']})),
                'survey_scenes':'; '.join(sorted({u['scene'] for u in row['observed_in_surveys']}))})
    (root/'asset_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'prompt_asset_changes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','before','after']);writer.writeheader();writer.writerows(changes)
    review=root/'review.html'
    if review.exists():
        text=review.read_text();start='<!-- asset-alignment-start -->';end='<!-- asset-alignment-end -->'
        if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
        block=(start+'<section><h2>Native asset alignment</h2><p>'
            +f"{len(changes)} revised instructions; {len(models)} native models inventoried; "
            +f"{summary['models_with_any_task_proposal']} models in primary, alternate or context proposals. "
            +'These are authoring plans, not instantiated or physically validated tasks.</p>'
            +'<p><a href="prompt_asset_changes.csv">Before/after prompts</a> · '
            +'<a href="asset_coverage.csv">Every native model</a> · '
            +'<a href="asset_coverage_summary.json">Coverage and remaining gaps</a></p></section>'+end)
        review.write_text(text.replace('<table>',block+'<table>',1))
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--count',type=int,default=100);ap.add_argument('--refresh-review',action='store_true');ap.add_argument('--semantic-review',action='store_true');ap.add_argument('--asset-inventory',action='store_true');ap.add_argument('--asset-coverage',action='store_true');args=ap.parse_args()
    if args.asset_coverage:
        review_asset_coverage();return
    if args.asset_inventory:
        inventory_native_assets();return
    if args.semantic_review:
        review_semantic_catalogue();return
    if args.refresh_review:
        print(json.dumps(refresh_review()['physics_counts'],indent=2));return
    if args.count!=100:ap.error('This balanced pilot recipe currently defines exactly 100 episodes.')
    common=yaml.safe_load((ROOT/'assets/tasks/vla_difficulty_v2/common.yaml').read_text())
    data={k:asset(k) for k in ITEMS};layouts=[]
    for path in sorted((ROOT/'runs/vla_difficulty_v2/_survey').glob('*.json')):
        if '.candidates.' in path.name:continue
        doc=json.loads(path.read_text())
        if not doc.get('entities') or not doc.get('footprint_boxes'):continue
        try:layouts.append(Layout(path))
        except (FileNotFoundError,ValueError) as e:print('Skip survey',path.stem,str(e),flush=True)
    print('Loaded',len(layouts),'survey layouts',flush=True)
    scene_count=Counter();rows=[];rejections=[]
    for family,roomtypes,sourcekind,targets,goalkind,quota in FAMILIES:
        options=[]
        for layout in layouts:
            rooms=sorted({r for o in layout.entities.values() for r in o.get('in_rooms',[]) if roomkind(r) in roomtypes})
            for room in rooms:
                for target in targets:
                    for src in sources(layout,room,sourcekind,data[target]['size']):
                        destrooms=[room]
                        if goalkind=='sink':destrooms=[r for r in layout.map.names.values() if roomkind(r)=='kitchen']
                        elif goalkind=='bathroom':destrooms=[r for r in layout.map.names.values() if roomkind(r)=='bathroom']
                        for destroom in destrooms:
                            for dest in surfaces(layout,destroom,'sink' if goalkind=='sink' else 'surface'):
                                if dest==src['entity']:continue
                                dist=float(np.linalg.norm(boxcenter(layout.entities[src['entity']])[:2]-boxcenter(layout.entities[dest])[:2]))
                                if dist>8:continue
                                options.append((layout,room,target,src,destroom,dest,dist))
        chosen=[];used=set()
        while options and len(chosen)<quota:
            target_counts=Counter(t[2] for t in chosen)
            options.sort(key=lambda o:(scene_count[o[0].scene],target_counts[o[2]],o[6],o[0].scene,o[1],o[2]))
            option=options.pop(0);layout,room,target,src,droom,dest,dist=option
            key=(layout.scene,room,target,src['entity'],src.get('link'),dest)
            if key in used:continue
            # Never count language-only paraphrases as separate data.
            route=layout.route(src['entity'],room,dest,droom)
            if route is None:continue
            used.add(key);chosen.append(option);scene_count[layout.scene]+=1
            tid=f'dense_search_{len(rows)+1:03d}'
            audit=write_recipe(tid,option,family,sourcekind,goalkind,data,common,route,len(chosen)-1)
            rows.append(audit)
            print(tid,family,layout.scene,target,flush=True)
        if len(chosen)<quota:rejections.append(dict(family=family,requested=quota,generated=len(chosen),reason='No further source/destination pair passed conservative offline route screening'))
    manifest=dict(version=2,source='BEHAVIOR-1K native assets; newly authored search tasks, not official task instances',
        requested_count=100,generated_count=len(rows),scene_count=len(scene_count),family_counts=dict(Counter(r['family'] for r in rows)),
        scene_counts=dict(scene_count),rejections=rejections,
        substitutions=["Red-car trunk and garage transfers are deferred: native trunk containment/color and a tolerant route were not verified. Breakfast grocery baskets provide the yogurt/milk discrimination tasks.",
          "Native labelled supplements replace unspecified medicines; no unverified medicine names are invented.",
          "Native garden/kitchen baskets replace balcony-only layouts when no balcony is present. Cross-room shopping-basket-to-bathroom routes did not pass static screening; bathroom-local shopping baskets are used.",
          "Meat tasks stop at delivery to the sink; thawing is not evaluated."],tasks=rows)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    refresh_review()
    print(json.dumps({k:manifest[k] for k in ('generated_count','scene_count','family_counts','rejections')},indent=2))
    if len(rows)!=100:raise SystemExit('Catalogue incomplete: preserve rejected quotas; do not silently fill with paraphrases.')

if __name__=='__main__':main()
