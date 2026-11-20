"""Offline dist + structural choice audit; no simulator/model execution.

Physical configurations, not language variants, are the unit of analysis.
Uses surveyed native relationships; scores retain provisional distance proxies.
"""
from pathlib import Path
import sys, json, csv, hashlib, math
from collections import Counter
import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(Path(__file__).parent))
from analyze_official_tasks import graph_choice, assign_bands
from analyze_survey import checker_from
from rummagebench.authoring.tasks.room_map import RoomMap
OUT=ROOT/'build/vla_difficulty_v2/difficulty'
ASSETS=Path('/home/ubuntu/ygwang/behavior-datasets/behavior-1k-assets')
RES=.1

class Map:
 def __init__(self,scene,survey):
  rm=RoomMap.load(ASSETS/'scenes'/scene,ASSETS/'metadata/room_categories.txt',resolution=RES)
  cells=np.argwhere(rm.room_ins>0)
  points=np.array([rm.to_world(int(r),int(c)) for r,c in cells])
  chk=checker_from(survey); free=np.zeros(len(points),bool)
  for a in np.radians([0,45,90,135]):
   for i in range(0,len(points),128):
    p=points[i:i+128];free[i:i+128]|=chk.free_many(p[:,0],p[:,1],np.full(len(p),a))
  self.points=points[free]; cells=cells[free];self.tree=cKDTree(self.points)
  ids={tuple(rc):i for i,rc in enumerate(cells)};rr=[];cc=[]
  for (r,c),i in ids.items():
   for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
    j=ids.get((r+dr,c+dc))
    if j is not None:rr.append(i);cc.append(j)
  self.graph=csr_matrix((np.full(len(rr),RES),(rr,cc)),shape=(len(ids),len(ids)))
  self.cache={}
 def node(self,p):
  dist,idx=self.tree.query(p[:2])
  if dist>.16:raise ValueError('approach_pose_off_grid')
  return int(idx)
 def distances(self,idx):
  if idx not in self.cache:self.cache[idx]=dijkstra(self.graph,directed=False,indices=idx)
  return self.cache[idx]
 def metric(self,groups):
  nodes=[sorted({self.node(p) for p in group}) for group in groups]
  if any(not x for x in nodes):raise ValueError('missing_approach_pose')
  matrix=np.zeros((len(nodes),len(nodes)))
  for i in range(len(nodes)):
   for j in range(i):
    d=min(float(self.distances(n)[nodes[j]].min()) for n in nodes[i])
    if not math.isfinite(d):raise ValueError('disconnected_survey_grid')
    matrix[i,j]=matrix[j,i]=max(d,1e-9)
  return float(minimum_spanning_tree(matrix).sum())

def structural(survey,v):
 # Explicit relations and room membership have different meanings. A nested
 # object keeps its room metadata, but only its relational parents form edges.
 excluded=set(survey.get('excluded_authoring_entities', ['probe_mug']))
 entities={n:e for n,e in survey['entities'].items() if n not in excluded}
 evidence=[dict(r) for r in survey.get('relations', [])
           if r['parent'] in entities and r['child'] in entities]
 target=v['target_placement'];child=target.get('entity','target_obj')
 entities[child]={}
 evidence=[r for r in evidence if r['child']!=child]
 evidence.append(dict(parent=target['receptacle'],child=child,
                      relation=target['relation'],source='declared_placement'))
 pairs={(r['parent'],r['child']) for r in evidence}
 def reachable(start,goal,edges):
  todo=[start];seen=set()
  while todo:
   n=todo.pop()
   if n==goal:return True
   if n in seen:continue
   seen.add(n);todo.extend(b for a,b in edges if a==n)
  return False
 cycles=[list(e) for e in sorted(pairs) if reachable(e[1],e[0],pairs)]
 # Predicate closures can contain A->B->C and A->C. Keep immediate
 # relationships; otherwise shortest paths bypass genuine nesting.
 reduced={e for e in pairs if not reachable(e[0],e[1],pairs-{e})} if not cycles else pairs
 edges=set(reduced);parented={b for a,b in reduced};missing=[]
 for name,e in entities.items():
  if name in parented:continue
  rooms=e.get('in_rooms') or ([e['room_at_center']] if e.get('room_at_center') else [])
  if not rooms:missing.append(name)
  for room in rooms:
   edges.add(('house','room:'+room));edges.add(('room:'+room,name))
 # Rooms remain present even if all their contents have relational parents.
 for e in entities.values():
  for room in e.get('in_rooms',[]):edges.add(('house','room:'+room))
 result=graph_choice(edges,list(entities))
 result.update(objects_without_room=missing,relation_evidence=evidence,
               transitive_edges_removed=[list(e) for e in sorted(pairs-reduced)],
               cycles=cycles,excluded_entities=sorted(excluded),
               relations_complete=survey.get('relations_complete',False))
 if cycles or not result['relations_complete']:
  result['choice']=None
 return result

def main():
 OUT.mkdir(exist_ok=True)
 manifest_path=ROOT/'build/vla_difficulty_v2/manifest.json'
 status_path=ROOT/'runs/vla_difficulty_v2_expansion/validation/summary.json'
 variants=json.loads(manifest_path.read_text())['variants']
 status=json.loads(status_path.read_text())['configs']
 rows=[];graphs={};maps={};inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [manifest_path,status_path]}
 for v in variants:
  if v['language_level']!='L2':continue
  scene=v['scene'];key=v['task_id']+'|'+v['spatial_level']
  if scene not in maps:
   p=ROOT/f'runs/vla_difficulty_v2/_survey/relations_refresh/{scene}.json'
   if not p.exists():p=ROOT/f'runs/vla_difficulty_v2/_survey/{scene}.json'
   survey=json.loads(p.read_text())
   if not survey.get('ok'):
    raise ValueError(f'Failed scene survey: {p}: {survey.get("error")}')
   inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
   geometry_path=ROOT/f'runs/vla_difficulty_v2/_survey/{scene}.json'
   geometry=json.loads(geometry_path.read_text())
   inputs[str(geometry_path.relative_to(ROOT))]=hashlib.sha256(geometry_path.read_bytes()).hexdigest()
   native_geometry=set(geometry['entities'])-set(geometry.get('excluded_authoring_entities',['probe_mug']))
   native_relations=set(survey['entities'])-set(survey.get('excluded_authoring_entities',['probe_mug']))
   if native_geometry != native_relations:
    raise ValueError(f'Geometry / relation survey inventories differ: {scene}')
   print('Building geometry graph:',scene,flush=True)
   maps[scene]=(Map(scene,geometry),survey)
  grid,survey=maps[scene]
  basepath=ROOT/f"assets/tasks/vla_difficulty_v2/base_tasks/{v['task_id']}.yaml"
  inputs[str(basepath.relative_to(ROOT))]=hashlib.sha256(basepath.read_bytes()).hexdigest()
  base=yaml.safe_load(basepath.read_text());S=v['spatial_level']
  structure=structural(survey,v);graphs[key]=structure
  row=dict(config=key,task_id=v['task_id'],scene=scene,family=v['family'],spatial_level=S,
           acceptance=status.get(key,{}).get('status','pending'),distance_m=None,distance_normalized=None,
           choice=structure['choice'],choice_target=next((p['cost'] for p in structure['paths'] if p['object']=='target_obj'),None),
           relation_object_count=structure['object_count'],unmapped_objects=len(structure['objects_without_room']),
           difficulty_band='unclassified',difficulty_score=None,issues='snapshot relation graph; provisional distance proxy')
  row['native_relation_count']=len(survey.get('relations',[]))
  row['mean_relation_depth']=structure['mean_path_depth']
  if not structure['relations_complete']:row['issues']+='; incomplete_native_relation_capture'
  if structure['cycles']:row['issues']+='; cyclic_native_relations'
  if structure['unreachable_objects']:row['issues']+='; unreachable_relation_nodes'
  groups=[[[v['start']['x'],v['start']['y']]],base['spatial'][S]['grasp_hints']]
  if base['family']!='fetch':groups.append(base['place_hints'])
  try:row['distance_m']=grid.metric(groups);row['distance_normalized']=row['distance_m']/1.
  except ValueError as e:row['issues']+='; '+str(e)
  rows.append(row)
 for source in [Path(__file__),ROOT/'scripts/analyze_official_tasks.py',ROOT/'scripts/v2/analyze_survey.py',ROOT/'scripts/v2/survey_scene.py']:
  inputs[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
 for scene in maps:
  for p in (ASSETS/'scenes'/scene/'layout').glob('floor_*_0.png'):
   inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 reference=[r for r in rows if r['acceptance']=='accepted']
 thresholds=assign_bands(reference)
 for r in rows:
  if r['acceptance']=='accepted' or r['distance_normalized'] is None or r['choice'] is None:continue
  scaled=[]
  for f in ['distance_normalized','choice']:
   s=thresholds['scales'][f];scaled.append(max(0.,(r[f]-s['low'])/(s['high']-s['low'])) if s['high']>s['low'] else .5)
  score=sum(scaled)/2;r['difficulty_score']=score
  r['difficulty_band']='easy' if score<=thresholds['easy_max'] else 'medium' if score<=thresholds['medium_max'] else 'hard'
  r['borderline']=min(abs(score-thresholds['easy_max']),abs(score-thresholds['medium_max']))<=.03
 metadata=dict(unit='physical configuration (L1/L2/L3 share metrics)',reference='accepted configs with complete proxies',
  distance='MST of minimum 4-neighbor 0.1m grid distances between start and certified GRASP/PLACE approach-pose sets; optimistic proxy, not an optimal execution plan; yaw-transition costs excluded',
  normalization='1m fixed reference; historical MOVE maximum, not current MOVE_TO limit',
  choice='mean over native object nodes plus target: min root path sum(1+log2(outdegree)); native Inside/OnTop immediate relationships and declared target placement; room edges only for objects without relational parents; authoring probe excluded',
  score='equal mean of nonnegative p10-p90-scaled dist and choice; tertile thresholds fitted to accepted physical configurations; ties not split',
  limitations=['Whole-scene choice can remain similar across tasks sharing the same background; incomplete or cyclic relation graphs are unclassified.','Native relations cover available Inside/OnTop states, not the mesh/link assembly tree; unsupported predicates are not inferred. Target placement is declared, not measured per task instance.','Original room mask restricts indoor cells; edited doorway/map mismatch can disconnect grid.','Survey geometry and approach poses are snapshots, not current runtime feasibility certificates.','Architecture/fixtures included; unknown root membership makes the graph incomplete rather than silently dropping objects.','Language variants are not independent samples; bands are provisional.'],
  thresholds=thresholds,inputs_sha256=inputs,counts=dict(total=len(rows),accepted=len(reference),accepted_classified=sum(r['difficulty_band']!='unclassified' for r in reference),bands=dict(Counter(r['difficulty_band'] for r in reference))))
 metadata['all_counts']=dict(total=len(rows),classified=sum(r['difficulty_band']!='unclassified' for r in rows),
    bands=dict(Counter(r['difficulty_band'] for r in rows)),
    missing_choice=sum(r['choice'] is None for r in rows),
    missing_distance=sum(r['distance_normalized'] is None for r in rows))
 metadata['all_variant_bands']=dict(Counter(
    next(r['difficulty_band'] for r in rows if r['config']==v['task_id']+'|'+v['spatial_level'])
    for v in variants))
 (OUT/'metrics.json').write_text(json.dumps(dict(protocol=metadata,rows=rows),indent=2))
 (OUT/'choice_graphs.json').write_text(json.dumps(graphs))
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 for band in ['easy','medium','hard','unclassified']:
  with (OUT/f'all_{band}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   w.writerows(r for r in rows if r['difficulty_band']==band)
  with (OUT/f'{band}.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   w.writerows(r for r in reference if r['difficulty_band']==band)
 bykey={r['config']:r for r in rows}
 from human_oracle import SMOKE_TASKS
 smoke=[]
 for i,vid in enumerate(SMOKE_TASKS,1):
  task,ls=vid.split('__');r=bykey[task+'|'+ls.split('_')[1]]
  smoke.append(dict(number=i,variant=vid,dist=r['distance_normalized'],choice=r['choice'],
      score=r['difficulty_score'],band=r['difficulty_band'],issues=r['issues']))
 with (OUT/'human_smoke.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(smoke[0]));w.writeheader();w.writerows(smoke)
 with (OUT/'variant_assignments.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['variant_id','physical_config','acceptance','difficulty_band','difficulty_score']);w.writeheader()
  for v in variants:
   key=v['task_id']+'|'+v['spatial_level'];r=bykey[key]
   w.writerow(dict(variant_id=v['variant_id'],physical_config=key,**{k:r[k] for k in ['acceptance','difficulty_band','difficulty_score']}))
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(16,11),layout='constrained')
 colors={'easy':'#21918c','medium':'#e0a12d','hard':'#c65365','unclassified':'#999999'}
 markers=['o','s','^','v','D','P','X','<','>','h','p','*']
 valid=[r for r in reference if r['difficulty_band']!='unclassified']
 for index,scene in enumerate(sorted(maps)):
  marker=markers[index % len(markers)]
  rs=[r for r in valid if r['scene']==scene]
  axes[0,0].scatter([r['distance_normalized'] for r in rs],[r['choice'] for r in rs],c=[colors[r['difficulty_band']] for r in rs],marker=marker,alpha=.7,label=scene,s=38)
 axes[0,0].set(xlabel='dist (grid MST / 1 m)',ylabel='choice (recorded relation graph)',title='Accepted configurations: color = provisional band')
 axes[0,0].legend(fontsize=8)
 for b in ['easy','medium','hard']:
  axes[0,1].hist([r['difficulty_score'] for r in valid if r['difficulty_band']==b],bins=np.linspace(0,max(r['difficulty_score'] for r in valid)+.01,20),alpha=.7,color=colors[b],label=b)
 for k in ['easy_max','medium_max']:axes[0,1].axvline(thresholds[k],ls='--',color='#333333')
 axes[0,1].set(xlabel='Combined score',ylabel='Physical configurations',title='Reference distribution (not semantic task count)');axes[0,1].legend()
 for ax,field,title in [(axes[1,0],'scene','Bands by scene'),(axes[1,1],'spatial_level','Bands by original spatial condition')]:
  labels=sorted({r[field] for r in reference});bottom=np.zeros(len(labels))
  for b in colors:
   count=np.array([sum(r[field]==l and r['difficulty_band']==b for r in reference) for l in labels]);ax.bar(range(len(labels)),count,bottom=bottom,label=b,color=colors[b]);bottom+=count
  ax.set_xticks(range(len(labels)),[l.replace('house_double_floor_upper','house upper').replace('hotel_suite_large','hotel suite') for l in labels]);ax.set(title=title,ylabel='Physical configurations');ax.legend(fontsize=8)
  if field=='scene':ax.tick_params(axis='x',labelrotation=55,labelsize=8)
 fig.suptitle('dist + choice: provisional difficulty audit\nAccepted configs only; native relation hierarchy — provisional benchmark labels',fontsize=14)
 for ext in ['png','pdf']:fig.savefig(OUT/f'difficulty_overview.{ext}',dpi=180)
 summary={**metadata['counts'],'by_scene':{s:dict(Counter(r['difficulty_band'] for r in reference if r['scene']==s)) for s in maps},'choice_range':[min(r['choice'] for r in reference if r['choice'] is not None),max(r['choice'] for r in reference if r['choice'] is not None)],'unclassified':[r['config'] for r in reference if r['difficulty_band']=='unclassified']}
 summary['structure_audit']={s:dict(native_relations=len(survey.get('relations',[])),
  relation_errors=len(survey.get('relation_errors',[])),
  choice_range=[min(r['choice'] for r in reference if r['scene']==s and r['choice'] is not None),
                max(r['choice'] for r in reference if r['scene']==s and r['choice'] is not None)])
  for s,(_,survey) in maps.items() if any(r['scene']==s and r['choice'] is not None for r in reference)}
 summary['all_counts']=metadata['all_counts']
 summary['all_variant_bands']=metadata['all_variant_bands']
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
