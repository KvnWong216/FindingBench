"""Resumeable serial residential expansion. Authoring and oracle only, no VLM.

Run inside ygbench: PYTHONPATH=src python -u scripts/v2/expand_dataset.py
Existing base tasks and reviewed scene phrases are preserved. Missing spatial
conditions are omitted, never manufactured or counted as validation successes.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,signal,fcntl
import yaml
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/vla_difficulty_v2_expansion'
SURVEY=ROOT/'runs/vla_difficulty_v2/_survey'
DESIGN=ROOT/'assets/tasks/vla_difficulty_v2'
SCENES=['Rs_int','Pomaria_2_int','house_double_floor_lower',
 'Beechwood_1_int','Pomaria_1_int','Wainscott_1_int','Ihlen_0_int',
 'Wainscott_0_int','Pomaria_0_int','Merom_1_int','Benevolence_2_int',
 'Ihlen_1_int','Benevolence_1_int','Merom_0_int','house_single_floor',
 'hotel_suite_small','Benevolence_0_int']

def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
 global OUT
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--count-per-scene',type=int,default=20)
 p.add_argument('--scenes',nargs='+',default=SCENES)
 p.add_argument('--skip-validation',action='store_true')
 p.add_argument('--parallel',type=int,default=1,choices=[1,2,3])
 p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
 a=p.parse_args()
 shared_out=OUT
 if a.worker:
  if len(a.scenes)!=1:p.error('A worker owns exactly one scene')
  OUT=shared_out/'workers'/a.scenes[0]
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {'scenes':{}}
 state.update(pid=os.getpid(),status='running',requested_layouts=20)
 atomic(OUT/'progress.json',state)
 if a.parallel>1 and not a.worker:
  todo=list(a.scenes);running={}
  state['parallel']=a.parallel
  while todo or running:
   while todo and len(running)<a.parallel:
    scene=todo.pop(0)
    log=(OUT/f'{scene}.worker.log').open('a')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--scenes',scene,
         '--count-per-scene',str(a.count_per_scene),'--skip-validation']
    proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);log.close()
    running[scene]=proc
    print('Started worker',scene,proc.pid,flush=True)
    time.sleep(10) # stagger simulator initialization
   for scene,proc in list(running.items()):
    progress=OUT/'workers'/scene/'progress.json'
    if progress.exists():
     worker=json.loads(progress.read_text())
     state['scenes'].update(worker.get('scenes',{}))
    if proc.poll() is not None:
     state['scenes'].setdefault(scene,{})['worker_returncode']=proc.returncode
     del running[scene]
   state['active_workers']={k:v.pid for k,v in running.items()}
   atomic(OUT/'progress.json',state);time.sleep(3)
  if not a.skip_validation:
   tasks=sorted(p.stem for p in (DESIGN/'base_tasks').glob('X*.yaml'))
   state['status']='validating';atomic(OUT/'progress.json',state)
   with (OUT/'validation.log').open('a') as log:
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/v2/run_batch.py'),
      '--jobs',str(a.parallel),'--tasks',*tasks,'--runs-dir',str(OUT/'validation')],
      cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  state['status']='finished_with_statuses';atomic(OUT/'progress.json',state)
  return
 def run(scene,stage,args,timeout=3600):
  row=state['scenes'].setdefault(scene,{})
  started=time.time()
  row.update(stage=stage,status='running');atomic(OUT/'progress.json',state)
  print(scene,stage,flush=True)
  # Generators share the category counter, phrase file and master manifest.
  writer=None
  if stage in {'generate','compile','phrases'}:
   writer=(shared_out/'dataset_write.lock').open('w')
   fcntl.flock(writer,fcntl.LOCK_EX)
  with (OUT/f'{scene}.{stage}.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=124
  if writer is not None:writer.close()
  row.update(returncode=rc,status='ok' if rc==0 else 'failed',elapsed_s=round(time.time()-started,1))
  row.setdefault('stage_timings_s',{})[stage]=row['elapsed_s']
  atomic(OUT/'progress.json',state)
  return rc==0
 seeds={}
 for line in (SURVEY/'more_scenes.txt').read_text().splitlines():
  _,s,x,y,yaw,rec=line.split();seeds[s]=([float(x),float(y),float(yaw)],rec)
 for scene in a.scenes:
  prefix=f'X{SCENES.index(scene)+1:02d}'
  existing=sorted((DESIGN/'base_tasks').glob(prefix+'[0-9][0-9][0-9].yaml'))
  if existing:
   state['scenes'].setdefault(scene,{}).update(status='generated',tasks=len(existing));continue
  loc=ROOT/f'runs/vla_difficulty_v2/_locations/{scene}.json'
  # Reuse measured sites where at least one target grasp succeeded.
  old=json.loads(loc.read_text()) if loc.exists() else {}
  usable=any(t.get('relation_holds') and t.get('grasp',{}).get('ok') for t in old.get('targets',[]))
  if not usable:
   sp=SURVEY/f'{scene}.json'
   previous=json.loads(sp.read_text()) if sp.exists() else {}
   if scene in seeds:start,rec=seeds[scene]
   elif previous.get('start'):
    start=previous['start'];spec=yaml.safe_load(Path(previous['scenario']).read_text());rec=spec['placements'][0]['receptacle']
   else:
    native=json.loads((Path(os.environ['OMNIGIBSON_DATA_PATH'])/'behavior-1k-assets/scenes'/scene/'json'/f'{scene}_best.json').read_text())
    info=native['objects_info']['init_info'];rec=next(iter(info));pos=native['state']['registry']['object_registry'][rec]['root_link']['pos'];start=[pos[0],pos[1],0]
   if sp.exists() and not (OUT/f'{scene}.original_survey.json').exists():
    (OUT/f'{scene}.original_survey.json').write_bytes(sp.read_bytes())
   if not run(scene,'survey',['scripts/v2/survey_scene.py','--scene',scene,'--start',*map(str,start),'--probe-on',rec,'--unplaced-probe'],1800):continue
   if not run(scene,'candidates',['scripts/v2/analyze_survey.py',scene],1200):continue
   phrases=yaml.safe_load((DESIGN/'phrases.yaml').read_text())
   if not phrases.get(scene,{}).get('furniture'):
    if not run(scene,'phrases',['scripts/v2/make_phrases.py','--scenes',scene],300):continue
   candidates=json.loads((SURVEY/f'{scene}.candidates.json').read_text())
   if not candidates['candidates'] or not candidates['starts']:
    state['scenes'][scene].update(status='blocked',reason='no reachable candidate sites or starts');atomic(OUT/'progress.json',state);continue
   run(scene,'certify',['scripts/v2/certify_locations.py','--scene',scene],7200)
  if not run(scene,'generate',['scripts/v2/generate_tasks.py','--scenes',scene,'--prefix',prefix,'--count',str(a.count_per_scene),'--allow-partial','--diverse-objects'],1200):continue
  tasks=sorted((DESIGN/'base_tasks').glob(prefix+'[0-9][0-9][0-9].yaml'))
  state['scenes'].setdefault(scene,{}).update(status='generated' if tasks else 'blocked',tasks=len(tasks))
  if tasks:
   run(scene,'compile',['scripts/v2/compile_variants.py','--only',*[p.stem for p in tasks]],600)
  atomic(OUT/'progress.json',state)
 if not a.skip_validation:
  # One public-protocol oracle rollout per new physical configuration, serial.
  tasks=[p.stem for p in (DESIGN/'base_tasks').glob('X*.yaml')]
  if tasks:run('all_new','validate',['scripts/v2/run_batch.py','--jobs','1','--tasks',*tasks,'--runs-dir',str(OUT/'validation')],7*86400)
 state['status']='finished_with_statuses';atomic(OUT/'progress.json',state)

if __name__=='__main__':main()
