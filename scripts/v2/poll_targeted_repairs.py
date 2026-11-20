"""Poll two native lanes and execute reviewed causal repairs exactly once.

This is a finite repair playbook, not an autonomous code-writing expert.
Unresolved cases retain their evidence and require a substantive next repair.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import re

ROOT=Path(__file__).resolve().parents[2]


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.poll.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def alive(pid,identity):
    try:return identity.encode() in Path(f'/proc/{int(pid)}/cmdline').read_bytes()
    except (FileNotFoundError,ProcessLookupError,TypeError,ValueError):return False


def release_lane(out):
    reservation=read(out/'surface_probe_supervisor_pause.json');pid=reservation.get('pid')
    if alive(pid,str(out/'supervise.py')):
        os.kill(pid,signal.SIGCONT)
    controller=read(out/'controller_status.json').get('pid')
    if alive(controller,'scripts/v2/expert_batch_repair.py'):
        os.kill(controller,signal.SIGCONT)
    (out/'surface_probe_supervisor_pause.json').unlink(missing_ok=True)


def hold_lane(out):
    supervisor=read(out/'supervisor_status.json').get('pid')
    if alive(supervisor,str(out/'supervise.py')):
        os.kill(supervisor,signal.SIGSTOP)
        write(out/'surface_probe_supervisor_pause.json',dict(pid=supervisor,
            reason='Polling campaign owns this lane until its next reviewed repair',updated=time.time()))
    controller=read(out/'controller_status.json').get('pid')
    if alive(controller,'scripts/v2/expert_batch_repair.py'):
        # Running native children finish normally. Prevent the old controller
        # from launching unreviewed work between consecutive targeted probes.
        os.kill(controller,signal.SIGSTOP)


def validate_action_cases(action,tasks=None):
    """Reject stale job/case bindings before reserving or starting a worker."""
    cases=read(ROOT/action['cases']);tasks=action['tasks'] if tasks is None else tasks
    missing=[tid for tid in tasks if not cases.get(tid,{}).get('record',{}).get('scenario')]
    if missing:raise ValueError('ACTION_CASE_BINDING_MISMATCH: '+action['tag']+' missing '+','.join(missing))
    if action.get('runtime'):
        for tid in tasks:
            recipe=Path(cases[tid]['record']['scenario'])
            if recipe.is_file() and not runtime_accepts_authoring_hints(recipe,ROOT/action['runtime']):
                raise ValueError('ACTION_RUNTIME_SCHEMA_MISMATCH: '+action['tag']+' '+tid)


def runtime_accepts_authoring_hints(recipe,runtime):
    """Do not send a new private layout field to an older frozen schema."""
    text=Path(recipe).read_text()
    if not re.search(r'^\s+orientation_hint_xyzw\s*:',text,re.M):return True
    relative='src/rummagebench/core/scenario.py';path=Path(runtime)/relative
    expected=read(Path(runtime)/'release.json').get('source_sha256',{}).get(relative)
    if not (expected and path.is_file() and 'orientation_hint_xyzw:' in path.read_text()
            and hashlib.sha256(path.read_bytes()).hexdigest()==expected):return False
    import yaml
    try:document=yaml.safe_load(text) or {}
    except yaml.YAMLError:return False
    on_top=any(p.get('relation')=='on_top' and p.get('orientation_hint_xyzw') is not None
               for p in document.get('placements',[]))
    return not on_top or 'on_top orientation_hint_xyzw requires' in path.read_text()


def launch(action):
    validate_action_cases(action)
    out=ROOT/action['lane'];script=out/'clearance_canary.py'
    supervisor=read(out/'supervisor_status.json').get('pid')
    if not alive(supervisor,str(out/'supervise.py')):
        raise RuntimeError('Expected live lane supervisor: '+str(out))
    os.kill(supervisor,signal.SIGSTOP)
    write(out/'surface_probe_supervisor_pause.json',dict(pid=supervisor,
        reason='Reserved by finite targeted repair polling plan',updated=time.time()))
    command=['/usr/bin/python3','-u',str(script),'--runtime',str(ROOT/action['runtime']),
        '--tag',action['tag'],'--probes',*action['tasks'],'--cases',str(ROOT/action['cases']),
        '--keep-lane-paused']
    if action.get('diagnostic_only'):command.append('--diagnostic-only')
    if action.get('prepare_repairs'):command.append('--prepare-repairs')
    try:
        with (out/(action['tag']+'_supervisor.log')).open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        return dict(stage='running',pid=child.pid,started=time.time())
    except BaseException:
        release_lane(out);raise


def evidence_timestamp(directory):
    """A later reload rejection supersedes its own earlier initialization."""
    return max((path.stat().st_mtime for name in ('validation.json','restore_validation.json')
                if (path:=Path(directory)/name).is_file()),default=0.)


def reconcile(actions):
    from track_full_asset_queue import accepted_native_pair,known_authoring_issue
    primary=ROOT/'runs/expert100_20261110'
    path=primary/'external_cases.json';external=read(path);accepted=[];failed=[]
    for action in actions:
        out=ROOT/action['lane'];results=read(out/(action['tag']+'_results.json'))
        for tid,row in results.get('rows',{}).items():
            recipe=Path(row['scenario']);directory=out/(action['tag']+'_'+tid)/row['id']
            report_path=directory/'validation.json';report=read(report_path)
            if report.get('scenario_sha256')!=hashlib.sha256(recipe.read_bytes()).hexdigest():continue
            restored=read(directory/'restore_validation.json')
            success=accepted_native_pair(recipe,directory)
            if not success and report.get('status')=='initialization_validated' and not restored:continue
            is_reload=report.get('status')=='initialization_validated' and bool(restored)
            last=restored if is_reload else report
            stamp=evidence_timestamp(directory)
            prior=external.get(tid,{});prior_stamp=evidence_timestamp(prior.get('evidence','/nonexistent'))
            if success or not prior.get('accepted') and stamp>=prior_stamp:
                external[tid]=dict(id=tid,record=row,evidence=str(directory),accepted=success,
                    error=last.get('error',last.get('status')),source='reviewed_polling_probe_'+action['tag'],
                    evidence_updated=stamp)
            if tid in action['tasks']:
                (accepted if success else failed).append(dict(task=tid,evidence=str(directory),
                    stage='native_init_and_reload_passed' if success else
                        'independent_frozen_reload' if is_reload else last.get('stage','fresh_native_validation'),
                    error=last.get('error'),full_skill_completion='unverified'))
    if external!=read(path):write(path,external)
    return accepted,failed


def cohort_coverage():
    """Count immutable evidence for the original cohort, never job exit codes."""
    from track_full_asset_queue import accepted_native_pair,known_authoring_issue
    primary=ROOT/'runs/expert100_20261110'
    ids={case['id'] for case in read(primary/'source_index.json',[])}
    if len(ids)!=100:raise ValueError('Original expert cohort must contain exactly 100 tasks')
    candidates=[]
    for name in ['expert100_20261110','expert100_lane2_20261111']:
        lane=ROOT/'runs'/name
        for job in read(lane/'jobs.json').values():
            row=job.get('proposal',{}).get('record',{})
            if job.get('accepted') and job.get('task') in ids and row.get('scenario'):
                candidates.append((job['task'],Path(row['scenario']),lane/job['lane']/row['id']))
    for case in read(primary/'external_cases.json').values():
        row=case.get('record',{})
        if case.get('accepted') and case.get('id') in ids:
            candidates.append((case['id'],Path(row['scenario']),Path(case['evidence'])))
    for row in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[]):
        if row.get('parent_task_id') in ids:
            candidates.append((row['parent_task_id'],ROOT/row['scenario'],Path(row['play_scenario']).parent))
    passed={};issues={}
    for tid,recipe,evidence in candidates:
        if (tid not in passed or tid in issues) and accepted_native_pair(recipe,evidence):
            issue=known_authoring_issue(recipe)
            passed[tid]=dict(recipe=str(recipe),evidence=str(evidence),full_skill_completion='unverified')
            if issue:issues[tid]=issue
            else:issues.pop(tid,None)
    reviewable={tid:row for tid,row in passed.items() if tid not in issues}
    return dict(passed=len(reviewable),native_passed=len(passed),total=100,
        remaining=sorted(ids-set(reviewable)),known_authoring_issues=issues,
        evidence=reviewable,native_evidence=passed)


def global_coverage():
    """Count distinct authored tasks with reviewable physical evidence."""
    from track_full_asset_queue import reviewable_native_pair
    queue=read(ROOT/'runs/asset_full_queue_20261105/queue.json')
    ids={row['id'] for row in queue.get('tasks',[])}
    if len(ids)!=1000:raise ValueError('Full authored campaign must retain all 1000 task identities')
    passed=set()
    for row in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[]):
        tid=row.get('parent_task_id')
        if tid in ids and tid not in passed and reviewable_native_pair(ROOT/row['scenario'],Path(row['play_scenario']).parent):
            passed.add(tid)
    return dict(passed=len(passed),total=len(ids),remaining=sorted(ids-passed),
                full_skill_completion='unverified')


def needed_tasks(action,cohort,whole):
    if action.get('scope')=='regression':return list(action['tasks'])
    remaining=whole['remaining'] if action.get('scope')=='global_remaining' else cohort['remaining']
    return [tid for tid in action['tasks'] if tid in remaining]


def refresh_running_actions(actions,states):
    """Release finished lane reservations before considering ordered work."""
    for action in actions:
        key=action['tag'];current=states.get(key,{})
        if current.get('stage')!='running':continue
        lane=ROOT/action['lane']
        current['native_result']=read(lane/(key+'_status.json'))
        if alive(current['pid'],'clearance_canary.py'):continue
        try:os.waitpid(current['pid'],os.WNOHANG)
        except ChildProcessError:pass
        hold_lane(lane)
        current.update(stage='finished',finished=time.time())
        states[key]=current


def extend_reviewed_plan(plan_path,plan,states,whole):
    """Apply reviewed authoring rules to new evidence, never invent new code.

    Each immutable input/runtime/authoring revision is prepared at most once.
    Unchanged proposals are rejected by the canary; bounded attempts retain
    failed evidence for the next substantive expert revision.
    """
    config=plan.get('reviewed_followups')
    if not config or not whole:return False
    from expert_batch_repair import preparation_revision
    from track_full_asset_queue import accepted_native_pair
    by_tag={a['tag']:a for a in plan['actions']}
    for tag,tasks in config.get('requires_success',{}).items():
        action=by_tag.get(tag)
        if not action:return False
        results=read(ROOT/action['lane']/(tag+'_results.json'))
        for tid in tasks:
            row=results.get('rows',{}).get(tid)
            if not row or not results.get('results',{}).get(tid):return False
            directory=ROOT/action['lane']/(tag+'_'+tid)/row['id']
            if not accepted_native_pair(Path(row['scenario']),directory):return False
    runtime=ROOT/config['runtime'];release=read(runtime/'release.json')
    if not release.get('source_sha256'):return False
    for relative,digest in release['source_sha256'].items():
        if hashlib.sha256((runtime/relative).read_bytes()).hexdigest()!=digest:return False
    revision=preparation_revision()
    runtime_id=hashlib.sha256((runtime/'release.json').read_bytes()).hexdigest()
    pending={tid for a in plan['actions'] if states.get(a['tag'],{}).get('stage') not in ('finished','launch_error')
             for tid in a['tasks']}
    history=plan.setdefault('reviewed_followup_history',[])
    seen={r['identity'] for r in history};selected=[]
    # A deferred offline proposal has no new native report to reconcile.
    # Retain its original hash-matched failure instead of dropping that task.
    fallback=read(ROOT/config['fallback_cases']) if config.get('fallback_cases') else {}
    if isinstance(fallback,list):fallback={case['id']:case for case in fallback}
    external={**fallback,**read(plan_path.parent/'external_cases.json')}
    for tid in whole['remaining']:
        case=external.get(tid)
        if tid in pending or not case or case.get('accepted'):continue
        if sum(r['task']==tid and r['authoring_revision']==revision and r.get('runtime_identity')==runtime_id
               for r in history)>=config.get('max_preparations_per_task',3):continue
        recipe=Path(case['record']['scenario']);report=read(Path(case['evidence'])/'validation.json')
        if not recipe.is_file():continue
        if not runtime_accepts_authoring_hints(recipe,runtime):continue
        sha=hashlib.sha256(recipe.read_bytes()).hexdigest()
        if report.get('scenario_sha256')!=sha:continue
        identity=hashlib.sha256(json.dumps(dict(task=tid,recipe=sha,runtime=runtime_id,
            authoring_revision=revision,observed_error=case.get('error')),sort_keys=True).encode()).hexdigest()
        if identity in seen:continue
        selected.append((tid,case,dict(task=tid,identity=identity,authoring_revision=revision,
                        source_recipe_sha256=sha,source_evidence=case['evidence'],runtime_identity=runtime_id)))
    if not selected:return False
    size=config.get('batch_size',4);lanes=config['lanes']
    if not 1<=size<=6 or len(lanes)!=2:raise ValueError('Reviewed followups require two lanes and bounded batches')
    # Plans can retain a sparse sequence after reviewed jobs are withdrawn.
    # Counting entries can reuse an existing tag and overwrite its cases.
    sequence=1+max((int(a['tag'].removeprefix('reviewed_followup_')) for a in plan['actions']
        if a['tag'].startswith('reviewed_followup_') and a['tag'].removeprefix('reviewed_followup_').isdigit()),default=0)
    backlog={lane:sum(len(a['tasks']) for a in plan['actions'] if a['lane']==lane
        and states.get(a['tag'],{}).get('stage') not in ('finished','launch_error')) for lane in lanes}
    for offset in range(0,len(selected),size):
        group=selected[offset:offset+size];tag=f'reviewed_followup_{sequence:03}'
        lane=min(lanes,key=lambda value:backlog[value]);backlog[lane]+=len(group)
        cases_path=plan_path.parent/(tag+'_cases.json')
        write(cases_path,{tid:case for tid,case,_ in group})
        plan['actions'].append(dict(tag=tag,lane=lane,runtime=config['runtime'],
            tasks=[tid for tid,_,_ in group],cases=str(cases_path.relative_to(ROOT)),
            scope='global_remaining',prepare_repairs=True,diagnostic_only=True,
            requires_success=config.get('requires_success',{})))
        history.extend(record for _,_,record in group);sequence+=1
    write(plan_path,plan)
    return True


def adopt_reviewed_runtime(plan_path,plan,states):
    """Roll unstarted work forward only after a reviewed canary promotes it."""
    from expert_batch_repair import synchronize_verified_runtime
    primary=plan_path.parent
    approved={str((ROOT/value).resolve()) for value in plan.get('approved_runtime_candidates',[])}
    config=plan.get('reviewed_followups',{})
    current=(ROOT/config.get('runtime','')).resolve()
    def ancestry(path):
        chain=[];seen=set()
        while path not in seen:
            seen.add(path);chain.append(path)
            base=read(path/'release.json').get('base')
            if not base:break
            path=(ROOT/base).resolve()
        return chain
    current_ancestors=set(ancestry(current))
    used={(ROOT/entry['runtime']).resolve() for entry in plan.get('verified_runtime_adoptions',[])}
    candidates=[]
    for lane in dict.fromkeys([primary,*(ROOT/value for value in config.get('lanes',[]))]):
        override=read(lane/'runtime_override.json')
        runtime=Path(override.get('runtime','/nonexistent')).resolve()
        if str(runtime) not in approved or runtime in current_ancestors or runtime in used:continue
        candidates.append((len(ancestry(runtime)),lane,runtime))
    for _,lane,runtime in sorted(candidates,key=lambda row:row[0],reverse=True):
        if synchronize_verified_runtime(lane,primary):break
    else:return False
    # synchronize verifies exact sources, independent pairs, both controls,
    # and a successful probe; the producing action supplies scheduling gates.
    source=Path(read(primary/'runtime_override.json')['source'])
    tag=source.name.removesuffix('_results.json')
    results=read(source).get('results',{})
    probe=next(t for t,v in results.items() if v and t not in ('F035','F090'))
    gates={tag:['F035','F090',probe]}
    config.update(runtime=str(runtime.relative_to(ROOT)),requires_success=gates)
    for action in plan['actions']:
        if action.get('scope')!='global_remaining' or not action.get('prepare_repairs'):continue
        if states.get(action['tag'],{}).get('stage') in ('running','finished','launch_error'):continue
        action.update(runtime=config['runtime'],requires_success=gates)
    plan.setdefault('verified_runtime_adoptions',[]).append(dict(runtime=config['runtime'],proof=str(source),updated=time.time()))
    write(plan_path,plan)
    return True


def preparation_is_current(out,tag,tasks,revision):
    records=read(out/(tag+'_preparation.json'))
    return bool(tasks) and all(records.get(tid) and records[tid][-1].get('preparation_revision')==revision for tid in tasks)


def prepared_deferrals(action,tasks,revision):
    """Do not reserve a simulator for already diagnosed offline failures."""
    if not action.get('prepare_repairs') or not tasks:return None
    records=read(ROOT/action['lane']/(action['tag']+'_preparation.json'));reasons={}
    for tid in tasks:
        latest=(records.get(tid) or [{}])[-1];proposal=latest.get('proposal',{})
        if latest.get('preparation_revision')!=revision or proposal.get('status')!='deferred':return None
        reasons[tid]=proposal.get('reason','Offline proposal did not pass authoring screening')
    return reasons


def task_running_elsewhere(tag,tasks,states):
    requested=set(tasks)
    return any(other!=tag and state.get('stage')=='running' and
        bool(requested & set(state.get('tasks_run',[]))) for other,state in states.items())


def exclude_certified_external(tasks,external):
    """Avoid duplicate launches while the central catalogue catches up."""
    from track_full_asset_queue import reviewable_native_pair
    pending=[]
    for tid in tasks:
        case=external.get(tid,{})
        if (case.get('accepted') and case.get('record',{}).get('scenario') and case.get('evidence')
                and reviewable_native_pair(Path(case['record']['scenario']),Path(case['evidence']))):continue
        pending.append(tid)
    return pending


def update_launch_snapshot(out,plan,states,cohort,whole):
    """Keep the human-facing handoff consistent with live verified coverage."""
    path=out/'targeted_repair_launch.json'
    if not path.exists():return
    doc=read(path);override=read(out/'runtime_override.json');by_tag={a['tag']:a for a in plan['actions']}
    lanes=[]
    for lane in sorted({a['lane'] for a in plan['actions']}):
        active=[dict(tag=tag,pid=state.get('pid'),tasks=state.get('tasks_run',[]),
            runtime=by_tag[tag]['runtime']) for tag,state in states.items()
            if state.get('stage')=='running' and tag in by_tag and by_tag[tag]['lane']==lane]
        lanes.append(dict(name=lane,mode='one native task at a time',active=active))
    doc.update(updated=time.time(),scheduler_pid=os.getpid(),lanes=lanes,
        cohort_progress={k:cohort[k] for k in ('passed','total','remaining')},all_tasks_progress=whole,
        offline_producer_pid=read(out/'offline_preparation_status.json').get('pid'),
        completion_policy='Keep legacy generation queues paused after the native-pair target is complete.',
        live_progress_path=str(out/'polling_repair_status.json'))
    doc.pop('all_task_progress',None)
    doc.pop('private_probes',None)  # Live lane/runtime records supersede old probe snapshots.
    if override:
        doc.update(production_runtime=override['runtime'],production_controls=override.get('verified_by'),
            production_results=override.get('controls'))
    write(path,doc)


def complete_repair_campaign(actions):
    # Releasing these old queues would resume obsolete generation jobs after
    # all requested data are ready. Keep them parked for explicit future work.
    for lane in {a['lane'] for a in actions}:hold_lane(ROOT/lane)


def prepare_ahead(plan_path,watch=False):
    """One offline producer overlaps layout preparation with two native lanes."""
    out=plan_path.parent
    with (out/'offline_preparation.lock').open('a') as lock:
        # A persistent producer may wait for the existing finite pass to
        # finish; it never runs a second layout preparer concurrently.
        fcntl.flock(lock,fcntl.LOCK_EX|(0 if watch else fcntl.LOCK_NB))
        from expert_batch_repair import preparation_revision
        while not (out/'polling_stop.request').exists():
            plan=read(plan_path);completed=[];lookahead={}
            remaining=set(global_coverage()['remaining']);coverage_checked=time.monotonic()
            for action in plan['actions']:
                if action.get('scope')!='global_remaining' or not action.get('prepare_repairs'):continue
                state=read(out/'polling_repair_status.json').get('actions',{}).get(action['tag'],{})
                if state.get('stage') in ('running','finished','launch_error'):continue
                tasks=[tid for tid in action['tasks'] if tid in remaining]
                if not tasks:continue
                try:validate_action_cases(action,tasks)
                except ValueError as error:
                    completed.append(dict(tag=action['tag'],case_binding_error=str(error)));continue
                lane=action['lane'];lookahead[lane]=lookahead.get(lane,0)+1
                if lookahead[lane]>plan.get('offline_lookahead_per_lane',2):continue
                if preparation_is_current(ROOT/lane,action['tag'],tasks,preparation_revision()):continue
                if time.monotonic()-coverage_checked>30:
                    remaining=set(global_coverage()['remaining']);coverage_checked=time.monotonic()
                    tasks=[tid for tid in tasks if tid in remaining]
                    if not tasks:continue
                command=['/usr/bin/python3','-u',str(ROOT/action['lane']/'clearance_canary.py'),
                    '--runtime',str(ROOT/action['runtime']),'--tag',action['tag'],
                    '--cases',str(ROOT/action['cases']),'--probes',*tasks,'--prepare-only']
                write(out/'offline_preparation_status.json',dict(pid=os.getpid(),stage='preparing',
                    tag=action['tag'],tasks=tasks,completed=completed,updated=time.time(),native_workers=0))
                with (out/'offline_preparation.log').open('a') as log:
                    result=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
                completed.append(dict(tag=action['tag'],exit_code=result.returncode))
            finished=global_coverage()['passed']==1000
            write(out/'offline_preparation_status.json',dict(pid=os.getpid(),stage='complete_1000_of_1000' if finished else 'watching_for_reviewed_work' if watch else 'prepared_current_plan',
                completed=completed,updated=time.time(),native_workers=0))
            if not watch or finished:return
            time.sleep(30)


def main(plan_path,interval):
    out=plan_path.parent;state_path=out/'polling_repair_status.json';state=read(state_path)
    states=state.get('actions',{})
    with (out/'polling_repair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (out/'polling_stop.request').exists():
            plan=read(plan_path);actions=plan['actions'];errors=[]
            refresh_running_actions(actions,states)
            # Adopt finished case evidence before dispatch, so a later repair
            # does not re-run cases already certified by another proposal.
            reconcile(actions)
            adopt_reviewed_runtime(plan_path,plan,states)
            coverage=cohort_coverage()
            whole=global_coverage() if plan.get('target_total')==1000 else None
            extend_reviewed_plan(plan_path,plan,states,whole)
            from expert_batch_repair import preparation_revision
            revision=preparation_revision()
            external=read(out/'external_cases.json')
            target_reached=(whole['passed']==1000 if whole else coverage['passed']==100)
            for action in actions:
                key=action['tag'];current=states.get(key,{})
                if current.get('stage')=='running':
                    continue
                if current.get('stage') in ('finished','launch_error'):continue
                if target_reached:continue
                wait_pid=action.get('after_pid')
                if wait_pid and alive(wait_pid,'clearance_canary.py'):continue
                dependencies=action.get('after_tags',[])
                if any(states.get(tag,{}).get('stage')!='finished' for tag in dependencies):continue
                gates=action.get('requires_success',{})
                if any(not all(states.get(tag,{}).get('native_result',{}).get('results',{}).get(tid)
                               for tid in tasks) for tag,tasks in gates.items()):continue
                needed=needed_tasks(action,coverage,whole)
                if action.get('scope')=='global_remaining':needed=exclude_certified_external(needed,external)
                if not needed:
                    states[key]=dict(stage='finished',finished=time.time(),tasks_run=[],
                        reason='All requested cases already have reviewable native pairs; no duplicate replay')
                    continue
                if task_running_elsewhere(key,needed,states):continue
                deferred=prepared_deferrals(action,needed,revision)
                if deferred is not None:
                    states[key]=dict(stage='finished',finished=time.time(),tasks_run=[],tasks_deferred=deferred,
                        preparation_revision=revision,native_result=dict(stage='offline_deferred_no_native_launch',results={tid:False for tid in needed}),
                        reason='No valid changed authoring proposal; retain diagnosis without spending a native lane')
                    continue
                if any(states.get(a['tag'],{}).get('stage')=='running' for a in actions if a['lane']==action['lane']):continue
                try:
                    states[key]=launch(dict(action,tasks=needed))
                    states[key]['tasks_run']=needed
                except Exception as exc:
                    states[key]=dict(stage='launch_error',error=type(exc).__name__+': '+str(exc))
                    errors.append(states[key])
            try:passed,failed=reconcile(actions)
            except Exception as exc:passed=[];failed=[];errors.append(dict(reconciliation_error=str(exc)))
            pending=[a['tag'] for a in actions if states.get(a['tag'],{}).get('stage') not in ('finished','launch_error')]
            running=any(v.get('stage')=='running' for v in states.values())
            complete=target_reached and not running
            write(state_path,dict(pid=os.getpid(),updated=time.time(),interval_seconds=interval,
                stage=('complete_1000_of_1000' if whole else 'complete_100_of_100') if complete else
                    'polling_reviewed_repairs' if pending else 'playbook_exhausted_needs_next_causal_revision',
                actions=states,new_native_pairs=passed,failed_repairs=failed,errors=errors,
                cohort=coverage,all_tasks=whole,
                note='Native init/reload evidence required; measurement protocol is pinned in each runtime release. No identical action is automatically re-launched.'))
            update_launch_snapshot(out,plan,states,coverage,whole)
            if complete:
                complete_repair_campaign(actions)
                return
            # Remain available for a revised reviewed plan; no native retries
            # are scheduled merely because a polling interval elapsed.
            time.sleep(interval)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=30)
    parser.add_argument('--prepare-ahead',action='store_true')
    parser.add_argument('--prepare-ahead-watch',action='store_true')
    args=parser.parse_args()
    if args.interval<10:parser.error('Polling interval must be at least 10 seconds')
    if args.prepare_ahead or args.prepare_ahead_watch:prepare_ahead(args.plan.resolve(),watch=args.prepare_ahead_watch)
    else:main(args.plan.resolve(),args.interval)
