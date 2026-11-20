"""Evidence-led interventions for a fixed cohort of 100 stubborn tasks.

Prepare a diagnosis and bounded intervention per task, then serialize native
validation at a boundary of the existing queue. All original semantic fields
and inventories are retained. Passing means initialization plus fresh reload.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,fcntl,hashlib,json,os,re,signal,time,traceback
import shutil
from overnight_asset_queue import Queue,ROOT,read,write,wait_external,storage_scope_text,has_qualified_storage_location,native_policy_signature
from track_full_asset_queue import accepted_native_pair

PRIMARY=ROOT/'runs/overnight_expert_20261109'


def shard_owns_task(tid,index,count):
    if count<1 or not 0<=index<count:raise ValueError('INVALID_TASK_SHARD')
    return int.from_bytes(hashlib.sha256(str(tid).encode()).digest()[:8],'big')%count==index


def diagnose_case(case, attempts=0):
    error=str(case.get('error',''));log=Path(case['evidence'])/'native_validation.log'
    text=log.read_text(errors='replace') if log.exists() else ''
    def events(marker):
        rows=[]
        for line in text.splitlines():
            if marker in line:
                try:rows.append(json.loads(line.split(marker,1)[1]))
                except ValueError:pass
        return rows
    failed=events('[first-placement-failure] ');contacts=events('[inside-contact-audit] ')
    plans=events('[expert-layout-plan] ');support=events('[expert-support-plane] ')
    last=re.findall(r'\[placement\].*? start (\S+) (inside|on_top) (\S+) link=',text)
    first=failed[0] if failed else (dict(entity=last[-1][0],relation=last[-1][1],receptacle=last[-1][2]) if last else {})
    if 'EXPERT_LAYOUT' in error:category='restricted_packing'
    elif 'EXPERT_SUPPORT' in error:category='support_geometry'
    elif 'EXPERT_CONTAINER' in error or 'OPEN_STATE_CHANGED' in error:category='joint_interference'
    elif 'no Open state' in error or 'capability mismatch' in error.lower():category='capability'
    elif 'CONVERGED' in error or 'POST_SETTLE' in error:category='resting_state'
    elif first.get('relation')=='on_top' or 'timeout' in error:category='surface_or_timeout'
    else:category='native_placement'
    related=[s for s in support if s.get('receptacle')==first.get('receptacle')]
    return dict(category=category,error=error,native_rejections_in_current_queue=attempts,
        first_blocker=first,solver=plans[-1] if plans else None,
        support_measurements=dict(total=len(related),planar=sum(s['status']=='measured_plane' for s in related),
                                  reasons=dict(Counter(s.get('reason','measured_plane') for s in related))),
        first_contact=contacts[0] if contacts else None,
        evidence=str(log),inference_scope='proposal/model diagnosis; native physics acceptance unchanged')


def intervention_for(diagnosis, instruction):
    category=diagnosis['category']
    specific=has_qualified_storage_location(instruction) or bool(re.search(r'\bdrawer\b',storage_scope_text(instruction),re.I))
    return dict(method='relocate_storage' if category in ('joint_interference','capability','resting_state') else 'correct_proposal',
        clear_unstated_links=not specific,plan_surfaces=category=='surface_or_timeout',
        runtime_changes=['closed_native_bbox_XY','disjoint_measured_support_patches','bounded_CUDA_candidate_domains'],
        original_instruction_retained=True,explicit_location_scope_retained=True,
        diagnosis=category,reason=diagnosis['error'])


def native_runtime_unchanged(case,runtime):
    report=read(Path(case['evidence'])/'validation.json');sources=report.get('runtime_sources',{})
    if not sources:return False
    runtime=Path(runtime)
    for relative,digest in sources.items():
        path=runtime/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:return False
    settings=read(runtime/'release.json').get('enable',{})
    return all(report.get('runtime_environment',{}).get(k)==v for k,v in settings.items())


def preparation_revision():
    """Invalidate pending offline proposals after causal authoring changes.

    Evidence timestamps and changing logs do not invalidate preparation;
    executable rules, explicit data revisions and learned sampling choices do.
    """
    digest=hashlib.sha256()
    for name in ['scripts/v2/expert_storage_repair.py','scripts/v2/constructed_storage.py',
                 'scripts/v2/audit_dense_layouts.py','src/rummagebench/sim/omnigibson/inside_volume.py',
                 'build/expert_editorial_revisions/index.json','build/expert_support_overrides/index.json',
                 'build/expert_scene_contact_repairs/index.json',
                 'build/native_fillable_geometry/index.json',
                 'build/expert_surface_binding_audit/index.json']:
        path=ROOT/name
        if path.is_file():digest.update(name.encode());digest.update(path.read_bytes())
    sampling=read(ROOT/'build/verified_surface_sites/index.json').get('assets',{})
    digest.update(json.dumps(sorted(key for key,row in sampling.items() if row.get('native_sampling_evidence'))).encode())
    profiles=read(ROOT/'build/native_support_geometry/index.json').get('assets',{})
    domains={key:{field:value for field,value in profile['native_proposal_domains'].items()
                   if field in ['volumes','native_links','center_halfspaces','obstacles','overlap_groups','supports']}
             for key,profile in profiles.items() if profile.get('native_proposal_domains')}
    digest.update(json.dumps(domains,sort_keys=True).encode())
    openings={key:row['opened_footprint_offsets_m'] for key,row in profiles.items() if row.get('opened_footprint_offsets_m')}
    digest.update(json.dumps(openings,sort_keys=True).encode())
    return digest.hexdigest()[:12]


def authoring_revision(output):
    """Content identity, excluding changing logs/timestamps/evidence filenames."""
    digest=hashlib.sha256()
    paths=['scripts/v2/expert_storage_repair.py','scripts/v2/expert_batch_repair.py','scripts/v2/overnight_asset_queue.py','scripts/v2/constructed_storage.py',
           'scripts/v2/audit_dense_layouts.py','build/expert_support_overrides/index.json','build/expert_editorial_revisions/index.json',
           'build/native_fillable_geometry/index.json','build/expert_scene_contact_repairs/index.json',
           'build/expert_surface_binding_audit/index.json','scripts/v2/generate_dense_pilot.py',
           'build/verified_surface_sites/index.json',
           'scripts/v2/check_frozen_layout_runtime.py',
           'src/rummagebench/sim/omnigibson/inside_volume.py']
    for relative in paths:
        path=ROOT/relative
        if path.is_file():digest.update(relative.encode());digest.update(path.read_bytes())
    profiles=read(ROOT/'build/native_support_geometry/index.json').get('assets',{})
    knowledge={asset:{key:dict(status=value.get('status'),patches=len(value.get('patches',[])))
        for key,value in profile.get('components',{}).items()} for asset,profile in profiles.items()}
    digest.update(json.dumps(knowledge,sort_keys=True).encode())
    verified=read(ROOT/'build/verified_storage_sites/index.json')
    digest.update(json.dumps(dict(sites=sorted(verified.get('sites',{})),assets=sorted(verified.get('asset_successes',{}))),sort_keys=True).encode())
    staged=read(Path(output)/'staged_expert_candidates.json')
    schedule=read(Path(output)/'scheduling.json')
    digest.update(json.dumps({key:schedule.get(key) for key in ('cohort_only','shard_count','shard_index')},sort_keys=True).encode())
    digest.update(json.dumps({tid:proposal.get('record',{}).get('scenario_sha256') for tid,proposal in staged.items()
                              if proposal.get('status')=='proposed'},sort_keys=True).encode())
    runtime=read(Path(output)/'runtime_override.json').get('runtime')
    if runtime:
        release=Path(runtime)/'release.json'
        if release.is_file():digest.update(release.read_bytes())
    return digest.hexdigest()[:12]


def synchronize_verified_runtime(output,primary=None):
    """Promote only an immutable runtime with real probe and control pairs."""
    output=Path(output);primary=Path(primary or PRIMARY)
    override=read(output/'runtime_override.json');source=override.get('source')
    if not source:return False
    report=read(output/source);rows=report.get('rows',{});results=report.get('results',{})
    controls=['F035','F090'];probes=[t for t,v in results.items() if v and t not in controls]
    if not probes or not all(results.get(t) for t in controls):return False
    runtime=Path(override['runtime'])
    settings=read(runtime/'release.json').get('enable',{})
    for tid in controls+[probes[0]]:
        row=rows.get(tid)
        if not row:return False
        recipe=Path(row['scenario']);directory=output/recipe.parent.parent.name/row['id']
        if not accepted_native_pair(recipe,directory):return False
        validation=read(directory/'validation.json');sources=validation.get('runtime_sources',{})
        if not sources or any(not (runtime/p).is_file() or hashlib.sha256((runtime/p).read_bytes()).hexdigest()!=sha for p,sha in sources.items()):return False
        if any(validation.get('runtime_environment',{}).get(k)!=v for k,v in settings.items()):return False
    # The receiving lane must not resolve a relative evidence filename in
    # its own directory; those proofs belong to the producing lane.
    doc=dict(override,source=str((output/source).resolve()),verified_by=str((output/source).resolve()))
    if read(primary/'runtime_override.json')!=doc:write(primary/'runtime_override.json',doc)
    return True


def remaining_failure_cases(output,cohort):
    """Cover the other failures using hash-matched observations, not stale IDs."""
    output=Path(output)
    pending={t['id'] for t in read(ROOT/'runs/asset_full_queue_20261105/queue.json').get('tasks',[])
             if t.get('status')!='ready_for_human_review'}
    cohort_ids={c['id'] for c in cohort}
    candidates=[read(output/'surface_binding_repair_cases.json'),read(output/'latest_cases.json'),read(PRIMARY/'latest_cases.json'),
                {c['id']:c for c in read(PRIMARY/'source_index.json',[])}]
    cases=[];unbound=[]
    for tid in sorted(pending-cohort_ids):
        chosen=None
        for index in candidates:
            candidate=index.get(tid);row=candidate.get('record') if candidate else None
            if not row or not candidate.get('evidence'):continue
            recipe=ROOT/row['scenario'];report=read(Path(candidate['evidence'])/'validation.json')
            if recipe.is_file() and report.get('scenario_sha256')==hashlib.sha256(recipe.read_bytes()).hexdigest():
                chosen=dict(candidate,record=dict(row,scenario=str(recipe.resolve())));break
        if chosen is None:
            unbound.append(dict(id=tid,reason='No source recipe matching an observed native report; requires expert rebinding'))
            continue
        diagnosis=diagnose_case(chosen)
        chosen.update(expert_diagnosis=diagnosis,expert_policy=intervention_for(diagnosis,chosen['record']['instruction']))
        cases.append(chosen)
    return cases,dict(total_remaining=len(pending),cohort_remaining=len(pending&cohort_ids),
        other_remaining=len(pending-cohort_ids),queued=[c['id'] for c in cases],unbound=unbound,
        native_workers=1,scheduling='one other failure between expert attempts; drain the remainder each revision',
        acceptance='fresh native initialization plus independent frozen-state reload')


def select_cohort(limit):
    ready={r['parent_task_id'] for r in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[])}
    index={c['id']:c for c in read(PRIMARY/'source_index.json',[])}
    latest=read(PRIMARY/'latest_cases.json');jobs=read(PRIMARY/'jobs.json')
    attempts=Counter(j['task'] for j in jobs.values() if 'task' in j and j.get('exit_code') is not None and not j.get('accepted'))
    groups={};excluded=[]
    for tid,initial in index.items():
        if tid in ready:continue
        case=latest.get(tid,initial);row=case.get('record')
        if not row or not case.get('evidence'):continue
        path=ROOT/row['scenario'];build=read(Path(case['evidence'])/'validation.json')
        digest=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if not digest or digest!=build.get('scenario_sha256'):
            excluded.append(dict(id=tid,reason='No source recipe matching observed native report'));continue
        row=dict(row,scenario=str(path.resolve()))
        diagnosis=diagnose_case(case,attempts[tid])
        selected=dict(case,record=row,source_sha256=digest,expert_diagnosis=diagnosis,
                      expert_policy=intervention_for(diagnosis,row['instruction']))
        groups.setdefault(diagnosis['category'],[]).append(selected)
    quotas={'restricted_packing':40,'support_geometry':20,'native_placement':15,
            'joint_interference':10,'surface_or_timeout':5,'resting_state':5,'capability':5}
    cohort=[];scenes=Counter()
    def rank(case):return (-case['expert_diagnosis']['native_rejections_in_current_queue'],case['id'])
    for category,quota in quotas.items():
        added=0
        for case in sorted(groups.get(category,[]),key=rank):
            if len(cohort)>=limit or added>=quota:break
            scene=case['record']['scene']
            if scenes[scene]>=15:continue
            cohort.append(case);scenes[scene]+=1;added+=1
    ids={c['id'] for c in cohort}
    remaining=sorted([c for rows in groups.values() for c in rows if c['id'] not in ids],key=rank)
    cohort.extend(remaining[:max(0,limit-len(cohort))])
    if len(cohort)!=limit:raise ValueError(f'Only {len(cohort)} hash-matched failures available for requested {limit}')
    return cohort,excluded


class ExpertBatch(Queue):
    def __init__(self,args):
        super().__init__(args)
        self.runtime=Path(read(self.out/'runtime_override.json').get('runtime') or
                          ROOT/'build/generator_runtime_expert_layout_v11_20261110')
        self.primary_pid=None

    def owns_task(self,tid):
        config=read(self.out/'scheduling.json')
        return shard_owns_task(tid,int(config.get('shard_index',0)),int(config.get('shard_count',1)))

    def bound_identical_failures(self,proposal):
        """Require a real input/runtime repair after repeated physical failures."""
        if proposal.get('status')!='proposed':return proposal
        limit=int(read(self.out/'scheduling.json').get('max_identical_native_failures',2))
        if limit<1:return proposal
        row=proposal['record'];sha=row.get('scenario_sha256');evidence=set()
        runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
        from overnight_asset_queue import infrastructure_error
        roots={self.out,ROOT/'runs/expert100_20261110',ROOT/'runs/expert100_lane2_20261111'}
        for job in self.ledger.values():
            prior=job.get('proposal',{}).get('record',{})
            if job.get('accepted') or not sha or prior.get('scenario_sha256')!=sha:continue
            for root in roots:
                directory=root/job.get('lane','')/row['id']
                a=read(directory/'validation.json');b=read(directory/'restore_validation.json')
                report=b if a.get('status')=='initialization_validated' and b else a
                if (report.get('status')=='failed' and report.get('scenario_sha256')==sha
                        and not infrastructure_error(report.get('error',''))
                        and native_runtime_unchanged(dict(evidence=str(directory)),runtime)):
                    evidence.add(str(directory.resolve()))
            if len(evidence)>=limit:
                return dict(status='deferred',reason='Identical recipe and native runtime already failed twice; require a changed intervention before another native attempt.',
                            scenario_sha256=sha,evidence=sorted(evidence),physical_validation_required=True)
        return proposal

    def staged_expert_candidate(self,proposal,lane,case):
        """Install an expert-authored immutable candidate, still native-pending."""
        row=proposal['record'];source=Path(row['scenario'])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=row.get('scenario_sha256'):
            raise ValueError('STAGED_EXPERT_RECIPE_CHANGED')
        if row.get('parent_task_id')!=case['id']:raise ValueError('STAGED_EXPERT_TASK_MISMATCH')
        runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
        if (case.get('record',{}).get('scenario_sha256')==row['scenario_sha256'] and
                native_runtime_unchanged(case,runtime)):
            return dict(status='deferred',reason='This staged recipe already failed with the same native runtime')
        catalog=self.cat/lane;destination=catalog/row['id']/'scenario.yaml'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists() and destination.read_bytes()!=source.read_bytes():raise ValueError('STAGED_EXPERT_LANE_CHANGED')
        shutil.copy2(source,destination)
        installed=dict(row,scenario=str(destination.resolve()))
        write(catalog/'manifest.json',dict(tasks=[installed]))
        write(destination.parent/'expert_handoff.json',dict(source_recipe=str(source),
            source_sha256=row['scenario_sha256'],source_intervention=proposal.get('evidence'),
            physical_validation_required=True))
        return dict(status='proposed',record=installed,expert_authored_candidate=True)

    def refresh_external_cases(self):
        """Consume newer hash-bound probe evidence; do not revert data revisions."""
        changed=False
        for tid,case in read(self.out/'external_cases.json').items():
            row=case.get('record',{});recipe=Path(row.get('scenario','/nonexistent'))
            report=Path(case.get('evidence','/nonexistent'))/'validation.json'
            if not recipe.is_file() or not report.is_file():continue
            if read(report).get('scenario_sha256')!=hashlib.sha256(recipe.read_bytes()).hexdigest():continue
            previous=Path(self.latest.get(tid,{}).get('evidence','/nonexistent'))/'validation.json'
            if previous.is_file() and previous.stat().st_mtime_ns>=report.stat().st_mtime_ns:continue
            self.latest[tid]=case;changed=True
        if changed:write(self.out/'latest_cases.json',self.latest)

    def advance_other(self,background,revision,*,native_used=False,initial=False):
        """Hot-readable priority, measured in native expert attempts, not skips."""
        if self.stop_requested:return
        config=read(self.out/'scheduling.json')
        if config.get('cohort_only'):return
        ratio=max(1,min(32,int(config.get('cohort_native_attempts_per_other',1))))
        count=getattr(self,'_cohort_native_since_other',0)+int(native_used)
        self._cohort_native_since_other=count
        if count<ratio and not (initial and ratio==1):return
        case=next(background,None)
        if case is not None:
            self.run_other_failure(case,revision);self._cohort_native_since_other=0

    def run_other_failure(self,initial,revision):
        tid=initial['id'];phase='full_queue_'+revision;key=phase+':'+tid
        if read(self.out/'scheduling.json').get('cohort_only'):return
        if not self.owns_task(tid) or key in self.ledger or self.accepted(tid):return
        case=initial;lane=phase+'_'+tid;started=time.time()
        policy=dict(initial['expert_policy'],method='relocate_storage',reason=case.get('error'),plan_surfaces=True)
        case=dict(case,expert_policy=policy)
        self.status(stage='repair_other_failure',task=tid,phase=phase,runtime=str(self.runtime))
        try:
            runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
            staged=read(self.out/'staged_expert_candidates.json').get(tid,{})
            if staged.get('status')=='proposed':proposal=self.staged_expert_candidate(staged,lane,case)
            else:proposal=self.prepare(case,lane,'direct') if not native_runtime_unchanged(case,runtime) else dict(status='deferred')
            if proposal.get('status')!='proposed':proposal=self.prepare(case,lane,'expert',80)
        except Exception as exc:proposal=dict(status='error',reason=str(exc));traceback.print_exc()
        proposal=self.bound_identical_failures(proposal)
        result=dict(task=tid,phase=phase,lane=lane,proposal=proposal,started=started,scope='other_failed_task')
        if proposal.get('status')=='proposed':
            try:result.update(self.validate(proposal['record'],lane,case))
            except Exception as exc:result.update(status='error',error=str(exc));traceback.print_exc()
        result['finished']=time.time();self.ledger[key]=result;write(self.out/'jobs.json',self.ledger)
        current=[j for k,j in self.ledger.items() if k.startswith(phase+':')]
        write(self.out/'other_failures_status.json',dict(updated=time.time(),revision=revision,current_task=tid,
            attempted=len(current),native_attempted=sum('accepted' in j for j in current),
            native_passed=sum(j.get('accepted',False) for j in current),last_result=result,
            acceptance='fresh native initialization plus independent frozen-state reload'))

    def progress(self,cases,stage,**extra):
        ids={c['id'] for c in cases}
        outcomes=read(self.out/'prepared.json');accepted={j.get('task') for j in self.ledger.values() if j.get('accepted') and j.get('task') in ids}
        published={r.get('parent_task_id') for r in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[])}
        accepted |= ids & published
        static={tid for tid,value in outcomes.items() if value.get('status')=='proposed'}
        static.update(tid for tid,value in read(self.out/'static_retry_v4.json').items() if value.get('status')=='proposed')
        static.update(tid for tid,value in read(self.out/'staged_expert_candidates.json').items() if value.get('status')=='proposed' and tid in ids)
        write(self.out/'batch_status.json',dict(stage=stage,updated=time.time(),selected=len(cases),
            prepared=len(static),static_deferred=len(ids-static),
            native_passed=len(accepted),native_attempts=sum('accepted' in j and j.get('task') in ids for j in self.ledger.values()),
            native_workers=1,parallel_native_workers=int(read(self.out/'scheduling.json').get('shard_count',1)),
            shard_index=int(read(self.out/'scheduling.json').get('shard_index',0)),primary_queue_paused=self.primary_pid,**extra))
        ready={r['parent_task_id']:r for r in read(ROOT/'runs/asset_full_queue_20261105/human_ready.json').get('tasks',[]) if r.get('parent_task_id') in ids}
        temp=self.out/'cases.csv.tmp'
        with temp.open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['task','original_scene','original_instruction','failure_category','intervention','initial_evidence',
                'initialization_and_reload','accepted_recipe','latest_native_error','latest_native_evidence','current_scene','current_instruction'])
            for case in cases:
                tid=case['id'];latest=self.latest.get(tid,case);passed=tid in accepted
                writer.writerow([tid,case['record']['scene'],case['record']['instruction'],case['expert_diagnosis']['category'],
                    case['expert_policy']['method'],case['evidence'],'passed' if passed else 'not_passed',
                    ready.get(tid,{}).get('scenario',''),'' if passed else latest.get('error',''),latest.get('evidence',''),
                    ready.get(tid,latest.get('record',{})).get('scene',''),ready.get(tid,latest.get('record',{})).get('instruction','')])
        temp.replace(self.out/'cases.csv')

    def prepare_all(self,cases):
        prepared=read(self.out/'prepared.json')
        for case in cases:
            if not self.owns_task(case['id']):continue
            if self.stop_requested:return
            tid=case['id']
            if tid in prepared:continue
            self.status(stage='per_case_expert_preparation',task=tid)
            try:result=self.prepare(case,'initial_'+tid,'expert',10)
            except Exception as exc:result=dict(status='error',reason=str(exc));traceback.print_exc()
            prepared[tid]=result;write(self.out/'prepared.json',prepared)
            self.progress(cases,'preparing',current_task=tid)
        self.progress(cases,'prepared')

    def pause_primary(self):
        status=read(PRIMARY/'controller_status.json');pid=status.get('pid')
        if not pid:return
        cmd=Path(f'/proc/{pid}/cmdline')
        if not cmd.exists():return
        if b'overnight_asset_queue.py' not in cmd.read_bytes():raise RuntimeError('Primary queue identity changed')
        os.kill(pid,signal.SIGSTOP);self.primary_pid=pid
        self.status(stage='waiting_primary_native_boundary',primary_queue=pid)
        for child in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
            try:command=Path(f'/proc/{child}/cmdline').read_bytes()
            except FileNotFoundError:continue
            if b'expert_storage_repair.py' not in command:wait_external(int(child))

    def run_selected(self,cases):
        if self.args.gate_pid:
            self.status(stage='waiting_runtime_controls',wait_pid=self.args.gate_pid);wait_external(self.args.gate_pid)
        gate=read(PRIMARY/'expert_v8_canary_status.json')
        if not gate.get('results') or not all(gate['results'].values()):
            self.progress(cases,'runtime_controls_failed',controls=gate);return
        if not read(self.out/'runtime_override.json').get('runtime'):
            write(self.out/'runtime_override.json',dict(runtime=str(self.runtime),controls=gate['results']))
        synchronize_verified_runtime(self.out)
        self.pause_primary()
        prepared=read(self.out/'prepared.json')
        self.refresh_external_cases()
        try:
            if self.stop_requested:self.progress(cases,'interrupted');return
            for tid,lane in [('F035','control_v11_F035'),('F090','control_v11_F090')]:
                key=f'control_v11:{tid}'
                if key not in self.ledger:
                    source=ROOT/'scenarios/overnight_expert_20261109/expert_v8_canary_20261110'
                    row=next(r for r in read(source/'manifest.json')['tasks'] if r['parent_task_id']==tid)
                    control=dict(id=tid,record=row,evidence=str(PRIMARY/'expert_v8_canary_20261110'/row['id']))
                    self.progress(cases,'validating_runtime_controls',current_task=tid)
                    proposal=self.prepare(control,lane,'direct')
                    result=dict(task=tid,phase='runtime_control',proposal=proposal)
                    if proposal.get('status')=='proposed':result.update(self.validate(proposal['record'],lane,control))
                    self.ledger[key]=result;write(self.out/'jobs.json',self.ledger)
                if not self.ledger[key].get('accepted'):
                    self.progress(cases,'runtime_controls_failed',control=tid);return
            # Revisit earlier native failures after causal generator fixes,
            # without discarding their immutable failed attempts.
            priority=read(self.out/'priority_tasks.json',[])
            owned_cases=[c for c in cases if self.owns_task(c['id'])]
            ordered=sorted(owned_cases,key=lambda c:priority.index(c['id']) if c['id'] in priority else len(priority))
            revision=authoring_revision(self.out);revision_phase='expert_'+revision
            runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
            policy=native_policy_signature(read(Path(runtime)/'release.json').get('source_sha256',{}))
            runtime_phase='runtime_'+(policy or revision)[:12]
            other=[];unbound=[]
            if getattr(self.args,'include_all_failures',False) and not read(self.out/'scheduling.json').get('cohort_only'):
                other,plan=remaining_failure_cases(self.out,cases)
                other=[c for c in other if self.owns_task(c['id'])]
                unbound=[c['id'] for c in plan['unbound']]
                unbound=[tid for tid in unbound if self.owns_task(tid)]
                plan.update(shard_index=int(read(self.out/'scheduling.json').get('shard_index',0)),
                            shard_count=int(read(self.out/'scheduling.json').get('shard_count',1)),
                            queued=[c['id'] for c in other],unbound=[c for c in plan['unbound'] if self.owns_task(c['id'])])
                write(self.out/'all_failures_plan.json',dict(plan,revision=revision,updated=time.time(),runtime=str(self.runtime)))
            background=iter(c for c in other if 'full_queue_'+revision+':'+c['id'] not in self.ledger)
            self.advance_other(background,revision,initial=True)
            round_budget_deferred=False
            for initial in ordered:
                staged=read(self.out/'staged_expert_candidates.json').get(initial['id'])
                staged_phase=('candidate_'+staged['record']['scenario_sha256'][:12]) if staged and staged.get('status')=='proposed' else None
                phases=([staged_phase] if staged_phase else [])+[runtime_phase,'convex_v11','initial','fixture','alternative','release_links_v2','geometry_v2','fixture_site_v3',
                              'live_domains_v14','fixture_retry_v5',revision_phase]
                native_for_task=0
                for phase in phases:
                    if self.stop_requested:self.progress(cases,'interrupted');return
                    tid=initial['id'];key=f'{phase}:{tid}'
                    if key in self.ledger or self.accepted(tid):continue
                    budget=max(1,min(8,int(read(self.out/'scheduling.json').get('cohort_native_attempts_per_task_round',2))))
                    if native_for_task>=budget:
                        round_budget_deferred=True;break
                    self.refresh_external_cases()
                    case=self.latest.get(tid,initial);lane=f'{phase}_{tid}';started=time.time()
                    self.progress(cases,'expert_native_validation',current_task=tid,phase=phase)
                    if phase==staged_phase:
                        try:proposal=self.staged_expert_candidate(staged,lane,case)
                        except Exception as exc:proposal=dict(status='error',reason=str(exc));traceback.print_exc()
                    elif phase in ('convex_v11','live_domains_v14') or phase.startswith('runtime_'):
                        runtime=read(self.out/'runtime_override.json').get('runtime') or self.runtime
                        try:
                            if native_runtime_unchanged(case,runtime):
                                proposal=dict(status='deferred',reason='Observed recipe already failed with identical native sources and settings; no unchanged replay')
                            else:proposal=self.prepare(case,lane,'direct')
                        except Exception as exc:
                            proposal=dict(status='error',reason=str(exc));traceback.print_exc()
                    elif phase=='initial':proposal=prepared[tid]
                    else:
                        # Follow the newly observed blocker, not a stale source
                        # or a second copy of the preceding failed layout.
                        policy=dict(initial['expert_policy'],method='correct_proposal' if phase=='release_links_v2' else 'relocate_storage',
                                    reason=case.get('error'),plan_surfaces=initial['expert_policy']['plan_surfaces'])
                        case=dict(case,expert_policy=policy)
                        variants={'fixture':11,'alternative':12,'release_links_v2':20,'geometry_v2':21,'fixture_site_v3':30,'fixture_retry_v5':60}
                        try:proposal=self.prepare(case,lane,'expert',variants.get(phase,70))
                        except Exception as exc:
                            proposal=dict(status='error',reason=str(exc));traceback.print_exc()
                    proposal=self.bound_identical_failures(proposal)
                    result=dict(task=tid,phase=phase,lane=lane,proposal=proposal,started=started)
                    if proposal.get('status')=='proposed':
                        try:result.update(self.validate(proposal['record'],lane,case))
                        except Exception as exc:result.update(status='error',error=str(exc));traceback.print_exc()
                    result['finished']=time.time();self.ledger[key]=result;write(self.out/'jobs.json',self.ledger)
                    native_for_task+=int('accepted' in result)
                    self.progress(cases,'expert_native_validation',current_task=tid,phase=phase)
                    self.advance_other(background,revision,native_used='accepted' in result)
            for initial in background:
                if read(self.out/'scheduling.json').get('cohort_only'):break
                if round_budget_deferred:break
                if self.stop_requested:self.progress(cases,'interrupted');return
                self.run_other_failure(initial,revision)
            unresolved=[c['id'] for c in owned_cases if not self.accepted(c['id'])]
            write(self.out/'unresolved.json',unresolved)
            other_unresolved=[c['id'] for c in other if not self.accepted(c['id'])]+unbound
            write(self.out/'other_unresolved.json',other_unresolved)
            self.progress(cases,'complete' if not unresolved and not other_unresolved else 'needs_further_expert_work',
                          remaining=unresolved,other_remaining=other_unresolved,round_budget_deferred=round_budget_deferred)
        finally:
            if self.primary_pid:
                try:os.kill(self.primary_pid,signal.SIGCONT)
                except ProcessLookupError:pass
                self.primary_pid=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/expert100_20261110')
    parser.add_argument('--limit',type=int,default=100);parser.add_argument('--run',action='store_true')
    parser.add_argument('--include-all-failures',action='store_true',help='Interleave repairs of every other failed task using the same verified generator')
    parser.add_argument('--gate-pid',type=int);parser.add_argument('--native-timeout',type=int,default=600)
    args=parser.parse_args();args.wait_pid=None
    queue=ExpertBatch(args)
    signal.signal(signal.SIGTERM,queue.stop);signal.signal(signal.SIGINT,queue.stop)
    with (queue.out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=read(queue.out/'source_index.json',[])
        if not cases:
            cases,excluded=select_cohort(args.limit);write(queue.out/'source_index.json',cases)
            write(queue.out/'source_mismatch_exclusions.json',excluded)
            write(queue.out/'diagnosis.json',dict(selected=len(cases),categories=dict(Counter(c['expert_diagnosis']['category'] for c in cases)),
                scenes=dict(Counter(c['record']['scene'] for c in cases)),tasks=[dict(id=c['id'],instruction=c['record']['instruction'],
                diagnosis=c['expert_diagnosis'],intervention=c['expert_policy']) for c in cases]))
            with (queue.out/'cases.csv').open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['task','scene','instruction','failure_category','intervention','evidence'])
                for c in cases:writer.writerow([c['id'],c['record']['scene'],c['record']['instruction'],c['expert_diagnosis']['category'],c['expert_policy']['method'],c['evidence']])
        with (queue.out/'geometry_worker.log').open('a') as log:
            queue.helper_log=log;queue.start_helper()
            try:
                queue.prepare_all(cases)
                if args.run and not queue.stop_requested:queue.run_selected(cases)
            finally:
                if queue.helper and queue.helper.poll() is None:queue.helper.terminate();queue.helper.wait()


if __name__=='__main__':main()
