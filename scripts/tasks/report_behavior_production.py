"""Read-only live production counts from completed, hash-bound stage receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from _common import REPO
from rummagebench.authoring.tasks.plan import sha256_path


def environment_assessment(process, folder, acceptance):
    """Validate physical receipts; never infer readiness from policy outcomes."""
    def review(reason):
        return dict(status='needs_review', reason=reason)

    if 'environment_ready' not in process['stage_sha256']:
        return review('NO_ENVIRONMENT_ONLY_RECEIPT')
    result=json.loads((folder/'result.json').read_text())
    validation=process['validation']
    if (validation.get('restore') is not True or validation.get('profile')!='observed-motion-v3'
            or validation.get('mode')!='native-realization'):
        return review('MISSING_PHYSICAL_VALIDATION_PROFILE')
    expected={'background_realization':{'support_and_stability_passed_pending_family'},
              'background_restore':{'restore_verified_pending_family'},
              'background_after_target':{'daily_props_physically_ready'}}
    if acceptance=='execution-v1':
        expected['background_realization'].add('support_and_pose_passed_motion_diagnostic')
    for stage,statuses in expected.items():
        if stage not in process['stage_sha256']:
            return review('MISSING_STAGE:'+stage)
        evidence=json.loads((folder/f'{stage}.json').read_text())
        if evidence.get('status') not in statuses:
            return review('STAGE_REQUIRES_REVIEW:'+stage)
        if (evidence.get('assembly_sha256')!=process['assembly_sha256']
                or evidence.get('acceptance_policy','strict-v3')!=acceptance):
            raise ValueError('PRODUCTION_ENVIRONMENT_STAGE_BINDING_MISMATCH:'+stage)
        if stage=='background_restore':
            if evidence.get('checkpoint_sha256')!=sha256_path(folder/'restorable_background.pt'):
                raise ValueError('PRODUCTION_CHECKPOINT_CHANGED:'+process.get('id',folder.parent.name))
    if (result.get('scenario_sha256')!=process['scenario_sha256']
            or result.get('assembly_sha256')!=process['assembly_sha256']
            or result.get('acceptance_policy','strict-v3')!=acceptance
            or result.get('verify_background_restore') is not True
            or result.get('runner_sha256')!=validation.get('sources',{}).get('scripts/tasks/behavior_pipeline.py')
            or not result.get('runner_sha256')):
        raise ValueError('PRODUCTION_ENVIRONMENT_RESULT_BINDING_MISMATCH')
    if (process.get('returncode')!=0 or result.get('status')!='environment_ready'
            or process.get('status')!='environment_ready' or result.get('error')
            or result.get('process_error') or process.get('error')
            or result.get('environment_only') is not True or validation.get('environment_only') is not True):
        return review('RUNTIME_ERROR_OR_INCOMPLETE_RECEIPT')
    evidence=json.loads((folder/'environment_ready.json').read_text())
    if (evidence.get('status')!='environment_ready'
            or evidence.get('assembly_sha256')!=process['assembly_sha256']
            or evidence.get('policy_executed') is not False
            or any(evidence.get(k) is not True for k in ('public_reset','spawn_clear','target_relation'))
            or result.get('attempts')):
        return review('INVALID_ENVIRONMENT_ONLY_EVIDENCE')
    for key,name in [('checkpoint_sha256','environment_start.pt'),('image_sha256','environment_start.png')]:
        if evidence.get(key)!=sha256_path(folder/name):
            raise ValueError('PRODUCTION_ENVIRONMENT_ARTIFACT_CHANGED:'+name)
    return dict(status='ready', reason='PHYSICAL_RESET_CAMERA_AND_CHECKPOINT_VERIFIED')



def report(root):
    root=root.resolve();batch=root/'batch'
    status_path=root/'production_status.json'
    coordinator=json.loads(status_path.read_text()) if status_path.exists() else {}
    acceptance=coordinator.get('active_acceptance_policy','strict-v3')
    exports={}
    for p in sorted((root/'exports').glob('*/live/process.json')):
        d=json.loads(p.read_text());exports[p.parent.parent.name]=d['status']
    summary_path=batch/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    rows=json.loads((batch/'records.json').read_text()) if (batch/'records.json').exists() else []
    selected=set(summary.get('selected_ids',[]));checked=[];failures=Counter();pending=[]
    stage_counts=Counter();skill_failures=Counter();environments=[]
    for row in rows:
        if row['id'] not in selected:continue
        folders=list((batch/row['id']).glob('gpu_realized_*/process.json'))
        finished=[];finished_folders=[]
        for receipt in folders:
            process=json.loads(receipt.read_text())
            # The outer coordinator adds the validation identity after the
            # isolated process completes. Its earlier inner receipt is pending.
            if 'validation' not in process:continue
            if process['validation'].get('acceptance_policy','strict-v3')!=acceptance:continue
            if bool(coordinator.get('environment_only'))!=bool(process['validation'].get('environment_only')):continue
            if coordinator.get('active_validation_id') and receipt.parent.name!='gpu_realized_'+coordinator['active_validation_id']:continue
            if (process['scenario_sha256']!=row['scenario_sha256']
                    or process['assembly_sha256']!=row['assembly_sha256']
                    or process['result_sha256']!=sha256_path(receipt.parent/'result.json')):
                raise ValueError('PRODUCTION_RECEIPT_CHANGED:'+row['id'])
            for stage,digest in process['stage_sha256'].items():
                if sha256_path(receipt.parent/f'{stage}.json')!=digest:
                    raise ValueError('PRODUCTION_STAGE_CHANGED:'+row['id'])
            if process['status']=='execution_smoke_passed':
                required={'background_realization','background_after_target','background_restore'}
                if not required.issubset(process['stage_sha256']):
                    raise ValueError('PRODUCTION_SUCCESS_WITHOUT_BOUND_STAGES:'+row['id'])
                if (process['returncode']!=0 or not process['validation']['restore']
                        or process['validation']['profile']!='observed-motion-v3'
                        or process['validation']['mode']!='native-realization'):
                    raise ValueError('PRODUCTION_SUCCESS_WITHOUT_RESTORE:'+row['id'])
                result=json.loads((receipt.parent/'result.json').read_text())
                if (result.get('status')!='execution_smoke_passed'
                        or result.get('verify_background_restore') is not True
                        or result.get('acceptance_policy','strict-v3')!=acceptance):
                    raise ValueError('PRODUCTION_RESULT_NOT_PASSED:'+row['id'])
                restoration=json.loads((receipt.parent/'background_restore.json').read_text())
                if restoration['status']!='restore_verified_pending_family':
                    raise ValueError('PRODUCTION_RESTORE_NOT_PASSED:'+row['id'])
                if restoration.get('acceptance_policy','strict-v3')!=acceptance:
                    raise ValueError('PRODUCTION_RESTORE_POLICY_MISMATCH:'+row['id'])
                if restoration['checkpoint_sha256']!=sha256_path(receipt.parent/'restorable_background.pt'):
                    raise ValueError('PRODUCTION_CHECKPOINT_CHANGED:'+row['id'])
            finished.append(process)
            finished_folders.append(receipt.parent)
        if len(finished)>1:raise ValueError('AMBIGUOUS_PRODUCTION_VALIDATION:'+row['id'])
        if not finished:pending.append(row['id']);continue
        process=finished[0];checked.append(process)
        skill_failures.update(process.get('failures',{}))
        folder=finished_folders[0]
        environments.append(dict(id=row['id'], evidence=str(folder),
            **environment_assessment(process,folder,acceptance)))
        for stage,passed in {'background_realization':'support_and_stability_passed_pending_family',
                             'background_restore':'restore_verified_pending_family',
                             'background_after_target':'daily_props_unchanged_within_tolerance'}.items():
            if stage in process['stage_sha256']:
                evidence=json.loads((folder/f'{stage}.json').read_text())
                if (evidence['status']==passed or (stage=='background_after_target' and evidence['status']=='daily_props_physically_ready') or (stage=='background_realization' and acceptance=='execution-v1'
                        and evidence['status']=='support_and_pose_passed_motion_diagnostic')):stage_counts[stage]+=1
        if process['status'] not in ('execution_smoke_passed','environment_ready'):
            reason=process.get('error') or process['status']
            if reason=='EXIT_1_OR_MISSING_FINAL_RECEIPT' and process.get('failures'):
                reason='PUBLIC_SKILL_CHAIN_REJECTED'
            failures[reason]+=1
    counts=Counter(p['status'] for p in checked)
    ready=[e['id'] for e in environments if e['status']=='ready']
    result=dict(coordinator_status=coordinator.get('status','not_started'),
        readiness_policy='physical-environment-v1',
        acceptance_policy=acceptance,
        requested_candidates=coordinator.get('requested_candidates'),geometry_exports=exports,
        candidates=summary.get('generated',0),cpu_attempted=summary.get('attempted',0),
        cpu_failures=summary.get('failures',{}),by_house=summary.get('by_scene',{}),
        validated=len(checked),execution_passed=counts.get('execution_smoke_passed',0),
        environment_ready=len(ready),environment_ready_ids=ready,
        environment_needs_review=len(environments)-len(ready),environment_pending=len(pending),
        environment_review_reasons=dict(Counter(e['reason'] for e in environments if e['status']!='ready')),
        environment_assessments=environments,
        execution_failed=sum(p['status']!='execution_smoke_passed' for p in checked if not p['validation'].get('environment_only')),pending_gpu=len(pending),
        validation_failures=dict(failures),background_stage_passes=dict(stage_counts),
        skill_attempt_failures=dict(skill_failures),evaluation_ready_episodes=0,generated_new_floorplans=0,
        scope='physical environment readiness only; historical policy results are diagnostic; solvability and search certification pending')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--run',type=Path,required=True)
    report(parser.parse_args().run)
