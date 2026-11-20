"""Evaluate a locally served VLM on certified variants, without running an oracle."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]


def ensure_manifest(out, expected):
    path=out/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise RuntimeError('evaluation settings differ; choose a new output root')
    else:
        if any(out.glob('*/report.json')):
            raise RuntimeError('existing results have no run manifest; choose a new output root')
        path.write_text(json.dumps(expected,indent=2)+'\n')


def aggregate(out):
    rows=[]
    for report in sorted(out.glob('*/report.json')):
        r=json.loads(report.read_text());s=r.get('summary',{})
        meta=r.get('variant',{})
        rows.append({'variant':report.parent.name,'task':meta.get('task_id'),
                     'L':meta.get('language_level'),'S':meta.get('spatial_level'),
                     'valid':s.get('run_valid',False),'success':s.get('status')=='SUCCESS',
                     'steps':s.get('planning_steps_used'),'status':s.get('status'),
                     'error':r.get('error') or s.get('error')})
    def stats(items):
        valid=[r for r in items if r['valid']]
        return {'attempts':len(items),'valid':len(valid),'invalid':len(items)-len(valid),
                'successes':sum(r['success'] for r in valid),
                'success_rate':sum(r['success'] for r in valid)/len(valid) if valid else None}
    summary={'overall':stats(rows),'by_language':{},'by_space':{},'by_cell':{},'episodes':rows}
    for key,label in [('L','by_language'),('S','by_space')]:
        for level in sorted({str(r[key]) for r in rows}):
            summary[label][level]=stats([r for r in rows if str(r[key])==level])
    for r in rows:
        cell=f"{r['L']}/{r['S']}"
        summary['by_cell'][cell]=stats([x for x in rows if x['L']==r['L'] and x['S']==r['S']])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def episode(args,profile):
    from rummagebench.agents.local_vlm import LocalVLMPolicy,request_json,prompt_fingerprint,PROMPT_VERSION,source_fingerprint
    from rummagebench.agents.policy_adapter import PolicyAgent
    from rummagebench.evaluation.episode_log import run_visual_episode
    scenario=ROOT/'build/vla_difficulty_v1'/args.episode/'scenario.yaml'
    out=args.out_root/args.model/f'seed_{args.seed}'/args.episode
    out.mkdir(parents=True,exist_ok=False)
    if args.agent_policy=='humanclaw':
        from rummagebench.agents.humanclaw_vlm import HumanCLAWPolicy,VERSION,fingerprint
        policy_class=HumanCLAWPolicy;prompt_version=VERSION;prompt_sha=fingerprint()
    else:
        policy_class=LocalVLMPolicy;prompt_version=PROMPT_VERSION;prompt_sha=prompt_fingerprint()
    report={'model_used':False,'variant':json.loads(scenario.with_name('variant.json').read_text()),
            'scenario_sha256':hashlib.sha256(scenario.read_bytes()).hexdigest(),
            'prompt_sha256':prompt_sha, 'prompt_version':prompt_version,
            'agent_policy':args.agent_policy,'max_steps_override':args.max_steps,
            'source_sha256':source_fingerprint(),
            'seed':args.seed,'max_new_tokens':args.max_new_tokens,'history_steps':None if args.agent_policy=='humanclaw' else 60,
            'history_policy':'multiturn_full_text_public_feedback_current_images' if args.agent_policy=='humanclaw' else 'last_60_actions_feedback',
            'image_policy':'current full panorama plus optional current perspective detail; model-native processor',
            'profile':profile}
    try:
        health=request_json(args.endpoint+'/health')
        if health['model_id']!=profile['model_id'] or health['revision']!=profile['revision']:
            raise RuntimeError('model server profile/revision differs from evaluation profile')
        report['server']=health
        from rummagebench.environment.env import InteractiveSearchEnv
        env=InteractiveSearchEnv(scenario,seed=args.seed,mode='agent',
                                 feedback_mode='reasons' if args.agent_policy=='humanclaw' else 'classes')
        if args.agent_policy=='humanclaw':
            from rummagebench.core.reason_feedback_session import PROTOCOL
            report['feedback_protocol'] = PROTOCOL
        if args.max_steps is not None:
            env._session.set_planning_budget(args.max_steps)
        report['effective_max_steps']=env._session._scenario.termination.max_planning_steps
        policy=policy_class(args.endpoint,profile['model_id'],out/'model_calls.jsonl',
                              max_new_tokens=args.max_new_tokens,seed=args.seed)
        env._session.expose_embodiment = args.agent_policy=='humanclaw'
        report['observation_profile'] = 'panorama-detail-embodiment-reasons-v4.3' if args.agent_policy=='humanclaw' else 'panorama-detail-v4'
        agent=PolicyAgent(policy,name=args.model)
        report['summary']=run_visual_episode(env._session,agent,out/'episode',save_images=True,
            model_version=profile['model_id']+'@'+profile['revision'])
        if getattr(policy,'infrastructure_degraded',False):
            report['summary']['run_valid']=False
            report['summary']['error']='inference transport errors occurred; retained rollout excluded from score'
        report['model_used']=(out/'model_calls.jsonl').exists()
        events=[json.loads(line) for line in (out/'episode/events.jsonl').read_text().splitlines()]
        actions=[e for e in events if e.get('event_type')=='action']
        report['feedback_counts']=dict(Counter(e.get('feedback') for e in actions))
        if report['model_used']:
            calls=[json.loads(line) for line in (out/'model_calls.jsonl').read_text().splitlines()]
            report['inference_totals']={k:sum(c.get(k,0) for c in calls)
                                       for k in ('input_tokens','output_tokens','duration_s')}
            report['inference_totals']['calls']=len(calls)
    except Exception as error:
        import traceback
        report['error']=f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report.get('summary',{}).get('run_valid') else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8091')
    p.add_argument('--variants',nargs='+')
    p.add_argument('--all',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-new-tokens',type=int)
    p.add_argument('--agent-policy',choices=['strict','humanclaw'],default='strict')
    p.add_argument('--max-steps',type=int)
    p.add_argument('--timeout',type=int,default=7200,help='wall time per episode, seconds')
    p.add_argument('--out-root',type=Path,default=ROOT/'runs/vlm_evaluation')
    p.add_argument('--episode',help=argparse.SUPPRESS)
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();args.endpoint=args.endpoint.rstrip('/');args.out_root=args.out_root.resolve()
    if args.max_new_tokens is None:
        args.max_new_tokens=({'qwen3-vl-4b':4096,'internvl3.5-4b':2048}.get(args.model,1200) if args.agent_policy=='humanclaw' else 512)
    if args.max_steps is not None and args.max_steps<1:p.error('--max-steps must be positive')
    profile=json.loads((ROOT/'configs/models/local_vlm.json').read_text())['models'][args.model]
    out=args.out_root/args.model/f'seed_{args.seed}'
    if args.episode:return episode(args,profile)
    out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:print(json.dumps(aggregate(out),indent=2));return 0
    choices={x.parent.name for x in (ROOT/'build/vla_difficulty_v1').glob('*/scenario.yaml')}
    variants=sorted(choices) if args.all else (args.variants or ['T01_knife_fetch__L2_S1'])
    if args.all and args.variants:p.error('choose --all or --variants')
    if not set(variants)<=choices:p.error('unknown variant: '+str(set(variants)-choices))
    if any((out/v).exists() for v in variants):p.error('output already exists; use a fresh --out-root')
    from rummagebench.agents.local_vlm import request_json, source_fingerprint
    source_sha=source_fingerprint()
    health=request_json(args.endpoint+'/health',timeout=10)
    if health['model_id']!=profile['model_id'] or health['revision']!=profile['revision']:
        p.error('start the requested model/revision first')
    ensure_manifest(out,dict(agent_policy=args.agent_policy,profile=profile,seed=args.seed,
        max_new_tokens=args.max_new_tokens,max_steps=args.max_steps,source_sha256=source_sha,
        server={k:v for k,v in health.items() if k!='weight_memory_bytes'}))
    failures=0
    for variant in variants:
        if source_fingerprint()!=source_sha:
            raise RuntimeError('benchmark sources changed during evaluation; start a new run')
        log=out/(variant+'.log')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',args.model,'--endpoint',args.endpoint,
             '--episode',variant,'--seed',str(args.seed),'--max-new-tokens',str(args.max_new_tokens),
             '--out-root',str(args.out_root),'--agent-policy',args.agent_policy]
        if args.max_steps is not None:cmd+=['--max-steps',str(args.max_steps)]
        with log.open('w') as f:
            try:
                result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=args.timeout)
                code=result.returncode
            except subprocess.TimeoutExpired:code=-9
        if code:
            failures+=1
            dest=out/variant;dest.mkdir(exist_ok=True)
            if not (dest/'report.json').exists():
                meta=json.loads((ROOT/'build/vla_difficulty_v1'/variant/'variant.json').read_text())
                (dest/'report.json').write_text(json.dumps({'variant':meta,'error':f'worker exit {code}; see {log.name}'})+'\n')
        print(f'{variant}: exit={code}',flush=True)
        aggregate(out)
    return int(failures>0)

if __name__=='__main__':
    code=main()
    sys.stdout.flush();sys.stderr.flush()
    if '--episode' in sys.argv:os._exit(code)  # bounded Kit teardown, after persisted report
    raise SystemExit(code)
