"""One public image -> model -> action schema; this is not a task success score."""
import argparse
import base64
import json
from pathlib import Path
from pydantic import TypeAdapter
from rummagebench.agents.local_vlm import LocalVLMPolicy, request_json, prompt_fingerprint, PROMPT_VERSION
from rummagebench.agents.policy_adapter import parse_action
from rummagebench.core.public_types import PublicAction, PUBLIC_SKILLS
from rummagebench.core.scenario import load_scenario


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8091')
    p.add_argument('--scenario',type=Path,default=Path('build/vla_difficulty_v1/T01_knife_fetch__L2_S1/scenario.yaml'))
    p.add_argument('--image',type=Path,default=Path('runs/vla_difficulty_v1/T01_knife_fetch__L2_S1/start_panorama.png'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    profile=json.loads(Path('configs/models/local_vlm.json').read_text())['models'][a.model]
    scenario=load_scenario(a.scenario)
    report={'ok':False,'scope':'one static public image, not an episode score',
            'prompt_version':PROMPT_VERSION,'prompt_sha256':prompt_fingerprint()}
    try:
        server=request_json(a.endpoint+'/health')
        if server['model_id']!=profile['model_id'] or server['revision']!=profile['revision']:
            raise ValueError('server model/revision mismatch')
        report['server']=server
        obs=dict(instruction=scenario.instruction,frame_id='frame_000000',planning_step=0,
                 max_planning_steps=60,skill_library=PUBLIC_SKILLS,feedback=None,
                 image_png_b64=base64.b64encode(a.image.read_bytes()).decode())
        raw=LocalVLMPolicy(a.endpoint,profile['model_id'],a.out/'calls.jsonl')(obs,[])
        report['action']=parse_action(raw)
        TypeAdapter(PublicAction).validate_python(report['action'])
        report['ok']=True
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}'
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return int(not report['ok'])

if __name__=='__main__':raise SystemExit(main())
