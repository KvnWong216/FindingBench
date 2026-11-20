"""Paired frozen-state prompt probe, not a rollout or success-rate estimate."""
import argparse,base64,json,hashlib
from pathlib import Path
from pydantic import TypeAdapter,ValidationError
from rummagebench.agents.local_vlm import request_json
from rummagebench.agents.policy_adapter import parse_action
from rummagebench.core.public_types import PublicAction
from vlm_debug_prompt import build_debug_prompt

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    server=request_json('http://127.0.0.1:8091/health');rows=[]
    for task in ['T01_knife_fetch','T02_mug_to_counter','T03_fork_to_drawer']:
        ep=Path('runs/vlm_grid_v3/qwen3-vl-4b/seed_0')/(task+'__L2_S1')
        calls=[json.loads(l) for l in (ep/'model_calls.jsonl').read_text().splitlines()]
        for step,call in enumerate(calls[:2]):
            old=call['prompt'];body=old.split('\nCurrent observation:\n',1)[1]
            obs,tail=body.split('\nPrevious actions and their feedback (oldest first):\n',1)
            history=json.JSONDecoder().raw_decode(tail)[0];obs=json.loads(obs)
            image=(ep/'episode'/f'step_{step:03d}.png').read_bytes()
            for label,prompt in [('original',old),('candidate',build_debug_prompt(obs,history))]:
                result=request_json('http://127.0.0.1:8091/generate',dict(model=server['model_id'],prompt=prompt,
                    image_png_b64=base64.b64encode(image).decode(),max_new_tokens=512,seed=0))
                raw=parse_action(result['text']);errors=[]
                try:TypeAdapter(PublicAction).validate_python(raw)
                except ValidationError as e:errors=e.errors(include_url=False)
                row=dict(task=task,step=step,condition=label,prompt=prompt,image_sha256=hashlib.sha256(image).hexdigest(),
                    action=raw,schema_errors=errors,frame_matches=not raw.get('point') or raw['point']['frame_id']==obs['frame_id'],**result)
                rows.append(row)
                with (a.out/'calls.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(task,step,label,raw,flush=True)
    (a.out/'report.json').write_text(json.dumps({'scope':'six paired frozen observations; NOT task outcomes','server':server,'calls':len(rows)},indent=2))
if __name__=='__main__':main()
