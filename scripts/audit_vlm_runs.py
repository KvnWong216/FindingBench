"""Read-only attribution audit; excludes unfinished episodes, never repairs actions."""
import argparse
import json
from collections import Counter
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from rummagebench.core.public_types import PublicAction


def audit(root):
    results={}
    adapter=TypeAdapter(PublicAction)
    for folder in sorted(root.glob('*/seed_*')):
        totals=Counter();skills=Counter();reasons=Counter();feedback=Counter();bad=Counter();errors=Counter()
        incomplete=[];examples=[];hashes={k:set() for k in ['prompt_sha256','source_sha256']}
        loops=0;calls=0;truncated=0;first=Counter();images=Counter();executed_skills=Counter()
        for ep in sorted(x for x in folder.iterdir() if x.is_dir()):
            if not (ep/'report.json').exists():incomplete.append(ep.name);continue
            report=json.loads((ep/'report.json').read_text());summary=report.get('summary',{})
            totals['reports']+=1
            if not summary.get('run_valid'):totals['invalid_runs']+=1;continue
            totals['valid_runs']+=1;totals[summary['status']]+=1
            for key in hashes:hashes[key].add(report.get(key))
            events=[json.loads(l) for l in (ep/'episode/events.jsonl').read_text().splitlines()]
            previous=None
            for event in events:
                if event.get('event_type')!='action':continue
                raw=event.get('raw_action') or {};skill=raw.get('skill','MISSING')
                skills[skill]+=1;feedback[event.get('feedback')]+=1;reasons[event.get('private_reason') or 'NONE']+=1
                if event.get('executed'):executed_skills[skill]+=1
                if raw==previous:loops+=1
                previous=raw
                if event.get('private_reason')=='SCHEMA':
                    bad[json.dumps(raw,sort_keys=True)]+=1
                    try:adapter.validate_python(raw)
                    except ValidationError as exc:
                        for e in exc.errors():errors[str(e['loc'])+':'+e['type']]+=1
                if event.get('private_reason') and len(examples)<8:
                    examples.append({'episode':ep.name,'step':event.get('planning_step'),'action':raw,'reason':event.get('private_reason')})
            p=ep/'model_calls.jsonl'
            if p.exists():
                rows=[json.loads(l) for l in p.read_text().splitlines()];calls+=len(rows)
                truncated+=sum(bool(r.get('output_budget_reached')) for r in rows)
                if rows:first[rows[0]['text']]+=1
                images.update(r.get('image_sha256') for r in rows)
        results[folder.parent.name+'/'+folder.name]={
            'totals':dict(totals),'skills':dict(skills),'executed_skills':dict(executed_skills),
            'feedback':dict(feedback),'private_reasons':dict(reasons),'schema_errors':dict(errors),
            'common_schema_rejections':bad.most_common(8),'consecutive_identical_actions':loops,
            'calls':calls,'budget_reached_calls':truncated,'unique_image_hashes':len(images),
            'first_outputs':first.most_common(5),'incomplete_excluded':incomplete,
            'hashes':{k:sorted(v,key=str) for k,v in hashes.items()},'examples':examples}
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data={str(root):audit(root) for root in a.roots}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,indent=2)+'\n')
    for root,models in data.items():
        for name,d in models.items():print(name,d['totals'],'unfinished',d['incomplete_excluded'],'schema',d['schema_errors'],'truncated',d['budget_reached_calls'])
