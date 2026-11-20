"""Build a review queue, NOT an approved 1K-task benchmark."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'build/behavior/official_catalog.json'
    tasks=json.loads(source.read_text())['tasks']
    rows=[]
    for t in tasks:
        rows.append(dict(source_activity=t['activity_name'],source_definition=t['definition'],
            source_sha256=t['sha256'],source_title=t['activity_name'].replace('_',' '),
            rigid_audit=t['rigid_only'],local_instance_count=len(t['instances']),
            review_status='rigid_skill_review_required' if t['rigid_only'] else 'adaptation_or_replacement_required',
            proposed_high_level_instruction='',machine_goal='',allowed_initial_states='',
            interpretation_notes='',semantic_family='',human_oracle_evidence='',
            accepted=False))
    out=ROOT/'build/behavior/high_level_task_review.csv'
    # Never overwrite human decisions on repeat runs.
    if out.exists():raise FileExistsError(f'Review queue already exists: {out}')
    with out.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(review_seeds=len(rows),rigid_review_candidates=sum(r['rigid_audit'] for r in rows),
        approved_tasks=0,target_distinct_high_level_tasks=1000,output=str(out)),indent=2))

if __name__=='__main__':main()
