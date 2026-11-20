"""Propose local wall edits from native collider intervals; never auto-approve them."""
import argparse,json
from pathlib import Path
import numpy as np


def proposals(inventory, width=1.4):
    warps=[];review=[]
    for obj in inventory['objects']:
        if obj['category']!='walls' or not obj['boxes']:continue
        boxes=np.asarray(obj['boxes'],float);lo=boxes[:,0].min(0);hi=boxes[:,1].max(0)
        axis=int(np.argmax(hi[:2]-lo[:2]));normal=1-axis
        if hi[normal]-lo[normal]>.5:
            review.append(dict(entity=obj['name'],reason='Nonplanar or diagonal wall needs manual editing'));continue
        floor=lo[2];intervals=sorted((b[0,axis],b[1,axis]) for b in boxes if b[0,2]<floor+.5 and b[1,2]>floor+.05)
        merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]+.02:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        for left,right in zip(merged,merged[1:]):
            a,b=left[1],right[0];gap=b-a
            if not .55<gap<width:continue
            anchor_lo=(left[0]+a)/2;anchor_hi=(b+right[1])/2
            center=(a+b)/2;new_a=center-width/2;new_b=center+width/2
            if new_a<=anchor_lo+.05 or new_b>=anchor_hi-.05:
                review.append(dict(entity=obj['name'],opening=[a,b],reason='Insufficient adjoining wall for local widening'));continue
            warps.append(dict(entity=obj['name'],axis=axis,source=[anchor_lo,a,b,anchor_hi],target=[anchor_lo,new_a,new_b,anchor_hi],
                              measured_width_m=gap,requested_width_m=width,wall_plane=float((lo[normal]+hi[normal])/2)))
    return dict(scene_model=inventory['scene_model'],status='proposal_only_not_runtime_validated',wall_warps=warps,translations=[],manual_review=review,
                note='Door leaves/frames, furniture, cross-walls and task-state preservation require independent inspection before enabling.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inventory',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=proposals(json.loads(a.inventory.read_text()));a.output.write_text(json.dumps(result,indent=2));print(len(result['wall_warps']),'candidate wall edits;',len(result['manual_review']),'manual reviews')

if __name__=='__main__':main()
