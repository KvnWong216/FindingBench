"""Prepare a declared background-only clearance edit of the honey task."""
import json
from pathlib import Path
import yaml
from rummagebench.datasets.scene_derivation import digest,validate_scene_derivation

ROOT=Path(__file__).resolve().parents[1]

def main():
    cat=json.loads((ROOT/'build/behavior/official_catalog.json').read_text())
    task=next(t for t in cat['tasks'] if t['activity_name']=='store_honey')
    src=Path(task['instances'][0]['template']);data=json.loads(src.read_text())
    # Explicit instance IDs, not a runtime semantic deletion policy.
    removed=['straight_chair_amgwaw_0','straight_chair_amgwaw_1','straight_chair_amgwaw_2',
             'straight_chair_eospnr_0','straight_chair_eospnr_1','breakfast_table_skczfi_0']
    out=ROOT/'build/behavior/store_honey_open_route';out.mkdir(parents=True,exist_ok=True)
    records={}
    for name in removed:
        if name in data['metadata']['task']['inst_to_name'].values():raise ValueError('Task object removal')
        records[name]=dict(init_info=data['objects_info']['init_info'].pop(name),
                           state=data['state']['registry']['object_registry'].pop(name))
    derived=out/'scene.json';derived.write_text(json.dumps(data,indent=2)+'\n')
    manifest=out/'manifest.json'
    manifest.write_text(json.dumps(dict(scene_model='Rs_int',activity_name='store_honey',
        source_template=str(src),source_sha256=digest(src),derived_template=str(derived),
        derived_sha256=digest(derived),removed_objects=removed,removed_records=records,
        purpose='Remove non-task seating/dining bottleneck between living room and kitchen; retain task objects, goals and robot spawn.',
        status='requires_navigation_witness',official_instance_unmodified=True),indent=2)+'\n')
    validate_scene_derivation(manifest,'Rs_int','store_honey')
    spec=yaml.safe_load((ROOT/'scenarios/behavior_store_honey/scenario.yaml').read_text())
    spec['id']='behavior_store_honey_open_route';spec['scene']['task_scene_manifest']=str(manifest)
    dest=ROOT/'scenarios/behavior_store_honey_open_route/scenario.yaml';dest.parent.mkdir(exist_ok=True)
    dest.write_text(yaml.safe_dump(spec,sort_keys=False));print(dest)

if __name__=='__main__':main()
