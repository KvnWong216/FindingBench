"""Reconcile incremental recipes with reviewed bindings, retaining their layout.

Changes invalidate recipe hashes and therefore require new native validation.
No accepted physics results are copied to revised recipes.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import yaml
from rummagebench.core.scenario import load_scenario

ROOT=Path(__file__).resolve().parents[2]
PREFIXES=('comparison_sample','comparison_negative_','required_negative_',
          'initial_inventory_','carried_content_')


def synchronize(catalogue):
    manifest=json.loads((catalogue/'manifest.json').read_text())
    changed=[]
    for row in manifest['tasks']:
        binding=manifest.get('binding_reviews',{}).get(row['parent_task_id'])
        if not binding or binding['status']!='ready':continue
        path=ROOT/row['scenario'];before=path.read_text();spec=yaml.safe_load(before)
        objects={o['name']:o for o in spec['objects']}
        prior_placements={p['entity']:dict(p) for p in spec['placements']}
        old_targets=row['target_entities']
        source=next(p for p in spec['placements'] if p['entity']==old_targets[0])
        base_goal=next(g for g in spec['goal'] if g['entity']==old_targets[0])
        support=base_goal['receptacle']
        if binding['destination_kind']=='basket':
            support=next(p['receptacle'] for p in spec['placements'] if p['entity']==support)
        overrides=binding.get('target_model_overrides') or [objects[n]['category']+'/'+objects[n]['model'] for n in old_targets]
        used=Counter();keys=[]
        for category in binding['target_categories']:
            candidates=[k for k in overrides if k.split('/')[0]==category]
            if not candidates:raise ValueError('No reviewed model for '+row['id']+' '+category)
            keys.append(candidates[min(used[category],len(candidates)-1)]);used[category]+=1
        removed=set(old_targets)|{n for n in objects if n.startswith(PREFIXES)}
        spec['objects']=[o for o in spec['objects'] if o['name'] not in removed]
        spec['placements']=[p for p in spec['placements'] if p['entity'] not in removed]
        spec['goal']=[g for g in spec['goal'] if g['entity'] not in removed]
        names=[]
        def add(name,key,placement):
            category,model=key.split('/')
            prior=prior_placements.get(name,{})
            old=objects.get(name,{})
            # Binding reconciliation must not undo the compiler's per-item
            # native compartment assignment for an unchanged object/location.
            if (old.get('category')==category and old.get('model')==model
                and prior.get('relation')==placement.get('relation')
                and prior.get('receptacle')==placement.get('receptacle')):
                placement=prior
            spec['objects'].append(dict(name=name,category=category,model=model))
            spec['placements'].append(dict(placement,entity=name))
        for i,key in enumerate(keys):
            name=old_targets[i] if i<len(old_targets) else f'target_added_{i}'
            names.append(name);add(name,key,source)
            spec['goal'].append(dict(base_goal,entity=name))
        if binding.get('reference'):
            add('comparison_sample',keys[0],dict(relation='on_top',receptacle=support))
            row['reference_entity']='comparison_sample'
            for i,model in enumerate(binding.get('same_category_distractor_models',[])):
                add(f'comparison_negative_{i}',keys[0].split('/')[0]+'/'+model,source)
        else:row.pop('reference_entity',None)
        for i,key in enumerate(binding.get('required_distractor_models',[])):
            add(f'required_negative_{i}',key,source)
        for i,key in enumerate(binding.get('destination_inventory_models',[])):
            add(f'initial_inventory_{i}',key,dict(relation=base_goal['kind'],receptacle=base_goal['receptacle']))
        for i,key in enumerate(binding.get('source_contents_models',[])):
            name=f'carried_content_{i}'
            add(name,key,dict(relation='inside',receptacle=names[0]))
            spec['goal'].append(dict(kind='inside',entity=name,receptacle=names[0]))
        if binding.get('target_contents_policy')=='empty' and binding.get('source_contents_models'):
            raise ValueError('Contradictory empty-container binding: '+row['id'])
        dk=binding['destination_kind'];sk=binding['source_kind']
        noun={'desk':'desk','table':'table','basket':'shopping basket','sink':'sink','counter':'counter','cabinet':'cabinet top'}[dk]
        room=lambda r:r.replace('private_office','study').replace('_',' ')
        destination=f'the {noun} in front of you in the '+room(binding['destination_room_type'])
        sn={'drawer':'drawers','cabinet':'cupboard','fridge':'refrigerator','shelf':'shelves','counter':'counter','table':'table','desk':'desk','dresser':'dresser','basket':'basket','bowl':'fruit basket'}[sk]
        source_text=f'the {sn} in the '+room(binding['source_room_type'])
        # Basket delivery surfaces are selected from the table categories.
        reference_support='the table with the shopping basket' if dk=='basket' else destination
        spec['instruction']=binding['instruction'].format(destination=destination,source=source_text,reference_support=reference_support)
        spec['target']=dict(entity=names[0],category=keys[0].split('/')[0])
        def normalized(document):
            document=dict(document)
            for field in ('objects','placements','goal'):
                document[field]=sorted(document.get(field,[]),key=lambda v:json.dumps(v,sort_keys=True))
            return document
        serialized=yaml.safe_dump(spec,sort_keys=False)
        if normalized(spec)!=normalized(yaml.safe_load(before)):
            path.write_text(serialized);changed.append(row['id'])
        load_scenario(path)
        row.update(target_entities=names,target_count=len(names),target_key=keys[0],object_count=len(spec['objects']),
                   instruction=spec['instruction'],scenario_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   binding_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest())
        row['restoration_entities']=list(dict.fromkeys([o['name'] for o in spec['objects']
            if o['name'] not in names and not o.get('fixed_base')]+list(spec.get('initial_states',{}))))
        if row.get('adaptation'):
            row['adaptation'].update(after_instruction=spec['instruction'],target_assets=keys,reason=binding['adaptation_reason'])
        (path.parent/'authoring.private.json').write_text(json.dumps(row,indent=2))
    tmp=catalogue/'manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(catalogue/'manifest.json')
    return changed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(dict(revised=synchronize(a.catalog_root.resolve()))))
