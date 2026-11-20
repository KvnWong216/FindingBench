import importlib.util
import json
from pathlib import Path
from pydantic import TypeAdapter
from rummagebench.core.public_types import PublicAction


def load(name):
    path=Path(__file__).resolve().parents[2]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_audit_excludes_interrupted_episode_and_explains_schema(tmp_path):
    folder=tmp_path/'model'/'seed_0';ep=folder/'complete';(ep/'episode').mkdir(parents=True)
    (folder/'interrupted').mkdir()
    (ep/'report.json').write_text(json.dumps({'summary':{'run_valid':True,'status':'FAIL_MAX_STEPS'}}))
    (ep/'episode/events.jsonl').write_text(json.dumps({'event_type':'action','raw_action':{'skill':'VIEW'},'private_reason':'SCHEMA','feedback':'INVALID_ACTION'})+'\n')
    row=load('audit_vlm_runs').audit(tmp_path)['model/seed_0']
    assert row['totals']['valid_runs']==1
    assert row['incomplete_excluded']==['interrupted']
    assert row['schema_errors']=={"('VIEW', 'yaw'):missing":1}


def test_debug_examples_are_valid_and_use_current_frame():
    obs=dict(instruction='Get a mug',frame_id='frame_000017',planning_step=0,
             max_planning_steps=60,skill_library=[],feedback=None,hidden_target='SECRET')
    prompt=load('vlm_debug_prompt').build_debug_prompt(obs,[])
    examples=json.JSONDecoder().raw_decode(prompt.split('Action syntax examples for THIS frame:\n')[1])[0]
    assert len(examples)==8 and 'SECRET' not in prompt and 'CURRENT_FRAME' not in prompt
    for example in examples:
        TypeAdapter(PublicAction).validate_python(example)
        if 'point' in example:assert example['point']['frame_id']=='frame_000017'
