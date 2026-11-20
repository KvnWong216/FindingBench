import json
from types import SimpleNamespace
import pytest
from rummagebench.agents.local_vlm import build_prompt, LocalVLMPolicy
from rummagebench.agents.policy_adapter import parse_action, PolicyAgent, PolicyInfrastructureError


def observation():
    return dict(instruction='Get the mug', frame_id='frame_0', planning_step=0,
                max_planning_steps=60, skill_library=['VIEW','REPORT_DONE'],feedback=None,
                image_png_b64='image', hidden_target='SECRET', oracle_pose='SECRET')


def test_prompt_excludes_private_fields_and_preserves_public_history():
    prompt=build_prompt(observation(),[{'action':{'skill':'VIEW','yaw':30},'feedback':None,'target':'SECRET'}])
    assert 'SECRET' not in prompt
    assert 'frame_0' in prompt and 'Get the mug' in prompt
    assert '"yaw": 30' in prompt


def test_reasoning_and_ambiguous_actions():
    assert parse_action('<think>Maybe {"skill":"MOVE"}</think>{"skill":"REPORT_DONE"}') == {'skill':'REPORT_DONE'}
    assert parse_action('<think>{"skill":"REPORT_DONE"}') == {}
    assert parse_action('{"skill":"VIEW","yaw":0} {"skill":"REPORT_DONE"}') == {}
    assert parse_action('[{"skill":"REPORT_DONE"}]') == {}
    # Wrong frame and coordinate remain wrong: no evaluator-side repair.
    action={'skill':'GRASP','point':{'frame_id':'old','x':100,'y':-1}}
    assert parse_action(json.dumps(action))==action


def test_local_client_records_output_without_substituting_actions(monkeypatch,tmp_path):
    from rummagebench.agents import local_vlm
    sent=[]
    def request(url,body,timeout):
        sent.append(body)
        return {'model':'test-model','text':'not json','input_tokens':7,'output_tokens':2}
    monkeypatch.setattr(local_vlm,'request_json',request)
    agent=PolicyAgent(LocalVLMPolicy('http://localhost:1','test-model',tmp_path/'calls.jsonl'))
    assert agent.act(observation())=={}
    assert sent[0]['image_png_b64']=='image'
    assert 'SECRET' not in sent[0]['prompt']
    assert json.loads((tmp_path/'calls.jsonl').read_text())['text']=='not json'


def test_service_error_is_not_a_model_failure(visual_session,tmp_path):
    from rummagebench.evaluation.episode_log import run_visual_episode
    def policy(obs,history):raise PolicyInfrastructureError('CUDA out of memory')
    summary=run_visual_episode(visual_session,PolicyAgent(policy),tmp_path)
    assert summary['run_valid'] is False
    assert summary['planning_steps_used']==0
    assert 'infrastructure' in summary['error']


def test_wrong_model_response_is_rejected(monkeypatch,tmp_path):
    from rummagebench.agents import local_vlm
    monkeypatch.setattr(local_vlm,'request_json',lambda *a,**k:{'model':'other','text':'{}'})
    with pytest.raises(PolicyInfrastructureError):
        LocalVLMPolicy('http://localhost:1','test-model',tmp_path/'calls.jsonl')(observation(),[])


def test_aggregate_excludes_infrastructure_errors(tmp_path):
    import runpy
    from pathlib import Path
    aggregate=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/evaluate_local_vlm.py'))['aggregate']
    for name,valid,status in [('success',True,'SUCCESS'),('failure',True,'FAIL_MAX_STEPS'),('oom',False,'FAIL_MAX_STEPS')]:
        dest=tmp_path/name;dest.mkdir()
        (dest/'report.json').write_text(json.dumps({'variant':{'task_id':'T01','language_level':'L2','spatial_level':'S1'},
            'summary':{'run_valid':valid,'status':status}}))
    totals=aggregate(tmp_path)['overall']
    assert totals=={'attempts':3,'valid':2,'invalid':1,'successes':1,'success_rate':.5}


def test_http_error_preserves_server_diagnostic(monkeypatch):
    import io
    import urllib.error
    from rummagebench.agents import local_vlm
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('http://localhost/generate',500,'error',{},
                                     io.BytesIO(b'{"error":"LayerNorm dtype mismatch"}'))
    monkeypatch.setattr(local_vlm.urllib.request,'urlopen',fail)
    with pytest.raises(PolicyInfrastructureError,match='LayerNorm dtype mismatch'):
        local_vlm.request_json('http://localhost/generate',{})
