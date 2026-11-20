import json
import pytest
from rummagebench.agents.humanclaw_vlm import HumanCLAWPolicy, parse_plan
from rummagebench.agents.policy_adapter import PolicyInfrastructureError


def obs(step=0):
    return dict(instruction='Get mug', planning_step=step, frame_id=f'frame_{step}',
                image_png_b64='image', hidden='SECRET', feedback={'code':'EXECUTED'})


def policy(tmp_path, responses, sent):
    def request(url, body, timeout):
        sent.append(body)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return dict(model='test', text=r, input_tokens=10, output_tokens=4)
    return HumanCLAWPolicy('http://localhost', 'test', tmp_path/'calls.jsonl', request=request)


@pytest.mark.parametrize('action', [
    {'skill':'REPORT_DONE'}, {'skill':'MOVE'},
    {'skill':'TURN','angle_deg':-999}, {'skill':'VIEW'},
    {'skill':'GRASP','point':{'frame_id':'old','x':2,'y':-.3}},
    {'skill':'UNKNOWN','extra':True}])
def test_actions_pass_through_without_intervention(tmp_path, action):
    sent=[]
    p=policy(tmp_path,[json.dumps({'action':action})],sent)
    assert p(obs(),[]) == action
    assert len(sent)==1 and 'SECRET' not in sent[0]['prompt']
    assert 'EXECUTED' in sent[0]['prompt']
    decision=json.loads((tmp_path/'agent_decisions.jsonl').read_text())
    assert decision['intervention'] is False


@pytest.mark.parametrize('raw', ['bad','[{"action":{"skill":"MOVE"}}]',
    '{"action":{"skill":"MOVE"}} {"action":{"skill":"TURN"}}',
    '{"action":null}', '{"action_id":0,"action_name":"MOVE<20>"}'])
def test_invalid_output_has_no_retry_or_fallback(tmp_path, raw):
    sent=[]; p=policy(tmp_path,[raw],sent)
    assert p(obs(),[]) == {}
    assert len(sent)==1 and not p.infrastructure_degraded


def test_transport_failure_stops_without_action(tmp_path):
    sent=[]; p=policy(tmp_path,[PolicyInfrastructureError('OOM')],sent)
    with pytest.raises(PolicyInfrastructureError,match='OOM'):
        p(obs(),[])
    assert len(sent)==1 and p.infrastructure_degraded
    assert not (tmp_path/'agent_decisions.jsonl').exists()
    assert json.loads((tmp_path/'calls.jsonl').read_text())['error_kind']=='transport'


def test_full_memory_survives_ten_steps_and_reset(tmp_path):
    sent=[]; p=policy(tmp_path,[json.dumps({'action':{'skill':'VIEW','yaw':0},
        'mid_level_goal':'look'})]*13,sent)
    for i in range(12):p(obs(i),[])
    assert len(p.memory)==12 and 'look' in sent[-1]['prompt']
    context=json.loads(sent[-1]['prompt'].split('\nInput:\n')[1])
    assert [r['step'] for r in context['history']]==list(range(11))
    p(obs(0),[])
    assert len(p.memory)==1
    assert json.loads(sent[-1]['prompt'].split('\nInput:\n')[1])['history']==[]


def test_full_output_and_public_outcomes_survive_without_private_trace(tmp_path):
    sent=[]
    reason='Long reasoning\n' * 100 + 'NEXT: turn right after backing up.'
    raw='<think>Model returned reasoning</think>'+json.dumps({
        'action':{'skill':'MOVE','distance_cm':-10},
        'low_level_action_reasoning':reason, 'mid_level_goal':reason})
    p=policy(tmp_path,[raw,'malformed model output','{"action":{"skill":"REPORT_DONE"}}'],sent)
    history=[]
    for i in range(3):
        action=p(obs(i),history)
        history.append({'action':action,'feedback':{'code':'UNSAFE','message':'Blocked.'},
                        'failure_diagnostic_private':'SECRET'})
    context=json.loads(sent[-1]['prompt'].split('\nInput:\n')[1])
    first,second=context['history']
    assert first['raw_output']==raw
    assert first['low_level_action_reasoning']==reason
    assert first['mid_level_goal']==reason
    assert second['raw_output']=='malformed model output'
    assert second['action']=={}
    assert all(r['feedback']['message']=='Blocked.' for r in context['history'])
    assert 'SECRET' not in sent[-1]['prompt']
    turns=sent[-1]['messages']
    assert [m['role'] for m in turns]==['user','assistant','user','assistant','user']
    assert turns[1]['content']==raw
    assert turns[3]['content']=='malformed model output'
    assert 'SECRET' not in json.dumps(turns)
    assert json.loads(turns[-1]['content'].split('\n',1)[1])['previous_feedback']['message']=='Blocked.'
    logged=json.loads((tmp_path/'calls.jsonl').read_text().splitlines()[-1])
    assert logged['messages']==turns and logged['prompt_is_audit_only']


def test_evaluation_manifest_prevents_mixed_agent_results(tmp_path):
    import runpy
    from pathlib import Path
    import pytest
    ensure=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/evaluate_local_vlm.py'))['ensure_manifest']
    ensure(tmp_path,{'agent_policy':'strict'})
    ensure(tmp_path,{'agent_policy':'strict'})
    with pytest.raises(RuntimeError,match='settings differ'):
        ensure(tmp_path,{'agent_policy':'humanclaw'})
    (tmp_path/'run_manifest.json').unlink()
    (tmp_path/'episode').mkdir();(tmp_path/'episode/report.json').write_text('{}')
    with pytest.raises(RuntimeError,match='no run manifest'):
        ensure(tmp_path,{'agent_policy':'humanclaw'})


def test_output_wrappers_do_not_change_action():
    raw='{"action":{"skill":"MOVE","distance_cm":999}}'
    assert parse_plan('<think>reason</think>```json\n'+raw+'\n```')[1]['distance_cm']==999
    assert parse_plan('<think>'+raw)==({}, {})
    assert parse_plan('{"action":{"skill":"MOVE"},"action":{"skill":"TURN"}}')==({}, {})


def test_history_pairs_reasoning_with_real_feedback(tmp_path):
    sent=[]
    action={'skill':'MOVE','distance_cm':30}
    p=policy(tmp_path,[json.dumps({'action':action,'mid_level_goal':'approach'})]*2,sent)
    p(obs(),[])
    p(obs(1),[{'action':action,'feedback':{'code':'UNSAFE'},'hidden':'SECRET'}])
    context=json.loads(sent[-1]['prompt'].split('\nInput:\n')[1])
    assert context['history'][0]['action']==action
    assert context['history'][0]['feedback']['code']=='UNSAFE'
    assert context['history'][0]['mid_level_goal']=='approach'
    assert 'SECRET' not in sent[-1]['prompt']


def test_two_image_payload_and_prompt_metadata(tmp_path):
    sent=[];p=policy(tmp_path,['{"action":{"skill":"VIEW","yaw":0}}'],sent)
    observation=obs()
    observation['detail_view']=dict(frame_id='detail',image_png_b64='DETAIL_DATA',yaw_deg=90,pitch_deg=0,hfov_deg=90)
    p(observation,[])
    assert len(sent[0]['images'])==2
    assert sent[0]['images'][0]['image_png_b64']=='image'
    assert sent[0]['images'][1]['image_png_b64']=='DETAIL_DATA'
    assert 'DETAIL_DATA' not in sent[0]['prompt']
    context=json.loads(sent[0]['prompt'].split('\nInput:\n')[1])
    assert context['observation']['detail_view']['frame_id']=='detail'


def test_output_contract_repeated_without_selecting_action(tmp_path):
    sent=[]
    p=policy(tmp_path,['{"action":{"skill":"TURN","angle_deg":-999}}']*2,sent)
    for i in range(2):
        assert p(obs(i),[]) == {"skill":"TURN","angle_deg":-999}
    from rummagebench.agents.humanclaw_vlm import OUTPUT_CONTRACT
    latest=json.loads(sent[-1]['messages'][-1]['content'].split('\n',1)[1])
    assert latest['response_format']==OUTPUT_CONTRACT
    assert 'default' not in json.dumps(OUTPUT_CONTRACT)
    assert sent[-1]['messages'][1]['content']=='{"action":{"skill":"TURN","angle_deg":-999}}'


def test_format_failure_is_reported_after_attempt_without_repair(tmp_path):
    sent=[]
    raw='Action: ```json {"skill":"MOVE","distance_cm":25} ```'
    p=policy(tmp_path,[raw,'{"action":{"skill":"VIEW","yaw":0}}'],sent)
    assert p(obs(),[]) == {}
    assert p(obs(1),[]) == {"skill":"VIEW","yaw":0}
    turn=json.loads(sent[-1]['messages'][-1]['content'].split('\n',1)[1])
    assert 'no action was extracted' in turn['previous_response_format_feedback']
    assert sent[-1]['messages'][1]['content']==raw
    assert len(sent)==2


def test_native_context_exhaustion_retains_raw_output_without_submitting(tmp_path):
    def request(*_):
        return dict(model='test',text='{"action":',context_budget_exhausted=True)
    p=HumanCLAWPolicy('http://localhost','test',tmp_path/'calls.jsonl',request=request)
    with pytest.raises(PolicyInfrastructureError,match='ContextWindowExceeded'):
        p(obs(),[])
    assert json.loads((tmp_path/'calls.jsonl').read_text())['text']=='{"action":'
    assert not (tmp_path/'agent_decisions.jsonl').exists()


def test_compaction_keeps_archive_and_lets_model_choose_memory(tmp_path):
    from rummagebench.agents.humanclaw_vlm import PLANNER
    def request(url,body,timeout):
        if url.endswith('/count_tokens'):
            return dict(model='test',input_tokens=9000 if len(body['messages'])>5 else 4000)
        if body.get('stage')=='compaction':return dict(model='test',text='My earlier moves were blocked.')
        return dict(model='test',text='{"action":{"skill":"TURN","angle_deg":-999}}')
    p=HumanCLAWPolicy('http://localhost','test',tmp_path/'calls.jsonl',request=request,compact_threshold_tokens=6000)
    p.messages=[dict(role=r,content=c) for r,c in [('user',PLANNER+'\nold public observation'),
        ('assistant','raw malformed'),('user','public feedback blocked'),('assistant','raw2'),
        ('user','recent observation'),('assistant','recent exact output')]]
    p.memory=[dict(step=i,raw_output='original '+str(i)) for i in range(3)]
    assert p(obs(3),[])=={'skill':'TURN','angle_deg':-999}
    assert p.compaction_count==1 and len(p.memory)==4
    assert p.memory[0]['raw_output']=='original 0'
    assert len(p.messages)==4
    assert p.messages[1]['content']=='My earlier moves were blocked.'
    row=json.loads((tmp_path/'context_compactions.jsonl').read_text())
    assert 'raw malformed' in row['messages'][0]['content']
    assert 'recent exact output' in row['messages'][0]['content']
    assert row['retained_recent_pairs'] is None
    assert '400 words' not in row['messages'][0]['content']
    assert 'SECRET' not in json.dumps(row)
    assert row['input_tokens_after']<row['input_tokens_before']
    assert len((tmp_path/'agent_decisions.jsonl').read_text().splitlines())==1
    assert len((tmp_path/'calls.jsonl').read_text().splitlines())==1


@pytest.mark.parametrize('result', [dict(model='test'),dict(model='wrong',text='memory')])
def test_compaction_failure_is_atomic(tmp_path,result):
    def request(url,body,timeout):
        if url.endswith('/count_tokens'):return dict(model='test',input_tokens=9000)
        return result
    p=HumanCLAWPolicy('http://localhost','test',tmp_path/'calls.jsonl',request=request,compact_threshold_tokens=6000)
    original=[dict(role='user',content='public'),dict(role='assistant',content='raw')]
    p.messages=original.copy()
    with pytest.raises(PolicyInfrastructureError,match='compaction failed'):p(obs(1),[])
    assert p.messages==original and p.compaction_count==0
    assert not (tmp_path/'agent_decisions.jsonl').exists()
    assert (tmp_path/'context_compactions.jsonl').exists()


@pytest.mark.parametrize('memory', [
    dict(text='A nonreducing memory.'), dict(text=''),
    dict(text='Partial memory', output_budget_reached=True),
    dict(text='Memory at native limit', context_budget_exhausted=True)])
def test_memory_not_rejected_for_length_or_completion(tmp_path, memory):
    sent=[]
    def request(url,body,timeout):
        sent.append(body)
        if url.endswith('/count_tokens'):return dict(model='test',input_tokens=9000)
        return dict(model='test',**memory)
    p=HumanCLAWPolicy('http://localhost','test',tmp_path/'calls.jsonl',request=request,compact_threshold_tokens=6000)
    p.messages=[dict(role='user',content='public'),dict(role='assistant',content='raw')]
    compact=p.prepare_messages(p.messages+[dict(role='user',content='now')],obs(1))
    assert compact[1]['content']==memory['text'] and p.compaction_count==1
    row=json.loads((tmp_path/'context_compactions.jsonl').read_text())
    assert row['memory_accepted_without_content_check'] and not row['reduced_input']
    assert len((tmp_path/'context_budget.jsonl').read_text().splitlines())==2
    assert not (tmp_path/'agent_decisions.jsonl').exists()


def test_compaction_merges_previous_memory_without_harness_rewriting(tmp_path):
    from rummagebench.agents.humanclaw_vlm import PLANNER
    seen=[]
    def request(url,body,timeout):
        if url.endswith('/count_tokens'):
            return dict(model='test',input_tokens=9000 if len(body['messages'])>5 else 4000)
        seen.append(body)
        return dict(model='test',text='New self-authored memory.')
    p=HumanCLAWPolicy('http://localhost','test',tmp_path/'calls.jsonl',request=request,compact_threshold_tokens=6000)
    p.messages=[dict(role=r,content=c) for r,c in [('user',PLANNER+' memory marker'),
        ('assistant','Earlier memory: I believe X, not verified.'),('user','old feedback'),
        ('assistant','raw action'),('user','recent feedback'),('assistant','recent output')]]
    current=dict(role='user',content='current observation')
    compact=p.prepare_messages(p.messages+[current],obs(6))
    assert 'Earlier memory: I believe X, not verified.' in seen[0]['messages'][0]['content']
    assert compact[1]['content']=='New self-authored memory.'
    assert compact[-1]==current
