import base64
import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('human_ui',Path(__file__).parents[2]/'scripts/serve_play.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_recording_preserves_resets_and_actions(tmp_path):
    scenario=tmp_path/'scenario.yaml';scenario.write_text('id: example\n')
    root=tmp_path/'run';r=m.HumanRecorder(root,scenario,0,False)
    obs={'frame_id':'a','image_png_b64':base64.b64encode(b'png test fixture').decode()}
    r.record('initial',{'observation':obs})
    action={'skill':'VIEW','yaw':90,'pitch':0}
    r.record('step',{'action':action,'result':{'episode_status':'RUNNING','observation':obs}})
    r.record('reset',{'observation':obs});r.close()
    rows=[json.loads(x) for x in (root/'episode_000/events.jsonl').read_text().splitlines()]
    assert rows[1]['payload']['action']==action
    image=root/'episode_000'/rows[0]['payload']['observation']['image_png_file']
    assert image.read_bytes()==b'png test fixture'
    assert (root/'episode_001/events.jsonl').exists()
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['eligible_as_unassisted_human'] and not manifest['replay_verified']
    with pytest.raises(FileExistsError):m.HumanRecorder(root,scenario,0,False)

def test_active_action_rejects_second_submission():
    state=m.PlayState('unused',0);state.ready=True;state.executing=True
    assert 'error' in state.enqueue({'skill':'VIEW','yaw':0})
    assert 'error' in state.request_reset()
    assert state.state_payload()['pending']


@pytest.mark.parametrize('explicit_data_root',[False,True])
def test_human_launcher_preserves_pinned_runtime_environment(tmp_path,monkeypatch,explicit_data_root):
    import os,sys
    path=Path(__file__).parents[2]/'scripts/human_oracle.py'
    spec=importlib.util.spec_from_file_location('human_oracle_launch_test',path)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    root=tmp_path/'repo';runtime=root/'build/pinned';runtime.mkdir(parents=True)
    scenario=root/'scenario.yaml';scenario.write_text('id: example\n')
    data=tmp_path/'behavior-datasets'
    for name in ('behavior-1k-assets','omnigibson-robot-assets'):(data/name).mkdir(parents=True)
    monkeypatch.setattr(launcher,'ROOT',root)
    def selected_tasks(directory):
        assert directory==root/'runs/asset_full_queue_20261105'
        return [dict(scenario=scenario,runtime_root=runtime,
            runtime_environment={'YGBENCH_EXPERT_LAYOUT':'1','YGBENCH_GPU_PROPOSALS':'0'})]
    monkeypatch.setattr(launcher,'asset_tasks',selected_tasks)
    environment={'PYTHONPATH':'existing'}
    if explicit_data_root:environment['OMNIGIBSON_DATA_PATH']='/user/chosen/datasets'
    monkeypatch.setattr(os,'environ',environment)
    monkeypatch.setattr(sys,'argv',['human_oracle.py','--suite','asset','--pick','1'])
    calls=[];monkeypatch.setattr(os,'chdir',lambda path:calls.append(('cwd',path)))
    monkeypatch.setattr(os,'execv',lambda executable,args:calls.append(('exec',executable,args)))
    launcher.main()
    assert environment['YGBENCH_EXPERT_LAYOUT']=='1' and environment['YGBENCH_GPU_PROPOSALS']=='0'
    assert environment['OMNIGIBSON_DATA_PATH']==('/user/chosen/datasets' if explicit_data_root else str(data))
    assert calls[0]==('cwd',runtime)
    assert str(runtime/'scripts/serve_play.py') in calls[-1][2]


def test_asset_menu_retains_declared_native_environment(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import rummagebench.core.scenario as schema
    import rummagebench.authoring.tasks.household_state as states
    path=Path(__file__).parents[2]/'scripts/human_oracle.py'
    spec=importlib.util.spec_from_file_location('human_oracle_menu_test',path)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    flags={'YGBENCH_GPU_PROPOSALS':'0'}
    task=dict(play_scenario=str(tmp_path/'play.yaml'),scene='house',instruction='Find the soap.',runtime_environment=flags)
    (tmp_path/'human_ready.json').write_text(json.dumps(dict(tasks=[task])))
    monkeypatch.setattr(schema,'load_scenario',lambda _:SimpleNamespace(scene=SimpleNamespace(frozen_member_manifest='manifest.json')))
    monkeypatch.setattr(states,'load_frozen_member',lambda *args:{})
    assert launcher.asset_tasks(tmp_path)[0]['runtime_environment']==flags
