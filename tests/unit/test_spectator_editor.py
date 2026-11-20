"""Editor math, native topology contracts and transactional publication."""
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from urllib.request import Request, urlopen

import numpy as np
import pytest

from spectator import FlightCamera, ThreadingHTTPServer
import spectator_editor as editor


def test_native_rest_pose_priority_and_world_heading():
    from scipy.spatial.transform import Rotation
    rest = Rotation.from_euler('x', 90, degrees=True)
    poses = dict(rare=dict(prob=.1, rotation=[0,0,0,1]),
                 common=dict(prob=.9, rotation=rest.as_quat().tolist()))
    actual = Rotation.from_quat(editor.stable_orientation(poses, 45))
    expected = Rotation.from_euler('z', 45, degrees=True)*rest
    assert (actual.inv()*expected).magnitude() < 1e-10
    # Native assets with no orientation entries use the SDK identity fallback.
    assert np.allclose(editor.stable_orientation([], 0), [0,0,0,1])


def test_move_repairs_roll_pitch_without_discarding_horizontal_heading():
    from scipy.spatial.transform import Rotation
    tilted = Rotation.from_euler('xyz', [35,20,72], degrees=True).as_quat()
    actual = Rotation.from_quat(editor.stable_orientation([], None, tilted))
    assert np.allclose(actual.as_euler('xyz', degrees=True), [0,0,72])
    with pytest.raises(editor.EditError, match='角度'):
        editor.stable_orientation([], float('nan'))


def test_downward_projection_accepts_vertical_clicked_face_and_nearest_shelf():
    calls=[]; shelf=object(); floor=object()
    def raycast(start, end, **kwargs):
        calls.append((start,end,kwargs))
        return [dict(hit=True,distance=1.05,position=[1,2,0],normal=[0,0,1],rigidBody='floor'),
                dict(hit=True,distance=.5,position=[1,2,.55],normal=[0,0,1],rigidBody='shelf'),
                dict(hit=True,distance=.1,position=[1,2,.95],normal=[1,0,0],rigidBody='wall')]
    support,hit=editor.downward_surface([1,2,1],raycast,{'shelf':shelf,'floor':floor}.get,['moving'])
    assert support is shelf and hit['position'][2]==.55
    assert calls==[([1.,2.,1.05],[1.,2.,-9.],dict(only_closest=False,ignore_bodies=['moving']))]
    with pytest.raises(editor.EditError,match='正下方'):
        editor.downward_surface([1,2,1],lambda *a,**k:[],lambda x:None,[])


def test_vertical_face_gets_object_width_clearance_but_horizontal_face_stays_exact():
    assert np.allclose(editor.gravity_anchor([1,2,3],[1,0,0],[.05,.1,.2]),[1.058,2,3])
    assert np.allclose(editor.gravity_anchor([1,2,3],[0,-1,0],[.05,.1,.2]),[1,1.892,3])
    assert editor.gravity_anchor([1,2,3],[0,0,1],[.05,.1,.2]) == [1,2,3]


@pytest.mark.parametrize('raw,code,words', [
    ('Initial robot footprint overlaps an obstacle; invalid spawn','robot_blocked','机器人'),
    ('Physics state is invalid; run must not be scored','physics_invalid','无效'),
    ('CUDA out of memory','gpu_resource','显卡'),
    ('[Errno 28] No space left on device','disk_full','磁盘'),
    ('独立重载未通过，原数据未覆盖：restored pose mismatch','reload_mismatch','重新打开'),
    ('ValueError: 1 validation error for ScenarioSpec','configuration_invalid','任务配置'),
    ('物体不存在','no_selection','点击'),
    ('选中的物体没有原生 OPEN/CLOSE 能力，请点选柜门或抽屉','not_openable','不能开合'),
    ('未安装这个 BEHAVIOR 类别／型号','asset_not_installed','重新选择'),
    ('[Errno 2] No such file or directory','missing_file','找不到'),
])
def test_native_failures_have_actionable_chinese_explanations(raw,code,words):
    actual=editor.explain_error(RuntimeError(raw),'add')
    assert actual['code']==code and words in actual['message']
    assert raw not in actual['message']


def test_error_details_are_preserved_separately_from_plain_message(tmp_path):
    state=editor.EditorState({})
    obj=object.__new__(editor.NativeEditor)
    obj.state=state;obj.dirty=True;obj.pending=None;obj.error_log=tmp_path/'errors.jsonl'
    raw='tensor backend kernel unsupported'
    obj.failure(RuntimeError(raw),'move')
    error=state.status['editor']['last_error']
    assert error['detail']==raw and raw not in error['message']
    assert json.loads(obj.error_log.read_text())['id']==error['id']
    assert state.status['editor']['dirty']


def test_search_chinese_and_english_only_returns_installed_models(tmp_path):
    for category in ('bottle_of_soy_sauce', 'box_of_tissues', 'mug'):
        for model in ('aaa', 'bbb'):
            folder = tmp_path/category/model
            (folder/'misc').mkdir(parents=True); (folder/'usd').mkdir()
            (folder/'misc/metadata.json').write_text(json.dumps(dict(bbox_size=[.1,.1,.2])))
    index = editor.AssetIndex(tmp_path)
    assert index.search('酱油')['total'] == 2
    assert index.search('纸巾盒')['items'][0]['category'] == 'box_of_tissues'
    assert index.search('soy sauce')['total'] == 2
    assert index.search('aaa')['total'] == 3
    assert index.search('mug', offset=1, limit=1)['items'][0]['model'] == 'bbb'
    assert index.search('not_a_native_asset')['items'] == []
    with pytest.raises(ValueError): index.item('../secret', 'aaa')


def test_screen_ray_matches_flight_and_letterbox_aspect():
    camera = FlightCamera([1,2,3], 70, -25)
    origin, ray = editor.screen_ray(camera, [.5,.5], 90, 2)
    assert origin == [1,2,3] and np.allclose(ray, camera.basis()[0])
    _, right = editor.screen_ray(FlightCamera([0,0,0]), [1,.5], 90, 2)
    _, top = editor.screen_ray(FlightCamera([0,0,0]), [.5,0], 90, 2)
    assert np.allclose(right, np.array([1,-1,0])/np.sqrt(2))
    assert np.allclose(top, np.array([1,0,.5])/np.linalg.norm([1,0,.5]))
    for xy in ([float('nan'),0], [1.1,0], [True,.1], None):
        with pytest.raises(ValueError): editor.screen_ray(camera, xy, 90, 2)


def test_depth_unprojection_uses_actual_unequal_focal_lengths():
    camera=FlightCamera([1,2,3])
    k=[[800,0,480],[0,500,320],[0,0,1]]
    p=editor.calibrated_point(camera.position,camera.basis(),k,560,420,2)
    assert np.allclose(p,[3,1.8,2.6])


@pytest.mark.parametrize('relation', ['inside','on_top'])
def test_editor_placement_records_obey_native_recipe_schema(relation):
    from rummagebench.core.scenario import PlacementSpec
    record=editor.placement_record('mug',relation,'cabinet',[1.,2.], 'drawer')
    PlacementSpec.model_validate(record)
    assert ('position_xy' in record) == (relation=='on_top')
    assert record.get('link')==('drawer' if relation=='inside' else None)


def document():
    return dict(robot=dict(name='robot'), target=dict(entity='target'),
        goal=[dict(entity='target', receptacle='destination')],
        objects=[dict(name='target'),dict(name='clutter')],
        placements=[dict(entity='clutter', receptacle='shelf'),dict(entity='target', receptacle='shelf')],
        initial_states=dict(shelf=dict(open=False),clutter={}))


def test_deletion_preserves_goal_and_cannot_leave_dangling_supports():
    doc = document()
    for name in ('target','robot','destination','shelf'):
        with pytest.raises(ValueError): editor.delete_from_document(doc, name)
    result = editor.delete_from_document(doc,'clutter')
    assert result['goal'] == doc['goal'] and result['target'] == doc['target']
    assert result['objects'] == [dict(name='target')]
    assert result['placements'] == [dict(entity='target',receptacle='shelf')]
    assert 'clutter' not in result['initial_states']
    assert len(doc['objects']) == 2


def test_named_state_keeps_unrelated_bodies_and_persists_add_delete_topology():
    before = {0:dict(pos=[0],ori=[1],registry=dict(object_registry=dict(a=10,b=20),systems=dict(water=3)))}
    after = {0:dict(pos=[9],ori=[9],registry=dict(object_registry=dict(a=11,new=30),systems=dict(water=9)))}
    result = editor.compose_named_state(before,after,{'new'})
    assert result[0]['registry']['object_registry'] == dict(a=10,new=30)
    assert result[0]['registry']['systems'] == dict(water=3)
    assert result[0]['pos'] == [0] and after[0]['pos'] == [9]


def test_derived_runtime_is_immutable_and_does_not_modify_parent(tmp_path):
    base = tmp_path/'base'; path=base/'src/rummagebench/authoring/tasks/household_state.py'
    path.parent.mkdir(parents=True); path.write_text('def restore_frozen_member(backend,path,scenario): pass\n')
    (base/'configs').mkdir(); (base/'configs/a.json').write_text('{}')
    target=editor.freeze_runtime(base,tmp_path/'runtime')
    assert 'findingbench_native_editor_removed_objects' not in path.read_text()
    assert 'findingbench_native_editor_removed_objects' in (target/path.relative_to(base)).read_text()
    assert target == editor.freeze_runtime(base,tmp_path/'runtime')
    (target/path.relative_to(base)).write_text('changed')
    with pytest.raises(ValueError,match='固定'): editor.freeze_runtime(base,tmp_path/'runtime')


def test_http_only_enqueues_native_mutations():
    state=editor.EditorState(dict(parent_task_id='F001'))
    state.catalog=SimpleNamespace(search=lambda q,offset:dict(items=[],total=0,offset=offset))
    handler=type('LocalEditor',(editor.EditorHandler,),dict(state=state))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        root=f'http://127.0.0.1:{server.server_port}'
        assert '柜内'.encode() in urlopen(root).read()
        body=json.dumps(dict(op='add',category='mug',model='aaa',xy=[.5,.5])).encode()
        response=json.load(urlopen(Request(root+'/editor/command',data=body,method='POST')))
        assert response['ok'] and len(state.commands)==1
        assert not state.status['ready']  # no simulator action on the HTTP thread
    finally:
        server.shutdown();server.server_close()


def setup_release(tmp_path,monkeypatch):
    import spectator_cache
    directory=tmp_path/'dataset';directory.mkdir(); revision=directory/'revision';revision.mkdir()
    original=dict(id='variant',parent_task_id='F001',play_scenario='old',runtime_root='parent')
    replacement=dict(original,play_scenario='new',editor_cache_dir='cache')
    (directory/'human_ready.json').write_text(json.dumps(dict(tasks=[original])))
    monkeypatch.setattr(spectator_cache,'identity',lambda task:(task['play_scenario'],{}))
    monkeypatch.setattr(spectator_cache,'lookup',lambda folder,task:('cache',{}))
    return directory,original,replacement,revision


def test_publication_backs_up_and_replaces_exactly_selected_row(tmp_path,monkeypatch):
    directory,original,new,revision=setup_release(tmp_path,monkeypatch)
    editor.publish_revision(directory,original,new,revision)
    assert json.loads((directory/'human_ready.json').read_text())['tasks'] == [new]
    assert json.loads((revision/'backup/human_ready.json').read_text())['tasks'] == [original]
    assert (revision/'publication.json').exists()


def test_publication_refreshes_existing_release_file_hash_ledger(tmp_path,monkeypatch):
    directory,original,new,revision=setup_release(tmp_path,monkeypatch)
    (directory/'verification.json').write_text(json.dumps(dict(files={'human_ready.json':'old'},authored_asset_model_count=123)))
    editor.publish_revision(directory,original,new,revision)
    verification=json.loads((directory/'verification.json').read_text())
    assert verification['files']['human_ready.json']==editor.digest(directory/'human_ready.json')
    assert verification['authored_asset_model_count'] is None
    assert verification['pinned_runtime_count']==1


def test_cache_failure_and_concurrent_edit_cannot_overwrite_current_row(tmp_path,monkeypatch):
    import spectator_cache
    directory,original,new,revision=setup_release(tmp_path,monkeypatch)
    before=(directory/'human_ready.json').read_bytes()
    monkeypatch.setattr(spectator_cache,'lookup',lambda *_:None)
    with pytest.raises(ValueError,match='缓存'):editor.publish_revision(directory,original,new,revision)
    assert (directory/'human_ready.json').read_bytes()==before
    altered=dict(original,play_scenario='other_editor')
    (directory/'human_ready.json').write_text(json.dumps(dict(tasks=[altered])))
    with pytest.raises(ValueError,match='其他编辑器'):editor.publish_revision(directory,original,new,revision)
    assert json.loads((directory/'human_ready.json').read_text())['tasks']==[altered]


def test_reload_pose_match_rejects_missing_objects_or_drift():
    expected=dict(obj=dict(links=dict(root=([0,0,0],[0,0,0,1])),joints=[0]))
    assert editor.compare_world(expected,expected)['position_m']==0
    with pytest.raises(ValueError):editor.compare_world(expected,{})
    moved=dict(obj=dict(links=dict(root=([.01,0,0],[0,0,0,1])),joints=[0]))
    with pytest.raises(ValueError):editor.compare_world(expected,moved)
