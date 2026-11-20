import sys,random
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v2'))
from generate_tasks import language
from compile_variants import compose,physical_hash
from rummagebench.core.scenario import load_scenario
ROOT=Path(__file__).resolve().parents[2]


def test_local_condition_does_not_require_cross_room_counterpart(tmp_path):
    task=yaml.safe_load((ROOT/'assets/tasks/vla_difficulty_v2/base_tasks/H001.yaml').read_text())
    task['spatial']={'S1':task['spatial']['S1']}
    common=yaml.safe_load((ROOT/'assets/tasks/vla_difficulty_v2/common.yaml').read_text())
    hashes=[]
    for level in ['L1','L2','L3']:
        _,scenario,_=compose(common,task,level,'S1')
        p=tmp_path/f'{level}.yaml';p.write_text(yaml.safe_dump(scenario));load_scenario(p)
        hashes.append(physical_hash(scenario))
    assert len(set(hashes))==1


def test_language_supports_only_available_spatial_conditions():
    phrases={'scene':{'rooms':{'kitchen_0':'kitchen'},'furniture':{'desk':{'noun':'desk','room':'kitchen_0'}}}}
    target={'relation':'on_top','receptacle':'desk','probe_room':'kitchen_0'}
    c={'scene':'scene','start':{'room':'kitchen_0'},'family':'fetch','goal':None,'S1':target,'S2':None,'S3':None}
    l=language(c,{'category':'mug'},phrases,random.Random(0),'setting_table#0')
    assert set(l['L1'])=={'S1'}
    assert 'kitchen' in l['L1']['S1']


def test_unnamed_source_keeps_goal_only_language_without_fake_l1():
    phrases={'scene':{'rooms':{},'furniture':{}}}
    c={'scene':'scene','start':{'room':'living_room_2'},'family':'fetch','goal':None,
       'S1':{'relation':'on_top','receptacle':'unknown_table','probe_room':'living_room_2'},'S2':None,'S3':None}
    text=language(c,{'category':'mug'},phrases,random.Random(0),'setting_table#0')
    assert text['L1']=={}
    assert 'mug' in text['L2'] and 'mug' in text['L3']
    assert 'unknown_table' not in str(text) and 'living_room_2' not in str(text)


def test_furniture_comparisons_use_unique_landmarks_and_reject_ties():
    from make_phrases import distinguish
    def obj(cat,x):return {'category':cat,'aabb':[[x,0,0],[x+1,1,1]]}
    entities={'a':obj('bookcase',0),'b':obj('bookcase',4),'tv':obj('standing_tv',-2)}
    rooms={n:'room' for n in entities}
    assert distinguish('a',['a','b'],entities,rooms)['noun']=='bookcase closest to the TV'
    entities['b']=obj('bookcase',0)
    assert distinguish('a',['a','b'],entities,rooms) is None
