"""Semantic checks for evaluator-only scene difficulty proxies."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('official_axes', Path(__file__).parents[2]/'scripts/analyze_official_tasks.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_width_and_depth():
    flat = [('house',f'leaf{i}') for i in range(32)]
    assert m.graph_choice(flat,[f'leaf{i}' for i in range(32)])['choice'] == 6
    grouped = [('house',f'r{i}') for i in range(4)]
    grouped += [(f'r{i}',f'l{i}_{j}') for i in range(4) for j in range(8)]
    assert m.graph_choice(grouped,[f'l{i}_{j}' for i in range(4) for j in range(8)])['choice'] == 7
    chain = [('house','n0')]+[(f'n{i-1}',f'n{i}') for i in range(1,5)]
    # Average all object instances, not only leaves.
    assert m.graph_choice(chain,[f'n{i}' for i in range(5)])['choice'] == 3
    assert m.graph_choice([('house',str(i)) for i in range(10)],list(map(str,range(10))))['choice'] == pytest.approx(1+np.log2(10))


def test_cycles_duplicates_and_missing_nodes():
    edges = [('house','a'),('a','b'),('b','a'),('a','b')]
    assert m.graph_choice(edges,['a','b'])['choice'] == 1.5
    missing = m.graph_choice(edges,['a','b','c'])
    assert missing['choice'] is None
    assert missing['unreachable_objects'] == ['c']
    assert m.graph_choice([],[])['choice'] is None


def test_room_membership_shortcuts_nesting():
    edges = [('house','room'),('room','cabinet'),('room','cup'),('cabinet','cup')]
    result = m.graph_choice(edges,['cabinet','cup'])
    assert result['choice'] == 3
    assert result['mean_path_depth'] == 2
    assert m.graph_choice(edges[:-1],['cabinet','cup'])['choice'] == 3


def test_missing_data_and_ties():
    rows = [dict(distance_normalized=d,choice=c,difficulty_band='unclassified')
            for d,c in [(1,2),(1,2),(3,4),(9,8),(None,5),(2,None)]]
    result = m.assign_bands(rows)
    assert result['reference_count'] == 4
    assert rows[0]['difficulty_score'] == rows[1]['difficulty_score']
    assert rows[0]['difficulty_band'] == rows[1]['difficulty_band']
    assert rows[4]['difficulty_band'] == rows[5]['difficulty_band'] == 'unclassified'


def test_grid_distance_and_disconnected_locations(tmp_path):
    import cv2
    from rummagebench.authoring.tasks.room_map import RoomMap
    from unittest.mock import patch
    trav = np.ones((7,7), dtype=bool)
    # A wall with a single gap forces a detour.
    trav[1:,3] = False
    room_map = RoomMap(.1,7,np.ones((7,7),dtype=int),trav,{1:'kitchen_0'})
    with patch.object(RoomMap,'load',return_value=room_map):
        g = m.SceneGeometry(tmp_path,tmp_path/'rooms.txt')
    points = [room_map.to_world(5,1),room_map.to_world(5,5)]
    result = g.extent(points)
    assert result['distance_m'] > 1.0
    assert result['max_snap_m'] == pytest.approx(0)
    assert g.extent([points[0],points[0]])['distance_m'] == 0
    room_map.trav[0,3] = False
    with patch.object(RoomMap,'load',return_value=room_map):
        g = m.SceneGeometry(tmp_path,tmp_path/'rooms.txt')
    with pytest.raises(ValueError,match='disconnected'): g.extent(points)
    with pytest.raises(ValueError,match='within'): g.snap([100,100])


def test_generation_dependencies_and_candidate_sites():
    from types import SimpleNamespace
    def info(category):
        return {'class_module':'objects.dataset_object','args':{'category':category,'in_rooms':['kitchen_0']}}
    data={'objects_info':{'init_info':{'table1':info('table'),'table2':info('table'),
            'plate':info('plate'),'apple':info('apple')}},
          'metadata':{'task':{'inst_to_name':{'table.n.02_1':'table1','plate.n.04_1':'plate','apple.n.01_1':'apple'}}},
          'state':{'registry':{'object_registry':{}}}}
    initial=[('ontop','plate.n.04_1','table.n.02_1'),('ontop','apple.n.01_1','plate.n.04_1')]
    geometry=SimpleNamespace(map=SimpleNamespace(names={1:'kitchen_0'}))
    result=m.generation_choice(data,initial,geometry)
    apple=next(x for x in result['paths'] if x['object']=='object:apple')
    assert apple['depth']==4
    assert apple['candidate_counts']==[1,2,1,1]
    assert apple['cost']==5
    assert result['removed_room_shortcuts']==2
    assert ['room:kitchen_0','object:apple'] not in result['edges']
    # Adding a target child does not increase the plate's candidate-site count.
    data['objects_info']['init_info']['apple2']=info('apple')
    data['metadata']['task']['inst_to_name']['apple.n.01_2']='apple2'
    initial.append(('ontop','apple.n.01_2','plate.n.04_1'))
    result=m.generation_choice(data,initial,geometry)
    assert next(x for x in result['paths'] if x['object']=='object:apple')['cost']==5


def test_global_candidates_ignore_true_room_and_unrelated_background():
    from types import SimpleNamespace
    infos={n:{'class_module':'objects.dataset_object','args':{'category':c,'in_rooms':[r]}}
           for n,c,r in [('table','table','kitchen_0'),('other_table','table','bedroom_0'),
                         ('apple','apple','kitchen_0')]}
    data={'objects_info':{'init_info':infos},'metadata':{'task':{'inst_to_name':{
        'table.n.02_1':'table','apple.n.01_1':'apple'}}},'state':{'registry':{'object_registry':{}}}}
    geometry=SimpleNamespace(map=SimpleNamespace(names={1:'kitchen_0',2:'bedroom_0'}))
    initial=[('ontop','apple.n.01_1','table.n.02_1')]
    result=m.generation_choice(data,initial,geometry)
    target=next(x for x in result['paths'] if x['object']=='object:apple')
    assert target['candidate_counts']==[1,2,1]
    assert target['cost']==4  # 3 edges + log2(2 sites); no double-counted rooms
    assert target['private_room_conditioned_cost']==3
    infos['wall']={'class_module':'objects.dataset_object','args':{'category':'walls','in_rooms':['kitchen_0']}}
    other=m.generation_choice(data,initial,geometry)
    assert next(x for x in other['paths'] if x['object']=='object:apple')['cost']==4


def test_continuous_scores_do_not_turn_tiny_differences_into_rank_jumps():
    values=[0,1,2,3,4,4.001,5,6,7,8,9]
    rows=[dict(distance_normalized=4,choice=v) for v in values]
    m.assign_bands(rows)
    assert abs(rows[4]['difficulty_score']-rows[5]['difficulty_score'])<.001


def test_extreme_distance_is_not_clipped():
    rows=[dict(distance_normalized=v,choice=3) for v in range(1,21)]
    rows.append(dict(distance_normalized=100,choice=3))
    m.assign_bands(rows)
    assert rows[-1]['distance_normalized_scaled']>1
    assert rows[-1]['difficulty_score']>rows[-2]['difficulty_score']
