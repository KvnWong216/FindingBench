"""The choice graph must preserve nesting without leaking room shortcuts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v2'))
from analyze_difficulty import structural


def graph(relations=(), complete=True):
    return dict(entities={n: {'in_rooms': ['kitchen']} for n in ['cabinet', 'box', 'cup', 'probe_mug']},
                relations=list(relations), relations_complete=complete)


def relation(parent, child):
    return dict(parent=parent, child=child, relation='inside', source='omnigibson_state')


def target(parent='box'):
    return dict(target_placement=dict(entity='target_obj', receptacle=parent, relation='inside'))


def test_preserves_nesting_and_removes_room_and_transitive_shortcuts():
    r = structural(graph([relation('cabinet', 'box'), relation('box', 'cup'), relation('cabinet', 'cup')]), target())
    edges = {tuple(e) for e in r['edges']}
    assert ('cabinet', 'box') in edges and ('box', 'cup') in edges
    assert ('room:kitchen', 'box') not in edges
    assert ('cabinet', 'cup') not in edges
    assert not any('probe_mug' in e for e in edges)
    assert r['object_count'] == 4
    assert r['choice'] is not None


def test_cycle_is_not_a_valid_score():
    assert structural(graph([relation('cabinet', 'box'), relation('box', 'cabinet')]), target())['choice'] is None


def test_missing_relation_capture_is_not_silently_flattened():
    assert structural(graph(complete=False), target())['choice'] is None


def test_unknown_placement_parent_is_not_silently_dropped():
    assert structural(graph(), target('missing'))['choice'] is None


def test_multiple_true_parents_retained():
    r = structural(graph([relation('cabinet', 'cup'), relation('box', 'cup')]), target())
    assert {('cabinet', 'cup'), ('box', 'cup')} <= {tuple(e) for e in r['edges']}
