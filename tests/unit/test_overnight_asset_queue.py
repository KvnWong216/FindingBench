from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/v2'))
from overnight_asset_queue import Queue,should_curate_first


def test_camera_related_reload_failure_is_not_assumed_to_need_new_furniture():
    assert not should_curate_first('POST_SETTLE_RELATION_FAILED: comparison_sample')
    assert should_curate_first('LAYOUT_NOT_CONVERGED: settling.private.json')
    assert should_curate_first('placement failed verification after physics settled')
    assert should_curate_first('EXPERT_SUPPORT_UNRESOLVED: no measured floor')
    assert should_curate_first('EXPERT_LAYOUT_UNRESOLVED: infeasible_model')


def test_static_rejection_generates_fresh_candidate_before_native(tmp_path):
    q=object.__new__(Queue);q.out=tmp_path;q.latest={};q.ledger={}
    q.accepted=lambda tid:False;q.status=lambda **kw:None
    q.prepare=lambda *args,**kw:dict(status='deferred',reason='No feasible compartment')
    calls=[]
    q.generate=lambda case,lane:(calls.append(('generate',lane)) or dict(status='proposed',record={'id':'instance'}))
    q.validate=lambda row,lane,case:(calls.append(('native',lane)) or dict(accepted=False))
    q.run_case(dict(id='F001',error='LAYOUT_NOT_CONVERGED'), 'guided')
    assert calls==[('generate','initial_generate'),('native','initial_generate')]
    q.run_case(dict(id='F001',error='LAYOUT_NOT_CONVERGED'), 'generate')
    assert calls[-2:]==[('generate','generate'),('native','generate')]


def test_deferred_alternative_never_starts_native(tmp_path):
    q=object.__new__(Queue);q.out=tmp_path;q.latest={};q.ledger={}
    q.accepted=lambda tid:False;q.status=lambda **kw:None
    q.generate=lambda *args:dict(status='deferred',reason='No measured layout')
    def forbidden(*args):raise AssertionError('Static rejection must not launch a simulator')
    q.validate=forbidden
    q.run_case(dict(id='F001'),'generate')
    assert q.ledger['generate:F001']['proposal']['status']=='deferred'


def test_runtime_replay_uses_new_immutable_lane_and_rejects_paths(tmp_path):
    import pytest
    q=object.__new__(Queue);q.out=tmp_path;q.latest={};q.ledger={}
    q.accepted=lambda tid:False;q.status=lambda **kw:None
    calls=[]
    q.prepare=lambda case,lane,action:(calls.append((lane,action)) or dict(status='deferred'))
    q.run_case(dict(id='F001'),'replay_proposal_fixed_v7')
    assert calls==[('proposal_fixed_v7','direct')]
    with pytest.raises(ValueError,match='Invalid immutable replay lane'):
        q.run_case(dict(id='F001'),'replay_../../old')
def test_native_policy_signature_invalidates_geometry_experience_after_generator_change():
    from overnight_asset_queue import native_policy_signature
    key='src/rummagebench/sim/omnigibson/'
    sources={key+'backend.py':'old backend',key+'inside_volume.py':'floor-only'}
    before=native_policy_signature(sources)
    assert before==native_policy_signature(dict(reversed(list(sources.items()))))
    sources[key+'inside_volume.py']='measured stacking supports'
    assert native_policy_signature(sources)!=before
    assert native_policy_signature({}) is None


def test_invalid_candidate_shape_is_infrastructure_not_layout_failure():
    from overnight_asset_queue import infrastructure_error
    assert infrastructure_error('ValueError: INVALID_EXPERT_LAYOUT_CANDIDATE')
    assert not infrastructure_error('EXPERT_LAYOUT_INFEASIBLE')


def test_authoring_audit_blocks_only_the_exact_defective_recipe(tmp_path,monkeypatch):
    import json,hashlib
    import track_full_asset_queue as tracker
    monkeypatch.setattr(tracker,'ROOT',tmp_path)
    recipe=tmp_path/'scenario.yaml';recipe.write_text('old contact-body binding')
    registry=tmp_path/'build/expert_surface_binding_audit/index.json';registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps(dict(recipes={hashlib.sha256(recipe.read_bytes()).hexdigest():dict(reason='wrong contact body')})))
    assert tracker.known_authoring_issue(recipe)=='wrong contact body'
    recipe.write_text('correct actual countertop entity')
    assert tracker.known_authoring_issue(recipe) is None
