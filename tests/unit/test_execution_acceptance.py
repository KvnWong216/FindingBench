"""Synthetic acceptance-policy fixtures, not simulator success evidence."""
import copy
import json
from types import SimpleNamespace as NS
import pytest
from rummagebench.authoring.layout import indoor_realization as r


def jitter():
    return dict(passed=False,position_drift_m=.00002,rotation_drift_deg=.019,
        linear_speed_m_s=.001,angular_speed_rad_s=.042,support_samples=[True]*91,
        limits=dict(position_m=.005,rotation_deg=2.,linear_m_s=.01,angular_rad_s=.02))


def test_execution_policy_keeps_strict_motion_diagnostics_unchanged():
    m=jitter();original=copy.deepcopy(m)
    assert not r.motion_accepted(m,'strict-v3')
    assert r.motion_accepted(m,'execution-v1')
    assert m==original


@pytest.mark.parametrize('field,value',[('position_drift_m',.006),('rotation_drift_deg',3.),
    ('support_samples',[True,False]),('angular_speed_rad_s',float('nan'))])
def test_execution_policy_still_rejects_pose_support_and_invalid_state(field,value):
    m=jitter();m[field]=value
    assert not r.motion_accepted(m,'execution-v1')


def test_execution_background_and_target_check_continue_without_resampling_jitter(tmp_path,monkeypatch):
    from rummagebench.sim.omnigibson import state_io
    monkeypatch.setattr(state_io,'seed_everything',lambda *a:None)
    monkeypatch.setattr(state_io,'save_snapshot',lambda p,*a,**k:p.write_bytes(b'unit checkpoint'))
    monkeypatch.setattr(r,'_measurement',lambda *a,**k:{'daily':jitter()})
    monkeypatch.setattr(r,'relation_holds',lambda *a:True)
    monkeypatch.setattr(r,'place_entity',lambda *a:pytest.fail('jitter must not trigger rearrangement'))
    obj=NS(get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,1.]))
    monkeypatch.setattr(r,'resolve_object',lambda *a:obj)
    backend=NS(_env=NS(scene=None),dump_state=lambda:None)
    manifest=dict(layout_seed=1,manifest_sha256='unit',objects=[dict(placement=dict(name='daily',role='clutter',support_name='table'))])
    result=r.realize_background(backend,manifest,tmp_path,stability_profile='observed-motion-v3',acceptance_policy='execution-v1')
    assert result['status']=='support_and_pose_passed_motion_diagnostic'
    assert result['motion_diagnostic_entities']==['daily'] and not result['repairs']
    after=r.verify_after_target(backend,manifest,result,tmp_path)
    assert after['status']=='daily_props_unchanged_within_tolerance'
    assert after['motion_diagnostic_entities']==['daily']
    monkeypatch.setattr(r,'resolve_object',lambda *a:NS(get_position_orientation=lambda:([.02,0.,0.],[0.,0.,0.,1.])))
    with pytest.raises(ValueError,match='TARGET_INSERTION_DISTURBED_BACKGROUND'):
        r.verify_after_target(backend,manifest,result,tmp_path)
    # Environment-only production permits a new, settled physical layout.
    # It must not pretend that this is a matched-family background.
    after=r.verify_after_target(backend,manifest,result,tmp_path,require_matched_background=False)
    assert after['status']=='daily_props_physically_ready'
    assert after['background_pose_changed_entities']==['daily']
    assert after['require_matched_background'] is False
    monkeypatch.setattr(r,'relation_holds',lambda *a:False)
    with pytest.raises(ValueError,match='TARGET_INSERTION_DISTURBED_BACKGROUND'):
        r.verify_after_target(backend,manifest,result,tmp_path,require_matched_background=False)
