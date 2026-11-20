"""Synthetic trace fixtures only; not simulation evidence."""
from types import SimpleNamespace

import pytest

from rummagebench.authoring.layout.indoor_dynamics import summarize_trace


def trace():
    return [dict(sim_time=i*.1, objects={'prop':dict(
        pose=[0.,0.,0.,0.,0.,0.,(-1.)**i], linear_velocity=[0.,0.,0.],
        angular_velocity=[0.,0.,.03 if i==1 else 0.],
        physx_velocity=[0.,0.,0.,0.,0.,.03 if i==1 else 0.],
        physics_handle_valid=True, asleep=False, supported=True)}) for i in range(3)]


def test_quaternion_sign_flip_is_not_motion_and_velocity_spike_is_retained():
    result=summarize_trace(trace())['objects']['prop']
    assert result['max_rotation_drift_deg']==0.
    assert result['finite_difference_angular_max_rad_s']==0.
    assert result['angular_exceedances']==1
    assert result['getter_physx_max_abs_difference']==0.


@pytest.mark.parametrize('bad', ['stalled_clock','missing_object','nonfinite'])
def test_bad_trace_cannot_be_reported_as_stationary(bad):
    rows=trace()
    if bad=='stalled_clock':rows[1]['sim_time']=rows[0]['sim_time']
    if bad=='missing_object':rows[1]['objects']={}
    if bad=='nonfinite':rows[1]['objects']['prop']['angular_velocity'][0]=float('nan')
    with pytest.raises(ValueError):summarize_trace(rows)


@pytest.mark.parametrize('diagnostic_fails', [False, True])
def test_diagnostic_cannot_upgrade_failure_or_mask_original_error(tmp_path, monkeypatch, diagnostic_fails):
    from rummagebench.authoring.layout import indoor_realization as r, indoor_dynamics as d
    from rummagebench.sim.omnigibson import state_io
    import json
    monkeypatch.setattr(state_io,'seed_everything',lambda *a:None)
    monkeypatch.setattr(r,'_settled_measurement',lambda *a:({'p':dict(passed=False)}, {'p':True}, []))
    def diagnostic(*a, **kw):
        if diagnostic_fails:raise ValueError('unit diagnostic failed')
        (tmp_path/'background_dynamics.json').write_text('{}')
    monkeypatch.setattr(d,'diagnose_background',diagnostic)
    manifest=dict(layout_seed=1,manifest_sha256='unit',objects=[dict(placement=dict(name='p',role='clutter'))])
    with pytest.raises(ValueError,match='BACKGROUND_PERSISTENT_DYNAMICS'):
        r.realize_background(SimpleNamespace(),manifest,tmp_path,diagnostic_steps=360)
    assert json.loads((tmp_path/'background_realization.json').read_text())['status']=='failed'


def test_iteration_experiment_changes_only_added_dynamic_props(tmp_path, monkeypatch):
    from rummagebench.authoring.layout import indoor_dynamics as d
    prop=SimpleNamespace(name='prop',articulated=True,kinematic_only=False,solver_velocity_iteration_count=1)
    objects={'prop':prop,'native':SimpleNamespace(), 'target':SimpleNamespace()}
    visited=[]
    def resolve(scene,name):visited.append(name);return objects[name]
    monkeypatch.setattr(d,'resolve_object',resolve)
    backend=SimpleNamespace(_env=SimpleNamespace(scene=None),_sim=SimpleNamespace(
        editing_usd=lambda:pytest.fail('SDK setters own their edit contexts; do not nest')))
    manifest=dict(manifest_sha256='unit',objects=[dict(placement=dict(name='prop',role='clutter')),
                                                dict(placement=dict(name='native',role='surface'))])
    result=d.apply_velocity_iteration_experiment(backend,manifest,tmp_path,4)
    assert visited==['prop']
    assert result['objects']=={'prop':dict(before=1,after=4,articulated=True)}
    prop.kinematic_only=True
    with pytest.raises(ValueError,match='DYNAMIC_RIGID_PROPS'):
        d.apply_velocity_iteration_experiment(backend,manifest,tmp_path,1)
    assert prop.solver_velocity_iteration_count==4
