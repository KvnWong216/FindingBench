"""Synthetic unit backend only; physical acceptance requires OmniGibson."""
import json
from types import SimpleNamespace

import pytest

from rummagebench.authoring.layout import indoor_realization as realization


def test_target_independent_parking_bounds_rotated_native_geometry():
    native = dict(objects=[dict(center=[1., 2., 3.], size=[2., 4., 6.],
                               quaternion=[0., 0., 2**-.5, 2**-.5])])
    m = dict(living_context=dict(native=native), target_slot='left')
    first = realization.parking_position(m)
    m['target_slot'] = 'right'
    assert realization.parking_position(m) == first
    assert first == pytest.approx([8., 8., 11.])


@pytest.mark.parametrize('repair_succeeds', [True, False])
def test_only_invalid_props_retry_with_original_support_and_bounded_budget(tmp_path, monkeypatch, repair_succeeds):
    from rummagebench.sim.omnigibson import state_io
    seed_calls, snapshots, calls = [], [], []
    monkeypatch.setattr(state_io, 'seed_everything', seed_calls.append)
    def save_snapshot(*a, **k):
        snapshots.append((a, k))
        a[0].write_bytes(b'unit-only-checkpoint')
    monkeypatch.setattr(state_io, 'save_snapshot', save_snapshot)
    manifest = dict(layout_seed=42, manifest_sha256='unit-only', objects=[
        dict(placement=dict(name=n, support_name='original_table', role='clutter'))
        for n in ('stable', 'unsupported')])
    backend = SimpleNamespace(_env=SimpleNamespace(scene=None), dump_state=lambda: 'unit-state')
    obj = SimpleNamespace(get_position_orientation=lambda: ([0., 0., 0.], [0., 0., 0., 1.]))
    monkeypatch.setattr(realization, 'resolve_object', lambda *a: obj)
    monkeypatch.setattr(realization, '_measurement', lambda *a: {
        n: dict(passed=True) for n in ('stable', 'unsupported')})
    monkeypatch.setattr(realization, 'relation_holds', lambda b, name, *a:
                        name == 'stable' or (repair_succeeds and bool(calls)))
    monkeypatch.setattr(realization, 'place_entity', lambda *a: calls.append(a[1:]) or True)
    if repair_succeeds:
        result = realization.realize_background(backend, manifest, tmp_path)
        assert result['status'] == 'support_and_stability_passed_pending_family'
        assert len(calls) == len(snapshots) == 1
    else:
        # Even a sampler returning True cannot override failed live checks.
        with pytest.raises(ValueError, match='BACKGROUND_NATIVE_REALIZATION_FAILED'):
            realization.realize_background(backend, manifest, tmp_path)
        assert len(calls) == 3 and not snapshots
        assert json.loads((tmp_path/'background_realization.json').read_text())['status'] == 'failed'
    assert seed_calls == [42]
    assert all(c == ('unsupported', 'original_table', 'on_top', None) for c in calls)


def test_stability_uses_real_pose_history_and_velocity(monkeypatch):
    position = [0., 0., 0.]
    obj = SimpleNamespace(get_position_orientation=lambda: (list(position), [0., 0., 0., 1.]),
                          get_linear_velocity=lambda: [0., 0., 0.],
                          get_angular_velocity=lambda: [0., 0., 0.])
    monkeypatch.setattr(realization, 'resolve_object', lambda *a: obj)
    def settle(steps):
        position[0] += .001*steps
    backend = SimpleNamespace(_env=SimpleNamespace(scene=None), settle=settle)
    measurement = realization._measurement(backend, [dict(name='moving')])['moving']
    assert not measurement['passed']
    assert measurement['position_drift_m'] == pytest.approx(.023)


def test_intermediate_velocity_spike_cannot_hide_behind_quiet_last_step(monkeypatch):
    step=[0]
    obj=SimpleNamespace(get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,1.]),
        get_linear_velocity=lambda:[0.,0.,0.],
        get_angular_velocity=lambda:[0.,0.,.03 if step[0]==5 else 0.])
    monkeypatch.setattr(realization,'resolve_object',lambda *a:obj)
    def settle(n):step[0]+=n
    backend=SimpleNamespace(_env=SimpleNamespace(scene=None),settle=settle)
    measured=realization._measurement(backend,[dict(name='prop')])['prop']
    assert step[0]==24 and not measured['passed']
    assert measured['angular_speed_rad_s']==pytest.approx(.03)
    assert measured['samples']['angular_velocity'][-1]==[0.,0.,0.]


@pytest.mark.parametrize('settles', [True, False])
def test_velocity_only_failure_waits_boundedly_without_rearranging_valid_support(tmp_path, monkeypatch, settles):
    from rummagebench.sim.omnigibson import state_io
    calls=[]
    monkeypatch.setattr(state_io,'seed_everything',lambda *a:None)
    monkeypatch.setattr(state_io,'save_snapshot',lambda p,*a,**k:p.write_bytes(b'unit-state'))
    obj=SimpleNamespace(get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,1.]))
    monkeypatch.setattr(realization,'resolve_object',lambda *a:obj)
    monkeypatch.setattr(realization,'relation_holds',lambda *a:True)
    def measure(*a):
        calls.append(1)
        return {'daily':dict(passed=settles and len(calls)>=2,position_drift_m=0.,rotation_drift_deg=0.,
                             angular_speed_rad_s=.03 if len(calls)==1 or not settles else 0.)}
    monkeypatch.setattr(realization,'_measurement',measure)
    monkeypatch.setattr(realization,'place_entity',lambda *a:pytest.fail('must not rearrange a stable pose for velocity alone'))
    backend=SimpleNamespace(_env=SimpleNamespace(scene=None),dump_state=lambda:None)
    manifest=dict(layout_seed=31,manifest_sha256='unit',objects=[dict(placement=dict(name='daily',role='clutter',support_name='table'))])
    if settles:
        assert realization.realize_background(backend,manifest,tmp_path)['repairs']==[]
        assert len(calls)==2
    else:
        with pytest.raises(ValueError,match='BACKGROUND_PERSISTENT_DYNAMICS'):
            realization.realize_background(backend,manifest,tmp_path)
        assert len(calls)==3


@pytest.mark.parametrize('offset', [0., .02])
def test_target_insertion_cannot_silently_move_background(tmp_path, monkeypatch, offset):
    manifest=dict(manifest_sha256='unit', objects=[dict(placement=dict(
        name='daily',role='clutter',support_name='table'))])
    before=dict(status='support_and_stability_passed_pending_family',assembly_sha256='unit',
                actual_poses=dict(daily=[[0.,0.,0.],[0.,0.,0.,1.]]))
    obj=SimpleNamespace(get_position_orientation=lambda: ([offset,0.,0.],[0.,0.,0.,1.]),
                        get_linear_velocity=lambda: [0.,0.,0.],get_angular_velocity=lambda: [0.,0.,0.])
    backend=SimpleNamespace(_env=SimpleNamespace(scene=None))
    monkeypatch.setattr(realization,'resolve_object',lambda *a:obj)
    monkeypatch.setattr(realization,'relation_holds',lambda *a:True)
    if offset:
        with pytest.raises(ValueError,match='TARGET_INSERTION_DISTURBED_BACKGROUND'):
            realization.verify_after_target(backend,manifest,before,tmp_path)
    else:
        assert realization.verify_after_target(backend,manifest,before,tmp_path)['status']=='daily_props_unchanged_within_tolerance'
    receipt=json.loads((tmp_path/'background_after_target.json').read_text())
    assert receipt['failed_entities']==(['daily'] if offset else [])


def test_post_target_velocity_failure_does_not_claim_pose_displacement(tmp_path, monkeypatch):
    manifest=dict(manifest_sha256='unit',objects=[dict(placement=dict(name='daily',role='clutter',support_name='table'))])
    before=dict(status='support_and_stability_passed_pending_family',assembly_sha256='unit',
                actual_poses=dict(daily=[[0.,0.,0.],[0.,0.,0.,1.]]))
    obj=SimpleNamespace(get_position_orientation=lambda:([0.,0.,0.],[0.,0.,0.,1.]),
                        get_linear_velocity=lambda:[0.,0.,0.],get_angular_velocity=lambda:[0.,0.,.03])
    monkeypatch.setattr(realization,'resolve_object',lambda *a:obj)
    monkeypatch.setattr(realization,'relation_holds',lambda *a:True)
    with pytest.raises(ValueError,match='BACKGROUND_UNSTABLE_AFTER_TARGET'):
        realization.verify_after_target(SimpleNamespace(_env=SimpleNamespace(scene=None)),manifest,before,tmp_path)
    assert json.loads((tmp_path/'background_after_target.json').read_text())['failure_causes']=={'daily':['angular_velocity']}
