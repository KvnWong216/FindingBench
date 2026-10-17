"""Synthetic motion traces; real interventions are separate GPU evidence."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rummagebench.authoring.layout.indoor_stability import observed_motion


def trace():
    times = np.arange(361)/120
    poses = np.tile([0., 0., 0., 0., 0., 0., 1.], (len(times), 1))
    velocity = np.zeros((len(times), 6))
    return poses, times, velocity


def test_solver_velocity_is_retained_and_does_not_masquerade_as_motion():
    poses, times, velocity = trace()
    velocity[:, -1] = .08
    result = observed_motion(poses, times, velocity)
    assert result['passed']
    assert not result['reported_solver_velocity_check']['passed']
    assert result['reported_solver_velocity_check']['angular_speed_rad_s'] == .08


@pytest.mark.parametrize('mode', ['slide', 'tip', 'spike', 'drift', 'oscillation'])
def test_real_motion_rejects_even_with_zero_reported_velocity(mode):
    poses, times, velocity = trace()
    if mode == 'slide': poses[:, 0] = .05*times
    if mode == 'tip': poses[:, 3:] = Rotation.from_rotvec(np.c_[times, times*0, times*0]).as_quat()
    if mode == 'spike': poses[180, 0] = .001
    if mode == 'drift': poses[:, 0] = .003*times
    if mode == 'oscillation': poses[:, 0] = .001*np.sin(times*100)
    assert not observed_motion(poses, times, velocity)['passed']


@pytest.mark.parametrize('mode', ['short', 'gap', 'reverse', 'nan_pose', 'nan_velocity', 'bad_quaternion'])
def test_missing_or_invalid_evidence_cannot_pass(mode):
    poses, times, velocity = trace()
    if mode == 'short': poses, times, velocity = poses[:120], times[:120], velocity[:120]
    if mode == 'gap': times[180:] += .02
    if mode == 'reverse': times[180] = times[179]
    if mode == 'nan_pose': poses[0, 0] = np.nan
    if mode == 'nan_velocity': velocity[0, 0] = np.nan
    if mode == 'bad_quaternion': poses[0, 3:] = 0.
    with pytest.raises(ValueError): observed_motion(poses, times, velocity)


def test_float32_norm_noise_and_quaternion_sign_are_not_rotation():
    poses, times, velocity = trace()
    poses[::2, 6] = -.9999999
    assert observed_motion(poses, times, velocity)['angular_speed_rad_s'] == 0.


@pytest.mark.parametrize('fault', [None, 'missing_step', 'invalid_handle', 'support_lost', 'moving_link'])
def test_sampling_requires_every_physics_step_and_every_collision_link(monkeypatch, fault):
    from types import SimpleNamespace as NS
    from rummagebench.authoring.layout import indoor_stability as s
    count = [0]
    subscribed = []
    unsubscribed = []
    def link(moving=False):
        def pose():
            return [[.001*count[0] if moving else 0., 0., 0., 0., 0., 0., 1.]]
        view = NS(is_physics_handle_valid=lambda: fault != 'invalid_handle',
                  _physics_view=NS(get_transforms=pose, get_velocities=lambda: [[0.]*6]))
        return NS(collision_meshes={'synthetic': 1}, _rigid_prim_view=view)
    root, lid = link(), link(fault == 'moving_link')
    obj = NS(root_link=root, links={'root': root, 'lid': lid})
    monkeypatch.setattr(s, 'resolve_object', lambda *a: obj)
    monkeypatch.setattr(s, 'relation_holds', lambda *a: not (fault == 'support_lost' and count[0] == 40))
    def subscribe(callback, **kw):
        assert kw == dict(pre_step=False, order=1)
        subscribed.append(callback)
        return NS(unsubscribe=lambda: unsubscribed.append(True))
    def settle(n):
        for _ in range(4*n):
            count[0] += 1
            if not (fault == 'missing_step' and count[0] == 20): subscribed[0](1/120)
    backend = NS(_env=NS(scene=None), settle=settle, _sim=NS(
        get_physics_dt=lambda: 1/120, get_sim_step_dt=lambda: 1/30,
        _physics_context=NS(_physx_interface=NS(subscribe_physics_on_step_events=subscribe))))
    placements = [dict(name='synthetic', support_name='unit_table')]
    if fault in ('missing_step', 'invalid_handle'):
        with pytest.raises(ValueError): s.measure_background_motion(backend, placements)
    else:
        measures, trace = s.measure_background_motion(backend, placements)
        assert len(trace['samples']) == 361
        assert measures['synthetic']['passed'] == (fault is None)
    assert unsubscribed == [True]
