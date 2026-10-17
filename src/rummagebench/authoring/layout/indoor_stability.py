"""Versioned observed-motion check for native background props.

PhysX split impulses separate integrated motion from reported solver velocity.
Observe every physics step, retain both quantities, and never change physics.
This bounded observation is not a proof of indefinite stability or clearance.
"""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.tasks.household_physics import placement_stability
from rummagebench.authoring.tasks.native_occupancy import plain
from rummagebench.sim.omnigibson.entity_resolver import resolve_object
from rummagebench.sim.omnigibson.probe_support import relation_holds

PROFILE = 'observed-motion-physics-steps-v3'


def observed_motion(poses, times, reported_velocities):
    """Same displacement / speed limits, applied to realized step motion.

    Raw solver velocities stay in the receipt and must be finite. No smoothing,
    percentile filtering, or sleeping exemption hides a real motion spike.
    """
    poses = np.asarray(poses, float)
    times = np.asarray(times, float)
    raw = np.asarray(reported_velocities, float)
    if (poses.ndim != 2 or poses.shape[1:] != (7,) or len(poses) < 2
            or times.shape != (len(poses),) or raw.shape != (len(poses), 6)
            or not np.isfinite(times).all() or not np.isfinite(raw).all()
            or not np.isfinite(poses).all() or np.any(np.diff(times) <= 0)
            or times[-1]-times[0] < 2.99 or np.diff(times).max() > 1/60+1e-7):
        raise ValueError('CONTIGUOUS_PHYSICS_MOTION_SAMPLES_REQUIRED')
    # Validate poses, including quaternion norms, before normalization.
    legacy = placement_stability(poses,
        raw[np.linalg.norm(raw[:, :3], axis=1).argmax(), :3],
        raw[np.linalg.norm(raw[:, 3:], axis=1).argmax(), 3:])
    q = poses[:, 3:]/np.linalg.norm(poses[:, 3:], axis=1)[:, None]
    following = q[1:]*np.where(np.sum(q[:-1]*q[1:], axis=1) < 0, -1., 1.)[:, None]
    angle = 4*np.arctan2(np.linalg.norm(following-q[:-1], axis=1),
                         np.linalg.norm(following+q[:-1], axis=1))
    speeds = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)/np.diff(times)
    spins = angle/np.diff(times)
    result = placement_stability(poses, [float(speeds.max()), 0., 0.],
                                 [float(spins.max()), 0., 0.])
    result.update(profile=PROFILE, duration_seconds=float(times[-1]-times[0]),
                  physics_samples=len(times), reported_solver_velocity_check=legacy)
    return result


def measure_background_motion(backend, placements, *, steps=90):
    """Normal OG steps with read-only post-physics samples of every collider link.

    The callback cannot silently lose samples or swallow reader exceptions.
    OnTop is checked on each normal state-update step, never in a stale cache
    inside the physics callback. Initial support is checked as well.
    """
    objects = {p['name']: resolve_object(backend._env.scene, p['name']) for p in placements}
    links = {}
    for name, obj in objects.items():
        links[name] = {key: link for key, link in obj.links.items()
                       if link is obj.root_link or link.collision_meshes}
        if not links[name]: raise ValueError('MOTION_COLLIDER_LINKS_REQUIRED')
    dt = float(backend._sim.get_physics_dt())
    normal_dt = float(backend._sim.get_sim_step_dt())
    ratio = round(normal_dt/dt)
    if not 0 < dt <= 1/60 or abs(ratio*dt-normal_dt) > 1e-7 or steps*normal_dt < 3-1e-7:
        raise ValueError('MOTION_SAMPLING_CLOCK_UNSUPPORTED')
    samples = []
    errors = []
    def sample(elapsed=0.):
        try:
            if samples and abs(float(elapsed)-dt) > 1e-7:
                raise ValueError('PHYSICS_CALLBACK_DT_CHANGED')
            row = dict(time=len(samples)*dt, bodies={})
            for name, group in links.items():
                row['bodies'][name] = {}
                for key, link in group.items():
                    view = link._rigid_prim_view
                    if not view.is_physics_handle_valid(): raise ValueError('INVALID_PHYSICS_HANDLE')
                    row['bodies'][name][key] = dict(pose=plain(view._physics_view.get_transforms()[0]),
                        velocity=plain(view._physics_view.get_velocities()[0]))
            samples.append(row)
        except Exception as exc:
            errors.append(f'{type(exc).__name__}:{exc}')
    supports = {p['name']: [] for p in placements}
    def check_supports():
        for p in placements:
            supports[p['name']].append(relation_holds(backend, p['name'], p['support_name'], 'on_top'))
    sample()
    check_supports()
    subscription = backend._sim._physics_context._physx_interface.subscribe_physics_on_step_events(
        sample, pre_step=False, order=1)
    try:
        for _ in range(steps):
            backend.settle(1)
            check_supports()
    finally:
        subscription.unsubscribe()
    if errors: raise ValueError('MOTION_SAMPLING_FAILED:'+errors[0])
    if len(samples) != steps*ratio+1:
        raise ValueError('MOTION_PHYSICS_SAMPLE_COUNT_MISMATCH')
    times = [r['time'] for r in samples]
    measurements = {}
    for name, group in links.items():
        checks = {key: observed_motion([r['bodies'][name][key]['pose'] for r in samples], times,
            [r['bodies'][name][key]['velocity'] for r in samples]) for key in group}
        result = dict(profile=PROFILE, passed=all(c['passed'] for c in checks.values()) and all(supports[name]),
                      links=checks, support_samples=supports[name],
                      limits=next(iter(checks.values()))['limits'])
        for metric in ('position_drift_m', 'rotation_drift_deg', 'linear_speed_m_s', 'angular_speed_rad_s'):
            result[metric] = max(c[metric] for c in checks.values())
        measurements[name] = result
    return measurements, dict(profile=PROFILE, physics_dt=dt, normal_dt=normal_dt, samples=samples)
