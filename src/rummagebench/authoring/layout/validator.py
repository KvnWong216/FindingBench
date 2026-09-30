"""Candidate checks are separate from evidence-backed simulator acceptance.

Simulator callbacks receive each Placement (connectivity, observe and interaction
checks receive storage placements only). They must check the actual loaded scene.
Missing callbacks are not_run, never a pass. No simulator loader is supplied here. layout_applied uniquely receives the full
GeneratedLayout and must verify it matches the actual loaded scene.
"""
from __future__ import annotations
import math
from rummagebench.authoring.layout.generator import footprint, placements_overlap, access_strips_free, _point_near_rect


class LayoutValidator:
    def __init__(self, config, sim_checks: dict | None = None):
        self.config = config
        self.sim_checks = sim_checks or {}

    def validate(self, layout, catalog) -> dict:
        ps = layout.placements
        surfaces = {p.name: p for p in ps if p.role == 'surface'}
        def supported(p):
            s = surfaces.get(p.support_name)
            if s is None:
                return False
            (x0,y0),(x1,y1) = footprint(p)
            (u0,v0),(u1,v1) = footprint(s)
            return (u0 <= x0 <= x1 <= u1 and v0 <= y0 <= y1 <= v1
                    and math.isclose(p.position[2]-p.aabb_size[2]/2,
                                     s.position[2]+s.aabb_size[2]/2, abs_tol=1e-8))
        def catalog_match(p):
            try:
                entry = catalog.get(p.role, p.model_id)
                return entry.category == p.category and entry.aabb_size == p.aabb_size
            except ValueError:
                return False
        checks = {
            'unique_names': len({p.name for p in ps}) == len(ps),
            'furniture_within_regions': all(
                any(r[0] <= footprint(p)[0][0] and r[1] <= footprint(p)[0][1]
                    and footprint(p)[1][0] <= r[2] and footprint(p)[1][1] <= r[3]
                    for r in self.config.placement_regions)
                for p in ps if p.role != 'clutter'),
            'no_object_overlap': not any(placements_overlap(a,b,self.config.min_furniture_clearance_m)
                                        for i,a in enumerate(ps) for b in ps[i+1:]),
            'required_storage_units': sum(p.role=='storage' for p in ps) == self.config.storage_units,
            'required_work_surfaces': len(surfaces) == self.config.work_surfaces,
            'required_clutter': (sum(p.role=='clutter' for p in ps) == self.config.work_surfaces*self.config.clutter_per_surface
                                and all(sum(p.role=='clutter' and p.support_name==name for p in ps)
                                        == self.config.clutter_per_surface for name in surfaces)),
            'clutter_supported': all(supported(p) for p in ps if p.role=='clutter'),
            'floor_aligned_candidates': all(math.isclose(p.position[2]-p.aabb_size[2]/2, 0., abs_tol=1e-8)
                                            for p in ps if p.role!='clutter'),
            'robot_spawn_free': not any(_point_near_rect(*self.config.robot_spawn, footprint(p), self.config.min_robot_clearance_m)
                                       for p in ps if p.role!='clutter'),
            'storage_access_strips_free': access_strips_free(ps,self.config),
            'model_ids_in_catalog': all(catalog_match(p) for p in ps),
            'no_rejected_requests': not layout.rejected,
        }
        structural_valid = all(checks.values())
        evidence = {key: {'status': 'pass' if value else 'fail'} for key,value in checks.items()}
        # asset_available must resolve the exact category/model against the installed
        # dataset. In particular, a scene-instance name must not be guessed into an ID.
        required = {'layout_applied': [layout], 'support': ps, 'asset_available': ps, 'scene_collision': ps,
                    'search_connectivity': [p for p in ps if p.role=='storage'],
                    'observe_viewpoint': [p for p in ps if p.role=='storage'],
                    'stage_a_interaction': [p for p in ps if p.role=='storage']}
        for key, placements in required.items():
            callback = self.sim_checks.get(key)
            if callback is None:
                evidence[key] = {'status': 'not_run', 'reason': ('Exact category/model IDs have not been verified against the installed dataset'
                                                                   if key == 'asset_available' else 'No loaded-scene check supplied')}
                checks[key] = False
                continue
            try:
                passed = [callback(p) for p in placements]
                if any(type(value) is not bool for value in passed):
                    raise TypeError('Simulator checks must return explicit bool results')
                checks[key] = all(passed)
                evidence[key] = {'status': 'pass' if checks[key] else 'fail', 'checked': len(passed)}
            except Exception as exc:
                checks[key] = False
                evidence[key] = {'status': 'fail', 'reason': str(exc)}
        accepted = structural_valid and all(checks[key] for key in required)
        return {**checks, 'structural_valid': structural_valid, 'sim_accepted': accepted,
                'valid': accepted, 'checks': evidence}
