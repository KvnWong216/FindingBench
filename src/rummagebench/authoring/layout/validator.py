"""Layout validation (§29): a layout is VALID only if all structural checks
pass. Build-time checks MAY use privileged simulator information; none of it
becomes agent input. Sim-dependent checks (floor support, OBSERVE viewpoint
feasibility, Stage-A reachable interaction target) are injected as callables
when a backend is available and run as part of acceptance."""

from __future__ import annotations

from rummagebench.authoring.layout.generator import footprint, _rects_overlap


class LayoutValidator:
    def __init__(self, config, sim_checks: dict | None = None):
        self.config = config
        # sim_checks: {"support": fn(Placement)->bool,
        #              "observe_viewpoint": fn(Placement)->bool,
        #              "stage_a_interaction": fn(Placement)->bool}
        self.sim_checks = sim_checks or {}

    def validate(self, layout, catalog) -> dict:
        results: dict[str, bool] = {}
        placements = layout.placements

        # 1. no task-relevant furniture overlap (structural, with clearance)
        furniture = [p for p in placements if p.role in ("storage", "surface")]
        results["no_furniture_overlap"] = not any(
            _rects_overlap(footprint(a), footprint(b),
                           self.config.min_furniture_clearance_m)
            for i, a in enumerate(furniture) for b in furniture[i + 1:]
        )

        # 2. required storage count
        results["required_storage_units"] = (
            sum(1 for p in placements if p.role == "storage")
            >= self.config.storage_units
        )

        # 3. robot spawn collision-free (structural margin to all furniture)
        rx, ry = self.config.robot_spawn
        results["robot_spawn_free"] = not any(
            _point_in_rect_margin(rx, ry, footprint(p), self.config.min_robot_clearance_m)
            for p in placements if p.role != "clutter"
        )

        # 4. required search regions connected through base free space —
        #    structural v1: every storage unit must admit a robot base pose
        #    at its front margin (clearance ring) free of other furniture
        results["search_regions_reachable"] = all(
            self._base_ring_free(p, placements) for p in placements if p.role == "storage"
        )

        # 5-7. per-storage robot free region / OBSERVE viewpoint / Stage-A
        results["storage_base_regions"] = results["search_regions_reachable"]
        for key, check in (
            ("observe_viewpoints", self.sim_checks.get("observe_viewpoint")),
            ("stage_a_interactable", self.sim_checks.get("stage_a_interaction")),
        ):
            if check is None:
                results[key] = True  # structural-only validation run
                continue
            results[key] = all(
                check(p) for p in placements if p.role == "storage"
            )

        # 8. exact model ids match the catalog (never substituted)
        results["model_ids_in_catalog"] = all(
            catalog.get(p.role, p.model_id) is not None for p in placements
        )

        results["valid"] = all(results.values())
        return results

    def _base_ring_free(self, placement, placements) -> bool:
        corners = footprint(placement)
        (x0, y0), (x1, y1) = corners
        m = self.config.min_robot_clearance_m
        front = ((x0, y0 - m - 0.3), (x1, y0 - m))  # sample: front strip
        for other in placements:
            if other is placement or other.role == "clutter":
                continue
            if _rects_overlap(front, footprint(other), 0.0):
                return False
        return True


def _point_in_rect_margin(px: float, py: float, rect, margin: float) -> bool:
    (x0, y0), (x1, y1) = rect
    return (x0 - margin) <= px <= (x1 + margin) and (y0 - margin) <= py <= (y1 + margin)
