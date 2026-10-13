"""Exact costs ONLY for a declared finite, single-reveal search model.

Every candidate has a positive known prior. Travel costs are supplied by the
caller; each inspection perfectly reveals exactly one location. A negative
result removes that location. No lookahead knows the actual target. Real
visibility, routes, coupled reveals and skills require separate measurement.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


def interval_label(value: float, intervals: dict[str, list]) -> str | None:
    """Half-open intervals [lo, hi), with null meaning no upper limit."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative")
    matches = [name for name, (lo, hi) in intervals.items()
               if lo <= value and (hi is None or value < hi)]
    if len(matches) > 1:
        raise ValueError("overlapping difficulty intervals")
    return matches[0] if matches else None


def exact_search_cost(travel, priors, inspect_cost, *, finish_cost: float = 2.,
                      movement_weight: float = 1., operation_weight: float = 1.) -> dict:
    """DP over uninspected candidates and current node (last node = start).

    Returns a target-blind order conditional on successive negative results.
    Minimizes the declared weighted expected cost, reports both dimensions
    and worst-case cost. This optimum does NOT certify real-scene difficulty.
    finish_cost=2 models target GRASP + REPORT_DONE; inspection costs must
    explicitly count OPEN/OBSERVE/other required operations for this model.
    The caller's travel matrix must include the intended MOVE/TURN costs.
    """
    p, check, d = np.asarray(priors, float), np.asarray(inspect_cost, float), np.asarray(travel, float)
    n = len(p)
    if not 1 <= n <= 10 or p.shape != (n,) or check.shape != (n,) or d.shape != (n + 1, n + 1):
        raise ValueError("invalid finite search-model dimensions; at most 10 candidates")
    if (not np.isfinite(p).all() or np.any(p <= 0) or abs(float(p.sum()) - 1.) > 1e-8
            or not np.isfinite(d).all() or np.any(d < 0) or not np.isfinite(check).all() or np.any(check < 0)
            or any(not math.isfinite(x) or x < 0 for x in (finish_cost, movement_weight, operation_weight))
            or movement_weight + operation_weight <= 0):
        raise ValueError("invalid search-model costs or prior")
    if not np.allclose(np.diag(d), 0.):
        raise ValueError("travel diagonal must be zero")

    @lru_cache(None)
    def solve(mask, last):
        if mask == 0:
            return (0., ())
        remaining = [i for i in range(n) if mask & (1 << i)]
        mass = sum(p[i] for i in remaining)
        choices = []
        for i in remaining:
            downstream, order = solve(mask ^ (1 << i), i)
            miss = 1. - p[i] / mass
            value = (movement_weight * d[last, i] + operation_weight * check[i]
                     + (p[i] / mass) * operation_weight * finish_cost + miss * downstream)
            choices.append((float(value), (i,) + order))
        return min(choices)

    optimum, order = solve((1 << n) - 1, n)
    movement = operations = expected_move = expected_ops = 0.
    last = n
    cases = []
    for i in order:
        movement += d[last, i]
        operations += check[i]
        expected_move += p[i] * movement
        expected_ops += p[i] * (operations + finish_cost)
        cases.append({"location": i, "movement": float(movement), "operations": float(operations + finish_cost)})
        last = i
    return {"scope": "exact_declared_discrete_model", "order_on_negative_results": list(order),
            "expected_movement": float(expected_move), "expected_operations": float(expected_ops),
            "weighted_optimum": optimum, "movement_weight": movement_weight, "operation_weight": operation_weight,
            "worst_case_movement": float(movement), "worst_case_operations": float(operations + finish_cost),
            "per_target_outcomes": cases, "real_scene_difficulty_certified": False,
            "assumptions": ["known_positive_prior", "one_perfect_reveal_per_candidate",
                            "travel_matrix_known_before_search", "travel_cost_does_not_change_after_inspection",
                            "no_observe_bypass_or_coupled_reveal", "finish_is_grasp_and_report"]}


def euclidean_proxies(start_xy, candidates_xy) -> dict:
    points = np.asarray([*candidates_xy, start_xy], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid candidate positions")
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return {"scope": "euclidean_distance_proxy_only", "unit": "metres", "candidate_count": len(candidates_xy),
            "pairwise": distances[:-1, :-1].tolist(), "start_to_candidate": distances[-1, :-1].tolist(),
            "traversable_path_measured": False, "real_scene_difficulty_certified": False}
