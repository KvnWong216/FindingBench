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


def separable_search_cost(enter, leave, priors, inspect_cost, *, start_cost: float = 0.,
                          finish_cost: float = 2., movement_weight: float = 1.,
                          operation_weight: float = 1.) -> dict:
    """Closed-form optimum for d(start,j)=start_cost+enter[j], d(i,j)=leave[i]+enter[j].

    Each inspection reveals exactly one location, perfectly. The last successful
    inspection does NOT pay its leave cost. Pairwise exchange gives the order
    by increasing (wm*(enter+leave)+wo*inspection)/prior. This is a declared
    model, never an inference that a real kitchen has separable travel costs.
    Unlike subset DP, this O(n log n) solver has no ten-location limit.
    """
    a, b, p, c = (np.asarray(x, dtype=float) for x in (enter, leave, priors, inspect_cost))
    if (p.ndim != 1 or not len(p) or any(x.shape != p.shape for x in (a, b, c))
            or any(not np.isfinite(x).all() for x in (a, b, p, c))
            or np.any(p <= 0) or abs(math.fsum(p) - 1.) > 1e-8
            or any(np.any(x < 0) for x in (a, b, c))
            or any(not math.isfinite(x) or x < 0 for x in
                   (start_cost, finish_cost, movement_weight, operation_weight))
            or movement_weight + operation_weight <= 0):
        raise ValueError("invalid separable search-model costs or prior")
    q = movement_weight * (a + b) + operation_weight * c
    # Cross products avoid division overflow for very small positive priors.
    from functools import cmp_to_key
    from fractions import Fraction
    if not np.isfinite(q).all():
        raise ValueError("separable search-model weighted costs overflow")
    qp = [(Fraction(float(qi)), Fraction(float(pi))) for qi, pi in zip(q, p)]
    def compare(i, j):
        left, right = qp[i][0] * qp[j][1], qp[j][0] * qp[i][1]
        return -1 if left < right else 1 if left > right else (i > j) - (i < j)
    order = sorted(range(len(p)), key=cmp_to_key(compare))
    movement, operations, last = float(start_cost), 0., None
    cases = []
    for i in order:
        movement += float(a[i]) + (0. if last is None else float(b[last]))
        operations += float(c[i])
        cases.append({"location": i, "movement": movement, "operations": operations + finish_cost})
        last = i
    em = math.fsum(float(p[r["location"]]) * r["movement"] for r in cases)
    eo = math.fsum(float(p[r["location"]]) * r["operations"] for r in cases)
    objective = movement_weight * em + operation_weight * eo
    if not all(math.isfinite(v) for v in (em, eo, objective, movement, operations + finish_cost)):
        raise ValueError("separable search-model accumulated costs overflow")
    return {"scope": "exact_declared_separable_search_model", "solver": "pairwise_exchange_ratio",
            "order_on_negative_results": order, "expected_movement": em, "expected_operations": eo,
            "weighted_optimum": objective, "movement_weight": movement_weight,
            "operation_weight": operation_weight, "worst_case_movement": movement,
            "worst_case_operations": operations + finish_cost, "per_target_outcomes": cases,
            "real_scene_difficulty_certified": False,
            "assumptions": ["known_positive_prior", "one_perfect_reveal_per_candidate",
                            "declared_separable_static_travel", "no_departure_after_success",
                            "no_observe_bypass_or_coupled_reveal", "constant_finish_cost"],
            "cost_contract": "caller-supplied costs; movement must count MOVE+TURN, operations all other submitted calls"}


def separable_group_search_cost(groups, enter, leave, priors, inspect_cost, **kwargs) -> dict:
    """Exact separable model when each inspection reveals a disjoint group.

    Groups must be a partition of ALL target locations. They must come from
    established reveal sets, not a guess based on shared furniture. Overlap,
    partial visibility or history-dependent reveals require a different model.
    """
    p=np.asarray(priors,dtype=float)
    if (p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any(p<=0)
            or abs(math.fsum(p)-1.)>1e-8 or not groups
            or len(enter)!=len(groups) or len(leave)!=len(groups) or len(inspect_cost)!=len(groups)):
        raise ValueError('invalid complete grouped search model')
    flat=[i for group in groups for i in group]
    if (any(not group for group in groups) or any(type(i) is not int for i in flat)
            or len(flat)!=len(p) or set(flat)!=set(range(len(p)))):
        raise ValueError('reveal groups must partition the complete target set without overlap')
    result=separable_search_cost(enter,leave,[math.fsum(p[i] for i in g) for g in groups],inspect_cost,**kwargs)
    group_outcomes=result.pop('per_target_outcomes')
    result['order_of_reveal_groups_on_negative_results']=result.pop('order_on_negative_results')
    result['per_target_outcomes']=[{**row,'location':i,'reveal_group':row['location']}
                                   for row in group_outcomes for i in groups[row['location']]]
    result.update(scope='exact_declared_disjoint_group_separable_model',reveal_groups=[list(g) for g in groups],
                  assumptions=['known_positive_prior','declared_separable_static_travel',
                    'established_disjoint_perfect_reveal_groups','constant_finish_cost',
                    'no_history_dependent_or_overlapping_reveals'])
    return result


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
