"""Independent exhaustive checks for the declared analytical search model."""
import itertools

import numpy as np
import pytest

from rummagebench.authoring.tasks.difficulty import exact_search_cost, separable_search_cost,separable_group_search_cost


def objective(order, a, b, p, c, wm, wo, start=0., finish=2.):
    movement, operations, value, last = start, 0., 0., None
    for i in order:
        movement += a[i] + (0 if last is None else b[last])
        operations += c[i]
        value += p[i] * (wm * movement + wo * (operations + finish))
        last = i
    return value


@pytest.mark.parametrize('weights', [(1., 1.), (0., 1.), (1., 0.), (.25, 3.)])
def test_asymmetric_nonuniform_ratio_matches_exhaustive_and_existing_dp(weights):
    a, b, p, c = [1., 3., 2., 4.], [5., 0., 1., 2.], [.1, .4, .3, .2], [2., 5., 1., 3.]
    wm, wo = weights
    result = separable_search_cost(a, b, p, c, start_cost=3., movement_weight=wm, operation_weight=wo)
    best = min(objective(o, a, b, p, c, wm, wo, start=3.) for o in itertools.permutations(range(4)))
    travel = np.zeros((5, 5))
    for i in range(4):
        travel[4, i] = 3 + a[i]
        for j in range(4):
            if i != j:
                travel[i, j] = b[i] + a[j]
    dp = exact_search_cost(travel, p, c, movement_weight=wm, operation_weight=wo)
    assert result['weighted_optimum'] == pytest.approx(best)
    assert result['weighted_optimum'] == pytest.approx(dp['weighted_optimum'])
    assert result['real_scene_difficulty_certified'] is False


def test_twenty_one_symmetric_locations_closed_form():
    n, radius, check, finish = 21, 3., 2., 2.
    result = separable_search_cost([radius]*n, [radius]*n, [1/n]*n, [check]*n, finish_cost=finish)
    assert result['expected_movement'] == pytest.approx(n*radius)
    assert result['expected_operations'] == pytest.approx(check*(n+1)/2+finish)
    assert result['worst_case_movement'] == (2*n-1)*radius
    assert result['worst_case_operations'] == n*check+finish
    assert result['order_on_negative_results'] == list(range(n))


def test_last_location_has_no_return_trip_and_finish_is_always_paid():
    result = separable_search_cost([2.], [100.], [1.], [3.], start_cost=7., finish_cost=2.)
    assert result['expected_movement'] == 9.
    assert result['expected_operations'] == 5.


def test_spatial_and_operational_optima_can_have_different_orders():
    args = ([1., 20.], [1., 20.], [.5, .5], [20., 1.])
    assert separable_search_cost(*args, operation_weight=0)['order_on_negative_results'] == [0, 1]
    assert separable_search_cost(*args, movement_weight=0)['order_on_negative_results'] == [1, 0]


def test_zero_cost_candidates_and_subnormal_prior_are_ordered_without_division_overflow():
    result = separable_search_cost([0., 0., 1.], [0., 0., 0.], [1e-320, .5, .5], [0., 1e-320, 1.])
    assert result['order_on_negative_results'] == [0, 1, 2]


@pytest.mark.parametrize('kwargs', [
    {'priors': [0., 1.]}, {'enter': [-1., 2.]}, {'leave': [float('nan'), 1.]},
    {'inspect_cost': [1.]}, {'movement_weight': 0., 'operation_weight': 0.}, {'finish_cost': -1.},
])
def test_reject_invalid_contract(kwargs):
    values = dict(enter=[1., 2.], leave=[1., 2.], priors=[.5, .5], inspect_cost=[2., 2.])
    values.update(kwargs)
    with pytest.raises(ValueError):
        separable_search_cost(**values)


def test_complete_native_family_contract_is_exact_and_does_not_change_easy():
    from rummagebench.authoring.tasks.household import native_direct_tier
    from rummagebench.authoring.tasks.tiers import load_tier
    from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate
    from rummagebench.authoring.tasks.gates import structural_violations,SEARCH_WIDTH_MISMATCH
    easy=load_tier('easy_v1');native=native_direct_tier(21,48)
    assert easy.search.candidate_slot_count.max==4 and easy.budget.max_planning_steps==16
    assert native.tier=='household_direct' and native.search.candidate_slot_count.min==native.search.candidate_slot_count.max==21
    assert native.gates.start_visibility_max==easy.gates.start_visibility_max
    assert native.gates.modes['in_container']==easy.gates.modes['in_container']
    assert native.budget.search_reserve_steps==0 and not native.planner_enabled
    assert native.max_distractors is None and native.search.lookalike_count.max is None
    cert=SearchStructureCertificate('household_direct','in_container',1,21,1,1,0,0,
        'cabinet_interior',0,3)
    assert not structural_violations(cert,native)
    assert SEARCH_WIDTH_MISMATCH in [v.code for v in structural_violations(cert,easy)]
    cert.candidate_slot_count=20
    assert SEARCH_WIDTH_MISMATCH in [v.code for v in structural_violations(cert,native)]


def test_native_execution_contract_does_not_load_easy(monkeypatch):
    from rummagebench.authoring.tasks.household import native_direct_tier
    from rummagebench.authoring.tasks import tiers
    original=tiers.load_tier
    seen=[]
    def load(name):
        seen.append(name)
        assert name!='easy_v1'
        return original(name)
    monkeypatch.setattr(tiers,'load_tier',load)
    assert native_direct_tier(21,48).search.candidate_slot_count.min==21
    assert seen==['household_direct_v1']


def test_multiple_locations_revealed_together_do_not_manufacture_extra_search_cost():
    # UNIT declared groups: one OPEN reveals two locations in the first bank,
    # three in the second. No claim that a real shared furniture does this.
    grouped=separable_group_search_cost([[0,1],[2,3,4]],[3.,3.],[3.,3.],
        [.25,.25,1/6,1/6,1/6],[1.,1.])
    reference=separable_search_cost([3.,3.],[3.,3.],[.5,.5],[1.,1.])
    assert grouped['weighted_optimum']==pytest.approx(reference['weighted_optimum'])
    assert grouped['expected_operations']==pytest.approx(3.5) # 1.5 OPEN + GRASP + REPORT
    outcomes={r['location']:r for r in grouped['per_target_outcomes']}
    assert set(outcomes)==set(range(5))
    assert outcomes[0]['operations']==outcomes[1]['operations']
    assert not grouped['real_scene_difficulty_certified']


@pytest.mark.parametrize('groups',[[[0,1],[1,2]],[[0],[2]],[[0,1,2],[]],[[False],[1,2]]])
def test_grouped_model_refuses_overlap_omission_empty_or_alias(groups):
    with pytest.raises(ValueError):
        separable_group_search_cost(groups,[1.,1.],[1.,1.],[.2,.3,.5],[1.,1.])
