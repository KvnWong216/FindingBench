"""Synthetic unit data only. Real solver trials are recorded separately."""
import copy

import pytest

from rummagebench.authoring.layout.living_brief import compile_brief, allocate_counts
from rummagebench.authoring.layout.indoor_background import verify_result


def inputs():
    context = dict(objects=[dict(name=n, support=True) for n in ['a', 'b']],
                   retained_native_objects=['a', 'b'])
    brief = dict(version=1, author='unit test', description='Some surfaces can be empty.',
                 total_count=2, surfaces={n: dict(minimum=0, preferred=1, maximum=2,
                 fullness=.1, rationale='unit') for n in ['a', 'b']})
    return context, brief


@pytest.mark.parametrize('mutation', ['unknown_surface', 'missing_surface', 'hidden_target',
                                     'inverted_bounds', 'bad_total', 'nan', 'fractional_count'])
def test_invalid_briefs_rejected(mutation):
    context, b = inputs()
    if mutation == 'unknown_surface': b['surfaces']['extra'] = b['surfaces']['a'].copy()
    if mutation == 'missing_surface': del b['surfaces']['b']
    if mutation == 'hidden_target': b['target_location'] = 'a'
    if mutation == 'inverted_bounds': b['surfaces']['a']['minimum'] = 2
    if mutation == 'bad_total': b['total_count'] = 5
    if mutation == 'nan': b['surfaces']['a']['fullness'] = float('nan')
    if mutation == 'fractional_count': b['surfaces']['a']['maximum'] = 1.5
    with pytest.raises(ValueError): compile_brief(b, context)


def test_ranges_allow_empty_support_but_not_missing_evidence_or_wrong_total():
    context, b = inputs()
    b['surfaces']['a']['preferred'] = 2
    b['surfaces']['b']['preferred'] = 0
    assert compile_brief(b, context) == compile_brief(copy.deepcopy(b), copy.deepcopy(context))
    req = dict(living_brief=b, seed=31, count_allocation=allocate_counts(b, 31),
               native_context=context, assets={'bowl/unit': {}})
    placements = [dict(name=f'p{i}', category='bowl', model_id='unit', support_name='a',
                      role='clutter', position=[0., 0., 1.], orientation_deg=0.,
                      aabb_size=[.1, .1, .1]) for i in range(2)]
    result = dict(status='constraints_satisfied', placements=placements,
                  violations=dict(daily_items_a=0, daily_items_b=0))
    verify_result(req, result)
    missing = copy.deepcopy(result); del missing['violations']['daily_items_b']
    with pytest.raises(ValueError, match='CONSTRAINT_EVIDENCE'): verify_result(req, missing)
    wrong = copy.deepcopy(result); wrong['placements'].pop()
    with pytest.raises(ValueError, match='INCOMPLETE_LIVING'): verify_result(req, wrong)
    wrong = copy.deepcopy(result); wrong['placements'][0]['support_name'] = 'extra'
    with pytest.raises(ValueError, match='INCOMPLETE_LIVING'): verify_result(req, wrong)
    wrong = copy.deepcopy(result); wrong['placements'][0]['model_id'] = 'unapproved'
    with pytest.raises(ValueError, match='ASSET_NOT_APPROVED'): verify_result(req, wrong)


def test_allocation_optimal_bounded_and_deterministic():
    from itertools import product
    _, b = inputs()
    b['surfaces']['a']['preferred'] = 2
    b['surfaces']['b']['preferred'] = 2
    best = min(abs(a-2)+abs(c-2) for a, c in product(range(3), repeat=2) if a+c == 2)
    seen = set()
    for seed in range(30):
        counts = allocate_counts(b, seed)
        assert counts == allocate_counts(b, seed)
        assert sum(counts.values()) == b['total_count']
        assert sum(abs(n-b['surfaces'][k]['preferred']) for k, n in counts.items()) == best
        seen.add(tuple(counts.values()))
    assert seen == {(0, 2), (1, 1), (2, 0)}


def test_tampered_allocation_rejected_before_receipt():
    context, b = inputs()
    req = dict(living_brief=b, seed=31, count_allocation={'a': 2, 'b': 0},
               native_context=context)
    with pytest.raises(ValueError, match='ALLOCATION_MISMATCH'): verify_result(req, {})
