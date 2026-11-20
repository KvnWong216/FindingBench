# FindingBench household search contract

The generation problem has two distinct outputs: a reproducible physical task
family, and search costs for that family. An analytical search solution does
not establish physical feasibility.

The native pilot uses `assets/tasks/tiers/household_direct_v1.yaml`, an
independent physical task-class contract, not an inheritance of easy_v1 or a
difficulty label. It reuses the existing builder, public skills, state checks
and certification machinery. Candidate count is instantiated as exactly the
complete observed set; no easy-tier population cap, natural-lookalike exclusion
or easy search-reserve budget is inherited. The current kitchen-only,
direct-container structure is a pilot scope, not a universal restriction on
later task families. Structural variants and measured difficulty strata are
separate concepts.

## Physical family

Let B be the complete frozen household background, R the robot and camera
protocol, L the complete set of publicly plausible target locations, and
p_i > 0 their prior, with sum p_i = 1. Member i reconstructs B and places the
same target in i. Its target placement must pass the existing Inside/link,
collision, stability, visibility and public skill checks. Every member uses
the same observed background state, poses and geometry. A changed native
asset topology explicitly replaces L and p; it cannot retain the source
distribution or silently omit a failed location. All derivatives of one
source house share a split group.

A checkpoint is a reconstruction input A. The simulator's loaded world is
B = Restore(A); B need not have the same floating-point serialization as A.
The invariant is repeated restoration of the SAME A giving exactly the SAME
B, including per-object state and geometry, without advancing physics.
Dumping B and subsequently using B as the next input is a different recipe.
Target insertion then has a separate strict background invariance check.

## General search

The hidden target location X is sampled once from p. Public history h contains
only instructions, images and submitted actions/feedback. A policy pi(h) must
not receive X, evaluator object IDs or an oracle path. The physical state also
includes robot pose and door/object states, so costs and reveal sets can
change with history. A negative observation may rule out one, multiple or no
locations. The general problem is a partially observable sequential decision
problem; no closed-form ordering is asserted for it.

For a complete successful run at X=i, define S_i as the number of submitted
MOVE and TURN calls, and O_i as all other submitted public calls. Rejections
count. OPEN/MOVE already return images: an extra OBSERVE is not mandatory.
GRASP and REPORT_DONE count. Report expected (sum p_i S_i, sum p_i O_i), worst
case and per-member traces. Bin the two measured axes using the configured
half-open intervals. If any positive-prior member fails or exceeds the public
horizon, reject the family; do not renormalize the successful subset.

The production reference policy remains explicitly a measured reference,
not a proved optimum. It cannot use the hidden member to choose its order.
Geometric metadata available to an evaluator's reference planner is not
exposed to the evaluated model.

## Closed-form subproblem

Assume a perfect inspection reveals exactly one previously uninspected
location, a miss leaves all subsequent costs unchanged, and no OBSERVE bypass
or coupled reveal exists. With only misses before success, a deterministic
policy is a permutation sigma chosen without knowing X.

Let d(start,j)=s+a_j and d(i,j)=b_i+a_j for distinct i,j, where all costs are
nonnegative. Inspection i costs c_i operational calls; finishing costs f.
For nonnegative weights wm, wo, at least one positive, minimize
J = wm E[S] + wo E[O]. Put q_i = wm(a_i+b_i)+wo c_i and
P_k = sum of priors at ranks k through n. Then

    E[S] = s + sum_k P_k (a_sigma(k)+b_sigma(k)) - sum_i p_i b_i
    E[O] = sum_k P_k c_sigma(k) + f
    J    = wm*s - wm*sum_i p_i*b_i + wo*f + sum_k P_k*q_sigma(k).

Swapping adjacent i,j changes J(i,j)-J(j,i) by p_j*q_i-p_i*q_j.
Thus sorting by increasing q_i/p_i is globally optimal, in O(n log n).
Zero-cost candidates come first, with deterministic index ties. There is no
departure cost after the successful inspection. The solver implements this
contract explicitly; it never fits a separable matrix to a real scene and
calls the fit exact.

For n equally likely locations, a_i=b_i=r and c_i=c:

    E[S]=n*r;             max S=(2*n-1)*r
    E[O]=c*(n+1)/2+f;     max O=n*c+f.

Here r must be the declared MOVE+TURN cost of a branch, not metres or a
known-target shortest path. For example n=21,r=3,c=2,f=2 gives expected
(63,24) and worst (123,44) ONLY in this model. It is not a measurement of the
current generated kitchen. The public horizon applies to S_i+O_i jointly.

Different weights can give different permutations: there need not be one
order minimizing both axes. Report weights and both resulting costs. This
solver does not claim to enumerate a Pareto frontier. For a general static
travel matrix, existing subset DP remains exact for at most ten candidates;
it is not replaced by the ratio rule. Real dynamic kitchen costs continue to
require measured reference runs unless all model assumptions are established.

If a measured inspection reveals a disjoint group G of locations at once,
replace its prior by p_G=sum(i in G) p_i and apply the same ratio solution to
groups. Retain all member locations and their original priors. For example,
five locations split into two equally likely reveal groups with one OPEN per
group require 1.5 expected OPEN calls, not three. `separable_group_search_cost`
implements this partition case and rejects overlaps or omitted locations.
Shared furniture alone does not prove shared reveal. Overlapping or partial
reveal sets require a belief/state model and do not inherit this closed form.

## Code and reproducible CPU check

`difficulty.separable_search_cost` implements the ratio solution;
`exact_search_cost` retains the general static-matrix DP. Candidate-profile
admission uses the closed form for its explicitly zero-travel operation
model. Neither function grants physical acceptance or final difficulty.

```bash
python -m pytest tests/unit/test_separable_search.py tests/unit/test_infinigen_pipeline.py -q
```

Tests compare asymmetric, nonuniform cases against independent exhaustive
enumeration and the existing DP; cover distinct axis optima, all 21 symmetric
locations, zero costs, the terminal departure and invalid model inputs.

The physical pipeline order is: obtain all real native regions, bind each
region to an existing SceneSlot with an explicit fillable link, probe all
regions, freeze and reconstruct one background, insert and validate every
member, replay the target-blind public search, then bin actual costs. The
analytical solution removes search-order enumeration under its assumptions;
it does not replace any of these physical checks.
