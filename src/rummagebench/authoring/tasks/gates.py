"""CPU + GPU tier gates and the task-authoring rejection taxonomy.

Tier gates read ONLY the SearchStructureCertificate (plus its attached
SimulatorEvidence): a tier is a certification result, never a planner label.
Plan-level CPU gates (eligibility, semantic approval, schema, determinism)
live in cpu_plan_gates() and need the static assets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate
from rummagebench.authoring.tasks.tiers.base import Bound, TierSpec

# task-authoring rejection codes (kept verbatim; report.json histograms
# count these)
INVALID_SLOT = "INVALID_SLOT"
INVALID_PRIOR = "INVALID_PRIOR"
INVALID_OBJECT_CATEGORY = "INVALID_OBJECT_CATEGORY"
GEOMETRY_OVERFLOW = "GEOMETRY_OVERFLOW"
PLACEMENT_FAILED = "PLACEMENT_FAILED"
SETTLE_FAILED = "SETTLE_FAILED"
START_VISIBLE = "START_VISIBLE"
REVEAL_TOO_WEAK = "REVEAL_TOO_WEAK"
OPEN_NOT_REQUIRED = "OPEN_NOT_REQUIRED"
TARGET_PREMATURELY_GRASPABLE = "TARGET_PREMATURELY_GRASPABLE"
TARGET_NOT_GRASPABLE = "TARGET_NOT_GRASPABLE"
REARRANGEMENT_DEPTH_MISMATCH = "REARRANGEMENT_DEPTH_MISMATCH"
SEARCH_WIDTH_MISMATCH = "SEARCH_WIDTH_MISMATCH"
HORIZON_LOWER_BOUND = "HORIZON_LOWER_BOUND"
START_UNREACHABLE = "START_UNREACHABLE"  # no MOVE/TURN route to the first interaction pose
PHYSICAL_UNSOLVABLE = "PHYSICAL_UNSOLVABLE"
TIER_GATE_FAILED = "TIER_GATE_FAILED"

REJECTION_CODES = (
    INVALID_SLOT, INVALID_PRIOR, INVALID_OBJECT_CATEGORY, GEOMETRY_OVERFLOW,
    PLACEMENT_FAILED, SETTLE_FAILED, START_VISIBLE, REVEAL_TOO_WEAK,
    OPEN_NOT_REQUIRED, TARGET_PREMATURELY_GRASPABLE, TARGET_NOT_GRASPABLE,
    REARRANGEMENT_DEPTH_MISMATCH, SEARCH_WIDTH_MISMATCH, HORIZON_LOWER_BOUND,
    PHYSICAL_UNSOLVABLE, TIER_GATE_FAILED,
)


@dataclass(frozen=True)
class Rejection:
    code: str
    detail: str
    stage: str = "cpu"  # cpu | gpu

    def __post_init__(self):
        if self.code not in REJECTION_CODES:
            raise ValueError(f"unknown rejection code {self.code!r}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# tier structure (CPU): search structure + horizon lower bound
# ---------------------------------------------------------------------------

_STRUCTURE_FIELDS: tuple[tuple[str, str], ...] = (
    ("room_count", SEARCH_WIDTH_MISMATCH),
    ("candidate_slot_count", SEARCH_WIDTH_MISMATCH),
    ("containment_depth", TIER_GATE_FAILED),
    ("rearrangement_depth", REARRANGEMENT_DEPTH_MISMATCH),
    ("lookalike_count", TIER_GATE_FAILED),
)


def structural_violations(cert: SearchStructureCertificate,
                          spec: TierSpec) -> list[Rejection]:
    out: list[Rejection] = []
    if cert.mode not in spec.modes:
        out.append(Rejection(TIER_GATE_FAILED,
                             f"mode {cert.mode!r} not in {spec.name} modes"))
    for name, code in _STRUCTURE_FIELDS:
        bound: Bound = getattr(spec.search, name)
        value = getattr(cert, name)
        if not bound.contains(value):
            out.append(Rejection(code, f"{name}={value} outside "
                                       f"{bound.describe()} for {spec.name}"))
    mode_gate = spec.gates.modes.get(cert.mode)
    if mode_gate is not None and mode_gate.open_required and cert.open_depth < 1:
        out.append(Rejection(OPEN_NOT_REQUIRED,
                             f"mode {cert.mode} requires an OPEN reveal "
                             f"(open_depth={cert.open_depth})"))
    budget = spec.budget.max_planning_steps - spec.budget.search_reserve_steps
    if cert.minimum_required_interactions > budget:
        out.append(Rejection(
            HORIZON_LOWER_BOUND,
            f"lower bound {cert.minimum_required_interactions} > "
            f"{spec.budget.max_planning_steps} - reserve "
            f"{spec.budget.search_reserve_steps}"))
    return out


# ---------------------------------------------------------------------------
# simulator evidence (GPU)
# ---------------------------------------------------------------------------

def simulator_violations(cert: SearchStructureCertificate, spec: TierSpec,
                         use_generation_margin: bool = True) -> list[Rejection]:
    """Gate the measured evidence. Missing measurements are failures: the
    certificate can only be "certified" from positive evidence.

    use_generation_margin=True applies the stricter generator acceptance
    boundary; False checks the published benchmark threshold.
    """
    ev = cert.evidence
    if ev is None:
        return [Rejection(TIER_GATE_FAILED, "no simulator evidence attached", "gpu")]
    out: list[Rejection] = []

    def need(flag: Optional[bool], code: str, what: str) -> bool:
        if flag is not True:
            out.append(Rejection(code, f"{what}: {flag!r}", "gpu"))
            return False
        return True

    if not need(ev.builder_ok, PLACEMENT_FAILED, "builder"):
        return out
    need(ev.placements_verified, PLACEMENT_FAILED, "placement relations")
    need(ev.settle_ok, SETTLE_FAILED, "settling")

    if ev.start_visibility is None:
        out.append(Rejection(START_VISIBLE, "start visibility not measured", "gpu"))
    elif ev.start_visibility > spec.gates.start_visibility_max:
        out.append(Rejection(START_VISIBLE,
                             f"start visibility {ev.start_visibility:.3f} > "
                             f"{spec.gates.start_visibility_max}", "gpu"))

    gate = spec.gates.modes.get(cert.mode)
    if gate is None:
        out.append(Rejection(TIER_GATE_FAILED, f"no gates for mode {cert.mode}", "gpu"))
        return out
    if gate.open_required:
        # OPEN must be a causal reveal: closed => invisible AND ungraspable
        if ev.closed_visibility is None or ev.closed_visibility > 0.0:
            out.append(Rejection(OPEN_NOT_REQUIRED,
                                 f"closed-container visibility "
                                 f"{ev.closed_visibility!r} (must be 0)", "gpu"))
    if ev.pre_reveal_graspable is None:
        out.append(Rejection(TARGET_PREMATURELY_GRASPABLE,
                             "pre-reveal graspability not measured", "gpu"))
    elif ev.pre_reveal_graspable and not gate.pre_reveal_graspable:
        code = OPEN_NOT_REQUIRED if gate.open_required and cert.rearrangement_depth == 0 \
            else TARGET_PREMATURELY_GRASPABLE
        out.append(Rejection(code, "target graspable before the required reveal", "gpu"))

    threshold = (gate.generator_visibility_margin if use_generation_margin
                 else gate.revealed_visibility_min)
    if ev.revealed_visibility is None or ev.revealed_visibility < threshold:
        out.append(Rejection(REVEAL_TOO_WEAK,
                             f"revealed visibility {ev.revealed_visibility!r} < "
                             f"{threshold}", "gpu"))
    need(ev.post_reveal_graspable, TARGET_NOT_GRASPABLE, "post-reveal GRASP")

    if (ev.rearrangement_depth is not None
            and not spec.search.rearrangement_depth.contains(ev.rearrangement_depth)):
        out.append(Rejection(REARRANGEMENT_DEPTH_MISMATCH,
                             f"measured rearrangement depth {ev.rearrangement_depth}", "gpu"))
    if (ev.reachable_candidate_slot_count is not None
            and not spec.search.candidate_slot_count.contains(
                ev.reachable_candidate_slot_count)):
        out.append(Rejection(SEARCH_WIDTH_MISMATCH,
                             f"reachable candidate slots "
                             f"{ev.reachable_candidate_slot_count}", "gpu"))

    need(ev.oracle_solvable, PHYSICAL_UNSOLVABLE, "oracle solvability")
    min_rearr = spec.search.rearrangement_depth.min
    if min_rearr and ev.oracle_solvable:
        if ev.oracle_rearrangement_depth is None:
            out.append(Rejection(TARGET_PREMATURELY_GRASPABLE,
                                 "oracle rearrangement depth not measured", "gpu"))
        elif ev.oracle_rearrangement_depth < min_rearr:
            out.append(Rejection(TARGET_PREMATURELY_GRASPABLE,
                                 f"full-information oracle moves "
                                 f"{ev.oracle_rearrangement_depth} < {min_rearr} objects "
                                 f"(target graspable under its cover)", "gpu"))
    need(ev.replay_success, PHYSICAL_UNSOLVABLE, "certified plan replay")
    if ev.certified_execution_steps is None:
        out.append(Rejection(HORIZON_LOWER_BOUND,
                             "certified execution length not measured", "gpu"))
    elif ev.certified_execution_steps > spec.budget.max_planning_steps:
        out.append(Rejection(HORIZON_LOWER_BOUND,
                             f"certified execution {ev.certified_execution_steps} "
                             f"> {spec.budget.max_planning_steps}", "gpu"))
    return out


def certify(cert: SearchStructureCertificate, spec: TierSpec,
            use_generation_margin: bool = True) -> list[Rejection]:
    """Full GPU gate; on success marks the certificate certified for spec."""
    rejections = structural_violations(cert, spec)
    rejections += simulator_violations(cert, spec, use_generation_margin)
    if rejections:
        cert.status = "rejected"
        cert.tier_certified = None
    else:
        cert.status = "certified"
        cert.tier_certified = spec.tier
    return rejections


def classify(cert: SearchStructureCertificate,
             specs: Iterable[TierSpec]) -> list[str]:
    """Tiers whose structural contract the certificate satisfies (mode
    ignored: classification asks what the STRUCTURE is, not what was
    proposed). The tier contracts are disjoint, so this has <= 1 element."""
    out = []
    for spec in specs:
        violations = [r for r in structural_violations(cert, spec)
                      if not r.detail.startswith("mode ")
                      and r.code != OPEN_NOT_REQUIRED]
        if not violations:
            out.append(spec.tier)
    return out


def histogram(rejections: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejections:
        for code in r.get("codes", [r.get("code")]):
            if code:
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# plan-level CPU gates: eligibility, semantics, geometry, structure
# ---------------------------------------------------------------------------

def cpu_plan_gates(ctx, plan, cert: SearchStructureCertificate) -> list[Rejection]:
    """Everything decidable without a simulator. Schema + determinism are
    checked where the scenario / replan exist (compile.py, plan_tasks.py)."""
    from rummagebench.authoring.tasks.planner import fits, occupancy

    out: list[Rejection] = []
    el = ctx.eligibility
    ss = ctx.scenes.get(plan.scene)
    if ss is None:
        return [Rejection(INVALID_SLOT, f"unknown scene {plan.scene}")]
    slots = ss.by_id()
    per_slot: dict[str, list] = {}
    for o in plan.objects:
        if not el.is_eligible(o.category, o.model):
            out.append(Rejection(INVALID_OBJECT_CATEGORY,
                                 f"{o.entity}: {o.category}/{o.model} not whitelisted"))
            continue
        slot = slots.get(o.slot_id)
        if slot is None:
            out.append(Rejection(INVALID_SLOT, f"{o.entity}: unknown slot {o.slot_id}"))
            continue
        if not ctx.priors.is_approved(o.category, slot.slot_type, slot.room_type):
            out.append(Rejection(INVALID_PRIOR,
                                 f"{o.entity}: ({o.category}, {slot.slot_type}, "
                                 f"{slot.room_type}) not approved"))
        bbox = el.categories[o.category].model(o.model).bbox
        if not fits(bbox, slot):
            out.append(Rejection(GEOMETRY_OVERFLOW, f"{o.entity} does not fit {o.slot_id}"))
        per_slot.setdefault(o.slot_id, []).append(bbox)
    spec = ctx.tier
    for sid, boxes in per_slot.items():
        slot = slots.get(sid)
        if slot is None:
            continue
        cap = (spec.target_slot.max_fill_ratio if sid == plan.target.slot_id else
               spec.other_containers.max_fill_ratio if slot.is_container else
               spec.other_surfaces.max_area_ratio)
        occ = occupancy(boxes, slot)
        if occ > cap + 1e-9:
            out.append(Rejection(GEOMETRY_OVERFLOW, f"{sid} occupancy {occ:.2f} > {cap}"))
    if len({o.entity for o in plan.objects}) != len(plan.objects):
        out.append(Rejection(INVALID_OBJECT_CATEGORY, "duplicate entity names"))
    tslot = slots.get(plan.target.slot_id)
    if tslot is not None and plan.target.slot_id not in cert.candidate_slots:
        out.append(Rejection(SEARCH_WIDTH_MISMATCH,
                             "target slot is not among the certified candidate slots"))
    if sorted(plan.candidate_slots) != sorted(cert.candidate_slots):
        out.append(Rejection(SEARCH_WIDTH_MISMATCH,
                             "planner candidate set disagrees with the re-derived one"))
    out += structural_violations(cert, spec)
    return out
