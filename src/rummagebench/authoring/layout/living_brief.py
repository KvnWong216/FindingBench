"""Strict model-authored semantic proposals, grounded in native supports.

This is a structured companion to a language description, not an NL parser.
Neither the compiler nor its input needs a task, target, or hidden location.
Physical feasibility and naturalness remain downstream validation tasks.
"""
from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class SurfaceBrief(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    minimum: StrictInt = Field(ge=0, le=6)
    preferred: StrictInt = Field(ge=0, le=6)
    maximum: StrictInt = Field(ge=0, le=6)
    fullness: float = Field(ge=0, le=.5)
    rationale: str = Field(min_length=1, max_length=1000)

    @model_validator(mode='after')
    def ordered(self):
        if not self.minimum <= self.preferred <= self.maximum:
            raise ValueError('BRIEF_COUNT_ORDER')
        return self


class LivingBrief(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: StrictInt = Field(ge=1, le=1)
    author: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=6000)
    total_count: StrictInt = Field(ge=1, le=200)
    surfaces: dict[str, SurfaceBrief]


def compile_brief(raw: dict, context: dict) -> dict:
    brief = LivingBrief.model_validate(raw)
    supports = {o['name'] for o in context['objects'] if o['support']}
    # No silent dropping of an infeasible or unknown requested surface.
    if set(brief.surfaces) != supports:
        raise ValueError('BRIEF_SUPPORT_COVERAGE_MISMATCH')
    if not (sum(s.minimum for s in brief.surfaces.values()) <= brief.total_count
            <= sum(s.maximum for s in brief.surfaces.values())):
        raise ValueError('BRIEF_TOTAL_INFEASIBLE')
    return brief.model_dump()


def allocate_counts(brief: dict, seed: int) -> dict[str, int]:
    """Minimize L1 distance from preferred counts at the exact item budget.

    Dynamic programming over at most 200 items, then seeded uniform sampling
    among optimal allocations. This solves only count allocation, never the
    geometric feasibility or search difficulty problem.
    """
    surfaces = sorted(brief['surfaces'].items())
    total = brief['total_count']
    dp = [{} for _ in range(len(surfaces)+1)]
    dp[-1][0] = (0, 1)  # minimum cost, number of optimal completions
    for i in range(len(surfaces)-1, -1, -1):
        s = surfaces[i][1]
        for n in range(s['minimum'], s['maximum']+1):
            for tail, (cost, ways) in dp[i+1].items():
                size = n+tail
                if size > total: continue
                score = cost+abs(n-s['preferred'])
                old = dp[i].get(size)
                if old is None or score < old[0]: dp[i][size] = (score, ways)
                elif score == old[0]: dp[i][size] = (score, old[1]+ways)
    if total not in dp[0]: raise ValueError('BRIEF_TOTAL_INFEASIBLE')
    rng = random.Random(seed)
    counts = {}; remaining = total
    for i, (name, s) in enumerate(surfaces):
        pick = rng.randrange(dp[i][remaining][1])
        for n in range(s['minimum'], s['maximum']+1):
            suffix = dp[i+1].get(remaining-n)
            if suffix is None or abs(n-s['preferred'])+suffix[0] != dp[i][remaining][0]: continue
            if pick < suffix[1]:
                counts[name] = n; remaining -= n; break
            pick -= suffix[1]
    return counts
