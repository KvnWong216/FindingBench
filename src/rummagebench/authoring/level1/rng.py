"""§41: deterministic independent randomness streams.

Every stochastic decision derives from the environment seed via numpy
SeedSequence. Changing one stream (e.g. appearance) can never perturb
another (geometry/object/packing/physics), and worker completion order can
never change sampling (revision §15): stream generators are created from the
seed alone, never from wall-clock or filesystem state.
"""
from __future__ import annotations

import numpy as np
import zlib

STREAMS = ("geometry", "object", "packing", "appearance", "physics")


def derive_streams(environment_seed: int) -> Dict[str, np.random.Generator]:
    """One independent Generator per stream, keyed by stream name."""
    children = np.random.SeedSequence(int(environment_seed)).spawn(len(STREAMS))
    return {name: np.random.default_rng(child)
            for name, child in zip(STREAMS, children)}


def stage_seed(environment_seed: int, candidate_id: int, stage_id: str,
               retry_id: int = 0) -> int:
    """Revision §15: randomness derives from (dataset seed, candidate, stage,
    retry) so parallel workers reproduce identical proposals. The stage name
    is folded in through crc32 (stable across processes/versions)."""
    ss = np.random.SeedSequence(
        [int(environment_seed), int(candidate_id),
         zlib.crc32(stage_id.encode("utf-8")) & 0x7FFFFFFF, int(retry_id)])
    return int(ss.generate_state(1)[0])
