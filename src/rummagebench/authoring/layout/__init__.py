"""BEHAVIOR-1K layout generation (protocol §23-§31).

Layout generation defines the PHYSICAL environment; episode generation
defines the TASK. The same layout_seed + different episode_seed must support
different target placements without regenerating the room.

Three explicit seeds, never global random state (§24):
    layout_seed     furniture model selection + spatial arrangement
    episode_seed    target/distractor placement + initial task state
    appearance_seed lighting/material variation only

Phase order is mandatory (§32): layout -> validation -> reproducibility
tests -> ONLY THEN rendering/texture quality.
"""

from rummagebench.authoring.layout.spec import LayoutConfig, Placement
from rummagebench.authoring.layout.generator import LayoutGenerator
from rummagebench.authoring.layout.validator import LayoutValidator
from rummagebench.authoring.layout.manifest import build_manifest

__all__ = ["LayoutConfig", "Placement", "LayoutGenerator", "LayoutValidator", "build_manifest"]
