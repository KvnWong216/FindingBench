"""Tiered task factory (easy / medium).

Authoring-layer only: every episode compiles into the frozen ScenarioSpec;
nothing here changes Core (skills, grounding, feasibility, schema).

    SceneSlots + SlotEmbodimentOverlay + PlacementPriors + Eligibility
    + TierSpec -> Planner -> TaskPlan -> Compiler -> ScenarioSpec
    -> CPU gates -> GPU certification -> SearchStructureCertificate

Tier is a certification result, not a label: the planner proposes a tier,
the SearchStructureCertificate proves it (gates.py).
"""
