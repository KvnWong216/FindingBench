"""FindingBench Level-1 data factory.

Local interactive search: floor + one verified support (+ optional open-top
container) + ~10 ordinary BEHAVIOR objects + target, instantiated per real
robot embodiment. Occupancy-guided physical scene compilation:

    task grammar -> BEHAVIOR asset sampling -> occupancy-guided packing ->
    continuous pose lifting -> real mesh instantiation -> PhysX relaxation ->
    exact physical + visual certification -> immutable environment ->
    per-robot episode.

Voxel/occupancy validity is a PROPOSAL representation and never substitutes
for simulator certification. GPU-dependent stages run inside the certified
scripts; this package stays importable on CPU for tests and planning.
"""
