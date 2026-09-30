# CPU repair validation — 2026-09-30

Base: `88f099e4d1b0bf95ec4b9da2356ac9d2bd2070d9`.
This record covers the repair patch accompanying this file, not earlier
committed `build/` reports. Source hashes were stable during the final suite.

## Executed

- `python -m pytest -m 'not sim' -q -ra`: **230 passed, 13 deselected**, 69.09 s
- `python -m compileall -q src scripts`: passed
- `git diff --check`: passed
- `python -m pip check`: no broken requirements
- Editable `.[test,kinematics]` dependency resolution/install: passed
- Layout seeds 0–19: each produced 3 storage units, 1 surface and 3 supported
  clutter items, no rejected requests, structural validity true
- Missing loaded-scene checks: all seven explicitly `not_run`; simulator
  acceptance false for every generated candidate
- Explicit false support callback: invoked for all seven placements, failed
  acceptance as required
- Candidate CLI: exit 0; `--require-sim-accepted`: exit 1; same manifest hash

Environment: Python 3.12.14, NumPy 1.26.4, pin 2.7.0, hpp-fcl 2.4.4,
cmeel-boost 1.83.0, eigenpy 3.5.1, Pydantic 2.13.5, PyYAML 6.0.3,
pytest 9.1.1, Pillow 12.3.0. Pinocchio/HPP-FCL compatibility includes real
URDF IK and collision tests, not just import stubs.

## Not executed / not implemented

- No OmniGibson, torch or pxr in this environment; the 13 deselected simulator
  tests are **not passes**. No real-camera calibration, held-object restoration
  or end-to-end AGENT success is claimed
- The shipped scenario retains its historical RGB-only configuration because
  its host was documented as crashing with renderer annotators. Strict AGENT
  reset now rejects that configuration. A stable host must supply synchronized
  RGB, depth and renderer instance segmentation; physics-ray fallback remains
  evaluator diagnostic only
- Generated-layout-to-simulator construction, installed dataset model
  resolution and physical layout acceptance remain pending
- Dedicated AGENT evaluation runner and SR/ST aggregation remain pending;
  legacy CLI evaluation/certification is explicitly ORACLE

See `EXPERIMENT_PLAN.md` for the next gates. Rendering-quality work remains
deferred until simulator layout acceptance.
