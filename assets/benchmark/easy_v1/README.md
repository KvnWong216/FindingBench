# FindingBench easy_v1 (release)

220 certified easy episodes for the R1Pro, frozen from the task factory by
`scripts/tasks/promote_tasks.py`. Design and certification gates:
`TASK_TIERS_PLAN.md` §6–8, §41.

| | episodes | in_container | on_surface | scenes |
|---|---|---|---|---|
| dev | 155 | 102 | 53 | Pomaria_1 43, Wainscott_0 41, Merom_0 38, Ihlen_1 28, Ihlen_0 5 |
| test | 65 | 38 | 27 | Beechwood_0 62, Pomaria_2 3 (held-out scenes) |

| path | content |
|---|---|
| `scenarios/<id>.yaml` | standard `ScenarioSpec`; load with `rummagebench.core.scenario.load_scenario` |
| `split.json` | `dev` / `test` id lists + generalization statistics |
| `manifest.json` | per episode: split, instruction, mode, scene / room / room type, target, search-structure fields, oracle depth `d*`, certifier witness length, scenario hashes |
| `../../robots/r1pro/` | robot URDF + convex collision meshes the episodes were certified with (`urdf_path` in every scenario) |

## Running an agent

Every episode uses the standard Final Agent Protocol (README): RGB only, the
8 skills, `REPORT_DONE` to finish. The agent sees only the `instruction`
field. Everything else in the manifest (target, slot, mode) is for scoring
and analysis and must not be given to the agent.

```bash
export PYTHONPATH=src
# human sanity check (same information boundary as the agent)
python scripts/serve_play.py --scenario assets/benchmark/easy_v1/scenarios/<id>.yaml
# MCP (reset_episode / observe / act / episode_status)
python -m rummagebench.cli serve-mcp
```

Termination (in each scenario): 16 planning steps; **grasping a wrong object
ends the episode as a failure** (`fail_on_wrong_grasp: true`, the easy tier
has no rearrangement); an UNSAFE action fails; success = `REPORT_DONE` while
holding the target.

Report `easy/all`, `easy/in_container` and `easy/on_surface` separately
(§8). Use `mode` from the manifest.

## Provenance

Each scenario is byte-identical to the certified build candidate except for
the header comment and `kinematics.urdf_path` (which pointed into the gitignored
`build/`). `certified_scenario_sha256` in the manifest is the hash the
certificate recorded, and `scenario_sha256` is the hash of the released file. The
URDF hash equals the certified `urdf_hash`. Results need the Core
feasibility fixes in `TASK_TIERS_PLAN.md` appendix A (collision scale and
convex hulls A.6, GRASP approach check F5 and OPEN base footprint F8 A.8,
self-collision artefact pairs F9 A.8.2). Older checkouts give different
GRASP/OPEN verdicts.
