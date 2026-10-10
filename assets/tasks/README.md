# Task factory source assets

Hand-written rules and GPU-probed robot overlays for the tiered task factory
(code in `src/rummagebench/authoring/tasks/`, scripts in `scripts/tasks/`).

| path | content |
|---|---|
| `slot_rules_v1.yaml` | which furniture exposes searchable slots, link classification, landmark search regions |
| `priors/review_v1.yaml` | approved (object, slot class, room type) placement triples |
| `priors/manual_v1.yaml` | hand-approved common-sense triples for non-kitchen rooms (no BDDL evidence) |
| `priors/eligibility_rules_v1.yaml` | object whitelist rules (abilities, size, mass, lookalikes) |
| `tiers/<tier>_v1.yaml` | tier contract: search-structure bounds, gates, instruction templates |
| `embodiment/<scene>__<robot>.yaml` | robot anchors, feasibility and start poses, probed on GPU by `probe_embodiment.py` |

Regenerable files are written to `build/tasks/` (gitignored): scene slots,
mined / reviewed placement priors, the object whitelist, plans, candidates
and certificates. Released episodes are promoted to `assets/benchmark/<tier>/`.

```bash
export PYTHONPATH=src
python scripts/tasks/build_scene_slots.py         # build/tasks/scenes/
python scripts/tasks/measure_drawer_geometry.py --scene <scene>   # GPU, then rerun build_scene_slots.py
python scripts/tasks/mine_priors.py               # build/tasks/priors/placement_v1.yaml
python scripts/tasks/build_eligibility.py         # build/tasks/priors/eligibility_v1.yaml
python scripts/tasks/probe_embodiment.py --scene <scene>          # GPU
python scripts/tasks/plan_tasks.py --scenes <scenes> --count 20
python scripts/tasks/compile_tasks.py --scenes <scenes>
python scripts/tasks/certify_tasks.py --gpus 0 1                  # GPU
python scripts/tasks/split_tasks.py --test-scenes <held-out scenes>
python scripts/tasks/report_tasks.py
python scripts/tasks/promote_tasks.py              # release -> assets/benchmark/<tier>/
```
