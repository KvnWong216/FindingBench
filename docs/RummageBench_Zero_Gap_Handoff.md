# RummageBench v2 Zero-Gap Collaborator Handoff

> **Status (`2026-11-25`):** `RummageBench v2` (`runs/rummagebench_v2_20261124`) — `1,000` USD/PhysX episodes across `60` layouts (`50` residential + `10` commercial convenience marts) and `6` archetypes (`S1–S6`), stratified **`Core-120`** evaluation split, **TPT-Bench + Qwen3-VL-4B visual/semantic non-collapse verification**, and the **live MCP + Unitree G1 + cuRoboV2 harness** — is frozen and ready for immediate collaborator evaluation & development with **zero information gap**.

---

## 1. Where Every Asset Lives (Exact Single-Source Map)

| Layer | Location | What's Inside | Size |
| :--- | :--- | :--- | :--- |
| **Source Code, Configs & Docs** | GitHub [`KvnWong216/FindingBench`](https://github.com/KvnWong216/FindingBench) (`main`) | MCP harness, Unitree G1 `curobo` bridge, `rummagebench_v2_*` build/audit/vis scripts, `Core-120` & paid-API configs (`eval_v2_core120_api.json`), `sync_v2_assets.py`, `docs/` | Git repo |
| **Research Portal, History & Interactive Kanban** | OpenResearch (`orx`) — live `https://openresearch.sh/p/...` + [`RummageBench.orx`](../RummageBench.orx) | Docs `01`–`06`, interactive HTML dashboard, 18 high-res figures/contact sheets, full experiment tree (`Baseline v2`, `Ablation 1a/1b`, `Scale-Up 1c`), session transcripts | `~36 MB` |
| **Runtime USD/Robot Binary Closure** | Source server: `runs/rummagebench_v2_release_bundle/` (or HF Hub once uploaded) | Exact **4,057-file** texture-inclusive USD closure (`v2` 1,000 cases, 60 layouts, architectural shells, object USDs/textures, `inspect100`, `diversity_report/embeddings.npz`), plus `assets/robots/unitree_g1`, `references/curobo`, `references/RobotAgentBench` | `1.95 GB` compressed (`~4.6 GB` unpacked) |

---

## 2. One-Shot Collaborator Onboarding (Zero Info Gap)

### Step A — Clone Code & Open Research Context
```bash
# 1. Codebase (GitHub)
git clone git@github.com:KvnWong216/FindingBench.git RummageBench
cd RummageBench

# 2. (Optional, recommended) Import OpenResearch workspace with full docs 01-06 + experiment tree
orx project import <openresearch.sh link OR RummageBench.orx> --path ~/RummageBench_collab
cd ~/RummageBench_collab && orx up
```

### Step B — Pull Heavy USD & Robot Binary Assets (`1.95 GB` Exact Closure)
Copy the 4 verified `.tar.zst` archives + `SHA256SUMS` from the workstation (`runs/rummagebench_v2_release_bundle/`) — e.g. via `rsync`/`scp` (or Hugging Face repo if configured):
```bash
conda activate ygbench

# Copy bundle from workstation (or download from HF if hosted)
rsync -avP ubuntu@<workstation-ip>:/home/ubuntu/ygwang/FindingBench/runs/rummagebench_v2_release_bundle/ ./v2_bundle/

# Verify SHA-256, unpack all 4 archives cleanly, and auto-relocate embedded USD/JSON paths to your checkout path
python scripts/sync_v2_assets.py --from-dir ./v2_bundle
```
*(No unused legacy assets needed: `sync_v2_assets.py` checks SHA-256, extracts the exact 4,057 USD/texture files + `unitree_g1` + `references/{curobo,RobotAgentBench}`, and uses `pxr.Sdf.Layer` to rewrite embedded paths cleanly to your local repo root and refresh `scene_sha256`.)*

### Step C — Verify Your Environment & Run Paid LLM Eval on `Core-120`
```bash
# 1. CPU & Protocol Unit Tests (45 tests)
pytest tests/test_rab_*.py tests/unit/test_rab_mcp_lifecycle.py tests/unit/test_policy_adapter.py -q

# 2. Asset Closure Self-Check
python scripts/sync_v2_assets.py --verify-only

# 3. Inspect any scene visually (3D spectator UI)
FINDINGBENCH_RAB_RELEASE=runs/rummagebench_v2_20261124 python scripts/spectator.py --task rb_U283

# 4. Run Paid Frontier VLM/LLM Evaluation on Stratified Core-120 (supports --shard i/N & --resume)
export GEMINI_API_KEY="YOUR_KEY"   # or OpenAI/OpenRouter/Claude compatible gateway in eval_v2_core120_api.json
python scripts/run_rab_mcp_eval.py --config configs/benchmarks/eval_v2_core120_api.json --resume
```

---

## 3. Exact Snapshot of What Was Built & Verified (`2026-11-25`)

- **Dataset Architecture (`runs/rummagebench_v2_20261124`)**:
  - `1,000` episodes (`missing_scene_files: 0`, `inspect100 negative_z_count: 0`) over `60` base layouts (`50` households + `10` convenience stores) and `932 / 1,000` unique `archetype × layout × lighting × palette` domain-randomized scenes.
  - Stratified `Core-120` subset (`configs/benchmarks/rummagebench_v2_core120.json`): exact `20` episodes × `6` archetypes (`S1–S6`).
  - Lived-in realistic clutter: `130.11` visual objects/scene in supermarkets (`124–137`), `106.00` in households (`80–147`), `24.42` graspable dynamic items/scene (`21–32`), `39` curated target categories across `6` cognitive paradigms (`P1–P6`) and `600 in_prior / 400 out_of_prior_misplaced` hides.
- **Physics & MCP Bugfixes Verified**:
  - Fixed rotated furniture (`ID_24`) oblique collider shear deadlock via world-aligned `*_SolidTopGuide` normalization in [`scripts/rab_skills.py`](../scripts/rab_skills.py).
  - Fixed `SM_01..SM_10` commercial shelf units missing `0.01`/Y-up transform (`s_base`), `right_door` invisible guard rails (`Bin*Guard*`), `fridge_interior_shelves` multi-level population, `dark_sink_worktop` wings, and `S6` shelf clutter density (`36–46`).
  - End-to-end live Isaac Sim + Unitree G1 + cuRoboV2 hot-reload MCP smoke verified on both household (`rb_U283`) and supermarket (`rb_P327`) with `has_private_leak=False`.
- **Anti-Collapse Quantitative Proof ([`docs/RummageBench_v2_Diversity_NonCollapse.md`](RummageBench_v2_Diversity_NonCollapse.md))**:
  - Container family entropy `H_norm = 0.9953` (max family share down from `87.2%` in `v1` to `13.6%` in `v2`); kitchen cupboard entropy `0.9887` (max door `1.7%`).
  - Real CPU `Qwen3-VL-4B-Instruct` visual ViT `eRank = 78.80` (`n=100`) / `140.54` (`n=200`), Centered Vendi Score `37.91`, photometric luminance `[0.2084, 0.7091]`.
  - Vocabulary `|V|`: `257 -> 737` (`2.87x`), lexical entropy `5.88 -> 7.21 bits`, `Distinct-1` `+72.6%`, semantic `eRank`: `135.15 -> 164.88`.
- **Open-Research Collaborator Docs (`/home/ubuntu/.local/share/openresearch/files/rummagebench/`)**:
  - `01-project-overview-and-mechanisms.md` · `02-metrics-and-empirical-results.md` · `03-collaborator-sync-and-roadmap.md` · `04-literature-review-and-design-philosophy.md` · `05-dataset-v2-10pct-audit.md` · `06-diversity-and-non-collapse-analysis.md` · `rummagebench-interactive-overview.html`.
