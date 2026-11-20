"""Run compiled vla_difficulty_v1 variants sequentially (one simulator process
each, via scripts/run_vla_variant.py) and/or aggregate their reports.

    python scripts/run_vla_batch.py --variants T01_knife_fetch__L1_S1 T01_knife_fetch__L2_S2
    python scripts/run_vla_batch.py --all --skip-done
    python scripts/run_vla_batch.py --summarize-only

Aggregate: runs/vla_difficulty_v1/summary.json + summary.md, with the three
levels kept apart (Task Initialization / Scripted Execution / VLA = not run).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/vla_difficulty_v1"
RUNS = ROOT / "runs/vla_difficulty_v1"


def run_one(variant: str, timeout_s: int) -> int:
    log = RUNS / "logs" / f"{variant}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts/run_vla_variant.py"),
           str(BUILD / variant / "scenario.yaml")]
    t = time.time()
    with log.open("w") as f:
        try:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout_s, env=os.environ.copy()).returncode
        except subprocess.TimeoutExpired:
            code = -9
            f.write(f"\n[run_vla_batch] TIMEOUT after {timeout_s}s\n")
    print(f"{variant}: exit={code} ({time.time() - t:.0f}s)", flush=True)
    return code


def _blocked_tasks() -> set[str]:
    try:
        import yaml
        index = yaml.safe_load((ROOT / "assets/tasks/vla_difficulty_v1/variants.yaml").read_text())
        return set(index.get("blocked_tasks") or [])
    except Exception:
        return set()


def summarize() -> dict:
    blocked = _blocked_tasks()
    rows = []
    for vdir in sorted(BUILD.glob("*__L*_S*")):
        meta = json.loads((vdir / "variant.json").read_text())
        rep_path = RUNS / vdir.name / "report.json"
        row = {"variant": vdir.name, "task": meta["task_id"], "L": meta["language_level"],
               "S": meta["spatial_level"], "run": rep_path.exists(),
               "blocked": meta["task_id"] in blocked}
        if rep_path.exists():
            r = json.loads(rep_path.read_text())
            init = r.get("task_initialization", {})
            se = r.get("scripted_execution") or {}
            row.update(
                init_ok=init.get("ok"), s_gates_pass=init.get("s_gates_pass"),
                s_gates=init.get("s_gates"),
                start_px=f"{init.get('start_panorama_target_px')}/{init.get('start_best_view_target_px')}",
                start_room=init.get("start_room"), target_room=init.get("target_room"),
                null_baseline_ok=(r.get("null_baseline") or {}).get("ok"),
                scripted_status=se.get("status"), scripted_steps=se.get("planning_steps_used"),
                budget=se.get("max_planning_steps"), oracle_failure=se.get("oracle_failure"),
                px_after_open=se.get("target_px_after_open"),
                error=r.get("error"), report=str(rep_path.relative_to(ROOT)))
        rows.append(row)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "vla_task_success": "not run (no model adapter)",
           "variants": rows}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    md = ["# vla_difficulty_v1 simulation status", "",
          f"Generated {out['generated']} by `scripts/run_vla_batch.py --summarize-only`.",
          "Three levels are reported separately; **VLA task success: not run (no model adapter)**.", "",
          "| variant | init | S gates | start px (pano/view) | rooms (start→target) | null REPORT_DONE rejected | scripted | steps/budget | px after OPEN | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda v: "—" if v is None else ("✅" if v else "❌")
    for r in rows:
        if r.get("blocked"):
            last = f" (last run: {r.get('scripted_status')}, {str(r.get('error') or r.get('oracle_failure') or '')[:60]})" \
                if r["run"] else ""
            md.append(f"| {r['variant']} | BLOCKED (see base task verification_status){last} | | | | | | | | |")
            continue
        if not r["run"]:
            md.append(f"| {r['variant']} | not run | | | | | | | | |")
            continue
        note = r.get("error") or r.get("oracle_failure") or ""
        if r.get("s_gates_pass") is False:
            note = (note + " gates: " + ", ".join(k for k, v in (r.get("s_gates") or {}).items() if not v)).strip()
        md.append(
            f"| {r['variant']} | {yn(r.get('init_ok'))} | {yn(r.get('s_gates_pass'))} | {r.get('start_px')} | "
            f"{r.get('start_room')}→{r.get('target_room')} | {yn(r.get('null_baseline_ok'))} | "
            f"{r.get('scripted_status') or '—'} | {r.get('scripted_steps')}/{r.get('budget')} | "
            f"{'' if r.get('px_after_open') is None else r.get('px_after_open')} | {str(note)[:120]} |")
    (RUNS / "summary.md").write_text("\n".join(md) + "\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--skip-done", action="store_true", help="skip variants with a report.json")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--timeout", type=int, default=2400)
    a = p.parse_args()
    if not a.summarize_only:
        todo = a.variants or ([d.name for d in sorted(BUILD.glob("*__L*_S*"))] if a.all else [])
        for v in todo:
            if a.skip_done and (RUNS / v / "report.json").exists():
                continue
            run_one(v, a.timeout)
    s = summarize()
    print(f"summary: {sum(r['run'] for r in s['variants'])}/{len(s['variants'])} variants have reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
