"""Batch-certify vla_difficulty_v2 physical configurations, N in parallel.

One run per (base task, S) — the L2 variant's scenario (L1/L3 only change the
instruction; compile_variants.py verified this). Each run is its own
subprocess (scripts/v2/run_variant.py) with a wall-clock timeout; existing
reports are skipped unless --rerun.

    python scripts/v2/run_batch.py --jobs 3                # everything pending
    python scripts/v2/run_batch.py --tasks B001 B002 --rerun
    python scripts/v2/run_batch.py --summarize-only

Summary: runs/vla_difficulty_v2/summary.{json,md}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build/vla_difficulty_v2"
RUNS = ROOT / "runs/vla_difficulty_v2"


def physical_configs(manifest: dict) -> list[dict]:
    return [v for v in manifest["variants"] if v["language_level"] == "L2"]


def run_one(v: dict, timeout: int) -> subprocess.Popen:
    out = RUNS / v["variant_id"]
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("w")
    return subprocess.Popen([sys.executable, str(ROOT / "scripts/v2/run_variant.py"), str(ROOT / v["scenario"]),
                             "--out-root", str(RUNS)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)


def summarize(manifest: dict) -> dict:
    by_phys: dict[str, dict] = {}
    for v in physical_configs(manifest):
        rp = RUNS / v["variant_id"] / "report.json"
        r = json.loads(rp.read_text()) if rp.exists() else None
        key = (v["task_id"], v["spatial_level"])
        by_phys["|".join(key)] = {
            "variant_id": v["variant_id"], "status": "pending" if r is None else
            ("accepted" if r.get("accepted") else "rejected"),
            "reason": None if r is None or r.get("accepted") else _reason(r),
            "oracle_steps": (r or {}).get("scripted_execution", {}).get("planning_steps_used")}
    cells: Counter = Counter()
    tasks: dict[str, int] = {}
    for v in manifest["variants"]:
        st = by_phys["|".join((v["task_id"], v["spatial_level"]))]["status"]
        if st == "accepted":
            cells[f"{v['language_level']}_{v['spatial_level']}"] += 1
    for k, s in by_phys.items():
        t = k.split("|")[0]
        tasks[t] = tasks.get(t, 0) + (s["status"] == "accepted")
    full = sorted(t for t, n in tasks.items() if n == 3)
    summary = {"physical_configs": len(by_phys),
               "status": dict(Counter(s["status"] for s in by_phys.values())),
               "reasons": dict(Counter(s["reason"] for s in by_phys.values() if s["reason"])),
               "base_tasks_all_three_S_accepted": len(full),
               "accepted_variants": sum(cells.values()), "cells": dict(sorted(cells.items())),
               "configs": by_phys}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# vla_difficulty_v2 certification summary", "",
             f"physical configs: {summary['physical_configs']}  status: {summary['status']}",
             f"base tasks with S1+S2+S3 accepted: {len(full)} -> {3 * len(full) * 3} variants usable as full 9-cell sets",
             f"accepted variants (any): {summary['accepted_variants']}", "", "| cell | accepted |", "|---|---|"]
    lines += [f"| {k} | {n} |" for k, n in sorted(cells.items())]
    lines += ["", "rejection reasons:", *[f"- {k}: {n}" for k, n in summary["reasons"].items()]]
    (RUNS / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def _reason(r: dict) -> str:
    if r.get("error"):
        return "error: " + r["error"].split(":")[0]
    init = r.get("task_initialization", {})
    if not init.get("ok"):
        return "init"
    gates = init.get("s_gates") or {}
    failed = [g for g, ok in gates.items() if not ok]
    if failed:
        return "gate: " + ",".join(failed)
    if not r.get("null_baseline", {}).get("ok"):
        return "null_baseline"
    se = r.get("scripted_execution", {})
    if not se.get("success"):
        return "oracle: " + str(se.get("oracle_failure") or se.get("status"))[:60]
    return "oracle steps > budget/2"


def main() -> int:
    global RUNS, BUILD
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument("--tasks", nargs="*", default=[])
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--build-dir", type=Path, default=BUILD,
                   help="compiled variants + manifest.json (build/vla_difficulty_v2_moveto for MOVE_TO)")
    p.add_argument("--runs-dir", type=Path, default=RUNS,
                   help="results directory (2026-10-26: runs/vla_difficulty_v2_moveto for the MOVE_TO protocol)")
    p.add_argument("--configs-file", type=Path, default=None,
                   help="run only the physical configs listed (one 'TASK|S' per line)")
    p.add_argument("--only-accepted-in", type=Path, default=None,
                   help="run only physical configs accepted in this earlier results directory")
    a = p.parse_args()
    RUNS, BUILD = a.runs_dir, a.build_dir
    manifest = json.loads((BUILD / "manifest.json").read_text())
    if a.configs_file is not None:
        keep = {l.strip() for l in a.configs_file.read_text().splitlines() if l.strip()}
        manifest = {**manifest, "variants": [v for v in manifest["variants"]
                                             if f"{v['task_id']}|{v['spatial_level']}" in keep]}
    if a.only_accepted_in is not None:
        prev = json.loads((a.only_accepted_in / "summary.json").read_text())["configs"]
        keep = {k for k, c in prev.items() if c["status"] == "accepted"}
        manifest = {**manifest, "variants": [v for v in manifest["variants"]
                                             if f"{v['task_id']}|{v['spatial_level']}" in keep]}
    if not a.summarize_only:
        todo = [v for v in physical_configs(manifest)
                if (not a.tasks or v["task_id"] in a.tasks)
                and (a.rerun or not (RUNS / v["variant_id"] / "report.json").exists())]
        print(f"{len(todo)} physical configs to run, {a.jobs} in parallel", flush=True)
        running: list[tuple[subprocess.Popen, dict, float]] = []
        while todo or running:
            while todo and len(running) < a.jobs:
                v = todo.pop(0)
                running.append((run_one(v, a.timeout), v, time.time()))
                if a.jobs > 1:
                    time.sleep(20)   # Stagger concurrent starts; serial jobs need no delay.
            time.sleep(5)
            for item in list(running):
                proc, v, t = item
                if proc.poll() is None and time.time() - t > a.timeout:
                    proc.kill()
                    proc.wait()
                    rp = RUNS / v["variant_id"] / "report.json"
                    if not rp.exists():
                        rp.write_text(json.dumps({"variant": v, "accepted": False,
                                                  "error": f"Timeout: > {a.timeout} s"}) + "\n")
                if proc.poll() is not None:
                    running.remove(item)
                    print(f"done {v['variant_id']} rc={proc.returncode} {time.time() - t:.0f}s", flush=True)
                    summarize(manifest)
    s = summarize(manifest)
    print(json.dumps({k: s[k] for k in ("physical_configs", "status", "reasons",
                                        "base_tasks_all_three_S_accepted", "cells")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
