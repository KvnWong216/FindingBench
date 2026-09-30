#!/usr/bin/env python
"""Pre-flight memory probe for the production acceptance run (run BEFORE the
benchmark when GPUs are shared).

Loads the REAL knife_search_001 workload end to end — scene load, R1Pro
articulation export, URDF validation hook, pinocchio feasibility backend,
dynamic skill grounding, the full scripted_success action sequence — and
reports peak device memory. Gates the benchmark on the prompt requirement:

    our_peak_vram_mib + headroom (default 3072 MiB) <= free VRAM at start

Writes runs/memory_probe.json. Exit code 0 = OK, 3 = insufficient headroom,
1 = workload failure. Always exits via os._exit (Kit teardown segfaults).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def gpu_free_mib(gpu_index: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
         "-i", str(gpu_index)],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def our_gpu_mib() -> int | None:
    """Total CUDA memory reserved by THIS process across all visible GPUs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(os.getpid()):
            return int(parts[1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--headroom-mib", type=int, default=3072)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    free_at_start = gpu_free_mib(args.gpu)
    print(f"[probe] GPU {args.gpu}: {free_at_start} MiB free at start", flush=True)

    report: dict = {
        "gpu_index": args.gpu,
        "free_at_start_mib": free_at_start,
        "headroom_required_mib": args.headroom_mib,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        import torch

        from rummagebench.adapters.python_api import create_session
        from rummagebench.core.types import Action

        t0 = time.time()
        session = create_session(
            REPO_ROOT / "scenarios" / "knife_search_001" / "scenario.yaml",
            run_dir=None,
            seed=0,
            mode="oracle",
        )
        observation = session.reset()
        print(f"[probe] session ready in {time.time()-t0:.0f}s", flush=True)

        # real grounding + execution over the full scripted trajectory
        t0 = time.time()
        for raw in session.scenario.agent.scripted_success:
            result = session.act(Action.from_dict(raw))
            print(
                f"[probe] step {result.planning_step} {result.action['skill']} "
                f"-> {result.failure_reason.value}", flush=True
            )
        report["episode_status"] = session.status().value
        report["trajectory_seconds"] = round(time.time() - t0, 1)

        report["torch_max_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 2**20, 1
        ) if torch.cuda.is_available() else None
        report["process_peak_vram_mib"] = our_gpu_mib()
        report["free_at_end_mib"] = gpu_free_mib(args.gpu)

        peak = report["process_peak_vram_mib"] or report["torch_max_reserved_mib"] or 0
        report["headroom_after_mib"] = free_at_start - peak
        report["passed"] = peak + args.headroom_mib <= free_at_start
    except Exception as e:  # noqa: BLE001 - probe reports, does not crash silently
        import traceback

        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()[-4000:]
        report["passed"] = False

    out_dir = REPO_ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in
                      ("episode_status", "process_peak_vram_mib",
                       "torch_max_reserved_mib", "headroom_after_mib",
                       "passed", "error")}, indent=2), flush=True)
    print(f"[probe] PROBE {'OK' if report.get('passed') else 'FAILED'}", flush=True)
    return 0 if report.get("passed") else 3


if __name__ == "__main__":
    code = main()
    os._exit(code)
