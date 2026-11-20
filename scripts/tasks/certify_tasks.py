"""Certify compiled task candidates on GPU.

    # dispatcher: every compiled candidate without a certificate, one fresh
    # simulator process per episode, round-robin over the given GPUs
    PYTHONPATH=src python scripts/tasks/certify_tasks.py --tier easy_v1 --gpus 0 1 3

    # one episode in this process (what the dispatcher runs)
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/tasks/certify_tasks.py \
        --tier easy_v1 --worker fb_0123456789abcdef

Outputs (<build_root>/<tier>/):
  search_certificates/<id>.json  evidence attached, status certified|rejected
  certificates/<id>.json         full record: evidence details, protocol
                                 witness trace, oracle d* certificate, gate
                                 rejections, provenance
  gpu_rejections.jsonl           rebuilt by the dispatcher after each run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import ASSETS, REPO, host_config

from rummagebench.authoring.tasks import gates
from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay, overlay_path
from rummagebench.authoring.tasks.plan import TaskPlan, git_state, sha256_path
from rummagebench.authoring.tasks.search_certificate import SearchStructureCertificate
from rummagebench.authoring.tasks.slots import SceneSlots
from rummagebench.authoring.tasks.tiers import tier_spec_path
from rummagebench.authoring.tasks.tiers.base import load_tier_spec


# simulator/host failures that say nothing about the episode (env_factory
# NOTE_ON_SLUDGE_SEGFAULT, PhysX teleport corruption, CUDA/Warp errors):
# recorded as INVALID_RUN and retried, never as a task rejection
INFRA_ERROR_MARKERS = ("sludge system", "CUDA", "Warp", "Physics state is invalid",
                       "Robot physics pose", "nonfinite", "Segmentation",
                       "No space left on device")
# Kit / OmniGibson write ~340 MB of USD scratch per process into TMPDIR and
# never clean it (workers leave via os._exit): 2000 runs filled the root
# filesystem (2026-10-06). Workers use a per-task dir on the data disk.
WORKER_TMP_ROOT = REPO / "runs" / "tmp_workers"


def is_infra_error(error: str | None) -> bool:
    return bool(error) and any(m in error for m in INFRA_ERROR_MARKERS)


def worker(args, out: Path) -> int:
    import shutil
    import tempfile

    tmp = WORKER_TMP_ROOT / args.worker
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
    tempfile.tempdir = str(tmp)
    try:
        return _worker(args, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _exit_clean(args) -> None:
    import shutil

    sys.stdout.flush()
    shutil.rmtree(WORKER_TMP_ROOT / args.worker, ignore_errors=True)
    os._exit(0)


def _worker(args, out: Path) -> int:
    from rummagebench.authoring.tasks.certify import CERTIFIER_VERSION, TaskCertifier
    from rummagebench.authoring.tasks.room_map import RoomMap

    host = host_config(args.host_config)
    tid = args.worker
    plan = TaskPlan.load(out / "plans" / f"{tid}.json")
    scen_path = out / "candidates" / f"{tid}.yaml"
    spec = load_tier_spec(tier_spec_path(args.tier))
    ss = SceneSlots.load(host["build_root"] / "scenes" / f"{plan.scene}.yaml")
    op = overlay_path(ASSETS, plan.scene, plan.robot_id)
    planned_with = (plan.provenance.get("inputs") or {}).get("overlay_hash")
    if sha256_path(op) != planned_with:
        raise SystemExit(f"{tid}: overlay changed since planning — replan + recompile first")
    ov = SlotEmbodimentOverlay.load(op)
    rm = RoomMap.load(host["behavior_assets_root"] / "scenes" / plan.scene,
                      host["behavior_assets_root"] / "metadata" / "room_categories.txt")
    scert_path = out / "search_certificates" / f"{tid}.json"
    full_path = out / "certificates" / f"{tid}.json"
    structural = json.loads(scert_path.read_text())
    if full_path.exists():  # re-certification starts from the structural record
        structural = json.loads(full_path.read_text())["structural"]
    cert = SearchStructureCertificate.from_dict(structural)

    os.chdir(REPO)
    log = lambda m: print(f"{dt.datetime.now().isoformat(timespec='seconds')} [certify] {m}",
                          flush=True)
    log(f"{tid} {plan.scene}/{plan.room} {plan.mode} target={plan.target.category} "
        f"slot={plan.target.slot_id}")
    reveal_min = spec.gates.modes[plan.mode].generator_visibility_margin
    res = TaskCertifier(ss, ov, rm, log, reveal_min=reveal_min).certify(plan, cert, scen_path)
    if is_infra_error(res.error):
        full_path.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        if full_path.exists():
            prev = json.loads(full_path.read_text())
            attempts = prev.get("invalid_run_attempts", [])
        full_path.write_text(json.dumps({
            "task_id": tid, "tier": args.tier, "status": "invalid_run",
            "structural": structural, "error": res.error, "details": res.details,
            "invalid_run_attempts": attempts + [res.error[:300]]}, indent=1, sort_keys=True))
        log(f"{tid}: INVALID_RUN (infrastructure): {res.error[:200]}")
        sys.stdout.flush()
        _exit_clean(args)
    va = res.details.get("view_anchor")
    if va and not va.get("rejected"):
        # the oracle certified WITH the witness's viewpoint anchor: the
        # candidate file must carry it so a replay reproduces the certificate
        import yaml

        from rummagebench.authoring.tasks.compile import with_view_anchors
        text = scen_path.read_text(encoding="utf-8")
        header = "".join(l for l in text.splitlines(True) if l.startswith("#"))
        doc = with_view_anchors(yaml.safe_load(text), {va["name"]: va})
        scen_path.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=100),
                             encoding="utf-8")
    cert.attach_evidence(res.evidence)
    rejections = gates.certify(cert, spec)
    if res.error:
        rejections.append(gates.Rejection(gates.TIER_GATE_FAILED,
                                          f"certifier error: {res.error}", "gpu"))
        cert.status, cert.tier_certified = "rejected", None
    record = {
        "task_id": tid, "tier": args.tier, "status": cert.status,
        "tier_certified": cert.tier_certified,
        "rejections": [r.to_dict() for r in rejections],
        "structural": structural, "search_certificate": cert.to_dict(),
        "details": res.details, "witness_trace": res.witness_trace,
        "oracle_certificate": res.oracle_certificate, "error": res.error,
        "provenance": {**plan.provenance, "certifier": {
            "version": CERTIFIER_VERSION, **git_state(REPO),
            "scenario_sha256": sha256_path(scen_path), "overlay_sha256": sha256_path(op),
            "date": dt.date.today().isoformat()}},
    }
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(json.dumps(record, indent=1, sort_keys=True))
    scert_path.write_text(json.dumps(cert.to_dict(), indent=1, sort_keys=True))
    log(f"{tid}: {cert.status} {[r.code for r in rejections]} "
        f"steps={res.evidence.certified_execution_steps} d*={res.evidence.oracle_depth} "
        f"vis start/closed/revealed={res.evidence.start_visibility}/"
        f"{res.evidence.closed_visibility}/{res.evidence.revealed_visibility} "
        f"[{res.details.get('seconds')}s]")
    sys.stdout.flush()
    _exit_clean(args)  # Kit teardown may segfault after the record is written


def dispatch(args, out: Path) -> int:
    ids = args.tasks or sorted(p.stem for p in (out / "candidates").glob("fb_*.yaml"))

    def needs_run(t):
        path = out / "certificates" / f"{t}.json"
        if args.force or not path.exists():
            return True
        rec = json.loads(path.read_text())
        return (rec.get("status") == "invalid_run"
                and len(rec.get("invalid_run_attempts", [])) <= args.max_retries)
    todo = [t for t in ids if needs_run(t)]
    print(f"{len(todo)}/{len(ids)} candidates to certify on GPUs {args.gpus}", flush=True)
    logs = out / "certify_logs"
    logs.mkdir(parents=True, exist_ok=True)
    free = list(args.gpus) * args.per_gpu
    no_record: dict[str, int] = {}  # timeouts / crashes leave no record
    running: list[tuple[subprocess.Popen, str, str, float]] = []
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    if args.threads:
        # OpenMP / BLAS pools default to one thread per core (~130 per worker)
        # and busy-wait; the worker's IK / collision math is single-threaded
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            env[var] = str(args.threads)
    while todo or running:
        while todo and free:
            tid, gpu = todo.pop(0), free.pop(0)
            cmd = [sys.executable, __file__, "--tier", args.tier, "--worker", tid]
            if args.host_config:
                cmd += ["--host-config", args.host_config]
            f = (logs / f"{tid}.log").open("w")
            p = subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                 stdout=f, stderr=subprocess.STDOUT)
            running.append((p, tid, gpu, time.time()))
        time.sleep(5)
        for item in list(running):
            p, tid, gpu, started = item
            if p.poll() is None:
                if time.time() - started > args.timeout:
                    p.kill()  # reaped on the next pass; no record -> NO_RECORD
                continue
            running.remove(item)
            free.append(gpu)
            done = (out / "certificates" / f"{tid}.json")
            status = json.loads(done.read_text())["status"] if done.exists() else "NO_RECORD"
            print(f"{dt.datetime.now().isoformat(timespec='seconds')} {tid} gpu{gpu} "
                  f"exit={p.returncode} {status}", flush=True)
            if status == "NO_RECORD":
                no_record[tid] = no_record.get(tid, 0) + 1
                if no_record[tid] <= args.max_retries:
                    todo.append(tid)  # bounded by --max-retries
            elif status == "invalid_run" and needs_run(tid):
                todo.append(tid)  # bounded by --max-retries (record attempts)
    # rebuild the GPU rejection log from the certificate records
    with (out / "gpu_rejections.jsonl").open("w") as rej:
        for path in sorted((out / "certificates").glob("fb_*.json")):
            r = json.loads(path.read_text())
            if r["status"] not in ("certified", "invalid_run"):
                rej.write(json.dumps({"stage": "gpu", "task_id": r["task_id"],
                                      "codes": sorted({x["code"] for x in r["rejections"]}),
                                      "rejections": r["rejections"]}) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="easy_v1")
    ap.add_argument("--worker", help="certify this task id in-process")
    ap.add_argument("--tasks", nargs="*", help="dispatcher: only these task ids")
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent workers per GPU")
    ap.add_argument("--timeout", type=float, default=5400.0)
    ap.add_argument("--threads", type=int, default=4,
                    help="OpenMP/BLAS threads per worker (0: library default)")
    ap.add_argument("--force", action="store_true", help="re-certify existing records")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="re-runs of an INVALID_RUN (infrastructure) episode")
    ap.add_argument("--host-config")
    args = ap.parse_args(argv)
    out = host_config(args.host_config)["build_root"] / args.tier
    return worker(args, out) if args.worker else dispatch(args, out)


if __name__ == "__main__":
    raise SystemExit(main())
