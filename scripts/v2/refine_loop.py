"""Replace failed objects and re-certify, round by round (after run_batch.py).

A base task counts only when S1, S2 and S3 are all accepted. Failures are
split by cause (runs/vla_difficulty_v2/<id>__L2_<S>/report.json):

  object-level  S1 RGB contrast, placement relation not holding, GRASP /
                PLACE infeasible for this object, object too large so it is
                visible from the start in S2 / S3
  layout-level  everything else (OPEN infeasible, route UNSAFE, errors) ->
                the base task is dropped, never patched

Object-level failures keep the layout and the official sub-goal: a new model
of one of that sub-goal's object categories is chosen (failed models are
excluded for good), the instruction nouns are swapped, the task is
recompiled and its three physical configurations are re-run.

    python scripts/v2/refine_loop.py --rounds 4 --jobs 3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v2"))
from generate_tasks import noun_of  # noqa: E402

DESIGN = ROOT / "assets/tasks/vla_difficulty_v2"
RUNS = ROOT / "runs/vla_difficulty_v2"
EXCLUDE = DESIGN / "failed_objects.txt"
DROPPED = DESIGN / "dropped_tasks.json"
OBJECT_LEVEL = ("gate: rgb_visible_at_start", "init", "gate: invisible_at_start",
                "feasible GRASP(target_obj)", "feasible PLACE(")


def reasons(task_id: str) -> dict[str, str | None]:
    sys.path.insert(0, str(ROOT / "scripts/v2"))
    from run_batch import _reason
    out = {}
    for S in ("S1", "S2", "S3"):
        rp = RUNS / f"{task_id}__L2_{S}" / "report.json"
        if not rp.exists():
            out[S] = "pending"
            continue
        r = json.loads(rp.read_text())
        out[S] = None if r.get("accepted") else _reason(r)
    return out


def pool() -> list[dict]:
    models = json.loads((ROOT / "build/vla_difficulty_v2/object_pool_official.json").read_text())["models"]
    return [m for m in models if m["accepted"] and max(m["bbox_size"]) <= 0.16 and m["bbox_size"][2] <= 0.16
            and min(m["bbox_size"]) >= 0.05 and (m.get("colourfulness") or 0) >= 40]


def official_categories(task: dict) -> list[str]:
    pairs = json.loads((ROOT / "build/vla_difficulty_v2/official_pairs.json").read_text())["pairs"]
    sub = task["source"]["official_subgoal"]
    for pr in pairs:
        if f"({pr['relation']} {pr['object_synset']} {pr['receptacle_synset']})" == sub:
            return pr["object_categories"]
    return [task["target"]["category"]]


def swap_object(task: dict, excluded: set, natives: set) -> bool:
    cats = set(official_categories(task))
    cands = [m for m in pool() if m["category"] in cats and (m["category"], m["model"]) not in excluded
             and m["category"] not in natives]
    if not cands:
        return False
    cands.sort(key=lambda m: (m["category"] == task["target"]["category"], -(m.get("colourfulness") or 0)))
    m = cands[0]
    old_noun, new_noun = task["target"]["noun"], noun_of(m["category"])

    def sub(text):
        return re.sub(rf"\bthe {re.escape(old_noun)}\b", f"the {new_noun}", text)

    lang = task["language"]
    lang["L1"] = {S: sub(t) for S, t in lang["L1"].items()}
    lang["L2"], lang["L3"] = sub(lang["L2"]), sub(lang["L3"])
    task["target"].update(category=m["category"], model=m["model"], noun=new_noun,
                          colour_estimate=m.get("colour"))
    task["source"]["object"] = f"BEHAVIOR-1K {m['category']}/{m['model']}"
    task["attempt"] = int(task.get("attempt", 1)) + 1
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--jobs", type=int, default=3)
    a = p.parse_args()
    py = sys.executable
    excluded = {tuple(x.split("/")) for x in EXCLUDE.read_text().split()} if EXCLUDE.exists() else set()
    dropped = json.loads(DROPPED.read_text()) if DROPPED.exists() else {}
    for rnd in range(1, a.rounds + 1):
        replaced = []
        for path in sorted((DESIGN / "base_tasks").glob("*.yaml")):
            task = yaml.safe_load(path.read_text())
            tid = task["task_id"]
            if tid in dropped:
                continue
            rs = reasons(tid)
            if "pending" in rs.values() or not any(rs.values()):
                continue
            failed = [r for r in rs.values() if r]
            if not all(any(k in r for k in OBJECT_LEVEL) for r in failed):
                dropped[tid] = {"reasons": rs, "kind": "layout-level"}
                continue
            failed_model = (task["target"]["category"], task["target"]["model"])
            excluded.add(failed_model)
            survey = json.loads((ROOT / f"runs/vla_difficulty_v2/_survey/{task['scene']}.json").read_text())
            natives = {e["category"] for e in survey["entities"].values() if not e.get("fixed_base")}
            if int(task.get("attempt", 1)) >= 4 or not swap_object(task, excluded, natives):
                dropped[tid] = {"reasons": rs, "kind": "objects exhausted"}
                continue
            task.setdefault("history", []).append({"failed_model": "/".join(failed_model), "reasons": rs})
            path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True))
            for S in ("S1", "S2", "S3"):
                subprocess.run(["rm", "-rf", str(RUNS / f"{tid}__L2_{S}")], check=True)
            replaced.append(tid)
        EXCLUDE.write_text("\n".join(sorted(f"{c}/{m}" for c, m in excluded)) + "\n")
        DROPPED.write_text(json.dumps(dropped, indent=1) + "\n")
        print(f"round {rnd}: {len(replaced)} tasks get a new object, {len(dropped)} dropped", flush=True)
        if not replaced:
            break
        subprocess.run([py, str(ROOT / "scripts/v2/compile_variants.py")], check=True, cwd=ROOT)
        subprocess.run([py, str(ROOT / "scripts/v2/run_batch.py"), "--jobs", str(a.jobs), "--tasks", *replaced],
                       check=True, cwd=ROOT)
    subprocess.run([py, str(ROOT / "scripts/v2/run_batch.py"), "--summarize-only"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
