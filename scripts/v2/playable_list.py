"""vla_difficulty_v2: list of certified (oracle-verified) variants for human play.

Reads build/vla_difficulty_v2/manifest.json and runs/vla_difficulty_v2/summary.json
(python scripts/v2/run_batch.py --summarize-only first). A variant is listed
when its physical configuration (task, S) was accepted; L1/L2/L3 share it.

    python scripts/v2/playable_list.py
Writes runs/vla_difficulty_v2/playable.md and playable.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build/vla_difficulty_v2"
RUNS = ROOT / "runs/vla_difficulty_v2"
PLAY = "bash scripts/play_behavior.sh --scenario {}"


def main() -> int:
    global RUNS, BUILD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=RUNS)
    ap.add_argument("--build-dir", type=Path, default=BUILD)
    args = ap.parse_args()
    RUNS, BUILD = args.runs_dir, args.build_dir
    manifest = json.loads((BUILD / "manifest.json").read_text())
    summary = json.loads((RUNS / "summary.json").read_text())
    configs = summary["configs"]
    rows = []
    for v in manifest["variants"]:
        c = configs.get(f"{v['task_id']}|{v['spatial_level']}")
        if not c or c["status"] != "accepted":
            continue
        scen = yaml.safe_load((ROOT / v["scenario"]).read_text())
        goal = v["goal"][0]
        rows.append({
            "variant_id": v["variant_id"], "task_id": v["task_id"], "scene": v["scene"],
            "family": v["family"], "L": v["language_level"], "S": v["spatial_level"],
            "s2_kind": v.get("s2_kind") or "", "object": v["target"]["noun"],
            "start_room": v["start"]["room"], "object_room": v["target_room"],
            "object_location": v["location_phrase"],
            "goal": goal["kind"] + (f" {goal['receptacle']}" if "receptacle" in goal else ""),
            "instruction": scen["instruction"].strip(), "oracle_steps": c["oracle_steps"],
            "command": PLAY.format(v["scenario"]),
        })
    rows.sort(key=lambda r: (r["scene"], r["task_id"], r["S"], r["L"]))
    with (RUNS / "playable.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tasks = defaultdict(set)
    for r in rows:
        tasks[r["task_id"]].add(r["S"])
    full = {t for t, s in tasks.items() if len(s) == 3}
    by_scene = Counter(r["scene"] for r in rows)
    lines = ["# vla_difficulty_v2：可试玩任务清单", "",
             "只列出已通过三级验证的变体：初始化 + S 门禁、空报告被拒、特权脚本 oracle 只用公共动作完成。",
             "**尚未经过人工试玩，也没有模型实测。** 同一 (任务, S) 的 L1/L2/L3 物理布局完全相同，只有指令不同。",
             "",
             *(["> 协议：只用 MOVE_TO 验证（点地面＝朝向不变走过去；点家具/物体＝自动站到它前面并面向它），"
                "与当前试玩界面一致。抓取/放置门禁未改。"] if "moveto" in RUNS.name else
               ["> 注意：这些变体是用 MOVE/TURN 协议验证的（oracle 路线由 MOVE/TURN 组成）。2026-10-26 19:13 起公共协议改为",
                "> MOVE_TO（无 MOVE/TURN，朝向不变），当前试玩界面也是新协议，**在新协议下是否仍可完成尚未验证**。"]),
             "", f"- 变体：**{len(rows)}** 条，基础任务 {len(tasks)} 个，其中 S1/S2/S3 全部通过 {len(full)} 个（★）",
             f"- 由 `python scripts/v2/playable_list.py` 生成；完整表格见同目录 `playable.csv`",
             "- 试玩：`" + PLAY.format("build/vla_difficulty_v2/<variant_id>/scenario.yaml") + "`（按提示在浏览器打开）",
             "", "| 场景 | 变体数 |", "|---|---|"]
    lines += [f"| {s} | {n} |" for s, n in sorted(by_scene.items())]
    cells = Counter(f"{r['L']}_{r['S']}" for r in rows)
    lines += ["", "| | S1 | S2 | S3 |", "|---|---|---|---|"]
    lines += [f"| {L} | " + " | ".join(str(cells[f'{L}_{S}']) for S in ("S1", "S2", "S3")) + " |"
              for L in ("L1", "L2", "L3")]
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["scene"], r["task_id"])].append(r)
    scene = None
    for (sc, t), rs in by_task.items():
        if sc != scene:
            scene = sc
            lines += ["", f"## {sc}"]
        r0 = rs[0]
        lines += ["", f"### {t}{' ★' if t in full else ''}：{r0['family']}，{r0['object']}，目标 `{r0['goal']}`", "",
                  "| 变体 | 起点房间 | 物体位置 | oracle 步数 | 指令 |", "|---|---|---|---|---|"]
        for r in rs:
            where = f"{r['object_location']}（{r['object_room']}）" + (f" [{r['s2_kind']}]" if r["s2_kind"] else "")
            lines.append(f"| `{r['variant_id']}` | {r['start_room']} | {where} | {r['oracle_steps']} | "
                         f"{r['instruction'].replace('|', '/')} |")
    (RUNS / "playable.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"variants": len(rows), "base_tasks": len(tasks), "full_9_cell": len(full),
                      "by_scene": by_scene, "cells": dict(sorted(cells.items()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
