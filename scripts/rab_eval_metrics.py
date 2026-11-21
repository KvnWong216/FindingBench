#!/usr/bin/env python3
"""Offline FindingBench RAB v1 metrics (findingbench_rab_pinhole_v1).

Evaluator-private: reads ``worker_trace/actions/*.json`` written by
``scripts/rab_skills.py``. Run ONLY on terminated episodes.

Core orthogonal metrics
    SR        success episodes / valid episodes
    ST        mean planning steps over successful episodes (ST_all also given)
    EFF_phys  EXECUTED / called over physical skills
              {MOVE_TO, TURN, OPEN, CLOSE, GRASP, PLACE}  (EFF_all also given)
    RES       mean terminal restoration score (auxiliary, never folded into SR)

Failure analysis (all labels deterministic, read from simulator records)
    Layer A   every refused skill call -> exactly one error prototype ->
              exactly one brain capability. Rates sum to 1 - EFF.
    Layer B   every failed episode -> exactly one terminal attribution.
              Shares sum to 1 - SR.

Infrastructure exceptions are never charged to the agent: such calls are
removed from EFF numerator and denominator and reported separately; episodes
with no terminal record are excluded from SR and reported as incomplete.

Usage
    python scripts/rab_eval_metrics.py RUN_DIR [RUN_DIR ...] \
        [--label NAME ...] [--out OUT_DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rab_feedback import TASK_STAGES, detail_for

PHYSICAL_SKILLS = ("MOVE_TO", "TURN", "OPEN", "CLOSE", "GRASP", "PLACE")
POINT_SKILLS = ("MOVE_TO", "CHECK", "OPEN", "CLOSE", "GRASP", "PLACE")
ALL_SKILLS = ("VIEW", "TURN", "MOVE_TO", "CHECK", "OPEN", "CLOSE", "GRASP", "PLACE", "REPORT_DONE")

# ---------------------------------------------------------------------------
# Layer A: error prototype -> brain capability (one-to-one, exhaustive)
# ---------------------------------------------------------------------------
CAPABILITIES: dict[str, dict[str, Any]] = {
    "spatial": dict(
        name="Spatial & Workspace Awareness",
        prototypes=("GRASP_TOO_FAR", "PLACE_TOO_FAR", "ARTICULATION_TOO_FAR",
                    "ARM_IK_TARGET_FAILED", "ARM_IK_PLACE_FAILED", "ARM_REACH_FAILED"),
        eligible=("GRASP", "PLACE", "OPEN", "CLOSE"),
    ),
    "collision": dict(
        name="Collision & Safety Clearance",
        prototypes=("NAV_NO_CLEAR_ROUTE", "NAV_SEGMENT_CLEARANCE_BLOCKED",
                    "TURN_RADIAL_CLEARANCE_BLOCKED", "BASE_MOTION_SHORTFALL",
                    "HAND_PATH_BLOCKED"),
        eligible=("MOVE_TO", "TURN", "GRASP", "PLACE"),
    ),
    "physical": dict(
        name="Physical Affordance & Stability",
        prototypes=("UNSAFE_OVERSIZED_GRASP", "UNSAFE_PLACEMENT_UNSTABLE",
                    "CLICK_NON_HORIZONTAL_SURFACE"),
        eligible=("GRASP", "PLACE"),
    ),
    "grounding": dict(
        name="Visual Grounding & Affordance",
        prototypes=("CLICK_NON_INTERACTABLE", "CLICK_NON_ARTICULATED"),
        eligible=POINT_SKILLS,
    ),
    "state": dict(
        name="State Tracking & Protocol Memory",
        prototypes=("PRECONDITION_HAND_OCCUPIED", "PRECONDITION_HAND_EMPTY",
                    "PRECONDITION_ALREADY_OPEN", "PRECONDITION_ALREADY_CLOSED",
                    "PLACE_ON_SELF", "STALE_FRAME_ID", "SCHEMA_INVALID",
                    "EPISODE_ALREADY_TERMINATED", "UNSUPPORTED_SKILL"),
        eligible=ALL_SKILLS,
    ),
}
PROTOTYPE_TO_CAPABILITY = {p: k for k, v in CAPABILITIES.items() for p in v["prototypes"]}
INFRA_PROTOTYPES = {"INFRA_EXCEPTION"}

# ---------------------------------------------------------------------------
# Layer B: terminal attribution -> brain capability
# ---------------------------------------------------------------------------
TERMINAL: dict[str, tuple[str, str]] = {
    # label: (episode status family, brain capability)
    "WRONG_TARGET_DELIVERED": ("FAIL_FALSE_COMPLETION", "Fine-grained Recognition & Instruction Grounding"),
    "WRONG_TARGET_MANIPULATED": ("FAIL_FALSE_COMPLETION", "Fine-grained Recognition & Instruction Grounding"),
    "WRONG_DESTINATION": ("FAIL_FALSE_COMPLETION", "Goal / Destination Grounding"),
    "PARTIAL_DELIVERY": ("FAIL_FALSE_COMPLETION", "Goal Completeness Verification"),
    "TARGET_NOT_RELEASED": ("FAIL_FALSE_COMPLETION", "Completion Verification"),
    "TARGET_UNSTABLE": ("FAIL_FALSE_COMPLETION", "Completion Verification"),
    "PREMATURE_DONE": ("FAIL_FALSE_COMPLETION", "Completion Verification (Hallucinated Done)"),
    "TARGET_NEVER_ENGAGED": ("FAIL_MAX_STEPS", "Semantic Search & Exploration"),
    "CONTAINER_OPENED_TARGET_NOT_ENGAGED": ("FAIL_MAX_STEPS", "In-container Visual Search"),
    "TARGET_ENGAGED_NOT_DELIVERED": ("FAIL_MAX_STEPS", "Long-horizon Execution & Error Recovery"),
}

LEGACY_TEXT = (  # free-text reasons emitted by pre-taxonomy worker builds
    ("hand path blocked by", "HAND_PATH_BLOCKED"),
    ("released object did not settle", "UNSAFE_PLACEMENT_UNSTABLE"),
    ("cannot place an object on itself", "PLACE_ON_SELF"),
    ("Traceback (most recent call last)", "INFRA_EXCEPTION"),
    ("validation error", "SCHEMA_INVALID"),
)


def prototype_of(code: str, private_error: str | None) -> str:
    """Map one action record to its deterministic error prototype."""
    if code == "EXECUTED":
        return "EXECUTED"
    if private_error:
        head = private_error.split(":", 1)[0].strip()
        if head in PROTOTYPE_TO_CAPABILITY or head in INFRA_PROTOTYPES:
            return head
        for needle, proto in LEGACY_TEXT:
            if needle in private_error:
                return proto
    return f"UNLABELED_{code}"


def capability_of(proto: str) -> str:
    if proto in INFRA_PROTOTYPES:
        return "infra"
    return PROTOTYPE_TO_CAPABILITY.get(proto, "unlabeled")


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------
def find_episodes(root: Path) -> list[Path]:
    """Return action directories under ``root`` (one per episode)."""
    root = Path(root)
    if (root / "actions").is_dir():
        return [root / "actions"]
    return sorted({p.parent for p in root.rglob("actions/0001.json")})


def load_episode(actions_dir: Path) -> dict[str, Any]:
    trace_dir = actions_dir.parent
    manifest_path = trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    records = [json.loads(p.read_text()) for p in sorted(actions_dir.glob("*.json"))]
    calls, terminal = [], None
    for rec in records:
        result = rec.get("result") or {}
        fb = result.get("feedback") or {}
        code = fb.get("code", "UNKNOWN")
        raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else {}
        skill = str(raw.get("skill", "UNPARSEABLE"))
        proto = prototype_of(code, rec.get("private_error"))
        detail_code = fb.get("detail_code")
        if detail_code is None and code != "EXECUTED":
            detail_code, _ = detail_for(code, proto if proto != f"UNLABELED_{code}" else rec.get("private_error"), skill=skill)
        calls.append(dict(skill=skill, code=code, prototype=proto, detail_code=detail_code,
                          capability=capability_of(proto), step=result.get("planning_step")))
        if result.get("terminal") and terminal is None:
            measurements = rec.get("measurements") or {}
            terminal = dict(
                status=result.get("episode_status"),
                success=bool(result.get("success")),
                restoration=result.get("restoration", rec.get("restoration")),
                steps=result.get("planning_step"),
                attribution=measurements.get("terminal_attribution"),
            )
    return dict(task_id=manifest.get("id", trace_dir.parent.name), path=str(trace_dir),
                calls=calls, terminal=terminal)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _ratio(n: float, d: float) -> float | None:
    return round(n / d, 4) if d else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def aggregate(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    episodes = list(episodes)
    valid = [e for e in episodes if e["terminal"] is not None]
    incomplete = [e["path"] for e in episodes if e["terminal"] is None]
    success = [e for e in valid if e["terminal"]["success"]]
    failed = [e for e in valid if not e["terminal"]["success"]]

    # ---- skill-level -------------------------------------------------------
    calls = [c for e in episodes for c in e["calls"]]
    agent_calls = [c for c in calls if c["capability"] != "infra"]
    infra_calls = len(calls) - len(agent_calls)
    phys = [c for c in agent_calls if c["skill"] in PHYSICAL_SKILLS]
    n_exec_all = sum(c["code"] == "EXECUTED" for c in agent_calls)
    n_exec_phys = sum(c["code"] == "EXECUTED" for c in phys)
    refused = [c for c in agent_calls if c["code"] != "EXECUTED"]

    proto_counts = Counter(c["prototype"] for c in refused)
    cap_counts = Counter(c["capability"] for c in refused)
    skill_called = Counter(c["skill"] for c in agent_calls)
    skill_exec = Counter(c["skill"] for c in agent_calls if c["code"] == "EXECUTED")
    skill_proto: dict[str, Counter] = defaultdict(Counter)
    for c in refused:
        skill_proto[c["skill"]][c["prototype"]] += 1

    layer_a = {}
    for key, spec in CAPABILITIES.items():
        n_err = cap_counts.get(key, 0)
        eligible = sum(skill_called.get(s, 0) for s in spec["eligible"])
        layer_a[key] = dict(
            capability=spec["name"],
            errors=n_err,
            rate_over_all_calls=_ratio(n_err, len(agent_calls)),   # sums to 1-EFF_all
            share_of_failures=_ratio(n_err, len(refused)),         # sums to 1
            eligible_calls=eligible,
            axis_score=None if not eligible else round(1 - n_err / eligible, 4),
            prototypes={p: proto_counts.get(p, 0) for p in spec["prototypes"]},
        )
    unlabeled = {p: n for p, n in proto_counts.items() if capability_of(p) == "unlabeled"}
    if unlabeled:
        n_err = sum(unlabeled.values())
        layer_a["unlabeled"] = dict(capability="Unlabeled (legacy trace)", errors=n_err,
                                    rate_over_all_calls=_ratio(n_err, len(agent_calls)),
                                    share_of_failures=_ratio(n_err, len(refused)),
                                    eligible_calls=None, axis_score=None, prototypes=unlabeled)

    # ---- episode-level -----------------------------------------------------
    term_counts = Counter((e["terminal"]["attribution"]
                           or f"UNATTRIBUTED_{e['terminal']['status']}") for e in failed)
    layer_b = {
        label: dict(
            count=n,
            share_of_failed=_ratio(n, len(failed)),
            share_of_episodes=_ratio(n, len(valid)),
            status=TERMINAL.get(label, ("?", "?"))[0],
            capability=TERMINAL.get(label, ("?", "Unattributed (legacy trace)"))[1],
        )
        for label, n in term_counts.most_common()
    }

    # ---- task-stage funnel & repeat-same-error rate (RSER) ----------------
    task_funnel = {}
    for stage_key, stage_name, stage_skills in TASK_STAGES:
        s_calls = [c for c in agent_calls if c["skill"] in stage_skills]
        s_exec = sum(c["code"] == "EXECUTED" for c in s_calls)
        s_fail = [c for c in s_calls if c["code"] != "EXECUTED"]
        s_details = Counter(c["detail_code"] for c in s_fail if c.get("detail_code"))
        task_funnel[stage_key] = dict(
            stage=stage_name,
            skills=list(stage_skills),
            called=len(s_calls),
            executed=s_exec,
            errors=len(s_fail),
            eff=_ratio(s_exec, len(s_calls)),
            share_of_failures=_ratio(len(s_fail), len(refused)),
            detail_codes=dict(s_details.most_common()),
        )

    repeat_counts: Counter[str] = Counter()
    for ep in episodes:
        ep_agent = [c for c in ep["calls"] if c["capability"] != "infra"]
        for prev, cur in zip(ep_agent, ep_agent[1:]):
            if prev["code"] != "EXECUTED" and cur["code"] != "EXECUTED":
                d_prev, d_cur = prev.get("detail_code"), cur.get("detail_code")
                if d_prev and d_prev == d_cur:
                    repeat_counts[d_cur] += 1
    total_repeats = sum(repeat_counts.values())
    rser = dict(
        repeat_errors=total_repeats,
        refused_calls=len(refused),
        rate=_ratio(total_repeats, len(refused)),
        by_detail_code=dict(repeat_counts.most_common()),
    )

    steps_succ = [e["terminal"]["steps"] for e in success if e["terminal"]["steps"] is not None]
    steps_all = [e["terminal"]["steps"] for e in valid if e["terminal"]["steps"] is not None]
    res = [e["terminal"]["restoration"] for e in valid if e["terminal"]["restoration"] is not None]

    return dict(
        episodes=dict(total=len(episodes), valid=len(valid), success=len(success),
                      failed=len(failed), incomplete_infra=len(incomplete)),
        core=dict(
            SR=_ratio(len(success), len(valid)),
            SR_ci95=wilson(len(success), len(valid)),
            ST_succ=round(sum(steps_succ) / len(steps_succ), 2) if steps_succ else None,
            ST_all=round(sum(steps_all) / len(steps_all), 2) if steps_all else None,
            EFF_phys=_ratio(n_exec_phys, len(phys)),
            EFF_all=_ratio(n_exec_all, len(agent_calls)),
            RES=round(sum(res) / len(res), 4) if res else None,
            RSER=rser["rate"],
        ),
        calls=dict(total=len(calls), agent=len(agent_calls), physical=len(phys),
                   refused=len(refused), infra_excluded=infra_calls),
        per_skill={s: dict(called=skill_called[s], executed=skill_exec.get(s, 0),
                           eff=_ratio(skill_exec.get(s, 0), skill_called[s]),
                           failures=dict(skill_proto.get(s, {})))
                   for s in sorted(skill_called)},
        task_funnel=task_funnel,
        rser=rser,
        layer_a=layer_a,
        layer_b=layer_b,
        incomplete_paths=incomplete,
        per_episode=[dict(task_id=e["task_id"], path=e["path"], **(e["terminal"] or {}),
                          calls=len(e["calls"]),
                          refused=sum(c["code"] != "EXECUTED" for c in e["calls"]))
                     for e in episodes],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def to_markdown(reports: dict[str, dict[str, Any]]) -> str:
    out = ["# FindingBench RAB v1 — Evaluation Report", "",
           "## Core orthogonal metrics", "",
           "| Run | Episodes | SR ↑ (95% CI) | ST_succ ↓ | ST_all | EFF_phys ↑ | EFF_all | RES ↑ | RSER ↓ |",
           "|---|---|---|---|---|---|---|---|---|"]
    for label, r in reports.items():
        c, e = r["core"], r["episodes"]
        ci = c["SR_ci95"]
        ci_s = f" ({_pct(ci[0])}–{_pct(ci[1])})" if ci else ""
        out.append(f"| {label} | {e['valid']} (+{e['incomplete_infra']} infra) | {_pct(c['SR'])}{ci_s} | "
                   f"{c['ST_succ'] if c['ST_succ'] is not None else '—'} | "
                   f"{c['ST_all'] if c['ST_all'] is not None else '—'} | {_pct(c['EFF_phys'])} | "
                   f"{_pct(c['EFF_all'])} | {_pct(c['RES'])} | {_pct(c.get('RSER'))} |")
    for label, r in reports.items():
        rser = r.get("rser") or {}
        rep_s = ", ".join(f"`{k}`×{v}" for k, v in (rser.get("by_detail_code") or {}).items()) or "—"
        out += ["", f"## {label}", "",
                "### Task-Stage Funnel (task-centric proprioceptive breakdown)", "",
                f"Repeat-Same-Error Rate (RSER): {_pct(rser.get('rate'))} "
                f"({rser.get('repeat_errors', 0)}/{rser.get('refused_calls', 0)}; repeats: {rep_s})", "",
                "| Task stage | Called | Executed | EFF | Share of failures | Proprioceptive detail codes |",
                "|---|---|---|---|---|---|"]
        for st in (r.get("task_funnel") or {}).values():
            dc = ", ".join(f"`{k}`×{v}" for k, v in st["detail_codes"].items()) or "—"
            out.append(f"| {st['stage']} | {st['called']} | {st['executed']} | {_pct(st['eff'])} | "
                       f"{_pct(st['share_of_failures'])} | {dc} |")
        out += ["", "### Layer A — skill-level error prototypes → brain capability", "",
                f"Agent calls: {r['calls']['agent']} · refused: {r['calls']['refused']} · "
                f"infra excluded: {r['calls']['infra_excluded']}", "",
                "| Brain capability | Errors | Rate / all calls | Share of failures | Axis score | Prototypes (n>0) |",
                "|---|---|---|---|---|---|"]
        for spec in r["layer_a"].values():
            protos = ", ".join(f"`{p}`×{n}" for p, n in spec["prototypes"].items() if n) or "—"
            out.append(f"| {spec['capability']} | {spec['errors']} | {_pct(spec['rate_over_all_calls'])} | "
                       f"{_pct(spec['share_of_failures'])} | {_pct(spec['axis_score'])} | {protos} |")
        out += ["", "### Layer B — terminal failure attribution", "",
                "| Attribution | Status | Brain capability | Count | Share of failed | Share of episodes |",
                "|---|---|---|---|---|---|"]
        for label_b, spec in r["layer_b"].items():
            out.append(f"| `{label_b}` | {spec['status']} | {spec['capability']} | {spec['count']} | "
                       f"{_pct(spec['share_of_failed'])} | {_pct(spec['share_of_episodes'])} |")
        if not r["layer_b"]:
            out.append("| — | — | — | 0 | — | — |")
        out += ["", "### Per-skill execution", "", "| Skill | Called | Executed | EFF | Failure prototypes |",
                "|---|---|---|---|---|"]
        for skill, s in r["per_skill"].items():
            fails = ", ".join(f"`{p}`×{n}" for p, n in sorted(s["failures"].items(), key=lambda x: -x[1])) or "—"
            out.append(f"| {skill} | {s['called']} | {s['executed']} | {_pct(s['eff'])} | {fails} |")
    return "\n".join(out) + "\n"


def plot(reports: dict[str, dict[str, Any]], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    written = []
    axes_keys = ["SR", *CAPABILITIES]
    labels = ["Task Success (SR)", *[CAPABILITIES[k]["name"].replace(" & ", "\n& ") for k in CAPABILITIES]]
    angles = np.linspace(0, 2 * np.pi, len(axes_keys), endpoint=False).tolist()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, polar=True)
    for label, r in reports.items():
        vals = [r["core"]["SR"] or 0.0] + [(r["layer_a"][k]["axis_score"] or 0.0) for k in CAPABILITIES]
        ax.plot(angles + angles[:1], vals + vals[:1], linewidth=2, label=label)
        ax.fill(angles + angles[:1], vals + vals[:1], alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("FindingBench RAB v1 — Brain Capability Radar", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    p = out_dir / "fig_capability_radar.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    # Stacked decomposition of (1 - EFF_all) by capability.
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(reports), 5))
    names = list(reports)
    bottom = np.zeros(len(names))
    keys = [*CAPABILITIES, "unlabeled"]
    for key in keys:
        vals = np.array([(reports[n]["layer_a"].get(key, {}).get("rate_over_all_calls") or 0.0) for n in names])
        if not vals.any():
            continue
        title = CAPABILITIES[key]["name"] if key in CAPABILITIES else "Unlabeled (legacy)"
        ax.bar(names, vals, bottom=bottom, label=title)
        bottom += vals
    ax.set_ylabel("Refused calls / all agent calls  (= 1 − EFF_all)")
    ax.set_title("Layer A — skill failure decomposition")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    p = out_dir / "fig_layer_a_decomposition.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (one per model/config)")
    ap.add_argument("--label", nargs="*", default=None, help="display label per run directory")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: first run dir / metrics)")
    args = ap.parse_args()
    labels = args.label or [p.resolve().name for p in args.runs]
    if len(labels) != len(args.runs):
        ap.error("--label count must match run directories")
    reports = {}
    for label, run in zip(labels, args.runs):
        episodes = [load_episode(d) for d in find_episodes(run)]
        if not episodes:
            print(f"[warn] no episodes under {run}")
            continue
        reports[label] = aggregate(episodes)
    if not reports:
        raise SystemExit("no episodes found")
    out = args.out or (args.runs[0] / "metrics")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    md = to_markdown(reports)
    (out / "metrics_report.md").write_text(md)
    figs = plot(reports, out)
    print(md)
    print(f"[ok] wrote {out / 'metrics.json'}, {out / 'metrics_report.md'}" + (f", {', '.join(figs)}" if figs else ""))


if __name__ == "__main__":
    main()
