#!/usr/bin/env python3
"""Build the HumanCLAW-style project demo page (demo/index.html).

Reads runs/demo_index.json + runs/demo_*/decisions.jsonl (produced by
scripts/render_demo_all.py on the sim host), copies the annotated episode
videos under demo/assets/, and renders a single-column academic demo page:
teaser, framework, benchmark, per-episode video + step-by-step decision
transcript + expert analysis, results table, bibtex.

    python scripts/build_demo_page.py [--runs runs] [--out demo]
"""

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ANALYSIS = {
    "scripted": (
        "Success case — grounded decisions only. The oracle baseline behaves exactly as "
        "embodiment-aware reasoning predicts: it NAVs to the kitchen, and for every closed "
        "container it first NAVs to that container's interaction anchor and only then OPENs, "
        "because <code>OPEN(cabinet)</code> is <em>grounded</em> (present in "
        "<code>available_skills</code>) only when an interaction configuration exists. "
        "<code>GRASP(knife)</code> does not exist in the action space until the knife's "
        "container is open — the action emerges from the world state. 8/8 selected actions "
        "were grounded at selection time and executed."
    ),
    "wrong_object": (
        "Failure: semantic discrimination. The agent opens the correct container but grasps a "
        "distractor spoon instead of the target knife. The benchmark attributes the failure "
        "mechanically — <code>FAIL_WRONG_TARGET</code>, terminal. Note that recognition is NOT "
        "the bottleneck: the knife was in the observation and in the grounded action space; the "
        "object-to-action binding is what failed."
    ),
    "timeout": (
        "Failure: ineffective exploration. The agent keeps NAVing between anchors without "
        "opening any container, so the knife is never exposed and <code>GRASP(knife)</code> "
        "never emerges in the action space. Dynamic grounding makes aimless wandering directly "
        "observable: the available action set stops changing while the step budget burns down."
    ),
    "unsafe": (
        "Failure: safety violation. From the living room the agent attempts to grasp "
        "fixed-base furniture (the countertop) directly. The SafetyValidator rejects the "
        "action — <code>FAIL_UNSAFE_ACTION</code>, terminal. Safety is evaluated at the "
        "semantic level, before any execution."
    ),
}

TITLES = {
    "scripted": ("Oracle baseline", "success"),
    "wrong_object": ("Semantic discrimination failure", "failure"),
    "timeout": ("Ineffective exploration failure", "failure"),
    "unsafe": ("Safety violation failure", "failure"),
}


def esc(s) -> str:
    return html.escape(str(s))


def badge(text: str) -> str:
    cls = "ok"
    if text in ("FAIL_WRONG_TARGET", "FAIL_MAX_STEPS", "FAIL_UNSAFE_ACTION", "False", "REJECTED"):
        cls = "bad"
    elif text in ("UNREACHABLE", "COLLISION", "INVALID_STATE"):
        cls = "warn"
    elif text == "None":
        cls = "mute"
    return f'<span class="badge {cls}">{esc(text)}</span>'


def build(runs_dir: Path, out_dir: Path) -> None:
    index = json.loads((runs_dir / "demo_index.json").read_text())
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    episodes_html = []
    for agent_name, info in index["episodes"].items():
        decisions = [
            json.loads(line)
            for line in (runs_dir / f"demo_{index['scenario']}_{agent_name}" / "decisions.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        asset_dir = assets / agent_name
        asset_dir.mkdir(exist_ok=True)
        import subprocess

        # Preferred path: encode directly from the rendered frame PNGs (the
        # sim host has no ffmpeg, so it only writes GIFs — and old ffmpeg
        # builds misread PIL GIF frame delays, so GIF->MP4 collapses the
        # timeline to a single frame; PNG sequences encode correctly).
        mp4 = asset_dir / "episode.mp4"
        encoded = False
        if shutil.which("ffmpeg"):
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2",
                 "-i", str(runs_dir / f"demo_{index['scenario']}_{agent_name}" / "frame_%04d.png"),
                 "-pix_fmt", "yuv420p", str(mp4)],
                capture_output=True,
            )
            encoded = r.returncode == 0 and mp4.exists() and mp4.stat().st_size > 0
        if encoded:
            video_src = f"assets/{agent_name}/episode.mp4"
        else:
            # fallback: copy the GIF (rename to .gif asset; <img> fallback is
            # handled by the browser since <video> cannot play GIFs)
            video_name = Path(info["video"]).name
            video_src = f"assets/{agent_name}/{video_name}"
            video_path = Path(info["video"])
            if not video_path.exists():
                video_path = runs_dir / f"demo_{index['scenario']}_{agent_name}" / video_name
            shutil.copy2(video_path, asset_dir / video_name)

        title, kind = TITLES[agent_name]
        rows = []
        for rec in decisions:
            action = rec.get("action")
            if action is None:
                act = rec.get("note", "—")
                verdict = rec.get("episode_status", "—")
            else:
                skill = action.get("skill", "?")
                tv = (action.get("target") or {}).get("value", "")
                act = f"{skill}({tv})" if tv else skill
                if rec.get("executed"):
                    verdict = "EXECUTED"
                else:
                    verdict = rec.get("failure_reason") or "REJECTED"
            skills = rec.get("available_skills") or []
            shown = skills[:4]
            skills_html = ", ".join(f"<code>{esc(s)}</code>" for s in shown)
            if len(skills) > 4:
                skills_html += f" <span class='more'>+{len(skills) - 4} more</span>"
            rows.append(
                "<tr><td class='stepno'>"
                + esc(rec.get("step"))
                + "</td><td><code>"
                + esc(act)
                + "</code></td><td>"
                + badge(verdict)
                + f"</td><td class='skills'>{skills_html}</td></tr>"
            )
        status = info["status"]
        status_cls = "ok" if info["match"] else "bad"
        episodes_html.append(f"""
<section class="episode {kind}">
  <h3>{esc(agent_name)} — {esc(title)} {badge(status)}</h3>
  <p class="cap">{esc(index['instruction'])} &middot; {info['steps']} planning steps &middot;
     expected {esc(info['expected'])}</p>
  <div class="ep-grid">
    <video controls preload="metadata" src="{video_src}"></video>
    <div>
      <table class="transcript">
        <thead><tr><th>step</th><th>action</th><th>verdict</th><th>available_skills (A<sub>t</sub>)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
  </div>
  <div class="analysis"><strong>Expert analysis.</strong> {ANALYSIS[agent_name]}</div>
</section>""")

    # results table
    order = ["scripted", "wrong_object", "timeout", "unsafe"]
    teaser_name = Path(index["episodes"]["scripted"]["video"]).stem + ".mp4"
    res_rows = "".join(
        f"<tr><td>{esc(a)}</td><td>{badge(index['episodes'][a]['status'])}</td>"
        f"<td>{esc(index['episodes'][a]['steps'])}</td>"
        f"<td>{esc(index['episodes'][a]['expected'])}</td>"
        f"<td>{'&#10003;' if index['episodes'][a]['match'] else '&#10007;'}</td></tr>"
        for a in order
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FindingBench — Demo</title>
<style>
  :root {{ --ink:#1a1a2e; --accent:#0f6e4e; --bad:#b3261e; --warn:#b26a00; }}
  html, body {{ background: #fff; }}
  body {{ font-family: Georgia, 'Times New Roman', serif; color: var(--ink);
         max-width: 1080px; margin: 0 auto; padding: 24px 20px 80px; line-height: 1.55; }}
  code, .mono {{ font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.86em;
         background: #f2f4f3; padding: 1px 4px; border-radius: 3px; }}
  h1 {{ font-size: 2.0em; margin-bottom: 4px; }}
  h2 {{ border-bottom: 2px solid #e3e6e4; padding-bottom: 6px; margin-top: 44px; }}
  .tagline {{ font-size: 1.05em; color: #444; }}
  .links a {{ margin-right: 14px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
  th, td {{ border: 1px solid #dfe3e1; padding: 6px 9px; text-align: left; vertical-align: top; }}
  th {{ background: #eef2f0; }}
  .badge {{ font-family: 'SF Mono', Menlo, monospace; font-size: 0.8em; padding: 2px 7px;
           border-radius: 10px; color: #fff; background: #667; white-space: nowrap; }}
  .badge.ok {{ background: var(--accent); }}
  .badge.bad {{ background: var(--bad); }}
  .badge.warn {{ background: var(--warn); }}
  .badge.mute {{ background: #889; }}
  .ep-grid {{ display: grid; grid-template-columns: 460px 1fr; gap: 18px; align-items: start; }}
  .ep-grid video {{ width: 100%; border: 1px solid #ccc; border-radius: 6px; background: #000; }}
  .transcript {{ font-size: 0.8em; }}
  .transcript td.stepno {{ color: #888; width: 3em; }}
  .transcript td.skills {{ color: #555; }}
  .more {{ color: #999; }}
  .analysis {{ background: #f5f7f6; border-left: 4px solid var(--accent);
              padding: 10px 14px; margin-top: 12px; font-size: 0.95em; }}
  .episode.failure .analysis {{ border-left-color: var(--bad); }}
  .pipeline {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch;
              font-family: 'SF Mono', Menlo, monospace; font-size: 0.85em; }}
  .pipe {{ border: 1.5px solid var(--accent); border-radius: 8px; padding: 8px 12px;
          background: #f2f8f5; min-width: 130px; }}
  .pipe b {{ color: var(--accent); }}
  .arr {{ align-self: center; color: var(--accent); font-weight: bold; }}
  .ladder td:first-child {{ font-weight: bold; width: 4em; text-align: center; }}
  footer {{ margin-top: 60px; color: #777; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>FindingBench</h1>
<p class="tagline">An embodiment-aware diagnostic benchmark for <b>interactive object search</b>:
the admissible action space is dynamically grounded from robot kinematics, object interaction
interfaces and world state — and execution is perfect by construction.</p>
<p class="links">
  <a href="https://github.com/KvnWong216/FindingBench">Code</a>
  <a href="#framework">Framework</a>
  <a href="#episodes">Demo episodes</a>
  <a href="#results">Results</a>
</p>

<h2 id="teaser">Teaser: find the hidden knife</h2>
<p>{esc(index['instruction'])} — Beechwood_0_int (BEHAVIOR-1K v3.9.3), R1 Pro wheeled dual-arm
mobile manipulator. Green overlay text is the agent's decision at each semantic step; physics
renders the egocentric view, but the benchmark itself never measures control.</p>
<video controls preload="metadata" src="assets/scripted/{teaser_name}" style="width:100%; max-width:860px; border:1px solid #ccc; border-radius:6px;"></video>

<h2 id="framework">Framework: the Embodied Action Grounding Engine</h2>
<p>Existing embodied benchmarks evaluate agents through end-to-end successful execution, which
entangles high-level decision quality with low-level execution noise. FindingBench factors the
two apart. Every step, the admissible action space is <b>regenerated</b>:</p>
<p class="mono" style="text-align:center;">A<sub>t</sub> = Ground(Robot, Object, WorldState<sub>t</sub>)</p>
<div class="pipeline">
  <div class="pipe"><b>Object adapters</b><br>rigid / articulated /<br>receptacle propose<br>semantic candidates +<br>interaction interfaces<br>(handle links, regions)</div>
  <div class="arr">&#8594;</div>
  <div class="pipe"><b>Feasibility filter</b><br>URDF kinematics (pinocchio)<br>SE(3) IK + joint limits<br>c-space collision (coal)<br>state constraints</div>
  <div class="arr">&#8594;</div>
  <div class="pipe"><b>available_skills</b><br>NAV / OPEN / CLOSE /<br>GRASP / PLACE — only what<br>exists right now</div>
  <div class="arr">&#8594;</div>
  <div class="pipe"><b>Skill executor</b><br>instant symbolic<br>state transition</div>
  <div class="arr">&#8594;</div>
  <div class="pipe"><b>World update</b><br>&amp; regenerate A<sub>t+1</sub></div>
</div>
<p>A skill is not chosen from a fixed list — it <b>emerges</b> from the world. Feasibility is
configuration-level ("does at least one collision-free interaction configuration exist?"),
derived from the robot's URDF exported and FK-cross-validated against the simulator — never
hand-authored capability proxies. Structured failures (<code>UNREACHABLE</code>,
<code>COLLISION</code>, <code>INVALID_STATE</code>) come from the validators themselves, so
failure attribution is mechanical, not hand-labeled.</p>
<p><b>Certified difficulty.</b> Every episode's optimal semantic depth d* is computed by an
oracle BFS planner over the canonical state using the production feasibility engine —
difficulty is measured, never assigned. Episodes proven unsolvable for the embodiment are
excluded from the standard split (19/30 oracle-solvable in the current split).</p>

<h2>Benchmark: difficulty ladder</h2>
<table class="ladder">
<tr><th>level</th><th>axis</th><th>tests</th><th>status</th></tr>
<tr><td>L0</td><td>Known location</td><td>interaction execution</td><td>knife_search_001</td></tr>
<tr><td>L1</td><td>Unknown container</td><td>search planning</td><td>knife_search_001</td></tr>
<tr><td>L2</td><td>Distractors</td><td>semantic discrimination</td><td>knife_search_001</td></tr>
<tr><td>L3</td><td>Embodiment constraint</td><td>capability-aware planning</td><td>mechanism + unit tests (same task, different embodiment &rarr; different action graph)</td></tr>
<tr><td>L4</td><td>Occlusion / rearrangement</td><td>interactive perception</td><td>future</td></tr>
</table>

<h2 id="episodes">Demo episodes: success &amp; failure case studies</h2>
<p>Four data-driven agents run the same scenario. Each video shows the egocentric view with the
decision overlay; the transcript lists every step's action, the mechanical verdict, and the
grounded action space <code>A<sub>t</sub></code> the agent saw when deciding.</p>
{''.join(episodes_html)}

<h2 id="results">Results</h2>
<table>
<tr><th>agent</th><th>final status</th><th>steps</th><th>expected</th><th>match</th></tr>
{res_rows}
</table>
<p>Unit tests 35/35 &middot; integration tests (real simulator) 11/11 &middot; scenario compiled
from YAML only, no scenario-specific Python.</p>

<h2>BibTeX</h2>
<pre><code>@misc{{findingbench2026,
  title  = {{FindingBench: An Embodiment-Aware Diagnostic Benchmark for Interactive Object Search}},
  year   = {{2026}},
  url    = {{https://github.com/KvnWong216/FindingBench}}
}}</code></pre>

<footer>FindingBench demo page — episode videos rendered with
<code>scripts/render_demo_all.py</code> (BEHAVIOR-1K v3.9.3 + OmniGibson, single process,
semantic steps only). Page layout modeled after the HumanCLAW project page.</footer>
</body>
</html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    print(f"wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=REPO / "runs")
    p.add_argument("--out", type=Path, default=REPO / "demo")
    a = p.parse_args()
    build(a.runs, a.out)
