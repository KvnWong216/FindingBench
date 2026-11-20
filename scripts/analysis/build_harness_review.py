"""Read-only trajectory publication. Does not import or call a model or simulator."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from failure_explanations import explain, sweep

TASK = "T01_knife_fetch__L1_S1"
DEFAULT_SUITE = ROOT / "runs/harness_v42_all_models_single_task"
DEFAULT_OUT = ROOT / "runs/harness_v42_review"


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not yet have completed its last line.
            break
    return result


def write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def relative(path, out):
    return os.path.relpath(path, out)


def patterns(steps):
    longest = current = 0
    for row in steps:
        e = row["event"]
        blocked = e.get("private_reason") in ("MOVE_BLOCKED", "TURN_BLOCKED") and not e.get("executed")
        current = current + 1 if blocked else 0
        longest = max(longest, current)
    return dict(longest_consecutive_rejected_motion=longest,
                skill_counts=dict(Counter((s["event"].get("raw_action") or {}).get("skill", "MALFORMED")
                                          for s in steps)),
                interpretation="Observable action pattern only; not a cognitive root-cause verdict.")


def normalize(episode, out, probe):
    calls = {r["planning_step"] + 1: r for r in lines(episode / "model_calls.jsonl")}
    decisions = {r["planning_step"] + 1: r for r in lines(episode / "agent_decisions.jsonl")}
    events = [r for r in lines(episode / "episode/events.jsonl") if r.get("event_type") == "action"]
    rows, yaw = [], math.pi  # fixed certified scenario initial quaternion [0,0,1,0]
    for event in events:
        step = event["planning_step"]
        call = calls.get(step, {})
        decision = decisions.get(step, {})
        diagnostic = None
        action = event.get("raw_action") or {}
        if event.get("private_reason") in ("MOVE_BLOCKED", "TURN_BLOCKED") or (
                event.get("executed") and action.get("skill") in ("MOVE", "TURN")):
            diagnostic = sweep(event, yaw, probe["obstacles"])
            if diagnostic:
                diagnostic.update(
                    evidence="historical_geometry_reconstruction",
                    verified_live=False,
                    prediction_matches_execution=diagnostic["predicted_safe"] == bool(event["executed"]),
                    caveat="Historical L2_S1 scene probe, rounded to millimeters; current task L1_S1. "
                           "Furniture/state differences possible. Agreement is corroboration, not exact contact proof.")
        before = episode / "episode" / f"step_{step-1:03d}.png"
        after = episode / "episode" / f"step_{step:03d}.png"
        detail = before.with_name(before.stem + "_detail.png")
        if event.get("failure_diagnostic_private") is not None:
            diagnostic = event["failure_diagnostic_private"]
        rows.append(dict(
            step=step, input_step=step-1, yaw_before=yaw,
            panorama=relative(before, out) if before.exists() else None,
            panorama_after=relative(after, out) if after.exists() else None,
            detail=relative(detail, out) if detail.exists() else None,
            model_call=call, decision=decision, event=event, diagnostic=diagnostic,
            human_explanation=event.get("public_feedback_message") or explain(event),
            feedback_was_supplied=bool(event.get("public_feedback_message")),
            # Historical object IDs are shown only in the evaluator diagnostic panel.
        ))
        if event.get("executed") and action.get("skill") == "TURN":
            yaw += math.radians(action["angle_deg"])
    return rows


STYLE = """body{margin:0;background:#f4f6fa;color:#172338;font:16px system-ui}
main{max-width:1200px;margin:auto;padding:28px}a{color:#1459a3}
header,section,article{background:white;padding:20px;border-radius:12px;margin-bottom:18px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f7;padding:14px;max-height:620px;overflow:auto}
.note{color:#586579}.bad{color:#a5222a}.good{color:#177345}button,select,input{font:inherit;padding:8px}
img{max-width:100%;display:block}canvas{width:100%;background:#edf1f5}
nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2;box-shadow:0 2px 8px #bbc}
table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd;text-align:left}
h2{font-size:21px}h3{font-size:18px}details{margin:14px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:700px){.grid{display:block}}"""

VIEWER = r"""
<!doctype html><html lang="en"><meta charset="utf-8"><title>MODEL · Step-by-step trajectory</title>
<style>STYLE</style><main><header><a href="index.html">← All models</a>
<h1 id="title"></h1><p id="status"></p><p id="patterns"></p><p class="note">Read-only analysis: actual model input is shown above. External views below are labeled as synchronized exo, independent replay, or geometric illustration. Actual feedback and offline attribution are separate. Instance IDs and external views are not model inputs. Reasoning shows only text actually returned by the model.</p></header>
<nav><button id="prev">Previous</button> <input id="slider" type="range" min="0" value="0">
<button id="next">Next</button> <button id="play">Play</button> <span id="step"></span></nav>
<section><h2>1. Ego panorama before the action</h2><img id="ego">
<details id="detailbox"><summary>Current supplementary VIEW pinhole image</summary><img id="detail"></details>
<h2 id="exoTitle">2. Exo geometry at the pre-action pose</h2>
<div id="realbox" hidden><img id="realexo"><p id="exoNote"></p></div>
<canvas id="exo" width="1100" height="460"></canvas>
<p class="note" id="mapnote">Blue: executor base footprint and heading. Red: predicted first blocking AABB. Orange: predicted rejected sample pose. Gray: scene bounds. Units: meters. This is not a contact measurement or camera image.</p>
</section>
<section><h2>Skills, parameters and outcomes</h2><div id="action"></div><p id="result"></p>
<p id="explanation"></p><details><summary>Offline attribution and evidence quality</summary><pre id="diag"></pre></details>
<details id="replaybox"><summary>v4.3 independent replay feedback and live geometry evidence</summary><pre id="replay"></pre></details>
<details><summary>Raw execution event (all fields retained)</summary><pre id="event"></pre></details></section>
<section><h2>Model-returned reasoning and plans</h2><div id="reasoning"></div>
<details open><summary>Full raw output (including returned think sections; unabridged)</summary><pre id="raw"></pre></details>
<details><summary>Full input log, image labels, tokens and latency; messages is authoritative when present</summary><pre id="call"></pre></details></section>
<section><h2>3. Ego panorama after the action</h2><img id="after"></section>
<details><summary>Logged model calls without matched execution events</summary><pre id="unmatched"></pre></details>
</main><script id="data" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
let at=0,timer=null;
$('title').textContent=D.model+' · '+D.task;
$('status').textContent=D.label+'; recorded '+D.steps.length+' steps. Version: '+D.version;
$('patterns').textContent='Longest consecutive rejected motion sequence: '+D.patterns.longest_consecutive_rejected_motion+
 ' steps; skill counts: '+JSON.stringify(D.patterns.skill_counts)+'. Behavioral statistics do not establish cognitive root causes.';
$('slider').max=Math.max(0,D.steps.length-1);
$('unmatched').textContent=JSON.stringify(D.unmatched_calls,null,2);
function dump(x){return JSON.stringify(x,null,2)}
function draw(r){
 const c=$('exo'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 const [x,y]=r.event.robot_pose_before, scale=80, ox=550,oy=230;
 const px=v=>ox+(v-x)*scale, py=v=>oy-(v-y)*scale;
 const hits=new Set((r.diagnostic?.blockers||[]).map(o=>o.entity));
 for(const o of D.obstacles){
  if(o.lo[2]>.5||o.hi[2]<.05)continue;
  g.fillStyle=hits.has(o.entity)?'#df6262':'#c3cbd4';
  g.fillRect(px(o.lo[0]),py(o.hi[1]),(o.hi[0]-o.lo[0])*scale,(o.hi[1]-o.lo[1])*scale);
  if(hits.has(o.entity)){g.fillStyle='#8c1626';g.font='14px sans-serif';g.fillText(o.entity,px(o.lo[0]),py(o.hi[1])-7);}
 }
 function body(a,b,t,color){
  g.save();g.translate(px(a),py(b));g.rotate(-t);g.strokeStyle=color;g.lineWidth=3;
  g.strokeRect(-32,-32,64,64);g.beginPath();g.moveTo(0,0);g.lineTo(52,0);
  g.lineTo(41,-7);g.moveTo(52,0);g.lineTo(41,7);g.stroke();g.restore();
 }
 body(x,y,r.yaw_before,'#1467c9');
 if(r.diagnostic?.sample_pose)body(...r.diagnostic.sample_pose,'#e39816');
 g.fillStyle='#172338';g.font='16px sans-serif';g.fillText('OFFLINE EXTERNAL GEOMETRY — not a camera observation',16,24);
 g.fillText(`base (${x.toFixed(3)}, ${y.toFixed(3)}) m | yaw ${(r.yaw_before*180/Math.PI).toFixed(1)}°`,16,445);
}
function show(){
 if(!D.steps.length){$('step').textContent='Waiting for this model to produce a trajectory';return;}
 const r=D.steps[at];$('slider').value=at;$('step').textContent=`Step ${r.step} / ${D.steps.length}`;
 $('ego').src=r.panorama||'';$('after').src=r.panorama_after||'';
 $('detailbox').hidden=!r.detail;if(r.detail)$('detail').src=r.detail;
 $('realbox').hidden=!r.exo_rgb;if(r.exo_rgb)$('realexo').src=r.exo_rgb;
 $('exoTitle').textContent=r.exo_rgb?(r.exo_synchronous?'2. Exo synchronized follow camera':'2. Exo camera: independent replay'):'2. Exo geometry illustration (not a camera image)';
 $('exoNote').textContent=r.exo_synchronous?'Recorded synchronously in this run; not supplied to the model.':'Independent diagnostic replay, not simultaneous recording of the original evaluation; not supplied to the model.';
 $('exo').hidden=!!r.exo_rgb;$('mapnote').hidden=!!r.exo_rgb;
 $('replaybox').hidden=!r.replay_diagnostic;$('replay').textContent=dump(r.replay_diagnostic);
 $('action').textContent=dump(r.event.raw_action);
 $('result').textContent=`${r.event.feedback} · executed=${r.event.executed} · ${r.event.episode_status}`;
 $('result').className=r.event.executed?'good':'bad';
 $('explanation').textContent=(r.feedback_was_supplied?'Feedback actually returned to the model: ':'Offline explanation for human review only: ')+r.human_explanation;
 $('diag').textContent=dump(r.diagnostic||{note:'Insufficient geometry evidence; collision objects and reach distances remain unknown.'});
 $('event').textContent=dump(r.event);
 $('raw').textContent=r.model_call.text||'No raw model output was recorded';
 $('call').textContent=dump(r.model_call);
 $('reasoning').replaceChildren();
 for(const [key,name] of Object.entries({visible_state:'Visual assessment',mid_level_progress_analysis:'Progress assessment',
  mid_level_goal:'Intermediate goal',low_level_action_reasoning:'Action rationale'})){
  const h=document.createElement('h3'),p=document.createElement('pre');
  h.textContent=name;p.textContent=r.decision.planner?.[key]??'Field absent or could not be parsed';
  $('reasoning').append(h,p);
 }
 draw(r);
}
$('prev').onclick=()=>{at=Math.max(0,at-1);show()};
$('next').onclick=()=>{at=Math.min(D.steps.length-1,at+1);show()};
$('slider').oninput=e=>{at=Number(e.target.value);show()};
$('play').onclick=()=>{
 if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}
 else{timer=setInterval(()=>{if(at>=D.steps.length-1){clearInterval(timer);timer=null;return;}at++;show()},1500);$('play').textContent='Pause'}
};show();
</script></html>
"""


def publish(suite, out):
    out.mkdir(parents=True, exist_ok=True)
    state = read(suite / "suite.json", {})
    probe = read(ROOT / "runs/vla_difficulty_v1/_probe/scene_probe.json")
    rows = []
    models = list(state.get("requested_models", []))
    sources = [(m, suite / m / "seed_0" / TASK, state["models"].get(m, {}), "") for m in models]
    # A completed, same-protocol example stays separate from the active cohort.
    sources.append(("qwen3-vl-4b-reference", ROOT / "runs/harness_v42_ondemand_pilot/qwen3-vl-4b/seed_0" / TASK,
                    {"status": "complete"}, "Earlier independent run with the same protocol; excluded from this cohort score"))
    instrumented = ROOT / "runs/harness_v43_instrumented_qwen_verified"
    if (instrumented / "report.json").exists():
        ir = read(instrumented / "report.json")
        sources.append(("qwen3-vl-4b-v43", instrumented, {"status":ir.get("status")},
                        "Independent experiment with a new protocol; not pooled with v4.2"))
    for name, episode, status, note in sources:
        steps = normalize(episode, out, probe)
        report = read(episode / "report.json", {})
        label = status.get("status", "pending") + (" · " + note if note else "")
        if state.get("status") == "interrupted" and status.get("status") in ("evaluating", "loading"):
            label = "interrupted · Manually interrupted; episode incomplete"
        if state.get("status") == "interrupted" and status.get("status") == "pending":
            label = "not_run · Queue stopped; not run"
        if report.get("summary"):
            label += " · " + str(report["summary"].get("status"))
        exo_manifest = read(out / "exo_replay/manifest.json", {})
        replay_events = {e["planning_step"]: e for e in lines(out / "exo_replay/replay_events.jsonl")
                         if e.get("event_type") == "action"}
        for step in steps:
            sync_exo = episode / "exo" / f"step_{step['input_step']:03d}_exo.png"
            if sync_exo.exists():
                step["exo_rgb"] = relative(sync_exo, out)
                step["exo_synchronous"] = True
            exo = out / "exo_replay" / f"step_{step['input_step']:03d}_exo.png"
            if name == "qwen3-vl-4b-reference" and exo.exists() and exo_manifest.get("camera_usable",True):
                step["exo_rgb"] = relative(exo, out)
                step["exo_replay_manifest"] = exo_manifest
                step["replay_diagnostic"] = replay_events.get(step["step"])
        action_steps = {r["input_step"] for r in steps}
        unmatched_calls = [c for c in lines(episode / "model_calls.jsonl") if c["planning_step"] not in action_steps]
        payload = dict(model=name, task=TASK, label=label,
                       version=report.get("prompt_version", state.get("prompt_version")),
                       report=report, source=str(episode), steps=steps, obstacles=probe["obstacles"],
                       unmatched_calls=unmatched_calls, patterns=patterns(steps))
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        write(out / f"{name}.json", serialized + "\n")
        page = VIEWER.replace("STYLE", STYLE).replace("MODEL", html.escape(name)).replace(
            "PAYLOAD", serialized.replace("<", "\\u003c"))
        write(out / f"{name}.html", page)
        counts = dict(Counter(r["event"].get("feedback") for r in steps))
        rows.append(dict(model=name, label=label, steps=len(steps), feedback=counts))
        md = [f"# {name} · {TASK}", label, f"Source: `{episode}`", ""]
        for r in steps:
            md += [f"## Step {r['step']}", "### Full model output", "````text",
                   r["model_call"].get("text", "Not recorded"), "````", "### Skill and parameters",
                   "```json", json.dumps(r["event"]["raw_action"], ensure_ascii=False), "```",
                   f"Execution outcome: {r['event']['feedback']}; executed={r['event']['executed']}",
                   f"Explanation ({'actually returned' if r['feedback_was_supplied'] else 'offline'}): {r['human_explanation']}", "### Raw execution event",
                   "```json", json.dumps(r["event"], ensure_ascii=False, indent=2), "```", ""]
        write(out / f"{name}.md", "\n".join(md))
    table = "".join(f'<tr><td><a href="{r["model"]}.html">{r["model"]}</a></td>'
                    f'<td>{html.escape(r["label"])}</td><td>{r["steps"]}</td>'
                    f'<td>{html.escape(str(r["feedback"]))}</td></tr>' for r in rows)
    stamp = datetime.now(timezone.utc).isoformat()
    index = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>FindingBench experiment audit</title>
<style>{STYLE}</style><main><header><h1>FindingBench · Harness and trajectories</h1>
<p>Snapshot UTC: {stamp} · Queue: {html.escape(state.get('status','unknown'))}</p>
<p>One trajectory per model for this task; models not run are explicitly marked. Refresh the page after the generator updates the files.</p>
<p><a href="../../docs/harness_experiment_disclosure_v42.md">Full experiment disclosure</a> ·
<a href="qwen3-vl-4b-reference.html">Completed v4.2 Qwen reference</a> ·
<a href="qwen3-vl-4b-v43.html">v4.3 independent experiment trajectory</a></p>
<p><a href="../harness_v43_instrumented_qwen_verified/media/episode.mp4">Experiment video</a> ·
<a href="../harness_v43_instrumented_qwen_verified/media/episode_attention.mp4">True attention video</a>
(Preview after five steps; updated when the experiment ends)</p>
<p>Reasoning displays contain only model-returned text. Missing actions or outputs are never reconstructed as success or hidden reasoning.</p></header>
<section><table><tr><th>Model</th><th>Status</th><th>Recorded steps</th><th>Outcome counts</th></tr>{table}</table></section>
<section><h2>Attribution evidence boundaries</h2><p>Original feedback, private execution reasons, and offline geometry reconstructions are kept separate.
Historical probes use L2_S1; the current task is L1_S1. AABB coordinates were rounded to millimeters. Reconstruction agreement is not evidence of physical contact.
Out of reach may involve pose, joint limits, or IK search, not just distance. Unrecorded subcauses remain unknown.</p>
<p>The historical cohort received four feedback classes. Revised v4.3 natural-language feedback uses a separate protocol and was not injected into old trajectories.</p></section></main></html>"""
    write(out / "index.html", index)
    write(out / "review_summary.json", json.dumps(dict(updated_utc=stamp, suite=state, rows=rows), ensure_ascii=False, indent=2))
    instrumented_state = read(instrumented / "run_state.json", {})
    return state.get("status") == "running" or instrumented_state.get("status") in ("loading", "evaluating")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    while True:
        running = publish(a.suite, a.out)
        if not a.watch or not running:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
