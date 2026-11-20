"""Export every completed step, then assemble full videos when the run ends."""
import argparse
import json
from pathlib import Path
import time
from compose_instrumented_video import compose, rows

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("run",type=Path)
a=p.parse_args();run=a.run.resolve()
while True:
    path=run/"run_state.json"
    try:state=json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        time.sleep(2);continue
    events=[r for r in rows(run/"episode/events.jsonl") if r.get("event_type")=="action"]
    exported=[]
    for event in events:
        step=event["planning_step"]
        out=run/"media"/f"step_{step:03d}"
        if not (out/"video_manifest.json").exists():
            report=compose(run,only_steps={step},output=out)
            if not report.get("steps"):break
            print(f"Exported step {step}",flush=True)
        exported.append(step)
        # Refresh the live index after EVERY exported step, including a backlog.
        media=run/"media"
        content='<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>Live step exports</title><h1>Live step exports</h1><p>Each step includes all returned reasoning, action, outcome, video, and separate action-token attention maps.</p><ol>'
        for n in exported:
            attention_link=(f' | <a href="step_{n:03d}/episode_attention.mp4">Attention video</a>' if (media/f"step_{n:03d}"/"episode_attention.mp4").exists() else " | True attention unavailable")
            content+=f'<li>Step {n}: <a href="step_{n:03d}/index.html">Full detail and token maps</a> | <a href="step_{n:03d}/episode.mp4">Video</a>{attention_link}</li>'
        content+='</ol>'
        tmp=media/"live.html.tmp";tmp.write_text(content);tmp.replace(media/"live.html")
    if state.get("status") in ("complete","error","interrupted"):
        compose(run)
        break
    time.sleep(2)
