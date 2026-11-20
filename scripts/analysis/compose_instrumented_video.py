"""Publish ego/exo/reasoning videos and measured image attention maps."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
W,H=1600,1440


def rows(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text().splitlines():
        try:result.append(json.loads(line))
        except ValueError:break
    return result


def wrap(text,width=75):
    result=[]
    for line in str(text).splitlines():
        result.extend(textwrap.wrap(line,width=width,replace_whitespace=False) or [""])
    return result


def attention_maps(call,pictures,output,step):
    meta=call.get("attention")
    if not meta or meta.get("aggregation") != "none":
        return {}, "No unaveraged decision-token attention available."
    queries=meta.get("selected_queries",[])
    if not queries:return {}, "No parsed action-value token available; no attention map fabricated."
    weights=np.load(meta["npz"])["image_attention"]
    merge=meta["spatial_merge_size"]
    primary=next((q for q in reversed(queries) if q["field"]=="skill"),queries[0])
    result={}; entries=[]
    from matplotlib import colormaps
    for q in queries:
        vector=weights[q["token_index"]]
        scale=max(float(vector.max()),1e-12)  # One scale shared by both images.
        offset=0
        for index,(t,h,w) in enumerate(meta["grids"]):
            if t != 1:raise ValueError("Static-image attention export requires one temporal slice")
            count=t*h*w//merge**2
            heat=vector[offset:offset+count].reshape(h//merge,w//merge)
            offset+=count
            original=pictures[index]
            normalized=np.clip(heat/scale,0,1)
            rgb=(colormaps["inferno"](normalized)[...,:3]*255).astype(np.uint8)
            color=Image.fromarray(rgb).resize(original.size,Image.Resampling.NEAREST)
            alpha=Image.fromarray((normalized*190).astype(np.uint8)).resize(original.size,Image.Resampling.NEAREST)
            overlay=Image.composite(color,original,alpha)
            stem=f"step_{step:03d}_{q['field']}_token_{q['token_index']:04d}_image_{index+1}"
            overlay.save(output/(stem+"_attention.png"))
            np.save(output/(stem+"_probabilities.npy"),heat)
            entries.append(dict(**q,image=index+1,image_mass=float(heat.sum()),
                display_scale_max_probability=scale,overlay=stem+"_attention.png",
                probabilities=stem+"_probabilities.npy"))
            if q is primary:result[index]=overlay
    (output/f"step_{step:03d}_attention_selection.json").write_text(json.dumps(entries,indent=2))
    label=(f"Predicting skill token {primary['token_text']!r} | L{primary['layer']} H{primary['head']} | "
           f"image mass={primary['image_attention_mass']:.4f} | no averaging; not causal attribution")
    return result,label


def mark_point(image,action,frame_id):
    image=image.copy()
    point=action.get("point")
    if point and point.get("frame_id")==frame_id:
        x=point["x"]*image.width;y=point["y"]*image.height
        d=ImageDraw.Draw(image)
        d.ellipse((x-13,y-13,x+13,y+13),outline="#00ff9d",width=4)
        d.line((x-22,y,x+22,y),fill="#00ff9d",width=3)
        d.line((x,y-22,x,y+22),fill="#00ff9d",width=3)
    return image


def frame(pano,exo,detail,lines,header,footer):
    page=Image.new("RGB",(W,H),"#121c2d")
    d=ImageDraw.Draw(page)
    small=ImageFont.truetype(FONT,18);title=ImageFont.truetype(FONT,21)
    d.text((12,5),header,font=title,fill="white")
    page.paste(pano.resize((1600,800)),(0,36))
    exo=exo.copy();exo.thumbnail((800,544))
    page.paste(exo,((800-exo.width)//2,872))
    d.text((12,840),"EXO · Synchronized follow camera · Not supplied to VLM",font=small,fill="#8fcedf")
    if detail is not None:
        thumb=detail.copy();thumb.thumbnail((300,225))
        page.paste(thumb,(485,1185))
        d.text((485,1161),"Current VIEW supplement",font=small,fill="white")
    d.rectangle((800,836,1599,1439),fill="#f3f5f8")
    for i,line in enumerate(lines):
        d.text((816,848+i*22),line,font=small,fill="#172338")
    d.text((12,1412),footer,font=small,fill="#e5b951")
    return page


def compose(run, only_steps=None, output=None):
    out=output or run/"media";out.mkdir(parents=True,exist_ok=True)
    calls={c["planning_step"]+1:c for c in rows(run/"model_calls.jsonl")}
    decisions={c["planning_step"]+1:c for c in rows(run/"agent_decisions.jsonl")}
    events=[r for r in rows(run/"episode/events.jsonl") if r.get("event_type")=="action"]
    normal_frames=[];heat_frames=[];audit=[]
    has_attention=any(c.get("attention") for c in calls.values())
    for e in events:
        step=e["planning_step"];c=calls.get(step)
        if only_steps is not None and step not in only_steps:continue
        ego_path=run/"episode"/f"step_{step-1:03d}.png"
        exo_path=run/"exo"/f"step_{step-1:03d}_exo.png"
        if not c or not ego_path.exists() or not exo_path.exists():continue
        pano=Image.open(ego_path).convert("RGB")
        exo=Image.open(exo_path).convert("RGB")
        detail_path=ego_path.with_name(ego_path.stem+"_detail.png")
        detail=Image.open(detail_path).convert("RGB") if detail_path.exists() else None
        context=json.loads(c["prompt"].split("\nInput:\n")[-1])["observation"]
        previous=context.get("feedback")
        prev_text=previous["message"] if previous else "Initial observation; no previous action."
        plan=decisions.get(step,{}).get("planner",{})
        action=e["raw_action"]
        text=["Previous outcome:",prev_text]
        think=re.findall(r"<think>(.*?)</think>",c.get("text",""),re.S)
        for t in think:text += ["Model-returned reasoning:",t]
        for key,title in [("visible_state","Visual assessment"),("mid_level_progress_analysis","Progress assessment"),
                          ("mid_level_goal","Short-term goal"),("low_level_action_reasoning","Action rationale")]:
            if key in plan:text += [title+":",str(plan[key])]
        if not plan:text+=["Parsing failed; raw output:",c.get("text","")]
        text+=["Selected skill and parameters:",json.dumps(action,ensure_ascii=False),
               "Execution outcome:",e.get("public_feedback_message",e["feedback"])]
        wrapped=wrap("\n".join(text))
        chunks=[wrapped[i:i+25] for i in range(0,len(wrapped),25)]
        heat,heat_label=attention_maps(c,[pano]+([detail] if detail else []),out,step)
        fid=context["frame_id"]
        main=mark_point(pano,action,fid)
        heatmain=mark_point(heat.get(0,pano),action,fid)
        heatdetail=heat.get(1,detail)
        if detail:
            heatdetail=mark_point(heatdetail,action,context["detail_view"]["frame_id"])
            detail=mark_point(detail,action,context["detail_view"]["frame_id"])
        for part,chunk in enumerate(chunks):
            number=len(normal_frames)
            title=f"STEP {step} · EGO PANORAMA / action input · page {part+1}/{len(chunks)}"
            footer=f"Skill: {action.get('skill','INVALID')}  |  {e['feedback']}  |  point marker = actual submitted point only"
            image=frame(main,exo,detail,chunk,title,footer)
            path=out/f"frame_{number:04d}.jpg";image.save(path,quality=90);normal_frames.append(path)
            heat_footer=heat_label
            image=frame(heatmain,exo,heatdetail,chunk,title+" · ATTENTION",heat_footer)
            path=out/f"attention_frame_{number:04d}.jpg";image.save(path,quality=90);heat_frames.append(path)
        audit.append(dict(step=step,pages=len(chunks),action=action,feedback=e["feedback"],
                          input_frame_id=fid,attention=c.get("attention"),
                          original_output=c.get("text"),text_lines=wrapped))
    if not normal_frames:return {"steps":0}
    ffmpeg=Path("/home/ubuntu/miniforge3/envs/ygbench/bin/ffmpeg")
    for prefix,name in ([("frame","episode"),("attention_frame","episode_attention")] if has_attention else [("frame","episode")]):
        tmp=out/f"{name}.tmp.mp4"
        subprocess.run([str(ffmpeg),"-y","-loglevel","error","-framerate","1/6","-i",
                        str(out/f"{prefix}_%04d.jpg"),"-t",str(len(normal_frames)*6),
                        "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
                        "-threads","2","-r","12","-movflags","+faststart",str(tmp)],check=True)
        tmp.replace(out/f"{name}.mp4")
    report=dict(steps=len(audit),pages=len(normal_frames),seconds_per_page=6,
                note="All returned reasoning is paginated, not silently truncated. Playback time is not simulator time.",
                attention_available=has_attention,
                attention_note="True model attention unavailable for this run." if not has_attention else "Single predictive token / single head; maximum visual mass across layers and heads. No averaging. Raw probabilities and selection metadata retained. Not causal saliency.",
                rows=audit)
    import html
    panels=[]
    for row in audit:
        step=row["step"]
        maps=sorted(out.glob(f"step_{step:03d}_*_attention.png"))
        panels.append(f"<h2>Step {step}</h2><pre>{html.escape(chr(10).join(row['text_lines']))}</pre>"+
            ''.join(f'<figure><img loading="lazy" src="{im.name}"><figcaption>{html.escape(im.stem)}</figcaption></figure>' for im in maps))
    (out/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Step export</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>Step exports</h1><p>Individual action-value tokens, single selected head; no averaging. '
        'Selection: largest image attention mass, not verified causal importance. '
        'Layer/head indices and raw masses are in attention_selection JSON files.</p>'+''.join(panels)+'</html>')
    (out/"video_manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path)
    a=p.parse_args();d=compose(a.run.resolve());print(json.dumps({k:v for k,v in d.items() if k!="rows"},ensure_ascii=False))
