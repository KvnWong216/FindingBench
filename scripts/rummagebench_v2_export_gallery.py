#!/usr/bin/env python3
"""Export interactive HTML gallery + contact-sheet grids + full audit summary
for the 10% (100-episode) RummageBench v2 inspection set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

ROOT = Path("/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace")


def build_contact_grid(items: list[dict], base_dir: Path, out_path: Path, cols: int = 3):
    """Build a crisp side-by-side (Initial | Target) contact sheet for visual overview."""
    pair_w, pair_h = 560, 178
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * pair_w + (cols + 1) * 10, rows * pair_h + (rows + 1) * 10), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    for idx, rec in enumerate(items):
        r, c = divmod(idx, cols)
        x0 = 10 + c * (pair_w + 10)
        y0 = 10 + r * (pair_h + 10)
        draw.rectangle([x0, y0, x0 + pair_w, y0 + pair_h], fill=(30, 41, 59), outline=(71, 85, 105), width=1)

        im_i_p = base_dir / rec["initial_image"]
        im_t_p = base_dir / rec["target_image"]
        if im_i_p.exists() and im_t_p.exists():
            im_i = Image.open(im_i_p).resize((272, 150), Image.BILINEAR)
            im_t = Image.open(im_t_p).resize((272, 150), Image.BILINEAR)
            canvas.paste(im_i, (x0 + 5, y0 + 24))
            canvas.paste(im_t, (x0 + 283, y0 + 24))
        hdr = f"{rec['id']} | {rec['scene_archetype'][:18]} | {rec['layout_name'][:10]} | {rec['search_fixture'][:18]}"
        draw.text((x0 + 6, y0 + 5), hdr[:76], fill=(226, 232, 240))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-dir", default=str(ROOT / "runs/rummagebench_v2_20261124/inspect100"))
    args = ap.parse_args()

    idir = Path(args.inspect_dir)
    data = json.loads((idir / "inspect100_rendered.json").read_text())
    eps = data["episodes"]

    # 1. Build overview contact grids per Scene Archetype + 18-episode Hero Montage
    by_arch = {}
    for e in eps:
        by_arch.setdefault(e["scene_archetype"], []).append(e)

    grids_dir = idir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    hero_picks = []
    for arch, lst in sorted(by_arch.items()):
        build_contact_grid(lst[:9], idir, grids_dir / f"grid_{arch}.jpg", cols=3)
        hero_picks.extend(lst[:3])
    build_contact_grid(hero_picks[:18], idir, grids_dir / "hero_18_diverse_archetypes.jpg", cols=3)

    # 2. Generate Interactive Filterable HTML Gallery (`index.html` with relative image paths)
    arch_counts = Counter(e["scene_archetype"] for e in eps)
    cfam_counts = Counter(e["container_family"] for e in eps)
    para_counts = Counter(e["cognitive_paradigm"] for e in eps)

    cards_html = []
    for idx, e in enumerate(eps, 1):
        eid = html.escape(e["id"])
        arch = html.escape(e["scene_archetype"])
        cfam = html.escape(e["container_family"])
        para = html.escape(e["cognitive_paradigm"])
        lname = html.escape(e["layout_name"])
        lprof = html.escape(e["lighting_profile"])
        mpal = html.escape(e["material_palette"])
        fix = html.escape(e["search_fixture"])
        fix_p = html.escape(e.get("search_fixture_phrase", fix))
        prior = html.escape(e["spatial_prior_mode"])
        inst = html.escape(e["instruction"])
        orig_inst = html.escape(e["original_gpt6_instruction"])
        orig_scene = html.escape(e["gpt6_original_scene"])
        tlabels = ", ".join(html.escape(x) for x in e.get("target_labels", e["target_assets"]))
        tc = e.get("settled_target_center_m", [0, 0, 0])
        dest_p = html.escape(e["delivery_phrase"])
        jts = ", ".join(e.get("opened_joints", [])) or "Open / Occluded Zone"

        cards_html.append(f"""
        <article class="card" data-arch="{arch}" data-cfam="{cfam}" data-para="{para}">
          <div class="card-head">
            <div>
              <span class="idx">#{idx:03d}</span>
              <strong>{eid}</strong>
              <span class="badge arch">{arch}</span>
              <span class="badge cfam">{cfam}</span>
              <span class="badge para">{para}</span>
              <span class="badge prior">{prior}</span>
            </div>
            <div class="meta-right">
              Layout: <code>{lname}</code> · Light: <code>{lprof}</code> · Palette: <code>{mpal}</code> · Objects: <strong>{e.get('total_visual_objects', e['total_objects'])}</strong> (dyn {e['total_objects']} + mess {e.get('static_lived_in_clutter_count', 0)})
            </div>
          </div>
          <div class="img-row">
            <figure>
              <img src="{e['initial_image']}" loading="lazy" alt="{eid} initial scene">
              <figcaption><strong>(1) 原始场景 (Initial Scene t=0 + Robot Ego)</strong> — Spawn: <code>{e['robot_spawn_zone']}</code> {e['robot_initial_xyyaw']}</figcaption>
            </figure>
            <figure>
              <img src="{e['target_image']}" loading="lazy" alt="{eid} target location">
              <figcaption><strong>(2) 目标位置 (GT Target Location)</strong> — Fixture: <code>{fix}</code> ({fix_p}) · Joint: <code>{jts}</code> · XYZ: <code>{tc}</code> · Target: <strong>{tlabels}</strong></figcaption>
            </figure>
          </div>
          <div class="task-box">
            <div class="inst-main"><strong>(3) 自然语言任务 (RummageBench Instruction):</strong> {inst}</div>
            <div class="inst-sub">
              <span><strong>Delivery Surface:</strong> {dest_p} (<code>{e['delivery_fixture']}</code>)</span> ·
              <span><strong>GPT-6 Source (<code>{e['parent_task_id']}</code>, {orig_scene}):</strong> <em>{orig_inst}</em></span>
            </div>
          </div>
        </article>
        """)

    arch_opts = "".join(f'<option value="{k}">{k} ({v})</option>' for k, v in sorted(arch_counts.items()))
    cfam_opts = "".join(f'<option value="{k}">{k} ({v})</option>' for k, v in sorted(cfam_counts.items()))
    para_opts = "".join(f'<option value="{k}">{k} ({v})</option>' for k, v in sorted(para_counts.items()))

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RummageBench v2 — 10% (100-Episode) Multi-Scene & Domain-Randomization Audit</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#0b1120; color:#e2e8f0; margin:0; padding:24px; }}
  h1 {{ margin:0 0 8px 0; font-size:24px; color:#f8fafc; }}
  .sub {{ color:#94a3b8; margin-bottom:18px; font-size:14px; line-height:1.5; }}
  .toolbar {{ position:sticky; top:0; z-index:20; background:#111827ee; backdrop-filter:blur(8px); border:1px solid #334155; border-radius:10px; padding:12px 16px; margin-bottom:20px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }}
  select, input {{ background:#1e293b; color:#f8fafc; border:1px solid #475569; border-radius:6px; padding:7px 10px; font-size:13px; }}
  .count-pill {{ margin-left:auto; background:#065f46; color:#d1fae5; padding:5px 12px; border-radius:999px; font-weight:600; font-size:13px; }}
  .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px; margin-bottom:18px; }}
  .card-head {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:12px; align-items:center; }}
  .idx {{ color:#38bdf8; font-family:monospace; font-weight:700; margin-right:6px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; margin-left:5px; }}
  .badge.arch {{ background:#1d4ed8; color:#dbeafe; }}
  .badge.cfam {{ background:#047857; color:#d1fae5; }}
  .badge.para {{ background:#6d28d9; color:#ede9fe; }}
  .badge.prior {{ background:#b45309; color:#fef3c7; }}
  .meta-right {{ font-size:12px; color:#cbd5e1; }}
  code {{ background:#0f172a; padding:2px 5px; border-radius:4px; color:#7dd3fc; font-size:12px; }}
  .img-row {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  figure {{ margin:0; background:#0f172a; border-radius:8px; overflow:hidden; border:1px solid #334155; }}
  figure img {{ width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }}
  figcaption {{ padding:8px 10px; font-size:12px; color:#cbd5e1; line-height:1.4; }}
  .task-box {{ margin-top:12px; background:#0f172a; border-left:4px solid #10b981; padding:10px 14px; border-radius:6px; }}
  .inst-main {{ font-size:14px; color:#f8fafc; line-height:1.5; margin-bottom:6px; }}
  .inst-sub {{ font-size:12px; color:#94a3b8; line-height:1.45; }}
</style>
</head>
<body>
  <h1>RummageBench v2 — 10% (100 / 1,000 Episodes) 原始场景 · 目标位置 · 自然语言任务 抽检面板</h1>
  <div class="sub">
    覆盖 <strong>6 大异构场景原型</strong>（独立商业超市多通道货架 <code>SM_01..10</code>、10+ 多柜体厨房与内置烤箱、卧室床头柜与胡桃木书房柜、三角钢琴客厅影音厅、玄关衣帽快递台、后备箱尾门货舱与五层车库货架）×
    <strong>60 个物理基础布局</strong> × <strong>12 套室内材质纹理</strong> × <strong>8 种色温光照域随机化</strong> × <strong>10 类容器遮挡机理 (单类 ≤11%)</strong> × <strong>6 类 GPT-6 认知翻找范式</strong>。
  </div>
  <div class="toolbar">
    <label>场景原型: <select id="fArch"><option value="">全部 (6 类)</option>{arch_opts}</select></label>
    <label>容器/遮挡家族: <select id="fCfam"><option value="">全部 (10 类)</option>{cfam_opts}</select></label>
    <label>认知范式: <select id="fPara"><option value="">全部 (6 类)</option>{para_opts}</select></label>
    <input id="fText" type="search" placeholder="搜索 episode ID / 柜体 / 物体 / 指令关键词..." style="min-width:250px;">
    <span class="count-pill" id="visCount">显示 100 / 100 条</span>
  </div>
  <div id="list">{''.join(cards_html)}</div>
<script>
  const fArch = document.getElementById('fArch');
  const fCfam = document.getElementById('fCfam');
  const fPara = document.getElementById('fPara');
  const fText = document.getElementById('fText');
  const cards = Array.from(document.querySelectorAll('.card'));
  const visCount = document.getElementById('visCount');
  function applyFilter() {{
    const a = fArch.value, c = fCfam.value, p = fPara.value, q = fText.value.trim().toLowerCase();
    let shown = 0;
    for (const el of cards) {{
      const ok = (!a || el.dataset.arch === a)
              && (!c || el.dataset.cfam === c)
              && (!p || el.dataset.para === p)
              && (!q || el.textContent.toLowerCase().includes(q));
      el.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }}
    visCount.textContent = `显示 ${{shown}} / ${{cards.length}} 条`;
  }}
  [fArch, fCfam, fPara, fText].forEach(el => el.addEventListener('input', applyFilter));
</script>
</body>
</html>
"""
    (idir / "index.html").write_text(page, encoding="utf-8")
    print("Exported interactive gallery:", idir / "index.html")
    print("Exported contact sheets:", sorted(p.name for p in grids_dir.glob("*.jpg")))


if __name__ == "__main__":
    main()
