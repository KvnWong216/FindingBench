"""Read-only corpus audit and local gallery; never promotes benchmark data."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import html
import json
from pathlib import Path
import re

from _common import REPO
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json,sha256_bytes,sha256_path
from rummagebench.sim.omnigibson.behavior_adapter import load_manifest


def verified_process(folder, record):
    proc=json.loads((folder/'process.json').read_text())
    if (proc['scenario_sha256']!=record['scenario_sha256'] or proc['assembly_sha256']!=record['assembly_sha256']
            or proc['result_sha256']!=sha256_path(folder/'result.json')):
        raise ValueError('AUDIT_GPU_BINDING_CHANGED:'+record['id'])
    result=json.loads((folder/'result.json').read_text())
    if proc['status']=='execution_smoke_passed' and (proc['returncode']!=0 or result.get('status')!=proc['status']
            or result.get('scenario_sha256')!=record['scenario_sha256']
            or result.get('assembly_sha256')!=record['assembly_sha256']):
        raise ValueError('AUDIT_GPU_SUCCESS_NOT_PROVEN:'+record['id'])
    return proc


def report(root):
    root=root.resolve()
    summary=json.loads((root/'summary.json').read_text())
    records=json.loads((root/'records.json').read_text())
    selected=set(summary['selected_ids'])
    houses,targets,assets,positions,slots=Counter(),Counter(),Counter(),set(),set()
    cpu_failures=Counter(); failed_surfaces=Counter(); gpu=Counter(); images=[]; rows=[]; realizations=[]
    attempted_gpu=set()
    for r in records:
        if r['status']=='generation_failed':
            logs=sorted((root/r['id']).glob('attempt_*/solver.log'))
            detail=logs[-1].read_text(errors='replace') if logs else ''
            matches=re.findall(r"violations=(\{[^\n]+\})",detail)
            if matches:
                cpu_failures['SOLVER_CONSTRAINTS_UNSATISFIED']+=1
                failed_surfaces.update(ast.literal_eval(matches[-1]).keys())
            else:cpu_failures[r.get('reason','generation_failed')]+=1
        if r['id'] not in selected:continue
        m=load_manifest(Path(r['assembly']))
        if m['manifest_sha256']!=r['assembly_sha256'] or sha256_path(r['scenario'])!=r['scenario_sha256']:
            raise ValueError('AUDIT_ARTIFACT_CHANGED:'+r['id'])
        import yaml
        s=yaml.safe_load(Path(r['scenario']).read_text())
        target=s['objects'][0]; target_key=f"{target['category']}/{target['model']}"
        houses[r['scene']]+=1;targets[target_key]+=1;slots.add((r['scene'],r['target_slot']))
        pose_rows=[]
        for item in m['objects']:
            p=item['placement'];key=f"{p['category']}/{p['model_id']}";assets[key]+=1
            pose_rows.append([key,p['support_name'],[round(v/.02) for v in p['position']]])
        positions.add(sha256_bytes(canonical_json([r['scene'],sorted(pose_rows)])))
        folder=root/r['id']/'gpu';status='pending_gpu';detail=''
        if (folder/'process.json').exists():
            proc=verified_process(folder,r)
            attempted_gpu.add(r['id'])
            status=proc['status'];detail=proc.get('error') or str(proc.get('failures',{}))
        for variant in sorted((root/r['id']).glob('gpu_realized*')):
            if not (variant/'process.json').exists():continue
            proc=verified_process(variant,r)
            attempted_gpu.add(r['id'])
            stages={}
            for name in ['background_realization','background_after_target']:
                path=variant/f'{name}.json'
                if path.exists():
                    d=json.loads(path.read_text())
                    if (d.get('assembly_sha256')!=r['assembly_sha256']
                            or proc.get('stage_sha256',{}).get(name)!=sha256_path(path)):
                        raise ValueError('AUDIT_REALIZATION_BINDING_CHANGED:'+r['id'])
                    stages[name]=d['status']
            realizations.append(dict(id=r['id'],variant=variant.name,status=proc['status'],
                stages=stages,error=proc.get('error'),receipt=str((variant/'process.json').relative_to(root))))
            variant_image=variant/'scene_start.png'
            if variant_image.exists():
                images.append(dict(id=f"{r['id']}/{variant.name}",scene=r['scene'],status=proc['status'],
                                   image=str(variant_image.relative_to(root))))
        gpu[status]+=1
        image=folder/'scene_start.png'
        if image.exists():images.append(dict(id=r['id'],scene=r['scene'],status=status,image=str(image.relative_to(root))))
        rows.append(dict(id=r['id'],scene=r['scene'],target=target_key,objects=len(m['objects']),status=status,
                         scenario=str(Path(r['scenario']).relative_to(root)),detail=detail))
    audit=dict(generated=len(rows),distinct_pose_asset_backgrounds_without_orientation=len(positions),
        original_house_templates=len(houses),generated_new_floorplans=0,
        by_house=dict(houses),target_models=dict(targets),daily_asset_instances=dict(assets),
        distinct_target_slots=len(slots),cpu_failure_classes=dict(cpu_failures),
        unsatisfied_surface_constraints=dict(failed_surfaces),gpu_status=dict(gpu),
        evaluation_ready_episodes=0,split='development',images=images,
        optional_realization_diagnostics=realizations,
        distinct_gpu_attempted_candidates=len(attempted_gpu),pending_any_gpu=len(rows)-len(attempted_gpu),
        scope='CPU-verified native-home proposals; sampled known-target execution checks only')
    atomic_write_json(root/'audit.json',audit)
    esc=html.escape
    cards=''.join(f'<figure><img loading="lazy" src="{esc(i["image"])}"><figcaption>{esc(i["id"])} · {esc(i["scene"])} · {esc(i["status"])}</figcaption></figure>' for i in images)
    table=''.join('<tr>'+''.join(f'<td>{esc(str(r[k]))}</td>' for k in ['id','scene','target','objects','status','detail'])+
                  f'<td><a href="{esc(r["scenario"])}">scenario</a></td></tr>' for r in rows)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>FindingBench batch audit</title>
<style>body{{font:15px system-ui;margin:32px;color:#20252c}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;width:100%}}code{{background:#eee}}</style>
<h1>FindingBench：{len(rows)} 条候选数据审计</h1>
<p>{len(houses)} 套原生房屋；{len(positions)} 种不同资产与位置组合（忽略朝向仍不同）；新增房型 0；正式验收 0。</p>
<p>InfiniGen 求解日用品布置。墙体和大件家具来自原生房屋。下列图片均为真实 OmniGibson 输出；失败状态不表示通过验收。</p>
<p>评估端私有审计视图：scenario 含隐藏任务配置，不应直接作为模型输入。</p>
<p>原始分层抽查状态：</p><pre>{esc(json.dumps(dict(gpu),ensure_ascii=False,indent=2))}</pre><div class="grid">{cards}</div>
<h2>独立摆放修复诊断</h2><p>同一候选的附加诊断，不增加生成数量，不替换原始失败记录。</p>
<p>合计尝试 GPU 验证的不同候选：{len(attempted_gpu)}；尚未进行任何 GPU 验证：{len(rows)-len(attempted_gpu)}。</p>
<pre>{esc(json.dumps(realizations,ensure_ascii=False,indent=2))}</pre>
<h2>全部候选</h2><table><tr><th>ID</th><th>原生房屋</th><th>目标资产</th><th>新增用品</th><th>GPU 状态</th><th>诊断</th><th>配置</th></tr>{table}</table></html>'''
    (root/'index.html').write_text(page)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--batch',type=Path,required=True)
    report(p.parse_args().batch)
