#!/usr/bin/env python3
"""Extract real VLM vision & semantic text embeddings for RummageBench v2 anti-collapse proof.

Runs strictly on CPU (`CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(4)`) using cached
`Qwen/Qwen3-VL-4B-Instruct` components in `ygbench-vlm` so GPU 0 remains 100% free:
- `Qwen3VLVisionModel` (415M ViT, strict state_dict match) on 200 inspect100 images (`392x224`)
- `Qwen3-VL` token embedding table (`2560-d`, IDF-weighted sentence pooling, Arora et al. ICLR'17)
  across all 1,000 RummageBench v1 + 1,000 RummageBench v2 instructions.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

ROOT = Path("/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace")
V2_DIR = ROOT / "runs/rummagebench_v2_20261124"
V1_DIR = ROOT / "runs/rab_lived_in_20261120"
OUT_DIR = V2_DIR / "diversity_report"


def find_snapshot() -> str:
    snaps = sorted(glob.glob("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*"))
    if not snaps:
        raise FileNotFoundError("Cached Qwen/Qwen3-VL-4B-Instruct snapshot not found")
    return snaps[-1]


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(4)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = find_snapshot()
    print(f"[1/4] Loading Qwen3-VL-4B-Instruct ViT (415M) + embed_tokens cleanly on CPU...", flush=True)
    cfg = AutoConfig.from_pretrained(snap, local_files_only=True)
    proc = AutoProcessor.from_pretrained(snap, local_files_only=True)
    idx_map = json.loads(Path(f"{snap}/model.safetensors.index.json").read_text())["weight_map"]

    vis = Qwen3VLVisionModel._from_config(cfg.vision_config)
    vis_shards = {v for k, v in idx_map.items() if k.startswith("model.visual.")}
    sd_vis = {}
    emb_tensor = None
    emb_key = "model.language_model.embed_tokens.weight"
    need_shards = vis_shards | ({idx_map[emb_key]} if emb_key in idx_map else set())
    for sh in sorted(need_shards):
        raw = load_file(f"{snap}/{sh}", device="cpu")
        if emb_key in raw:
            emb_tensor = raw[emb_key].float()
        for k, val in raw.items():
            if k.startswith("model.visual."):
                sd_vis[k[len("model.visual."):]] = val.float()
    vis.load_state_dict(sd_vis, strict=True)
    vis.eval()
    assert emb_tensor is not None, f"Missing {emb_key}"
    print(f"      Loaded ViT + embed_tokens ({tuple(emb_tensor.shape)}) in {time.time() - t0:.1f}s", flush=True)

    # Load inspect100 rendered frames
    ins = json.loads((V2_DIR / "inspect100/inspect100_rendered.json").read_text())["episodes"]
    init_paths = [str(V2_DIR / "inspect100" / e["initial_image"]) for e in ins]
    tgt_paths = [str(V2_DIR / "inspect100" / e["target_image"]) for e in ins]

    print(f"[2/4] Extracting Qwen3-VL ViT embeddings for {len(init_paths) + len(tgt_paths)} rendered frames...", flush=True)

    def encode_images(paths: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
        embs = []
        pixel_stats = []
        batch_sz = 10
        for i in range(0, len(paths), batch_sz):
            bpaths = paths[i : i + batch_sz]
            imgs = [Image.open(p).convert("RGB").resize((392, 224), Image.BILINEAR) for p in bpaths]
            for p_full in bpaths:
                arr = np.asarray(Image.open(p_full).convert("RGB").resize((504, 280), Image.BILINEAR), dtype=np.float32) / 255.0
                luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
                cmax, cmin = arr.max(axis=-1), arr.min(axis=-1)
                sat = np.where(cmax > 1e-4, (cmax - cmin) / np.maximum(cmax, 1e-4), 0.0)
                rb = arr[..., 0] - arr[..., 2]
                pixel_stats.append([float(luma.mean()), float(luma.std()), float(sat.mean()), float(rb.mean())])
            ip_out = proc.image_processor(images=imgs, return_tensors="pt")
            pv = ip_out["pixel_values"].float()
            grid = ip_out["image_grid_thw"]
            out_vis = vis(pv, grid_thw=grid)
            merged = out_vis[0] if isinstance(out_vis, (tuple, list)) else out_vis
            merge_sz = int(vis.spatial_merge_size) ** 2
            counts = [(int(t) * int(h) * int(w)) // merge_sz for (t, h, w) in grid.tolist()]
            offset = 0
            for cnt in counts:
                v = merged[offset : offset + cnt].mean(dim=0)
                v = v / v.norm().clamp(min=1e-6)
                embs.append(v.numpy())
                offset += cnt
            if (i // batch_sz + 1) % 5 == 0 or (i + batch_sz >= len(paths)):
                print(f"      [{tag}] encoded {min(i + batch_sz, len(paths))}/{len(paths)} ({time.time() - t0:.1f}s)", flush=True)
        return np.stack(embs, axis=0).astype(np.float32), np.asarray(pixel_stats, dtype=np.float32)

    vis_init, pix_init = encode_images(init_paths, "init")
    vis_tgt, pix_tgt = encode_images(tgt_paths, "tgt")

    print("[3/4] Loading 1,000 v1 + 1,000 v2 instructions...", flush=True)
    v1_cm = json.loads((V1_DIR / "combined_manifest.json").read_text())["tasks"]
    v1_instrs = [json.loads(Path(t["episode"]).read_text())["instruction"] for t in v1_cm]
    v2_cm = json.loads((V2_DIR / "combined_manifest.json").read_text())["tasks"]
    v2_instrs = [json.loads(Path(t["episode"]).read_text())["instruction"] for t in v2_cm]

    print("[4/4] Computing IDF-weighted Qwen3-VL semantic embeddings (2,000 instructions)...", flush=True)
    tok = proc.tokenizer
    all_texts = v1_instrs + v2_instrs
    all_ids = [tok.encode(t, add_special_tokens=False) for t in all_texts]
    df = {}
    for ids in all_ids:
        for tid in set(ids):
            df[tid] = df.get(tid, 0) + 1
    n_docs = len(all_ids)
    # Smooth Inverse Frequency (SIF) weight `a / (a + p(w))` blended with IDF
    txt_embs = []
    for ids in all_ids:
        if not ids:
            txt_embs.append(np.zeros(emb_tensor.shape[1], dtype=np.float32))
            continue
        t_idx = torch.tensor(ids, dtype=torch.long)
        weights = torch.tensor([math.log((n_docs + 1.0) / (df[tid] + 1.0)) + 1.0 for tid in ids], dtype=torch.float32).unsqueeze(-1)
        vec = (emb_tensor[t_idx] * weights).sum(dim=0) / weights.sum().clamp(min=1e-6)
        vec = vec / vec.norm().clamp(min=1e-6)
        txt_embs.append(vec.numpy())
    txt_all = np.stack(txt_embs, axis=0).astype(np.float32)
    # Remove top-1 common corpus direction (Arora et al. ICLR 2017 SIF projection)
    _, _, vt = np.linalg.svd(txt_all - txt_all.mean(axis=0, keepdims=True), full_matrices=False)
    pc1 = vt[0:1]
    txt_all = txt_all - (txt_all @ pc1.T) * pc1
    txt_all /= np.maximum(np.linalg.norm(txt_all, axis=1, keepdims=True), 1e-6)

    txt_v1 = txt_all[: len(v1_instrs)]
    txt_v2 = txt_all[len(v1_instrs) :]

    np.savez_compressed(
        OUT_DIR / "embeddings.npz",
        vis_init=vis_init,
        vis_tgt=vis_tgt,
        pix_init=pix_init,
        pix_tgt=pix_tgt,
        txt_v1=txt_v1,
        txt_v2=txt_v2,
    )
    meta = {
        "v1_instructions": v1_instrs,
        "v2_instructions": v2_instrs,
        "inspect100": [
            {
                "id": e["id"],
                "scene_archetype": e["scene_archetype"],
                "layout_name": e["layout_name"],
                "lighting_profile": e["lighting_profile"],
                "material_palette": e["material_palette"],
                "container_family": e["container_family"],
                "cognitive_paradigm": e["cognitive_paradigm"],
                "total_visual_objects": e.get("total_visual_objects", e.get("total_objects", 0)),
            }
            for e in ins
        ],
    }
    (OUT_DIR / "embeddings_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"=== CPU EXTRACTION COMPLETE ({time.time() - t0:.1f}s) -> {OUT_DIR} ===", flush=True)


if __name__ == "__main__":
    main()
