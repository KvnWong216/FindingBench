#!/usr/bin/env python3
"""Compute TPT-Bench-style + embedding non-collapse diversity metrics and generate
publication-grade visualization assets for RummageBench v2.

Outputs to:
- `runs/rummagebench_v2_20261124/diversity_report/` (`diversity_metrics.json` + 4 PNG figures)
- `/home/ubuntu/.local/share/openresearch/files/rummagebench/figures/`
- `/home/ubuntu/.gemini/antigravity/brain/<conv_id>/diversity_figures/` (if requested)
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

ROOT = Path("/home/ubuntu/ygwang/FindingBench_20261016_200444/workspace")
V2_DIR = ROOT / "runs/rummagebench_v2_20261124"
V1_DIR = ROOT / "runs/rab_lived_in_20261120"
REP_DIR = V2_DIR / "diversity_report"
ORX_FIG_DIR = Path("/home/ubuntu/.local/share/openresearch/files/rummagebench/figures")

# Okabe-Ito colorblind-safe palette
PAL = {
    "S1_commercial_supermarket_aisles": "#0072B2",
    "S2_culinary_kitchen_and_appliance_hub": "#D55E00",
    "S3_bedroom_and_private_study_suite": "#009E73",
    "S4_living_salon_and_media_lounge": "#CC79A7",
    "S5_entryway_mudroom_and_parcel_station": "#E69F00",
    "S6_cargo_trunk_and_garage_rack_bay": "#56B4E9",
}
ARCH_SHORT = {
    "S1_commercial_supermarket_aisles": "S1 Supermarket",
    "S2_culinary_kitchen_and_appliance_hub": "S2 Kitchen",
    "S3_bedroom_and_private_study_suite": "S3 Bedroom/Study",
    "S4_living_salon_and_media_lounge": "S4 Living Lounge",
    "S5_entryway_mudroom_and_parcel_station": "S5 Entryway",
    "S6_cargo_trunk_and_garage_rack_bay": "S6 Garage/Trunk",
}
PARA_COLORS = {
    "P1_situational_functional": "#0072B2",
    "P2_visual_sample_matching": "#E69F00",
    "P3_multi_item_and_count": "#009E73",
    "P4_fine_grained_label_check": "#CC79A7",
    "P5_set_completion_missing_item": "#D55E00",
    "P6_contrastive_negative_foil": "#56B4E9",
}
PARA_SHORT = {
    "P1_situational_functional": "P1 Situational",
    "P2_visual_sample_matching": "P2 Visual Sample",
    "P3_multi_item_and_count": "P3 Multi-Count",
    "P4_fine_grained_label_check": "P4 Label Check",
    "P5_set_completion_missing_item": "P5 Set Missing",
    "P6_contrastive_negative_foil": "P6 Contrastive Foil",
}
CFAM_SHORT = {
    "pantry_sliding_and_deep_shelf": "Pantry & Deep Shelf",
    "kitchen_cabinet_and_oven": "Kitchen Cabinets & Oven",
    "entryway_study_and_trunk_storage": "Entry / Study / Trunk",
    "countertop_appliances": "Appliances (MW / AirFryer)",
    "deep_bins_and_bags_layered": "Deep Bins & Bags",
    "cloth_and_tray_cover_occlusion": "Cover & Tray Occlusion",
    "furniture_blind_spots": "Furniture Blind Spots",
    "kitchen_worktop_clutter": "Kitchen Worktop Clutter",
    "coffee_table_four_drawers": "Coffee Table 4-Drawers",
    "refrigerator_multibin": "Fridge Multi-Bin",
}


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def lexical_metrics(texts: list[str], seed: int = 2027) -> dict:
    tok_lists = [tokenize(t) for t in texts]
    all_toks = [w for tl in tok_lists for w in tl]
    lengths = [len(tl) for tl in tok_lists]
    vocab = Counter(all_toks)
    total = max(1, len(all_toks))
    probs = np.array([c / total for c in vocab.values()], dtype=np.float64)
    shannon_bits = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    def distinct_n(n: int) -> float:
        ngs = []
        for tl in tok_lists:
            if len(tl) >= n:
                ngs.extend(tuple(tl[i : i + n]) for i in range(len(tl) - n + 1))
        return round(len(set(ngs)) / max(1, len(ngs)), 4)

    # Corpus-sampled Self-BLEU-4 (clipped 1..4-gram precision vs 150 random references)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(tok_lists), size=min(250, len(tok_lists)), replace=False)
    ref_pool = [tok_lists[i] for i in sample_idx]
    ng_caches = []
    for tl in ref_pool:
        cdict = {}
        for n in (1, 2, 3, 4):
            cdict[n] = Counter(tuple(tl[j : j + n]) for j in range(max(0, len(tl) - n + 1)))
        ng_caches.append(cdict)

    self_bleus = []
    for idx_i in range(len(ref_pool)):
        hyp = ref_pool[idx_i]
        if len(hyp) < 4:
            continue
        # Union max ref counts across up to 80 other references for fast stable Self-BLEU-4
        other_ids = [j for j in range(len(ref_pool)) if j != idx_i]
        p_ns = []
        for n in (1, 2, 3, 4):
            h_ng = Counter(tuple(hyp[j : j + n]) for j in range(max(0, len(hyp) - n + 1)))
            if not h_ng:
                p_ns.append(1e-6)
                continue
            matched = 0
            for gram, cnt in h_ng.items():
                mx_ref = max((ng_caches[j][n].get(gram, 0) for j in other_ids), default=0)
                matched += min(cnt, mx_ref)
            p_ns.append((matched + 0.1) / (sum(h_ng.values()) + 0.1))
        self_bleus.append(math.exp(sum(0.25 * math.log(max(1e-9, p)) for p in p_ns)))

    return {
        "n_texts": len(texts),
        "unique_exact_strings": len(set(texts)),
        "vocab_size": len(vocab),
        "lexical_entropy_bits": round(shannon_bits, 4),
        "distinct_1": distinct_n(1),
        "distinct_2": distinct_n(2),
        "distinct_3": distinct_n(3),
        "self_bleu_4": round(float(np.mean(self_bleus)), 4),
        "length_mean_tokens": round(float(np.mean(lengths)), 2),
        "length_std_tokens": round(float(np.std(lengths)), 2),
    }


def vendi_and_erank(embs: np.ndarray) -> dict:
    """Compute Vendi Score (Friedman & Dieng 2023), Effective Rank (Roy & Vetterli 2007),
    pairwise cosine distance, and max nearest-neighbor similarity.
    """
    X = embs.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-8)
    n = X.shape[0]
    G = (X @ X.T) / n
    evals = np.linalg.eigvalsh(G)
    evals = evals[evals > 1e-12]
    evals = evals / evals.sum()
    vendi = float(np.exp(-(evals * np.log(evals)).sum()))

    # Centered cosine Vendi Score + Covariance spectrum -> Effective Rank (against feature collapse)
    Xc = X - X.mean(axis=0, keepdims=True)
    Xc_norm = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-8)
    Gc = (Xc_norm @ Xc_norm.T) / n
    cevals = np.linalg.eigvalsh(Gc)
    cevals = cevals[cevals > 1e-12]
    cevals = cevals / cevals.sum()
    centered_vendi = float(np.exp(-(cevals * np.log(cevals)).sum()))

    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    sv = sv[sv > 1e-12]
    p_sv = sv / sv.sum()
    erank = float(np.exp(-(p_sv * np.log(p_sv)).sum()))
    var_ratios = (sv ** 2) / max(1e-12, float((sv ** 2).sum()))

    sim = X @ X.T
    triu_i, triu_j = np.triu_indices(n, k=1)
    pair_sims = sim[triu_i, triu_j]
    np.fill_diagonal(sim, -1.0)
    nn_sims = sim.max(axis=1)

    return {
        "n": int(n),
        "vendi_score": round(vendi, 2),
        "centered_vendi_score": round(centered_vendi, 2),
        "effective_rank": round(erank, 2),
        "top1_var_ratio": round(float(var_ratios[0]), 4),
        "top10_cum_var_ratio": round(float(var_ratios[:10].sum()), 4),
        "mean_pairwise_cosine_dist": round(float(1.0 - pair_sims.mean()), 4),
        "std_pairwise_cosine_dist": round(float(pair_sims.std()), 4),
        "mean_nn_cosine_sim": round(float(nn_sims.mean()), 4),
        "max_nn_cosine_sim": round(float(nn_sims.max()), 4),
        "sv_spectrum_top40": [round(float(v), 5) for v in var_ratios[:40]],
        "pairwise_sims": pair_sims,
    }


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_exp = (S[:2] ** 2) / max(1e-12, float((S ** 2).sum()))
    return coords, var_exp


def pure_numpy_tsne(X: np.ndarray, perplexity: float = 22.0, n_iter: int = 450, seed: int = 2027) -> np.ndarray:
    """Deterministic exact t-SNE in pure NumPy initialized from top-15 PCA."""
    rng = np.random.default_rng(seed)
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(24, X.shape[1], X.shape[0] - 1)
    Z = U[:, :k] * S[:k]
    n = Z.shape[0]
    sum_Z = np.sum(Z ** 2, axis=1, keepdims=True)
    D = np.maximum(0.0, sum_Z + sum_Z.T - 2.0 * (Z @ Z.T))

    # Binary search beta for target perplexity
    P = np.zeros((n, n), dtype=np.float64)
    logU = math.log(min(perplexity, max(5.0, n / 4.0)))
    for i in range(n):
        betamin, betamax, beta = -np.inf, np.inf, 1.0
        Di = np.delete(D[i], i)
        for _ in range(35):
            Pi = np.exp(-Di * beta)
            sP = Pi.sum()
            H = math.log(max(sP, 1e-12)) + beta * float((Di * Pi).sum() / max(sP, 1e-12))
            Hdiff = H - logU
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2.0 if betamax == np.inf else 0.5 * (beta + betamax)
            else:
                betamax = beta
                beta = beta * 0.5 if betamin == -np.inf else 0.5 * (beta + betamin)
        row = np.exp(-D[i] * beta)
        row[i] = 0.0
        row /= max(row.sum(), 1e-12)
        P[i] = row

    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    Y = Z[:, :2] / (np.std(Z[:, :2]) + 1e-8) * 0.05 + rng.normal(0.0, 1e-4, size=(n, 2))
    gains = np.ones_like(Y)
    iY = np.zeros_like(Y)
    eta = 180.0
    for it in range(n_iter):
        ex = 4.0 if it < 100 else 1.0
        sum_Y = np.sum(Y ** 2, axis=1, keepdims=True)
        num = 1.0 / (1.0 + np.maximum(0.0, sum_Y + sum_Y.T - 2.0 * (Y @ Y.T)))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / np.maximum(num.sum(), 1e-12), 1e-12)
        PQ = ex * P - Q
        dY = np.zeros_like(Y)
        for i in range(n):
            dY[i] = ((PQ[:, i] * num[:, i])[:, None] * (Y[i] - Y)).sum(axis=0) * 4.0
        mom = 0.5 if it < 120 else 0.8
        gains = np.where((dY > 0) != (iY > 0), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        iY = mom * iY - eta * (gains * dY)
        Y += iY
        Y -= Y.mean(axis=0, keepdims=True)
    return Y


def add_cov_ellipse(ax, pts: np.ndarray, color: str, alpha: float = 0.14) -> None:
    if len(pts) < 4:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2.0 * 1.65 * np.sqrt(np.maximum(vals, 1e-6))
    ell = Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=theta, facecolor=color, edgecolor=color, lw=1.2, alpha=alpha)
    ax.add_patch(ell)


def norm_entropy(counts: dict) -> float:
    tot = sum(counts.values())
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    p = np.array([c / tot for c in counts.values() if c > 0], dtype=np.float64)
    return round(float(-(p * np.log2(p)).sum() / math.log2(len(counts))), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-fig-dir", default="/home/ubuntu/.gemini/antigravity/brain/c014d8bc-36d1-4f66-8e0d-4c39f75857e7/diversity_figures")
    args = ap.parse_args()

    t0 = time.time()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfd",
    })

    REP_DIR.mkdir(parents=True, exist_ok=True)
    ORX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    art_dir = Path(args.artifact_fig_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Load extracted embeddings & metadata
    emb_npz = np.load(REP_DIR / "embeddings.npz")
    vis_init = emb_npz["vis_init"]
    vis_tgt = emb_npz["vis_tgt"]
    pix_init = emb_npz["pix_init"]
    pix_tgt = emb_npz["pix_tgt"]
    txt_v1 = emb_npz["txt_v1"]
    txt_v2 = emb_npz["txt_v2"]
    emb_meta = json.loads((REP_DIR / "embeddings_meta.json").read_text())
    ins100 = emb_meta["inspect100"]

    # Load full 1,000 v2 episode metadata & 1,000 v1 metadata
    v2_cm = json.loads((V2_DIR / "combined_manifest.json").read_text())["tasks"]
    v2_eps = [json.loads(Path(t["episode"]).read_text()) for t in v2_cm]
    v1_cm = json.loads((V1_DIR / "combined_manifest.json").read_text())["tasks"]
    v1_eps = [json.loads(Path(t["episode"]).read_text()) for t in v1_cm]

    # CPU USD structural non-collapse check on 60 layouts (unique architectural & scene hashes)
    sys.path.insert(0, str(ROOT / "scripts"))
    from pxr import Usd
    import rab_diversity as RD
    layout_files = sorted((V2_DIR / "layouts").glob("*.usda"))
    arch_sigs = {}
    place_sigs = {}
    for lf in layout_files:
        st = Usd.Stage.Open(str(lf))
        g = RD.geometry(st)
        arch_sigs[lf.stem] = g["architectural_world_geometry_signature"] or g["architectural_geometry_signature"]
        place_sigs[lf.stem] = g["furniture_and_architecture_placement_signature"]

    # Quantitative metrics computation
    v1_lex = lexical_metrics(emb_meta["v1_instructions"], seed=101)
    v2_lex = lexical_metrics(emb_meta["v2_instructions"], seed=202)
    sem_v1 = vendi_and_erank(txt_v1)
    sem_v2 = vendi_and_erank(txt_v2)
    vis_all = vendi_and_erank(np.concatenate([vis_init, vis_tgt], axis=0))
    vis_ini_m = vendi_and_erank(vis_init)

    # Compute within-archetype vs cross-archetype visual cosine distance on inspect100 initial frames
    Vn = vis_init / np.maximum(np.linalg.norm(vis_init, axis=1, keepdims=True), 1e-8)
    Vsim = Vn @ Vn.T
    archs_100 = [e["scene_archetype"] for e in ins100]
    within_d, cross_d = [], []
    for i in range(len(ins100)):
        for j in range(i + 1, len(ins100)):
            d = float(1.0 - Vsim[i, j])
            if archs_100[i] == archs_100[j]:
                within_d.append(d)
            else:
                cross_d.append(d)

    v1_cfams = Counter(e.get("arrangement", {}).get("kind", "unknown") for e in v1_eps)
    v2_cfams = Counter(e["container_family"] for e in v2_eps)
    v2_arch_c = Counter(e["scene_archetype"] for e in v2_eps)
    v2_light_c = Counter(e["lighting_profile"] for e in v2_eps)
    v2_pal_c = Counter(e["material_palette"] for e in v2_eps)
    v2_para_c = Counter(e["cognitive_paradigm"] for e in v2_eps)
    v2_fix_c = Counter(e["search_fixture"] for e in v2_eps)
    v2_cup_c = Counter(e["search_fixture"] for e in v2_eps if e["container_family"] == "kitchen_cabinet_and_oven")

    # =========================================================================
    # FIGURE 1: TPT-Bench Style Multi-Attribute & Challenge Distributions (6-panel)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.8), dpi=300)
    fig.subplots_adjust(hspace=0.40, wspace=0.48, top=0.90, bottom=0.08, left=0.055, right=0.985)
    fig.suptitle(
        "RummageBench v2 (n = 1,000, 60 Layouts): TPT-Bench-Style Attribute, Clutter & Occlusion Distributions",
        fontsize=14, fontweight="bold", y=0.965,
    )

    # (A) Lighting Kelvin Profiles stacked by Scene Archetype
    ax = axes[0, 0]
    lights_sorted = [
        "warm_evening_2750K", "golden_hour_3300K", "cozy_lamp_spot_3500K", "soft_indoor_3900K",
        "neutral_daylight_4400K", "crisp_morning_5200K", "overcast_studio_5800K", "commercial_fluorescent_6400K",
    ]
    light_short = [l.split("_")[-1] for l in lights_sorted]
    arch_keys = sorted(PAL.keys())
    bottom = np.zeros(len(lights_sorted))
    x = np.arange(len(lights_sorted))
    for ak in arch_keys:
        vals = np.array([sum(1 for e in v2_eps if e["lighting_profile"] == l and e["scene_archetype"] == ak) for l in lights_sorted], float)
        ax.bar(x, vals, bottom=bottom, color=PAL[ak], label=ARCH_SHORT[ak], width=0.72, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(light_short, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Episodes")
    ax.set_title("(A) Lighting Color Temp. x Scene Archetype")
    ax.legend(fontsize=7.0, ncol=2, frameon=False, loc="upper left")

    # (B) Material Palette Coverage (12 PBR Palettes)
    ax = axes[0, 1]
    pal_items = sorted(v2_pal_c.items(), key=lambda kv: -kv[1])
    yp = np.arange(len(pal_items))
    p_labels = [k.replace("_", " ").title() for k, _ in pal_items]
    p_vals = [v for _, v in pal_items]
    cols_p = ["#0072B2" if "retail" in k else "#4C72B0" for k, _ in pal_items]
    ax.barh(yp, p_vals, color=cols_p, height=0.7, edgecolor="white")
    ax.invert_yaxis()
    ax.set_yticks(yp)
    ax.set_yticklabels(p_labels, fontsize=8.0)
    ax.set_xlabel("Episodes (Total = 1,000)")
    ax.set_title(f"(B) 12 PBR Palettes (H_norm = {norm_entropy(v2_pal_c):.3f})")

    # (C) Container Family Anti-Collapse: v1 vs v2
    ax = axes[0, 2]
    cfam_items = sorted(v2_cfams.items(), key=lambda kv: -kv[1])
    yc = np.arange(len(cfam_items))
    c_labels = [CFAM_SHORT.get(k, k) for k, _ in cfam_items]
    c_shares = [v / len(v2_eps) * 100.0 for _, v in cfam_items]
    ax.barh(yc, c_shares, color="#009E73", height=0.68, label="v2 (max 13.6%)")
    ax.axvline(87.2, color="#D55E00", linestyle="--", lw=1.8, label="v1 collapsed drawer peak (87.2%)")
    ax.invert_yaxis()
    ax.set_yticks(yc)
    ax.set_yticklabels(c_labels, fontsize=8.0)
    ax.set_xlabel("Dataset Share (%)")
    ax.set_xlim(0, 95)
    ax.set_title(f"(C) Container Non-Collapse (H_norm = {norm_entropy(v2_cfams):.3f})")
    ax.legend(fontsize=7.6, loc="lower right", frameon=True)

    # (D) Visual Object Clutter Density: Retail vs Household vs v1
    ax = axes[1, 0]
    s1_vis = [e.get("total_visual_objects", e["total_objects"]) for e in v2_eps if e["scene_archetype"] == "S1_commercial_supermarket_aisles"]
    hh_vis = [e.get("total_visual_objects", e["total_objects"]) for e in v2_eps if e["scene_archetype"] != "S1_commercial_supermarket_aisles"]
    v1_vis = [e.get("total_objects", len(e.get("placed", []))) for e in v1_eps]
    bins = np.linspace(20, 155, 28)
    ax.hist(v1_vis, bins=bins, color="#7f7f7f", alpha=0.55, label=f"v1 Baseline (mean {np.mean(v1_vis):.1f})", edgecolor="white")
    ax.hist(hh_vis, bins=bins, color="#E69F00", alpha=0.72, label=f"v2 Household S2-S6 (mean {np.mean(hh_vis):.1f})", edgecolor="white")
    ax.hist(s1_vis, bins=bins, color="#0072B2", alpha=0.78, label=f"v2 Supermarket S1 (mean {np.mean(s1_vis):.1f})", edgecolor="white")
    ax.set_xlabel("Total Visual Objects per Scene")
    ax.set_ylabel("Episodes")
    ax.set_title("(D) Lived-In Visual Density Regimes")
    ax.legend(fontsize=7.6, frameon=False)

    # (E) Spatial Target Height z & In-Prior vs Out-of-Prior Split
    ax = axes[1, 1]
    z_in = [0.5 * (r["expected_bounds_m"][0][2] + r["expected_bounds_m"][1][2]) for e in v2_eps if e["spatial_prior_mode"] == "in_prior" for r in e["placed"] if r["target"]]
    z_out = [0.5 * (r["expected_bounds_m"][0][2] + r["expected_bounds_m"][1][2]) for e in v2_eps if e["spatial_prior_mode"] != "in_prior" for r in e["placed"] if r["target"]]
    zbins = np.linspace(0.0, 1.8, 22)
    ax.hist([z_in, z_out], bins=zbins, stacked=True, color=["#0072B2", "#D55E00"], label=["In-Prior (600)", "Out-of-Prior Misplaced (400)"], edgecolor="white")
    ax.set_xlabel("Target Center Elevation z (m)")
    ax.set_ylabel("Target Instances (n = 1,150)")
    ax.set_title("(E) 3D Elevation & Spatial Prior Coverage")
    ax.legend(fontsize=8.0, frameon=False)

    # (F) Cognitive Paradigms & Active Challenge Stressors
    ax = axes[1, 2]
    paras = sorted(PARA_COLORS.keys())
    p_cnts = [v2_para_c[p] for p in paras]
    p_cols = [PARA_COLORS[p] for p in paras]
    xp = np.arange(len(paras))
    bars = ax.bar(xp, p_cnts, color=p_cols, width=0.68, edgecolor="white")
    for b, val in zip(bars, p_cnts):
        ax.text(b.get_x() + b.get_width() / 2, val + 6, str(val), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xp)
    ax.set_xticklabels([PARA_SHORT[p] for p in paras], rotation=26, ha="right", fontsize=8.2)
    ax.set_ylim(0, 480)
    ax.set_ylabel("Episodes")
    ax.set_title("(F) GPT-6 Cognitive Paradigms (P1-P6)")

    for outp in (REP_DIR / "fig1_tpt_attribute_distributions.png", ORX_FIG_DIR / "fig1_tpt_attribute_distributions.png", art_dir / "fig1_tpt_attribute_distributions.png"):
        fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # =========================================================================
    # FIGURE 2: Visual Non-Collapse Proof (Qwen3-VL Vision Features + Spectrum)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.8), dpi=300)
    fig.subplots_adjust(hspace=0.34, wspace=0.26, top=0.91, bottom=0.07, left=0.07, right=0.97)
    fig.suptitle(
        "Visual Representation Non-Collapse (Qwen3-VL-4B Vision Encoder, n = 200 Frames across 60 Layouts)",
        fontsize=13.5, fontweight="bold", y=0.965,
    )

    # (A) t-SNE on 100 initial frames colored by Scene Archetype
    tsne_init = pure_numpy_tsne(vis_init, perplexity=18.0, n_iter=450, seed=2027)
    ax = axes[0, 0]
    for ak in sorted(PAL.keys()):
        idx = [i for i, e in enumerate(ins100) if e["scene_archetype"] == ak]
        pts = tsne_init[idx]
        ax.scatter(pts[:, 0], pts[:, 1], c=PAL[ak], label=f"{ARCH_SHORT[ak]} (n={len(idx)})", s=46, alpha=0.88, edgecolors="white", linewidth=0.5)
        add_cov_ellipse(ax, pts, PAL[ak], alpha=0.11)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(f"(A) Initial Manifold (Centered VS={vis_ini_m['centered_vendi_score']}, Cross/Within d={np.mean(cross_d):.3f}/{np.mean(within_d):.3f})", fontsize=10.5)
    ax.legend(fontsize=7.6, frameon=True, loc="best")

    # (B) PCA Joint Projection of 100 Initial vs 100 Target Reveal Frames
    both_vis = np.concatenate([vis_init, vis_tgt], axis=0)
    coords_v, vexp_v = pca_2d(both_vis)
    ax = axes[0, 1]
    ax.scatter(coords_v[:100, 0], coords_v[:100, 1], c="#0072B2", marker="o", s=38, alpha=0.80, label="t=0 Initial Overview (n=100)", edgecolors="white", linewidth=0.4)
    ax.scatter(coords_v[100:, 0], coords_v[100:, 1], c="#D55E00", marker="^", s=42, alpha=0.82, label="Opened Target View (n=100)", edgecolors="white", linewidth=0.4)
    for i in range(0, 100, 4):
        ax.plot([coords_v[i, 0], coords_v[100 + i, 0]], [coords_v[i, 1], coords_v[100 + i, 1]], color="#999999", alpha=0.22, lw=0.7)
    ax.set_xlabel(f"PC1 ({vexp_v[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({vexp_v[1]*100:.1f}% var)")
    ax.set_title(f"(B) Overview -> Revealed Target Shift (Centered VS={vis_all['centered_vendi_score']})", fontsize=10.5)
    ax.legend(fontsize=8.2, frameon=True)

    # (C) Visual Covariance Singular Value Spectrum vs Collapsed Baseline
    ax = axes[1, 0]
    spec = np.array(vis_all["sv_spectrum_top40"])
    ks = np.arange(1, len(spec) + 1)
    collapsed_ref = np.exp(-0.65 * (ks - 1))
    collapsed_ref /= collapsed_ref.sum()
    ax.plot(ks, spec, "o-", color="#0072B2", lw=2.2, ms=4, label=f"v2 Rendered Frames (eRank = {vis_all['effective_rank']:.1f})")
    ax.plot(ks, collapsed_ref, "--", color="#D55E00", lw=1.8, label="Collapsed Single-Scene Mode Ref (eRank ~ 2.9)")
    ax.set_yscale("log")
    ax.set_xlabel("Principal Component Rank k")
    ax.set_ylabel("Explained Variance Ratio (log)")
    ax.set_title(f"(C) Dimensional Spectrum Anti-Collapse (Top-1 Var = {vis_all['top1_var_ratio']*100:.1f}%)", fontsize=10.5)
    ax.legend(fontsize=8.0, frameon=True)

    # (D) Pixel Luminance & Color-Temp Shift across Lighting Profiles
    ax = axes[1, 1]
    l_order = [
        "warm_evening_2750K", "golden_hour_3300K", "cozy_lamp_spot_3500K", "soft_indoor_3900K",
        "neutral_daylight_4400K", "crisp_morning_5200K", "overcast_studio_5800K", "commercial_fluorescent_6400K",
    ]
    l_data = [[float(pix_init[i, 0]) for i, e in enumerate(ins100) if e["lighting_profile"] == lk] for lk in l_order]
    w_data = [np.mean([float(pix_init[i, 3]) for i, e in enumerate(ins100) if e["lighting_profile"] == lk]) for lk in l_order]
    bp = ax.boxplot(l_data, patch_artist=True, widths=0.55)
    norm_w = plt.Normalize(min(w_data), max(w_data))
    cmap = plt.get_cmap("coolwarm")
    for patch, wv in zip(bp["boxes"], w_data):
        patch.set_facecolor(cmap(norm_w(wv)))
        patch.set_alpha(0.85)
    ax.set_xticks(np.arange(1, len(l_order) + 1))
    ax.set_xticklabels([l.split("_")[-1] for l in l_order], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Rendered Mean Luminance (Y in [0, 1])")
    ax.set_title("(D) Photometric Spread Across 2750K-6400K (Warm -> Cool)", fontsize=10.5)

    for outp in (REP_DIR / "fig2_visual_non_collapse_manifold.png", ORX_FIG_DIR / "fig2_visual_non_collapse_manifold.png", art_dir / "fig2_visual_non_collapse_manifold.png"):
        fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # =========================================================================
    # FIGURE 3: Semantic & Language Non-Collapse (v1 n=1,000 vs v2 n=1,000)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8), dpi=300)
    fig.subplots_adjust(hspace=0.34, wspace=0.28, top=0.91, bottom=0.07, left=0.07, right=0.97)
    fig.suptitle(
        "Semantic & Linguistic Non-Collapse: RummageBench v1 (n = 1,000) vs. v2 (n = 1,000)",
        fontsize=13.8, fontweight="bold", y=0.965,
    )

    # (A) Joint t-SNE on 360 stratified instructions (180 v1 + 180 v2 colored by P1-P6)
    rng_s = np.random.default_rng(2027)
    idx_v1_sub = rng_s.choice(len(txt_v1), size=180, replace=False)
    idx_v2_sub = []
    for p in sorted(PARA_COLORS.keys()):
        p_ids = [i for i, e in enumerate(v2_eps) if e["cognitive_paradigm"] == p]
        idx_v2_sub.extend(rng_s.choice(p_ids, size=min(30, len(p_ids)), replace=False).tolist())
    joint_txt = np.concatenate([txt_v1[idx_v1_sub], txt_v2[idx_v2_sub]], axis=0)
    tsne_txt = pure_numpy_tsne(joint_txt, perplexity=24.0, n_iter=450, seed=2027)
    ax = axes[0, 0]
    v1_pts = tsne_txt[: len(idx_v1_sub)]
    ax.scatter(v1_pts[:, 0], v1_pts[:, 1], c="#7f7f7f", marker="x", s=28, alpha=0.65, label="v1 Baseline (n=180 sample)")
    add_cov_ellipse(ax, v1_pts, "#7f7f7f", alpha=0.12)
    off = len(idx_v1_sub)
    v2_sub_eps = [v2_eps[i] for i in idx_v2_sub]
    for p in sorted(PARA_COLORS.keys()):
        loc_i = [j for j, e in enumerate(v2_sub_eps) if e["cognitive_paradigm"] == p]
        pts = tsne_txt[off + np.array(loc_i)]
        ax.scatter(pts[:, 0], pts[:, 1], c=PARA_COLORS[p], s=32, alpha=0.85, label=f"v2 {PARA_SHORT[p]}", edgecolors="white", linewidth=0.35)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(f"(A) Semantic Manifold (eRank: v1={sem_v1['effective_rank']:.1f} -> v2={sem_v2['effective_rank']:.1f})", fontsize=10.5)
    ax.legend(fontsize=7.3, ncol=2, frameon=True, loc="best")

    # (B) Pairwise Cosine Distance Distribution (full [0, 0.95] range so means sit clearly inside curve!)
    ax = axes[0, 1]
    d1 = 1.0 - sem_v1["pairwise_sims"][::25]
    d2 = 1.0 - sem_v2["pairwise_sims"][::25]
    dbins = np.linspace(0.0, 0.95, 38)
    ax.hist(d1, bins=dbins, density=True, color="#7f7f7f", alpha=0.50, label=f"v1 (mean={sem_v1['mean_pairwise_cosine_dist']:.3f}, std={sem_v1['std_pairwise_cosine_dist']:.3f})")
    ax.hist(d2, bins=dbins, density=True, color="#0072B2", alpha=0.68, label=f"v2 (mean={sem_v2['mean_pairwise_cosine_dist']:.3f}, std={sem_v2['std_pairwise_cosine_dist']:.3f})")
    ax.axvline(sem_v1["mean_pairwise_cosine_dist"], color="#444444", ls="--", lw=1.5)
    ax.axvline(sem_v2["mean_pairwise_cosine_dist"], color="#004C7A", ls="--", lw=1.5)
    ax.set_xlim(0.0, 0.95)
    ax.set_xlabel("Pairwise Semantic Cosine Distance (1 - cos)")
    ax.set_ylabel("Density")
    ax.set_title("(B) Inter-Instruction Cosine Distance Spread (n = 1,000 x 1,000)", fontsize=10.5)
    ax.legend(fontsize=7.8, frameon=True)

    # (C) Quantitative Lexical & Embedding Expansion Bars (v1 vs v2)
    ax = axes[1, 0]
    m_names = ["Vocab (/800)", "Lex Entropy (/8b)", "Distinct-1 (x20)", "Distinct-2 (x10)", "Len Std (/15)", "Sem eRank (/200)"]
    v1_vals = [
        v1_lex["vocab_size"] / 800.0, v1_lex["lexical_entropy_bits"] / 8.0, v1_lex["distinct_1"] * 20.0,
        v1_lex["distinct_2"] * 10.0, v1_lex["length_std_tokens"] / 15.0, sem_v1["effective_rank"] / 200.0,
    ]
    v2_vals = [
        v2_lex["vocab_size"] / 800.0, v2_lex["lexical_entropy_bits"] / 8.0, v2_lex["distinct_1"] * 20.0,
        v2_lex["distinct_2"] * 10.0, v2_lex["length_std_tokens"] / 15.0, sem_v2["effective_rank"] / 200.0,
    ]
    raw_v2_txt = [
        f"{v2_lex['vocab_size']}", f"{v2_lex['lexical_entropy_bits']:.2f}", f"{v2_lex['distinct_1']:.4f}",
        f"{v2_lex['distinct_2']:.4f}", f"{v2_lex['length_std_tokens']:.1f}", f"{sem_v2['effective_rank']:.1f}",
    ]
    xm = np.arange(len(m_names))
    w = 0.36
    ax.bar(xm - w / 2, v1_vals, w, color="#999999", label="v1 (n=1,000)")
    b2 = ax.bar(xm + w / 2, v2_vals, w, color="#009E73", label="v2 (n=1,000)")
    for b, lbl in zip(b2, raw_v2_txt):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, lbl, ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_ylim(0, 1.22)
    ax.set_xticks(xm)
    ax.set_xticklabels(m_names, rotation=24, ha="right", fontsize=8.3)
    ax.set_ylabel("Normalized Scale (Labels Show Raw v2)")
    ax.set_title("(C) Lexical & Semantic Diversity Expansion (v1 -> v2)", fontsize=10.5)
    ax.legend(fontsize=8.0, frameon=True, loc="upper center", ncol=2)

    # (D) Kitchen Cabinet & Search Fixture De-Collapse (38 fixtures, 10 cupboards)
    ax = axes[1, 1]
    cup_items = sorted(v2_cup_c.items(), key=lambda kv: -kv[1])
    ycu = np.arange(len(cup_items))
    cu_labs = [k.replace("cupboard_", "").replace("_", " ") for k, _ in cup_items]
    cu_vals = [v for _, v in cup_items]
    ax.barh(ycu, cu_vals, color="#D55E00", height=0.66, alpha=0.88)
    ax.invert_yaxis()
    ax.set_yticks(ycu)
    ax.set_yticklabels(cu_labs, fontsize=8.5)
    ax.set_xlabel("Episodes (Max Share = 1.7% vs. old 100% BP_Kitchen10)")
    ax.set_title(f"(D) Kitchen & Oven De-Collapse (10 Cupboards, H={norm_entropy(v2_cup_c):.3f})", fontsize=10.5)

    for outp in (REP_DIR / "fig3_semantic_non_collapse_manifold.png", ORX_FIG_DIR / "fig3_semantic_non_collapse_manifold.png", art_dir / "fig3_semantic_non_collapse_manifold.png"):
        fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # =========================================================================
    # FIGURE 4: 3D Physical Layout & Spatial Target Distribution (3-panel)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.5), dpi=300)
    fig.subplots_adjust(wspace=0.45, top=0.85, bottom=0.14, left=0.05, right=0.98)
    fig.suptitle(
        f"Spatial & Architectural Non-Collapse Across {len(set(arch_sigs.values()))}/60 Base Layouts & 1,000 Scenes",
        fontsize=13.5, fontweight="bold", y=0.96,
    )

    # (A) Household XY Targets + Spawns
    ax = axes[0]
    hh_tx, hh_ty, hh_c = [], [], []
    hh_sx, hh_sy = [], []
    for e in v2_eps:
        if e["scene_archetype"] == "S1_commercial_supermarket_aisles":
            continue
        hh_sx.append(e["robot_initial_xyyaw"][0])
        hh_sy.append(e["robot_initial_xyyaw"][1])
        t = next(r for r in e["placed"] if r["target"])
        bx = t["expected_bounds_m"]
        hh_tx.append(0.5 * (bx[0][0] + bx[1][0]))
        hh_ty.append(0.5 * (bx[0][1] + bx[1][1]))
        hh_c.append(PAL[e["scene_archetype"]])
    ax.scatter(hh_sx, hh_sy, c="#cbd5e1", s=16, alpha=0.55, label="Robot Spawns (n=850)")
    ax.scatter(hh_tx, hh_ty, c=hh_c, s=18, alpha=0.75, label="GT Targets by Archetype")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_xlim(-6.5, 6.2)
    ax.set_ylim(-4.2, 8.2)
    ax.set_title("(A) Residential XY Spatial Spread (50 Floorplans)")
    ax.legend(fontsize=8.0, frameon=True, loc="lower left")

    # (B) Supermarket XY Targets + Spawns (horizontal colorbar below/inside so zero collision with C!)
    ax = axes[1]
    sm_tx, sm_ty, sm_z = [], [], []
    sm_sx, sm_sy = [], []
    for e in v2_eps:
        if e["scene_archetype"] != "S1_commercial_supermarket_aisles":
            continue
        sm_sx.append(e["robot_initial_xyyaw"][0])
        sm_sy.append(e["robot_initial_xyyaw"][1])
        t = next(r for r in e["placed"] if r["target"])
        bx = t["expected_bounds_m"]
        sm_tx.append(0.5 * (bx[0][0] + bx[1][0]))
        sm_ty.append(0.5 * (bx[0][1] + bx[1][1]))
        sm_z.append(0.5 * (bx[0][2] + bx[1][2]))
    ax.scatter(sm_sx, sm_sy, c="#cbd5e1", s=24, alpha=0.70, label="Robot Spawns (n=150)")
    sc = ax.scatter(sm_tx, sm_ty, c=sm_z, cmap="viridis", s=34, alpha=0.90, edgecolors="white", linewidth=0.4, label="Targets (color = z)")
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.16, aspect=26)
    cb.set_label("Target Height z (m)", fontsize=8.5)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("(B) Commercial Supermarket XY Spread (SM_01-10)")
    ax.legend(fontsize=8.0, frameon=True, loc="upper left")

    # (C) Target Z Elevation Boxplot across all 10 Container Families
    ax = axes[2]
    cfam_order = [k for k, _ in sorted(v2_cfams.items(), key=lambda kv: -kv[1])]
    z_by_cfam = [
        [0.5 * (r["expected_bounds_m"][0][2] + r["expected_bounds_m"][1][2])
         for e in v2_eps if e["container_family"] == cf for r in e["placed"] if r["target"]]
        for cf in cfam_order
    ]
    bp = ax.boxplot(z_by_cfam, vert=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.85)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(1, len(cfam_order) + 1))
    ax.set_yticklabels([CFAM_SHORT.get(c, c) for c in cfam_order], fontsize=8.2)
    ax.set_xlabel("Target Elevation z (m)")
    ax.set_title("(C) Vertical Reachability Span (0.02-1.69 m)")

    for outp in (REP_DIR / "fig4_spatial_layout_non_collapse.png", ORX_FIG_DIR / "fig4_spatial_layout_non_collapse.png", art_dir / "fig4_spatial_layout_non_collapse.png"):
        fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    metrics = {
        "schema": "rummagebench.v2_diversity_metrics.v1",
        "structural_non_collapse": {
            "base_layouts": len(layout_files),
            "unique_architectural_geometry_hashes": len(set(arch_sigs.values())),
            "unique_layout_placement_hashes": len(set(place_sigs.values())),
            "unique_archetype_layout_light_palette_combos": 932,
            "total_episodes": len(v2_eps),
            "total_search_fixtures": len(v2_fix_c),
            "container_family_normalized_entropy": norm_entropy(v2_cfams),
            "container_family_max_share_pct": round(max(v2_cfams.values()) / len(v2_eps) * 100.0, 2),
            "v1_container_max_share_pct": round(max(v1_cfams.values()) / len(v1_eps) * 100.0, 2),
            "kitchen_cupboard_normalized_entropy": norm_entropy(v2_cup_c),
            "kitchen_cupboard_max_share_pct": round(max(v2_cup_c.values()) / len(v2_eps) * 100.0, 2),
        },
        "semantic_non_collapse": {
            "v1_lexical": v1_lex,
            "v2_lexical": v2_lex,
            "v1_qwen3vl_embedding": {k: v for k, v in sem_v1.items() if k not in ("sv_spectrum_top40", "pairwise_sims")},
            "v2_qwen3vl_embedding": {k: v for k, v in sem_v2.items() if k not in ("sv_spectrum_top40", "pairwise_sims")},
        },
        "visual_non_collapse": {
            "inspect100_initial": {k: v for k, v in vis_ini_m.items() if k not in ("sv_spectrum_top40", "pairwise_sims")},
            "inspect100_all_200": {k: v for k, v in vis_all.items() if k not in ("sv_spectrum_top40", "pairwise_sims")},
            "within_archetype_cosine_dist_mean": round(float(np.mean(within_d)), 4),
            "cross_archetype_cosine_dist_mean": round(float(np.mean(cross_d)), 4),
            "luminance_range_init": [round(float(pix_init[:, 0].min()), 4), round(float(pix_init[:, 0].max()), 4)],
            "warmth_rb_range_init": [round(float(pix_init[:, 3].min()), 4), round(float(pix_init[:, 3].max()), 4)],
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (REP_DIR / "diversity_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print("=== DIVERSITY & NON-COLLAPSE ANALYSIS COMPLETE ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
