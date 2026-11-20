"""RGB visibility of an entity in a private frame (evaluator-only).

The v1 S1 gate counted segmentation pixels only; a silver knife on a beige
counter passed with ~37 px but was invisible in RGB to people and to Qwen
(runs/vlm_diag_20261025). v2 additionally requires colour contrast: CIE Lab
distance between the median colour of the entity's pixels and the median of
a surrounding ring (the entity dilated by RING_PX, minus itself).
"""
from __future__ import annotations

import numpy as np

RING_PX = 6
# S1 gate (calibrated against the v1 knife: ~37 px, low contrast -> must fail)
S1_MIN_PX = 60
S1_MIN_DELTA_E = 15.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def entity_mask(frame, entity: str) -> np.ndarray:
    seg = np.asarray(frame.instance_segmentation)
    ids = [int(k) for k, v in frame.meta.get("instance_labels", {}).items() if v == entity]
    return np.isin(seg, ids) if ids else np.zeros(seg.shape[:2], bool)


def rgb_visibility(frame, entity: str) -> dict:
    mask = entity_mask(frame, entity)
    px = int(mask.sum())
    out = {"px": px, "delta_e": None}
    if px == 0:
        return out
    ring = _dilate(mask, RING_PX) & ~mask
    rgb = np.asarray(frame.rgb)[..., :3]
    if not ring.any():
        return out
    lab_obj = np.median(_srgb_to_lab(rgb[mask]), axis=0)
    lab_ring = np.median(_srgb_to_lab(rgb[ring]), axis=0)
    out["delta_e"] = round(float(np.linalg.norm(lab_obj - lab_ring)), 1)
    return out


def s1_visible(v: dict) -> bool:
    return v["px"] >= S1_MIN_PX and (v["delta_e"] or 0.0) >= S1_MIN_DELTA_E
