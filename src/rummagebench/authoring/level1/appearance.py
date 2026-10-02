"""§15: appearance stream — parameter sampling only.

Sampling is CPU work from the appearance stream; APPLYING the parameters
(lights, textures) is a GPU-stage responsibility that must never touch
physics, collision, transforms or task semantics (revision §1: renderer
quality settings stay fixed and are recorded, not randomized).
"""
from __future__ import annotations

from rummagebench.authoring.level1.config import load_appearance_config


def sample_appearance(appearance_stream, config: dict | None = None) -> dict:
    cfg = config if config is not None else load_appearance_config()
    ranges = cfg["randomize"]
    rng = appearance_stream

    def uniform(name: str) -> float:
        lo, hi = ranges[name]
        return float(lo + (hi - lo) * rng.random())

    params: dict[str, object] = {}
    if "light_direction_yaw_deg" in ranges:
        params["light_direction_yaw_deg"] = uniform("light_direction_yaw_deg")
    if "light_intensity_multiplier" in ranges:
        params["light_intensity_multiplier"] = uniform(
            "light_intensity_multiplier")
    if "light_color_temperature_k" in ranges:
        params["light_color_temperature_k"] = uniform(
            "light_color_temperature_k")
    if "ambient_multiplier" in ranges:
        params["ambient_multiplier"] = uniform("ambient_multiplier")
    if ranges.get("wall_color_palette"):
        palette = ranges["wall_color_palette"]
        params["wall_color"] = palette[int(rng.integers(0, len(palette)))]
    params["support_texture_swap"] = bool(ranges.get("support_texture_swap"))
    params["container_texture_swap"] = bool(ranges.get("container_texture_swap"))
    params["fixed_renderer"] = dict(cfg.get("fixed_renderer", {}))
    params["forbid"] = dict(cfg.get("forbid", {}))
    return params
