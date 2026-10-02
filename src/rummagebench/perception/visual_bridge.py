"""Visual bridge: agent 2D point -> simulator entity + private 3D surface point.

Pipeline (protocol §8, grounding_version=center_pixel_v2, revision §4):
    point -> synchronized private frame -> AUTHORITATIVE center-pixel
    instance selection (background/unknown/robot-self center is invalid)
    -> owning-object canonicalization (handle/door/link -> root entity)
    -> surface point from the selected object's OWN samples inside the
    11x11 window (their true pixel coordinates and depths) -> unprojection
    -> selected_surface_point_world

Segmentation decides WHICH object; depth decides WHERE the visible surface is;
the ObjectInterface — never the clicked point — decides where the ROBOT
interacts (protocol §9). The neighborhood NEVER overrides the center-pixel
choice: a thin target remains selectable against any background.
"""

from __future__ import annotations

import numpy as np

from rummagebench.perception.camera_geometry import (
    normalized_to_pixel,
    to_world,
    unproject_range,
    unproject_z_depth,
)


class VisualGroundingError(Exception):
    """Private subreasons; the public feedback is always INVALID_ACTION."""

    def __init__(self, subreason: str):
        super().__init__(subreason)
        self.subreason = subreason


# Selected-component version bound into certificates and click records
# (revision §4). Bump on any change to the picking contract.
GROUNDING_VERSION = "center_pixel_v2"


class VisualBridge:
    def __init__(
        self,
        patch_size: int = 11,
        min_valid_pixels: int = 20,
        dominant_ratio: float = 0.50,
    ):
        self.patch_size = patch_size
        self.min_valid_pixels = min_valid_pixels
        self.dominant_ratio = dominant_ratio

    # ------------------------------------------------------------ selection

    def select_instance(self, frame, x: float, y: float):
        """Authoritative selection: the renderer instance AT the center pixel.

        Revision §4: the neighborhood no longer chooses the object — a thin
        target surrounded by other surfaces stays selected. A background /
        unknown / robot-self center is NO_VISUAL_TARGET (public
        INVALID_ACTION); the click never snaps to a different object.
        """
        seg = np.asarray(frame.instance_segmentation)
        H, W = seg.shape[:2]
        u, v = normalized_to_pixel(x, y, W, H)
        center = seg[v, u]
        key = center.item() if hasattr(center, "item") else center
        if key in (0, "", None, "background", 0xFFFFFFFF):
            raise VisualGroundingError("NO_VISUAL_TARGET")
        return key, (u, v)

    # ------------------------------------------------------ surface point

    def surface_point(self, frame, instance, pixel: tuple[int, int]) -> np.ndarray:
        """Return an actual valid sample near the patch's median depth.

        The representative is restricted to the clicked instance and 11x11
        neighborhood; unrelated portions of the object cannot move it.
        """
        seg = np.asarray(frame.instance_segmentation)
        depth = np.asarray(frame.depth, dtype=float)
        if depth.shape != seg.shape or frame.depth_convention not in ("z_depth", "euclidean_range"):
            raise VisualGroundingError("INVALID_DEPTH_GEOMETRY")
        u, v = pixel
        r = self.patch_size // 2
        v0, v1 = max(0, v-r), min(seg.shape[0], v+r+1)
        u0, u1 = max(0, u-r), min(seg.shape[1], u+r+1)
        patch = depth[v0:v1, u0:u1]
        mask = (seg[v0:v1, u0:u1] == instance) & np.isfinite(patch) & (patch > 0)
        rows, cols = np.where(mask)
        if not len(rows):
            raise VisualGroundingError("NO_VALID_DEPTH")
        vals = patch[mask]
        median = np.median(vals)
        # Depth robustness first, then proximity to the selected pixel.
        order = np.lexsort(((cols+u0-u)**2 + (rows+v0-v)**2, np.abs(vals-median)))
        idx = order[0]
        pu, pv, d = int(cols[idx]+u0), int(rows[idx]+v0), float(vals[idx])
        unproject = unproject_z_depth if frame.depth_convention == "z_depth" else unproject_range
        return to_world(unproject(pu, pv, d, frame.camera_intrinsics), frame.camera_extrinsics)

    # ------------------------------------------------------------- resolve

    def resolve(self, frame, x: float, y: float, canonicalize):
        """Full pipeline. `canonicalize(instance) -> entity name | None` maps a
        renderer instance to the OWNING FindingBench entity (§8.3)."""
        instance, (u, v) = self.select_instance(frame, x, y)
        entity = canonicalize(instance)
        if entity is None:
            raise VisualGroundingError("NO_OWNING_ENTITY")
        point = self.surface_point(frame, instance, (u, v))
        return entity, point, {"pixel": [int(u), int(v)], "instance": str(instance)}
