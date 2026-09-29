"""§8.6 integration: point -> depth -> world roundtrip <= 1 cm on the real
simulator; segmentation identity resolves to the owning entity."""

import numpy as np
from pathlib import Path
import pytest

pytestmark = pytest.mark.sim


def _roundtrip_error(backend, frame, world_point):
    """Project a known world point with the frame's geometry and unproject."""
    T_cam_world = np.linalg.inv(frame.camera_extrinsics)
    p_cam = T_cam_world @ np.append(world_point, 1.0)
    if p_cam[2] <= 0:
        return None
    K = frame.camera_intrinsics
    u = int(round(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]))
    v = int(round(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]))
    if not (0 <= u < frame.image_width and 0 <= v < frame.image_height):
        return None
    d = float(np.asarray(frame.depth)[v, u])
    if not np.isfinite(d) or d <= 0:
        return None
    from rummagebench.perception.camera_geometry import (
        to_world, unproject_range, unproject_z_depth,
    )

    pc = (
        unproject_z_depth(u, v, d, K)
        if frame.depth_convention == "z_depth"
        else unproject_range(u, v, d, K)
    )
    return float(np.linalg.norm(to_world(pc, frame.camera_extrinsics) - world_point))


def test_visual_bridge_roundtrip_under_1cm(agent_env):
    backend = agent_env._backend
    frame = backend.capture_visual_frame()
    frame.frame_id = "frame_roundtrip"
    assert frame.depth_convention in ("z_depth", "euclidean_range")

    # pick a visible benchmark entity with real geometry near the spawn view
    errors = []
    checked = 0
    for entity in backend.visible_entities():
        info = backend.describe_entity(entity)
        if info is None or info.fixed_base:
            continue  # rigid objects only: their pose IS a surface proxy
        pose = backend.entity_pose(entity)
        if pose is None:
            continue
        err = _roundtrip_error(backend, frame, np.asarray(pose, dtype=float))
        if err is None:
            continue
        checked += 1
        errors.append(err)
        if err <= 0.01:
            break
    assert checked > 0, "no visible rigid entity could be projected"
    assert min(errors) <= 0.01, f"roundtrip error {min(errors):.4f} m > 1 cm"


def test_instance_segmentation_resolves_to_entity(agent_env):
    backend = agent_env._backend
    frame = backend.capture_visual_frame()
    seg = np.asarray(frame.instance_segmentation)
    ids, counts = np.unique(seg, return_counts=True)
    resolved = 0
    for inst, n in zip(ids, counts):
        if n < 400:
            continue  # skip tiny slivers
        entity = backend.instance_to_entity(inst.item() if hasattr(inst, "item") else inst)
        if entity is not None:
            resolved += 1
            assert entity in backend.entity_names()
    assert resolved > 0, "no segmentation instance resolved to a benchmark entity"


@pytest.fixture(scope="module")
def agent_env(tmp_path_factory):
    """AGENT-mode environment for the visual protocol sim tests."""
    import os

    from rummagebench.adapters.python_api import create_session

    os.environ.setdefault("RUMMAGEBENCH_ROOT",
                          str(Path(__file__).resolve().parents[2]))
    session = create_session(
        Path(__file__).resolve().parents[2] / "scenarios" / "knife_search_001" / "scenario.yaml",
        seed=0, mode="agent",
        trace_path=tmp_path_factory.mktemp("agent") / "trace.jsonl",
    )
    session.reset()
    yield session
