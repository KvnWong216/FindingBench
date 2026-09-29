"""Unit-test fixtures: a minimal in-memory SimBackend double.

These tests verify the benchmark core (schema, validators, session, grounding,
feasibility, failure branches) WITHOUT the simulator, which also validates
that the backend abstraction is real and OmniGibson is genuinely pluggable.

The double supports both feasibility engines:
- proxy mode: entity poses + whole-entity AABBs (reach radius + point check)
- configuration mode: articulation_info (configurable handle links/offsets),
  link poses/AABBs and per-entity collision geometries (AABB boxes)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "mini.yaml"

from rummagebench.sim.base import (
    ArticulationInfo,
    ArticulationJointInfo,
    EntityInfo,
    ResolvedTarget,
    SimBackend,
    WorldCollisionObject,
)
from rummagebench.core.types import TargetKind
from rummagebench.feasibility.ik_solver import Pose


class FakeBackend(SimBackend):
    """Minimal backend double for unit tests.

    entities: name -> EntityInfo; holding: entity currently grasped;
    opens: set of open entities; poses: name -> [x,y,z];
    aabbs: name -> (lo,hi); articulations: name -> ArticulationInfo;
    link_poses: (entity, link) -> [x,y,z]; link_aabbs: (entity, link) -> (lo,hi).
    """

    def __init__(
        self,
        entities: dict[str, EntityInfo],
        anchors: set[str],
        poses: dict[str, list[float]] | None = None,
        aabbs: dict[str, tuple[list[float], list[float]]] | None = None,
        articulations: dict[str, ArticulationInfo] | None = None,
        link_poses: dict[tuple[str, str], list[float]] | None = None,
        link_aabbs: dict[tuple[str, str], tuple[list[float], list[float]]] | None = None,
    ):
        self.entities = entities
        self.anchors = anchors
        self.poses = poses or {}
        self.aabbs = aabbs or {}
        self.articulations = articulations or {}
        self.link_poses = link_poses or {}
        self.link_aabbs_map = link_aabbs or {}
        self.holding_entity: str | None = None
        self.opens: set[str] = set()
        # container -> [entities inside it] (visibility model, §20)
        self.contains: dict[str, list[str]] = {}
        self._counter = 0
        # GRASP realization must never move the base: this records the base
        # pose at grasp time for the no-teleport regression test
        self.grasp_base_pose: tuple[list[float], list[float]] | None = None

    def setup(self, scenario):
        return {"ok": True}

    def reset(self) -> None:
        self.holding_entity = None
        self.opens = set()

    def get_observation(self) -> Any:
        self._counter += 1
        return np.full((2, 2, 3), self._counter % 255, dtype=np.uint8)

    def resolve_entity(self, name: str) -> ResolvedTarget:
        info = self.entities.get(name)
        if info is None:
            from rummagebench.core.errors import UnresolvableTargetError

            raise UnresolvableTargetError(f"unknown entity {name!r}")
        return ResolvedTarget(kind=TargetKind.ENTITY, entity=name, info=info)

    def teleport_robot(self, anchor) -> None:
        self._last_anchor_position = list(anchor.position)

    def robot_pose(self):
        return getattr(self, "_last_anchor_position", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]

    def set_open(self, entity: str, open_value: bool) -> bool:
        if entity not in self.entities or not self.entities[entity].openable:
            return False
        if open_value:
            self.opens.add(entity)
        else:
            self.opens.discard(entity)
        return True

    def is_open(self, entity: str) -> bool:
        return entity in self.opens

    def symbolic_grasp(self, entity: str) -> bool:
        if entity not in self.entities or not self.entities[entity].graspable:
            return False
        # realization records (never moves) the base pose
        self.grasp_base_pose = self.robot_pose()
        self.holding_entity = entity
        return True

    def is_holding(self, entity: str) -> bool:
        return self.holding_entity == entity

    def settle(self, steps: int = 10) -> None:
        pass

    def dump_state(self) -> Any:
        return {"opens": set(self.opens), "holding": self.holding_entity}

    def load_state(self, state: Any) -> None:
        self.opens = set(state["opens"])
        self.holding_entity = state["holding"]

    def render_snapshot(self, path: str) -> None:
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------- embodiment grounding

    def entity_names(self) -> list[str]:
        return list(self.entities.keys())

    def entity_pose(self, name: str) -> list[float]:
        return list(self.poses.get(name, [0.0, 0.0, 0.5]))

    def entity_aabb(self, name: str):
        return self.aabbs.get(name)

    def entity_pose6d(self, name: str) -> Pose | None:
        pos = self.entity_pose(name)
        return Pose.from_lists(pos)

    def held_count(self) -> int:
        return 1 if self.holding_entity is not None else 0

    def symbolic_place(self, entity: str, receptacle: str) -> bool:
        if self.holding_entity != entity:
            return False
        self.holding_entity = None
        return True

    def describe_entity(self, name: str) -> EntityInfo | None:
        return self.entities.get(name)

    # ------------------------------------------------- physical grounding API

    def articulation_info(self, entity: str) -> ArticulationInfo | None:
        return self.articulations.get(entity)

    def link_pose(self, entity: str, link: str) -> Pose | None:
        pos = self.link_poses.get((entity, link))
        if pos is None:
            return None
        return Pose.from_lists(pos)

    def link_aabb(self, entity: str, link: str):
        return self.link_aabbs_map.get((entity, link))

    def receptacle_region(self, entity: str):
        aabb = self.aabbs.get(entity)
        if aabb is None:
            return None
        lo, hi = aabb
        if entity in self.opens:
            from rummagebench.feasibility.interaction_target import InteractionRegion

            return InteractionRegion(lo=list(lo), hi=list(hi), kind="inside_volume")
        from rummagebench.feasibility.interaction_target import InteractionRegion

        return InteractionRegion(lo=list(lo), hi=list(hi), kind="top_surface")

    def visible_entities(self) -> list[str]:
        """Closed-container contents are not observable (§19/§20)."""
        hidden: set[str] = set()
        for container, entities in self.contains.items():
            if container not in self.opens:
                hidden.update(entities)
        return [n for n in self.entities if n not in hidden]

    def collision_geometries(self) -> list[WorldCollisionObject]:
        """AABB boxes for every entity (approximation level recorded)."""
        bodies: list[WorldCollisionObject] = []
        for name, aabb in self.aabbs.items():
            lo, hi = aabb
            center = [(a + b) / 2 for a, b in zip(lo, hi)]
            half = [(b - a) / 2 for a, b in zip(lo, hi)]
            body = WorldCollisionObject(
                entity=name,
                link=None,
                geometry=_coal_box(*half),
                pose=Pose.from_lists(center),
                category=self.entities[name].category if name in self.entities else "",
                approximation="aabb_primitive",
                aabb=(list(lo), list(hi)),
            )
            bodies.append(body)
        return bodies


def _coal_box(hx: float, hy: float, hz: float):
    """coal Box geometry (requires the pin wheel; unit tests import it)."""
    try:
        import coal
    except ImportError:
        import pinocchio as pin
        coal = pin.hppfcl
    return coal.Box(hx, hy, hz)


@pytest.fixture
def fake_backend():
    return FakeBackend(
        entities={
            "target_knife": EntityInfo(name="target_knife", category="knife"),
            "distractor_spoon": EntityInfo(name="distractor_spoon", category="spoon"),
            "cabinet_A": EntityInfo(
                name="cabinet_A", category="cabinet", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "cabinet_B": EntityInfo(
                name="cabinet_B", category="cabinet", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "drawer_A": EntityInfo(
                name="drawer_A", category="drawer", fixed_base=True, openable=True,
                is_receptacle=True,
            ),
            "countertop": EntityInfo(
                name="countertop", category="countertop", fixed_base=True
            ),
            "hot_pot": EntityInfo(name="hot_pot", category="pot_nonstick"),
        },
        anchors={"living_room", "kitchen"},
        poses={
            "target_knife": [3.05, 1.05, 0.35],
            "distractor_spoon": [3.02, 1.02, 0.30],
            "cabinet_A": [2.30, 2.30, 0.45],
            "cabinet_B": [3.00, 1.00, 0.45],
            "drawer_A": [3.65, 0.50, 0.30],
            "countertop": [2.80, 0.80, 0.60],
            "hot_pot": [3.30, 1.30, 0.90],
        },
        aabbs={
            "target_knife": ([3.02, 1.02, 0.32], [3.08, 1.08, 0.38]),
            "distractor_spoon": ([2.99, 0.99, 0.27], [3.05, 1.05, 0.33]),
            "hot_pot": ([3.24, 1.24, 0.84], [3.36, 1.36, 0.96]),
            "cabinet_B": ([2.7, 0.7, 0.0], [3.3, 1.3, 0.9]),
            "cabinet_A": ([2.0, 2.0, 0.0], [2.6, 2.6, 0.9]),
            "drawer_A": ([3.4, 0.2, 0.0], [3.9, 0.8, 0.6]),
            "countertop": ([2.4, 0.4, 0.5], [3.6, 1.6, 0.7]),
        },
        # articulated interaction interfaces: handle anchor 0.15 m in front of
        # each cabinet root (proxy tests never use these; config-mode tests do)
        articulations={
            "cabinet_A": ArticulationInfo(
                entity="cabinet_A",
                joints=[
                    ArticulationJointInfo(
                        name="cabinet_A_door_joint", joint_type="revolute",
                        parent_link="cabinet_A_body", child_link="cabinet_A_door",
                        axis=[0.0, 0.0, 1.0], limits=(0.0, 1.6),
                    )
                ],
                links=["cabinet_A_body", "cabinet_A_door"],
            ),
            "cabinet_B": ArticulationInfo(
                entity="cabinet_B",
                joints=[
                    ArticulationJointInfo(
                        name="cabinet_B_door_joint", joint_type="revolute",
                        parent_link="cabinet_B_body", child_link="cabinet_B_door",
                        axis=[0.0, 0.0, 1.0], limits=(0.0, 1.6),
                    )
                ],
                links=["cabinet_B_body", "cabinet_B_door"],
            ),
            "drawer_A": ArticulationInfo(
                entity="drawer_A",
                joints=[
                    ArticulationJointInfo(
                        name="drawer_A_slide_joint", joint_type="prismatic",
                        parent_link="drawer_A_body", child_link="drawer_A_front",
                        axis=[1.0, 0.0, 0.0], limits=(0.0, 0.3),
                    )
                ],
                links=["drawer_A_body", "drawer_A_front"],
            ),
        },
        link_poses={
            ("cabinet_A", "cabinet_A_body"): [2.30, 2.30, 0.45],
            ("cabinet_A", "cabinet_A_door"): [2.15, 2.30, 0.45],
            ("cabinet_B", "cabinet_B_body"): [3.00, 1.00, 0.45],
            ("cabinet_B", "cabinet_B_door"): [2.85, 1.00, 0.45],
            ("drawer_A", "drawer_A_body"): [3.65, 0.50, 0.30],
            ("drawer_A", "drawer_A_front"): [3.50, 0.50, 0.30],
        },
        link_aabbs={
            ("cabinet_A", "cabinet_A_door"): ([2.10, 2.25, 0.0], [2.20, 2.35, 0.9]),
            ("cabinet_B", "cabinet_B_door"): ([2.80, 0.95, 0.0], [2.90, 1.05, 0.9]),
            ("drawer_A", "drawer_A_front"): ([3.45, 0.45, 0.0], [3.55, 0.55, 0.6]),
        },
    )


# ============================================================================
# Visual-protocol fixtures (final AGENT protocol): a FakeBackend with
# synchronized synthetic camera frames (rgb/depth/seg_instance) so the whole
# public-protocol pipeline runs WITHOUT the simulator.
# ============================================================================

FRAME_H = FRAME_W = 64


class VisualFakeBackend(FakeBackend):
    """FakeBackend + protocol §7 private frame capture.

    Synthetic frames: each entity in `frame_entities` renders as a filled
    square at a fixed pixel slot with constant depth 1.0 m; camera K default
    90-deg HFOV, extrinsics identity, convention z_depth.
    """

    def __init__(self, *args, frame_entities=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_entities = frame_entities or {}  # entity -> (u, v, w, h)
        self._teleports: list[tuple] = []

    def teleport_robot(self, anchor) -> None:
        self._teleports.append((tuple(anchor.position), tuple(anchor.orientation)))
        self._last_anchor_position = list(anchor.position)

    def _synthetic_frame(self):
        from rummagebench.perception.frame_store import VisualFramePrivate
        from rummagebench.perception.camera_geometry import default_intrinsics

        seg = np.zeros((FRAME_H, FRAME_W), dtype=np.int64)
        depth = np.full((FRAME_H, FRAME_W), np.nan)
        rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        labels: dict[str, str] = {}
        for idx, (entity, (u, v, w, h)) in enumerate(self.frame_entities.items(), start=2):
            seg[v:v + h, u:u + w] = idx
            depth[v:v + h, u:u + w] = 1.0
            rgb[v:v + h, u:u + w] = (200, 200, 200)
            labels[str(idx)] = entity
        self._instance_labels = labels
        return VisualFramePrivate(
            frame_id="", rgb=rgb, depth=depth, instance_segmentation=seg,
            camera_intrinsics=default_intrinsics(FRAME_W, FRAME_H),
            camera_extrinsics=np.eye(4),
            image_width=FRAME_W, image_height=FRAME_H,
            depth_convention="z_depth",
        )

    # protocol API
    def capture_visual_frame(self):
        return self._synthetic_frame()

    def instance_to_entity(self, instance):
        labels = getattr(self, "_instance_labels", {})
        label = labels.get(str(instance.item() if hasattr(instance, "item") else instance))
        return label if label in self.entities else None

    def entity_visible_pixels(self, frame, entity: str) -> int:
        seg = np.asarray(frame.instance_segmentation)
        for key, label in getattr(self, "_instance_labels", {}).items():
            if label == entity:
                return int(np.count_nonzero(seg == int(key)))
        return 0


@pytest.fixture
def visual_backend(fake_backend):
    """FakeBackend with the mini-fixture world + synthetic frames:
    cabinet_B at left of frame, distractor_spoon at right, knife hidden
    (not in frame: closed-container contents never render)."""
    from rummagebench.core.scenario import load_scenario

    vb = VisualFakeBackend(
        fake_backend.entities,
        fake_backend.anchors,
        poses=fake_backend.poses,
        aabbs=fake_backend.aabbs,
        frame_entities={
            "cabinet_B": (4, 24, 20, 20),
            "distractor_spoon": (44, 24, 12, 12),
        },
    )
    vb._scenario = load_scenario(FIXTURE)
    return vb


@pytest.fixture
def visual_session(visual_backend):
    from rummagebench.core.scenario import load_scenario
    from rummagebench.core.visual_session import VisualProtocolSession

    scenario = load_scenario(FIXTURE)
    session = VisualProtocolSession(visual_backend, scenario)
    session.set_trace(None)
    session.reset()
    return session
