"""SceneSlots: scene-static searchable-space description.

Built on CPU from the BEHAVIOR scene JSON + per-model metadata.json, driven
by assets/tasks/slot_rules_v1.yaml. Contains NO robot-dependent data
(anchors, reachability): that is SlotEmbodimentOverlay.

Geometry is an approximation from metadata AABBs (recorded as
geometry.source) — good enough for CPU pruning; GPU probing confirms it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

SLOTS_VERSION = 1
CONTAINER_TYPES = ("drawer", "cabinet_interior")
_STRUCTURAL = {"walls", "floors", "ceilings"}


@dataclass(frozen=True)
class SlotGeometry:
    # usable inner extent (x, y, z) in the object frame, metres, scaled
    usable_extent: tuple[float, float, float]
    usable_volume: float
    # horizontal opening (two largest horizontal extents) for inside slots;
    # surface footprint for on_top slots
    opening_size: tuple[float, float]
    # approximate world height of the slot floor / surface top
    floor_z: float
    source: str = "metadata_aabb"


@dataclass(frozen=True)
class Slot:
    slot_id: str
    scene: str
    room: str
    room_type: str
    parent_entity: str
    parent_category: str
    parent_model: str
    relation: str  # inside | on_top
    slot_type: str  # drawer | cabinet_interior | countertop | table | desk
    link: Optional[str]
    requires_open: bool
    articulated: bool
    geometry: SlotGeometry
    # parent furniture root pose (world) — scene-static
    parent_position: tuple[float, float, float]
    parent_yaw: float
    classification: str = ""  # how slot_type was derived

    @property
    def is_container(self) -> bool:
        return self.slot_type in CONTAINER_TYPES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return _plain(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Slot":
        d = dict(d)
        g = dict(d.pop("geometry"))
        geom = SlotGeometry(usable_extent=tuple(g["usable_extent"]),
                            usable_volume=g["usable_volume"],
                            opening_size=tuple(g["opening_size"]),
                            floor_z=g["floor_z"], source=g.get("source", ""))
        d["parent_position"] = tuple(d["parent_position"])
        return cls(geometry=geom, **d)


@dataclass(frozen=True)
class Furniture:
    """Scene-static furniture record (for compiler initial_states / anchors)."""

    entity: str
    category: str
    model: str
    room: str
    room_type: str
    position: tuple[float, float, float]
    yaw: float
    openable: bool
    # scaled metadata bbox (object frame x, y, z), metres
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Landmark:
    """A visible fixed scene object an instruction may name (sink, fridge...)."""

    entity: str
    category: str
    name: str
    room: str
    position: tuple[float, float, float]
    yaw: float
    size: tuple[float, float, float]


@dataclass(frozen=True)
class Region:
    """Slot furniture right next to one landmark."""

    region_id: str
    room: str
    landmark: str  # landmark entity
    name: str  # landmark display name used in the instruction
    furniture: tuple[str, ...]
    # distance (m) to the nearest slot furniture OUTSIDE the region
    nearest_outside_m: Optional[float]
    clear: bool  # boundary unambiguous (clear band respected)
    unique_name: bool  # landmark name occurs once in the room

    @property
    def usable(self) -> bool:
        return bool(self.furniture) and self.clear and self.unique_name


@dataclass
class SceneSlots:
    scene: str
    version: int
    source: dict[str, Any]
    rooms: dict[str, str]  # room instance -> room type
    slots: list[Slot] = field(default_factory=list)
    furniture: list[Furniture] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    # room -> sorted categories of every other scene-native object (for
    # lookalike counting against objects the scene already contains)
    native_objects: dict[str, list[str]] = field(default_factory=dict)
    landmarks: list[Landmark] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    # entity -> world AABB [lo, hi] with every slot furniture closed, measured
    # in the simulator (measure_drawer_geometry.py): the static obstacle set
    # of skills/move.py, for CPU MOVE/TURN route checks; {} = not measured
    footprint_obstacles: dict[str, list[list[float]]] = field(default_factory=dict)

    def region(self, region_id: str) -> Region:
        for r in self.regions:
            if r.region_id == region_id:
                return r
        raise KeyError(f"{self.scene}: unknown region {region_id!r}")

    def by_id(self) -> dict[str, Slot]:
        return {s.slot_id: s for s in self.slots}

    def slots_in_room(self, room: str) -> list[Slot]:
        return [s for s in self.slots if s.room == room]

    def furniture_by_entity(self) -> dict[str, Furniture]:
        return {f.entity: f for f in self.furniture}

    def to_dict(self) -> dict[str, Any]:
        return _plain({
            "scene": self.scene,
            "version": self.version,
            "source": self.source,
            "rooms": dict(sorted(self.rooms.items())),
            "furniture": [asdict(f) for f in self.furniture],
            "slots": [s.to_dict() for s in self.slots],
            "skipped": self.skipped,
            "native_objects": {k: list(v) for k, v in sorted(self.native_objects.items())},
            "landmarks": [asdict(l) for l in self.landmarks],
            "regions": [asdict(r) for r in self.regions],
            "footprint_obstacles": dict(sorted(self.footprint_obstacles.items())),
        })

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneSlots":
        furn = []
        for f in d.get("furniture", []):
            f = dict(f)
            f["position"] = tuple(f["position"])
            f["size"] = tuple(f.get("size") or (0.0, 0.0, 0.0))
            furn.append(Furniture(**f))
        lms = [Landmark(**{**l, "position": tuple(l["position"]), "size": tuple(l["size"])})
               for l in d.get("landmarks") or []]
        regs = [Region(**{**r, "furniture": tuple(r["furniture"])})
                for r in d.get("regions") or []]
        return cls(scene=d["scene"], version=int(d["version"]),
                   source=dict(d.get("source", {})), rooms=dict(d["rooms"]),
                   slots=[Slot.from_dict(s) for s in d["slots"]],
                   furniture=furn, skipped=list(d.get("skipped", [])),
                   native_objects={k: list(v) for k, v in
                                   (d.get("native_objects") or {}).items()},
                   landmarks=lms, regions=regs,
                   footprint_obstacles=dict(d.get("footprint_obstacles") or {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "# Scene slots (generated by build_scene_slots.py).\n"
        path.write_text(header + yaml.safe_dump(self.to_dict(), sort_keys=False,
                                                width=100), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SceneSlots":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def room_type_of(room: str) -> str:
    """'kitchen_0' -> 'kitchen', 'private_office_0' -> 'private_office'."""
    return re.sub(r"_\d+$", "", room)


def yaw_from_quat_xyzw(q: list[float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def classify_link(extent: list[float], rules: dict[str, Any]) -> str:
    """'door' | 'drawer' | 'ambiguous' from a link AABB extent (x, y, z)."""
    thin = min(extent[0], extent[1])
    if thin <= rules["door_max_thickness_m"]:
        return "door"
    if thin >= rules["drawer_min_depth_m"] and extent[2] <= rules["drawer_max_height_m"]:
        return "drawer"
    return "ambiguous"


def build_scene_slots(scene: str, assets_root: Path, rules: dict[str, Any]) -> SceneSlots:
    scene_json = assets_root / "scenes" / scene / "json" / f"{scene}_best.json"
    doc = json.loads(scene_json.read_text(encoding="utf-8"))
    init = doc["objects_info"]["init_info"]
    registry = doc["state"]["registry"]["object_registry"]
    furn_rules = rules["furniture"]
    link_rules = rules["link_classification"]
    margin = float(rules["wall_margin_m"])
    version_file = assets_root / "VERSION"

    rooms: dict[str, str] = {}
    slots: list[Slot] = []
    furniture: list[Furniture] = []
    skipped: list[dict[str, Any]] = []
    native: dict[str, list[str]] = {}
    landmarks: list[Landmark] = []
    landmark_names = (rules.get("regions") or {}).get("landmarks") or {}
    for name in sorted(init):
        args = init[name]["args"]
        category = args.get("category")
        in_rooms = args.get("in_rooms") or []
        for r in in_rooms:
            rooms[r] = room_type_of(r)
        if category in landmark_names and len(in_rooms) == 1:
            lm = _landmark(name, args, registry[name]["root_link"], in_rooms[0],
                           landmark_names[category], assets_root)
            if lm is not None:
                landmarks.append(lm)
        if category not in furn_rules:
            if category not in _STRUCTURAL:
                for r in in_rooms:
                    native.setdefault(r, []).append(category)
            continue
        if len(in_rooms) != 1:
            skipped.append({"entity": name, "reason": f"in_rooms={in_rooms}"})
            continue
        room = in_rooms[0]
        model = args["model"]
        scale = [float(s) for s in (args.get("scale") or [1.0, 1.0, 1.0])]
        meta = json.loads((assets_root / "objects" / category / model / "misc"
                           / "metadata.json").read_text(encoding="utf-8"))
        root = registry[name]["root_link"]
        pos = tuple(round(float(v), 4) for v in root["pos"])
        yaw = round(yaw_from_quat_xyzw(root["ori"]), 4)
        openable = [l for l, tags in meta.get("link_tags", {}).items()
                    if "openable" in tags]
        bbox = [float(v) * s for v, s in zip(meta["bbox_size"], scale)]
        furniture.append(Furniture(entity=name, category=category, model=model,
                                   room=room, room_type=room_type_of(room),
                                   position=pos, yaw=yaw, openable=bool(openable),
                                   size=tuple(round(v, 4) for v in bbox)))
        bottom_z = pos[2] - bbox[2] / 2.0
        rule = furn_rules[category]
        common = dict(scene=scene, room=room, room_type=room_type_of(room),
                      parent_entity=name, parent_category=category,
                      parent_model=model, parent_position=pos, parent_yaw=yaw)

        if rule.get("surface"):
            ext = (bbox[0], bbox[1], 0.0)
            slots.append(Slot(
                slot_id=f"{scene}/{name}/on_top", relation="on_top",
                slot_type=rule["surface"], link=None, requires_open=False,
                articulated=bool(openable),
                geometry=_geom(ext, (bbox[0], bbox[1]), pos[2] + bbox[2] / 2.0),
                classification="surface_rule", **common))
        if rule.get("container"):
            if not openable:
                skipped.append({"entity": name, "reason": "container without openable links"})
                continue
            if args.get("fixed_base") is False:
                # Core grounds OPEN/CLOSE only on fixed-base articulated
                # furniture (object_interface.build_object_interface); a
                # movable cabinet is modelled as a rigid, graspable object
                skipped.append({"entity": name, "reason": "movable (fixed_base=False): "
                                "Core has no OPEN interface for it"})
                continue
            boxes = meta.get("link_bounding_boxes", {})
            has_door = False
            for link in sorted(openable):
                box = boxes.get(link, {}).get("collision", {}).get("axis_aligned")
                if box is None:
                    skipped.append({"entity": name, "link": link, "reason": "no link bbox"})
                    continue
                raw = [float(v) for v in box["extent"]]
                kind = classify_link(raw, link_rules)
                if kind == "door":
                    has_door = True
                    continue
                if kind == "ambiguous":
                    skipped.append({"entity": name, "link": link,
                                    "reason": f"ambiguous link extent {raw}"})
                    continue
                ext = [max(0.0, v * s - 2 * margin) for v, s in zip(raw, scale)]
                local_z = float(box["transform"][2][3]) * scale[2]
                slots.append(Slot(
                    slot_id=f"{scene}/{name}/{link}", relation="inside",
                    slot_type="drawer", link=link, requires_open=True,
                    articulated=True,
                    geometry=_geom(tuple(ext), (ext[0], ext[1]),
                                   pos[2] + local_z - raw[2] * scale[2] / 2.0),
                    classification="link_aabb:drawer", **common))
            if has_door:
                ext = [max(0.0, v - 2 * margin) for v in bbox]
                slots.append(Slot(
                    slot_id=f"{scene}/{name}/interior", relation="inside",
                    slot_type="cabinet_interior", link=None, requires_open=True,
                    articulated=True,
                    geometry=_geom(tuple(ext), (ext[0], ext[1]), bottom_z + margin),
                    classification="link_aabb:door", **common))

    source = {
        "scene_json": str(scene_json),
        "scene_json_sha256": hashlib.sha256(scene_json.read_bytes()).hexdigest(),
        "assets_version": (version_file.read_text().strip()
                           if version_file.exists() else None),
        "rules_version": int(rules["version"]),
    }
    return SceneSlots(scene=scene, version=SLOTS_VERSION, source=source,
                      rooms=rooms, slots=slots, furniture=furniture, skipped=skipped,
                      native_objects={r: sorted(c) for r, c in native.items()},
                      landmarks=landmarks,
                      regions=build_regions(scene, landmarks, furniture, slots,
                                            rules.get("regions") or {}))


def _landmark(name, args, root, room, display, assets_root) -> Optional[Landmark]:
    meta_path = (assets_root / "objects" / args["category"] / args["model"] / "misc"
                 / "metadata.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scale = [float(s) for s in (args.get("scale") or [1.0, 1.0, 1.0])]
    size = tuple(round(float(v) * s, 4) for v, s in zip(meta["bbox_size"], scale))
    return Landmark(entity=name, category=args["category"], name=display, room=room,
                    position=tuple(round(float(v), 4) for v in root["pos"]),
                    yaw=round(yaw_from_quat_xyzw(root["ori"]), 4), size=size)


def build_regions(scene: str, landmarks: list[Landmark], furniture: list[Furniture],
                  slots: list[Slot], rules: dict[str, Any]) -> list[Region]:
    """Slot furniture touching each landmark (oriented 2-D footprints)."""
    from rummagebench.authoring.tasks.regions import footprint, rect_distance

    if not rules:
        return []
    touch = float(rules["touch_m"])
    band = float(rules["clear_band_m"])
    with_slots = {s.parent_entity for s in slots}
    names: dict[tuple[str, str], int] = {}
    for lm in landmarks:
        names[(lm.room, lm.name)] = names.get((lm.room, lm.name), 0) + 1
    out = []
    for lm in landmarks:
        lrect = footprint(lm.position, lm.size, lm.yaw)
        members, outside = [], []
        for f in furniture:
            if f.room != lm.room or f.entity not in with_slots or f.entity == lm.entity:
                continue
            d = rect_distance(lrect, footprint(f.position, f.size, f.yaw))
            (members if d <= touch else outside).append((d, f.entity))
        nearest = min((d for d, _ in outside), default=None)
        out.append(Region(
            region_id=f"{scene}/{lm.room}/near_{lm.entity}", room=lm.room,
            landmark=lm.entity, name=lm.name,
            furniture=tuple(sorted(e for _, e in members)),
            nearest_outside_m=None if nearest is None else round(nearest, 4),
            clear=nearest is None or nearest > touch + band,
            unique_name=names[(lm.room, lm.name)] == 1))
    return sorted(out, key=lambda r: r.region_id)


def _geom(ext, opening, floor_z) -> SlotGeometry:
    ext = tuple(round(float(v), 4) for v in ext)
    vol = round(ext[0] * ext[1] * ext[2], 6)
    return SlotGeometry(usable_extent=ext, usable_volume=vol,
                        opening_size=tuple(round(float(v), 4) for v in opening),
                        floor_z=round(float(floor_z), 4))


def _plain(obj: Any) -> Any:
    """Tuples -> lists recursively (YAML/JSON canonical form)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


MEASURED_SOURCE = "sim_collision_v1"
MEASURE_FAILED_SOURCE = "sim_collision_v1_failed"


def apply_measured_geometry(ss: SceneSlots, measured: dict[str, Any], margin: float
                            ) -> dict[str, int]:
    """Replace metadata-AABB drawer geometry by simulator measurements
    (scripts/tasks/measure_drawer_geometry.py).

    metadata link_bounding_boxes are in the LINK frame, whose origin is only
    in the encrypted USD: the derived floor height is wrong, and the height
    ignores what sits above the drawer. Measured: floor = top of the drawer's
    floor slab; footprint = that slab (world AABB, rotated into the furniture
    frame); height = clearance to the lowest body above the footprint, or the
    drawer's own wall height when nothing is above. A drawer whose
    measurement failed gets a zero usable extent (nothing fits: its metadata
    geometry is known to be wrong); unmeasured slots keep their metadata
    geometry (counted, never silently mixed)."""
    import dataclasses

    counts = {"measured": 0, "failed": 0, "absent": 0}
    out = []
    for s in ss.slots:
        m = measured.get(s.slot_id)
        if s.slot_type != "drawer":
            out.append(s)
            continue
        if m is None:
            counts["absent"] += 1
            out.append(s)
            continue
        if not m.get("ok"):
            counts["failed"] += 1
            out.append(dataclasses.replace(s, geometry=dataclasses.replace(
                _geom((0.0, 0.0, 0.0), (0.0, 0.0), s.geometry.floor_z),
                source=MEASURE_FAILED_SOURCE)))
            continue
        (x0, y0), (x1, y1) = m["inner_xy"]
        dx, dy = x1 - x0, y1 - y0
        if abs(math.sin(s.parent_yaw)) > abs(math.cos(s.parent_yaw)):
            dx, dy = dy, dx  # world axes -> furniture frame
        height = (m["clearance_m"] if m.get("clearance_m") is not None
                  else m["walls_top_z"] - m["floor_top_z"])
        ext = (max(0.0, dx - 2 * margin), max(0.0, dy - 2 * margin),
               max(0.0, height - margin))
        geom = dataclasses.replace(_geom(ext, (ext[0], ext[1]), m["floor_top_z"]),
                                   source=MEASURED_SOURCE)
        out.append(dataclasses.replace(s, geometry=geom))
        counts["measured"] += 1
    ss.slots = out
    return counts
