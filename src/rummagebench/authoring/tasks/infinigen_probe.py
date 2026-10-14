"""Infinigen inputs for the existing SceneProber; CPU preparation never probes.

BEHAVIOR's PNG room maps and scene_json floor lookup are replaced by actual
exported kitchen floor triangles. Sampling proposes poses only. Live checks
retain the production footprint, IK, renderer, placement and reveal gates.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from rummagebench.authoring.level1.robot_registry import runtime_protocol_hash
from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


@dataclass
class GeometryRoomMap:
    resolution: float
    origin: tuple[float, float]
    trav: np.ndarray
    floor_z: float
    room: str = "kitchen_0"

    def _distance(self):
        from scipy.ndimage import distance_transform_edt
        # Explicit occupied padding: clearance cannot extend beyond the map.
        return distance_transform_edt(np.pad(self.trav, 1))[1:-1, 1:-1] * self.resolution

    def to_world(self, row, col):
        return (float(self.origin[0] + col * self.resolution),
                float(self.origin[1] + row * self.resolution))

    def free_points(self, room, clearance_m, spacing_m):
        if room != self.room:
            return []
        mask = self.trav & (self._distance() >= clearance_m)
        step = max(1, int(round(spacing_m / self.resolution)))
        return [self.to_world(r, c) for r in range(0, mask.shape[0], step)
                for c in range(0, mask.shape[1], step) if mask[r, c]]

    def clearest_points(self, k=10, room=None):
        if room not in (None, self.room):
            return []
        dist = self._distance()
        indices = np.argsort(-dist, axis=None, kind="stable")
        return [(float(dist.flat[i]), self.to_world(*np.unravel_index(i, dist.shape)))
                for i in indices if self.trav.flat[i]][:k]

    def to_dict(self):
        return {"resolution": self.resolution, "origin": list(self.origin),
                "floor_z": self.floor_z, "room": self.room, "trav": self.trav.astype(int).tolist(),
                "scope": "actual_floor_triangles_and_furniture_envelopes_pose_proposals_only"}

    @classmethod
    def from_dict(cls, doc):
        mask = np.asarray(doc["trav"], dtype=bool)
        if mask.ndim != 2 or not mask.any() or mask.size > 4_000_000:
            raise ValueError("INVALID_GEOMETRY_PROPOSAL_MAP")
        return cls(float(doc["resolution"]), tuple(doc["origin"]), mask, float(doc["floor_z"]), doc["room"])


def raster_floor(triangles, obstacles, resolution=.1):
    """Rasterise actual horizontal floor triangles; never infer a rectangular room."""
    tris = np.asarray(triangles, float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or not np.isfinite(tris).all() or not len(tris):
        raise ValueError("KITCHEN_FLOOR_TRIANGLES_REQUIRED")
    if not .02 <= resolution <= .25:
        raise ValueError("INVALID_MAP_RESOLUTION")
    z = float(tris[:, :, 2].mean())
    if np.ptp(tris[:, :, 2]) > .02:
        raise ValueError("SINGLE_LEVEL_FLOOR_REQUIRED")
    lo = tris[:, :, :2].min(axis=(0, 1)) - resolution
    hi = tris[:, :, :2].max(axis=(0, 1)) + resolution
    size = np.ceil((hi - lo) / resolution).astype(int) + 1
    if np.prod(size) > 4_000_000:
        raise ValueError("GEOMETRY_MAP_TOO_LARGE")
    xs, ys = np.meshgrid(lo[0] + np.arange(size[0]) * resolution,
                         lo[1] + np.arange(size[1]) * resolution)
    mask = np.zeros(xs.shape, bool)
    for tri in tris[:, :, :2]:
        a, b, c = tri
        determinant = np.cross(b - a, c - a)
        if abs(determinant) < 1e-10:
            continue
        u = ((xs-a[0])*(c-a)[1] - (ys-a[1])*(c-a)[0]) / determinant
        v = ((b-a)[0]*(ys-a[1]) - (b-a)[1]*(xs-a[0])) / determinant
        mask |= (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
    for low, high in obstacles:
        if high[2] >= z+.05 and low[2] <= z+.5:
            mask &= ~((xs >= low[0]-.03) & (xs <= high[0]+.03)
                      & (ys >= low[1]-.03) & (ys <= high[1]+.03))
    if not mask.any():
        raise ValueError("FLOOR_PROPOSAL_MAP_EMPTY")
    return GeometryRoomMap(resolution, tuple(lo), mask, z)


def assembly_floor_map(manifest, assembly):
    from pxr import Usd, UsdGeom
    from rummagebench.authoring.tasks.infinigen import read_solve_state

    meta = read_solve_state(Path(manifest["source"]["solve_state"]))
    if len(meta["kitchens"]) != 1:
        raise ValueError("SINGLE_KITCHEN_REQUIRED_FOR_PROBE")
    kitchen = meta["kitchens"][0]
    path = manifest["source_to_usd"].get(kitchen)
    if path is None:
        raise ValueError("KITCHEN_MESH_MAPPING_REQUIRED_FOR_PROBE")
    stage = Usd.Stage.Open(manifest["source"]["scene_usd"])
    xforms = UsdGeom.XformCache()
    triangles = []
    floor_path = manifest.get("source_component_paths",{}).get(kitchen,{}).get("floor")
    # Prefer the actual independently exported floor component. Existing
    # nested room geometry remains supported for earlier exports.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(floor_path or path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = np.asarray(xforms.GetLocalToWorldTransform(prim))
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset:offset+count]; offset += count
            for i in range(1, count-1):
                tri = points[[face[0], face[i], face[i+1]]]
                normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
                length = np.linalg.norm(normal)
                if length > 1e-10 and abs(normal[2])/length > .999:
                    triangles.append(tri)
    if not triangles:
        raise ValueError("KITCHEN_HORIZONTAL_FLOOR_GEOMETRY_REQUIRED")
    lowest = min(t[:, 2].mean() for t in triangles)
    # Either winding is accepted; live settling verifies the actual floor.
    floor = [t for t in triangles if t[:, 2].mean() <= lowest+.02]
    obstacles = []
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for obj in manifest["objects"]:
        if obj["role"] == "articulated_asset":
            cfg = obj["og_config"]
            # The replacement size is stored in its source world envelope.
            source_prim = stage.GetPrimAtPath(manifest["source_to_usd"][obj["source_id"]])
        elif "Semantics.Furniture" in obj.get("source_tags", []):
            source_prim = stage.GetPrimAtPath(obj["source_usd_prim"])
        else:
            continue
        bounds = cache.ComputeWorldBound(source_prim).ComputeAlignedRange()
        obstacles.append([list(bounds.GetMin()), list(bounds.GetMax())])
    return raster_floor(floor, obstacles)


def probe_fingerprint(assembly, robot_config, urdf, target_root):
    """Bind local resources and implementations; no license redistribution."""
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest
    manifest = load_assembly_manifest(assembly)
    robot = yaml.safe_load(robot_config.read_text())
    target = manifest["target_asset"]
    if target_root.parts[-3:] != ("objects", target["category"], target["model"]):
        raise ValueError("PROBE_TARGET_RESOURCE_PATH_MISMATCH")
    files = sorted(p for p in target_root.rglob("*") if p.is_file())
    if not files or not (target_root / "misc/metadata.json").is_file() or not urdf.is_file():
        raise ValueError("REAL_TARGET_RESOURCES_AND_EXPORTED_URDF_REQUIRED")
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(urdf).getroot()
        links = {link.attrib["name"] for link in tree.findall("link")}
        kin = robot["robot"]["kinematics"]
        if tree.tag != "robot" or not {kin["base_link"],kin["end_effector_link"]} <= links:
            raise ValueError("PROBE_URDF_LINK_IDENTITY_INVALID")
    except (ET.ParseError,KeyError) as exc:
        raise ValueError("PROBE_URDF_INVALID") from exc
    package = Path(__file__).resolve().parents[2]
    repo = package.parent.parent
    implementations = [*sorted((package/"sim/omnigibson").glob("*.py")),
                       Path(__file__), Path(__file__).with_name("probe.py"), Path(__file__).with_name("navigation.py"),
                       repo/"configs/robots/r1pro_symbolic.yaml", repo/"configs/simulator/omnigibson.yaml"]
    return {"assembly_sha256": sha256_path(assembly), "robot_config_sha256": sha256_path(robot_config),
            "urdf_sha256": sha256_path(urdf), "protocol_sha256": runtime_protocol_hash(),
            "probe_sources": {str(p.resolve()): sha256_path(p) for p in implementations},
            "target_asset": manifest["target_asset"], "target_resources": {str(p.resolve()): sha256_path(p) for p in files},
            "camera_protocol": {"selection": "current_head_rgb", "robot_config": robot["robot"]}}


def prepare_probe(spec, assembly: Path, repo: Path, assets_root: Path, output: Path):
    from rummagebench.authoring.tasks.probe import ProbeConfig, anchor_candidates
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly
    manifest = load_assembly_manifest(assembly)
    if manifest["scene_id"] != spec.base_layout_id or manifest["family_id"] != spec.family_id:
        raise ValueError("ASSEMBLY_SPEC_MISMATCH")
    if manifest["target_asset"] != spec.target.model_dump():
        raise ValueError("PROBE_TARGET_ASSET_MISMATCH")
    robot_path = (repo / spec.robot_config_ref).resolve()
    robot = yaml.safe_load(robot_path.read_text())
    if spec.robot_id != "r1pro" or robot["robot"]["model"] != "r1pro":
        raise ValueError("ONLY_CURRENT_R1PRO_PROTOCOL_SUPPORTED")
    urdf = (repo / robot["robot"]["kinematics"]["urdf_path"]).resolve()
    target_root = (assets_root / "objects" / spec.target.category / spec.target.model).resolve()
    fingerprint = probe_fingerprint(assembly, robot_path, urdf, target_root)
    room_map = assembly_floor_map(manifest, assembly)
    starts = room_map.clearest_points(10, "kitchen_0")
    if not starts or starts[0][0] < .5:
        raise ValueError("NO_CPU_START_POSE_PROPOSAL")
    slots = scene_slots_from_assembly(assembly)
    cfg = ProbeConfig()
    request = {"version": 1, "status": "pending_simulation", "gpu_executed": False,
        "assembly": str(assembly.resolve()), "robot_config": str(robot_path), "urdf": str(urdf),
        "target_root": str(target_root), "fingerprint": fingerprint,
        "required_slots": sorted(s.slot_id for s in slots.slots), "room_map": room_map.to_dict(),
        "probe_config": cfg.to_dict(), "expected_base_z": room_map.floor_z+.0053,
        "start_proposals": [[*p, room_map.floor_z+.0053, 0.] for _, p in starts],
        "interaction_proposals": {f.entity: [list(p) for p in anchor_candidates(f,cfg)] for f in slots.furniture},
        "path_source": "SceneProber/plan_route MOVE+TURN candidates; public protocol replay still required",
        "geometry_status": "proposals only; actual floor/spawn stability, collision and reach pending",
        "dependency_integrity": manifest["dependency_integrity"], "evaluation_ready_episodes": 0}
    request["request_sha256"] = sha256_bytes(canonical_json(request))
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/"request.json", request)
    return request


def load_probe_request(path):
    doc = json.loads(Path(path).read_text())
    payload = {k:v for k,v in doc.items() if k != "request_sha256"}
    if doc.get("request_sha256") != sha256_bytes(canonical_json(payload)):
        raise ValueError("PROBE_REQUEST_HASH_MISMATCH")
    if doc.get("version") != 1 or doc.get("status") != "pending_simulation" or doc.get("gpu_executed") is not False:
        raise ValueError("PENDING_PROBE_REQUEST_REQUIRED")
    now = probe_fingerprint(Path(doc["assembly"]), Path(doc["robot_config"]), Path(doc["urdf"]), Path(doc["target_root"]))
    if now != doc["fingerprint"]:
        raise ValueError("PROBE_INPUT_OR_IMPLEMENTATION_CHANGED")
    from rummagebench.sim.omnigibson.infinigen_adapter import scene_slots_from_assembly
    actual = sorted(s.slot_id for s in scene_slots_from_assembly(Path(doc["assembly"])).slots)
    if doc["required_slots"] != actual or len(actual) != len(set(actual)):
        raise ValueError("PROBE_COMPLETE_CANDIDATE_SET_REQUIRED")
    GeometryRoomMap.from_dict(doc["room_map"])
    return doc


def validate_probe_evidence(evidence, request, overlay):
    """A failed necessary position rejects the whole family; no subset acceptance."""
    if (evidence.get("status") != "probed" or evidence.get("gpu_executed") is not True
            or evidence.get("request_sha256") != request["request_sha256"]
            or evidence.get("fingerprint") != request["fingerprint"]
            or set(evidence.get("slots", {})) != set(request["required_slots"])
            or set(overlay.slots) != set(request["required_slots"])
            or not overlay.probed or not overlay.start_poses.get("kitchen_0")
            or not evidence.get("camera_sensor") or evidence.get("camera_valid") is not True):
        raise ValueError("COMPLETE_REAL_PROBE_EVIDENCE_REQUIRED")
    for sid in request["required_slots"]:
        inter = overlay.slots[sid]
        if (not inter.probed or not inter.usable(True) or inter.open_causal is not True
                or inter.reveal_visibility is None or inter.reveal_visibility < request["probe_config"]["reveal_visibility_min"]
                or evidence["slots"][sid] != inter.evidence):
            raise ValueError(f"WHOLE_FAMILY_PROBE_REJECTED:{sid}")


def verify_overlay_evidence(overlay):
    path = Path(overlay.source)
    if not path.is_file() or "probe_evidence_sha256:"+sha256_path(path) not in overlay.notes:
        raise ValueError("HASH_BOUND_REAL_PROBE_RECEIPT_REQUIRED")
    evidence = json.loads(path.read_text())
    request = load_probe_request(Path(evidence["request_path"]))
    if overlay.urdf_hash != request["fingerprint"]["urdf_sha256"] or Path(overlay.urdf_path).resolve() != Path(request["urdf"]):
        raise ValueError("PROBE_OVERLAY_URDF_MISMATCH")
    robot_doc = yaml.safe_load(Path(request["robot_config"]).read_text())
    expected_robot = robot_doc["robot"]
    expected_robot["kinematics"]["urdf_path"] = request["urdf"]
    if overlay.robot_config != expected_robot or overlay.feasibility != robot_doc.get("feasibility",{}):
        raise ValueError("PROBE_OVERLAY_ROBOT_CAMERA_CONFIG_MISMATCH")
    validate_probe_evidence(evidence,request,overlay)
    return request


def run_gpu_probe(request_path: Path, output: Path):
    """GPU ONLY. All simulation imports are delayed to this explicitly named entry."""
    request = load_probe_request(request_path)
    from rummagebench.authoring.tasks.probe import SceneProber, ProbeConfig, pose_from_xyyaw, slot_interaction
    from rummagebench.authoring.tasks.embodiment_slots import SlotEmbodimentOverlay
    from rummagebench.sim.omnigibson.infinigen_adapter import load_assembly_manifest, scene_slots_from_assembly
    from rummagebench.core.scenario import ScenarioSpec
    from rummagebench.core.session import BenchmarkSession
    from rummagebench.adapters.python_api import prepare_renderer_grounding, install_renderer_grounding
    from rummagebench.sim.omnigibson.backend import OmniGibsonBackend
    from rummagebench.sim.omnigibson.observation import select_head_rgb_sensor

    assembly = Path(request["assembly"]); manifest = load_assembly_manifest(assembly)
    ss = scene_slots_from_assembly(assembly); robot = yaml.safe_load(Path(request["robot_config"]).read_text())
    robot["robot"]["kinematics"]["urdf_path"] = request["urdf"]
    room_map = GeometryRoomMap.from_dict(request["room_map"])
    # Park the real target far outside every background object, independently
    # of its hidden family position; SceneProber uses the existing sampler.
    pose = pose_from_xyyaw(*request["start_proposals"][0]).to_anchor()
    scenario = ScenarioSpec.model_validate({"id": "probe_"+ss.scene, "instruction": "evaluator-only embodiment probe",
        "scene": {"model": ss.scene, "source":"infinigen", "assembly_manifest": str(assembly)},
        "robot":{**robot["robot"], "init_anchor":"start"}, "feasibility":robot["feasibility"],
        "anchors":{"start":pose}, "target":{"entity":"probe_object", "category":manifest["target_asset"]["category"]},
        "objects":[{"name":"probe_object", "category":manifest["target_asset"]["category"],
                    "model":manifest["target_asset"]["model"]}],
        "placements":[], "initial_states":{f.entity:{"open":False} for f in ss.furniture},
        "skills":["NAV","OPEN","CLOSE","GRASP","PLACE"]})
    prepare_renderer_grounding(scenario)
    output.mkdir(parents=True,exist_ok=False)
    backend = OmniGibsonBackend(seed=0)
    evidence = {"version":1, "status":"partial", "gpu_executed":False,
        "request_sha256":request["request_sha256"], "fingerprint":request["fingerprint"],
        "slots":{}, "furniture":{}, "evaluation_ready_episodes":0}
    try:
        from omnigibson.utils.asset_utils import get_dataset_path
        actual_target = Path(get_dataset_path("behavior-1k-assets"))/"objects"/manifest["target_asset"]["category"]/manifest["target_asset"]["model"]
        if actual_target.resolve() != Path(request["target_root"]):
            raise ValueError("RUNTIME_TARGET_ASSET_ROOT_DIFFERS_FROM_PROBE_REQUEST")
        backend.setup(scenario, task_seed_pose=lambda _backend,_spec: [10000.,10000.,10.])
        evidence["gpu_executed"] = True
        backend.validate_physics_state()
        install_renderer_grounding(backend)
        obs,_ = backend._env.get_obs()
        sensor, data = select_head_rgb_sensor(obs[0].get(scenario.robot.name,{}))
        from rummagebench.sim.omnigibson.observation import _to_numpy
        rgb = _to_numpy(data["rgb"])
        frame = backend.capture_visual_frame()
        camera_valid = (rgb.shape[:2] == (scenario.robot.image_height,scenario.robot.image_width)
                        and rgb.ndim == 3 and rgb.shape[2] in (3,4)
                        and np.isfinite(frame.camera_intrinsics).all() and np.isfinite(frame.camera_extrinsics).all())
        evidence.update(camera_sensor=sensor,camera_valid=bool(camera_valid),camera_rgb_shape=list(rgb.shape))
        cfg = ProbeConfig(**request["probe_config"])
        prober = SceneProber(backend, BenchmarkSession(backend,scenario), ss,room_map,"probe_object",cfg,
                             views_dir=output/"views",expected_base_z=request["expected_base_z"])
        starts = {"kitchen_0":prober.probe_start_poses("kitchen_0")}
        interactions = {}
        for furn in sorted(ss.furniture,key=lambda f:f.entity):
            slots = [s for s in ss.slots if s.parent_entity == furn.entity]
            fp = prober.probe_furniture(furn,slots)
            evidence["furniture"][furn.entity] = asdict(fp)
            for slot in slots:
                inter = slot_interaction(slot,fp)
                if inter is not None:
                    interactions[slot.slot_id] = inter
                    evidence["slots"][slot.slot_id] = inter.evidence
        overlay = SlotEmbodimentOverlay(ss.scene,"r1pro",request["urdf"],request["fingerprint"]["urdf_sha256"],
            "probed", str((output/"evidence.json").resolve()),robot["robot"],robot["feasibility"],starts,interactions,
            notes=["assembly_manifest_sha256:"+sha256_path(assembly)])
        evidence["status"] = "probed"
        validate_probe_evidence(evidence,request,overlay)
        # Recheck everything after the actual probe before committing results.
        load_probe_request(request_path)
        evidence["request_path"] = str(request_path.resolve())
        atomic_write_json(output/"evidence.json",evidence)
        overlay.notes.append("probe_evidence_sha256:"+sha256_path(output/"evidence.json"))
        overlay.save(output/"overlay.yaml")
    except Exception as exc:
        evidence.update(status="rejected",error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output/"evidence.json",evidence)
        raise
    finally:
        backend.close()
    return evidence
