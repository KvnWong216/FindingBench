"""CPU assembly preparation over real USD + existing BEHAVIOR interactive assets.

No simulation imports at module load. Whole-scene cabinets are explicitly
replaced, not labelled articulated. Replacement support/collision/open-space
checks remain pending. All unmatched geometry is retained as static background.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from rummagebench.authoring.level1.snapshot import atomic_write_json
from rummagebench.authoring.tasks.household import HouseholdTaskSpec
from rummagebench.authoring.tasks.infinigen import (inspect_export, read_solve_state, usd_dependency_ledger,
                                                    verify_usd_dependency_ledger, source_mesh_owner)
from rummagebench.authoring.tasks.plan import canonical_json, sha256_bytes, sha256_path


def _checked_reference(base: Path, ref: str) -> Path:
    p = (base / ref).resolve()
    if not p.is_relative_to(base.resolve()) or not p.is_file():
        raise ValueError(f"ASSEMBLY_ARTIFACT_MISSING_OR_OUTSIDE_BUNDLE:{ref}")
    return p


def load_assembly_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    payload = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    if sha256_bytes(canonical_json(payload)) != doc.get("manifest_sha256"):
        raise ValueError("ASSEMBLY_MANIFEST_HASH_MISMATCH")
    if doc.get("version") != 2 or doc.get("status") != "pending_simulation":
        raise ValueError("ASSEMBLY_SCHEMA_OR_STATUS_INVALID")
    if not doc.get("usd_dependency_ledgers"):
        raise ValueError("ASSEMBLY_DEPENDENCY_LEDGER_REQUIRED")
    for ref, digest in doc["bundle_artifacts"].items():
        if sha256_path(_checked_reference(path.parent, ref)) != digest:
            raise ValueError(f"ASSEMBLY_ARTIFACT_HASH_MISMATCH:{ref}")
    for ref, digest in doc["external_asset_sources"].items():
        if not Path(ref).is_file() or not digest or sha256_path(ref) != digest:
            raise ValueError(f"INTERACTIVE_ASSET_SOURCE_CHANGED:{ref}")
    for ledger in doc["usd_dependency_ledgers"]:
        verify_usd_dependency_ledger(ledger)
    if doc.get('native_region_catalog'):
        from rummagebench.authoring.tasks.infinigen_component_binding import verify_native_region_catalog
        verify_native_region_catalog(doc)
    return doc


def assembly_env_config(path: Path) -> dict:
    doc = load_assembly_manifest(path)
    objects = []
    for item in doc["objects"]:
        cfg = dict(item["og_config"])
        if cfg["type"] == "USDObject":
            cfg["usd_path"] = str(_checked_reference(Path(path).resolve().parent, cfg["usd_path"]))
        objects.append(cfg)
    return {"scene": {"type": "Scene", "use_floor_plane": False, "use_skybox": False,
                       "include_robots": False}, "objects": objects}



def native_category(source: dict) -> str | None:
    """Conservative factory evidence from the pinned solve_state, not visual truth."""
    mapping = {"BowlFactory": "bowl", "CupFactory": "teacup", "ForkFactory": "tablefork",
               "SpoonFactory": "tablespoon", "KnifeFactory": "tableknife", "PlateFactory": "plate",
               "WineglassFactory": "wineglass", "PanFactory": "pan", "PotFactory": "stockpot",
               "CanFactory": "can", "BottleFactory": "bottle", "FoodBagFactory": "food_bag",
               "FoodBoxFactory": "food_box", "JarFactory": "jar",
               "BookStackFactory": "book", "BookColumnFactory": "book",
               # The pinned factory's complete subfactory list contains
               # rocks, pinecones, shells, coral and animal figurines. This
               # coarse category does not claim visual semantic acceptance.
               "NatureShelfTrinketsFactory": "nature_shelf_trinket"}
    generator = source.get("generator") or ""
    found = [category for factory, category in mapping.items()
             if re.search(r"(?:^|[. <])" + factory + r"(?:[ >(.]|$)", generator)]
    return found[0] if len(found) == 1 else None


def replacement_root_position(center, scale, orientation, metadata):
    """Match DatasetObject.set_bbox_center_position_orientation's real frame.

    Environment._load_objects interprets `position` as root pose, whereas
    Infinigen supplies a bounding-box center. Missing offsets are unknown.
    This remains a pose proposal; runtime geometry must pass separately.
    """
    from rummagebench.feasibility.ik_solver import quat_rotate
    offset=np.asarray(metadata.get('base_link_offset'),float)
    center,scale,orientation=np.asarray(center,float),np.asarray(scale,float),np.asarray(orientation,float)
    if (offset.shape!=(3,) or center.shape!=(3,) or scale.shape!=(3,) or orientation.shape!=(4,)
            or not np.isfinite(np.r_[offset,center,scale,orientation]).all() or np.any(scale<=0)
            or not np.isclose(np.linalg.norm(orientation),1.,atol=1e-6)):
        raise ValueError('REAL_ASSET_BBOX_ROOT_OFFSET_REQUIRED')
    return (center+quat_rotate(orientation,-scale*offset)).tolist()


def propose_interactive_bindings(spec, scene: Path, solve: Path, receipt: Path,
                                assets_root: Path, approved_slots: list[Path]) -> dict:
    """Match source cabinets to exact models already used by easy_v1 slots.

    This is an asset proposal, not proof of a cavity, door clearance or skill.
    No asset is invented or fetched, and no target location enters ranking.
    Actual metadata replaces the old catalog's estimated furniture dimensions.
    """
    from pxr import Usd, UsdGeom, Gf
    from rummagebench.authoring.tasks.slots import SceneSlots
    from rummagebench.authoring.tasks.household_dataset import verify_generation_admission

    admission = verify_generation_admission(spec, receipt, solve)
    inspection = inspect_export(scene, solve)
    mapping = admission["measurement"]["candidate_mapping"]
    if set(mapping) != {loc.location_id for loc in spec.locations}:
        raise ValueError("INTERACTIVE_BINDING_COMPLETE_CANDIDATES_REQUIRED")
    allowed = set()
    for path in approved_slots:
        for slot in SceneSlots.load(path).slots:
            if (slot.room_type == "kitchen" and slot.slot_type == "cabinet_interior"
                    and slot.requires_open and slot.articulated):
                allowed.add((slot.parent_category, slot.parent_model))
    models, rejected = {}, []
    for category, model in sorted(allowed):
        root = assets_root / "objects" / category / model
        meta_path = root / "misc/metadata.json"
        usd = root / "usd" / f"{model}.usd"
        if not usd.is_file():
            usd = usd.with_name(f"{model}.encrypted.usd")
        try:
            meta = json.loads(meta_path.read_text())
            dims = np.asarray(meta["bbox_size"], float)
            if (dims.shape != (3,) or not np.isfinite(dims).all() or min(dims) <= 0
                    or not usd.is_file() or not any("openable" in t for t in meta.get("link_tags", {}).values())):
                raise ValueError("EXACT_OPENABLE_ASSET_METADATA_INVALID")
            replacement_root_position([0.,0.,0.],[1.,1.,1.],[0.,0.,0.,1.],meta)
            models[(category, model)] = (dims, meta_path, usd)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rejected.append({"category": category, "model": model, "reason": str(exc)})
    stage = Usd.Stage.Open(str(scene.resolve()))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xforms = UsdGeom.XformCache()
    bindings, failures, resources = [], [], {}
    for location, source in sorted(mapping.items()):
        path = inspection["source_to_usd"].get(source)
        if path is None:
            failures.append(f"CANDIDATE_SOURCE_NOT_EXPORTED:{source}")
            continue
        prim = stage.GetPrimAtPath(path)
        box = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        dims = ((np.asarray(box.GetMax()) - np.asarray(box.GetMin()))
                * np.abs(np.asarray(Gf.Transform(xforms.GetLocalToWorldTransform(prim)).GetScale())))
        choices = []
        for (category, model), (native, meta_path, usd) in models.items():
            scale = dims / native
            if np.isfinite(scale).all() and np.all((scale >= .5) & (scale <= 2.)):
                choices.append((float(np.abs(np.log(scale)).sum()), category, model, meta_path, usd))
        if not choices:
            failures.append(f"NO_APPROVED_INTERACTIVE_MODEL_FITS:{source}")
            continue
        _, category, model, meta_path, usd = min(choices)
        bindings.append(dict(location_id=location, source_id=source, category=category, model=model))
        for resource in (meta_path, usd):
            resources[str(resource.resolve())] = sha256_path(resource)
    return {"version": 1, "status": "rejected" if failures else "candidate",
            "bindings": bindings, "failures": failures, "rejected_assets": rejected,
            "source_hashes": {str(p.resolve()): sha256_path(p) for p in [scene, solve, receipt, *approved_slots]},
            "asset_hashes": resources, "target_conditioned": False,
            "selection": "minimum log scale distortion in exact easy_v1 cabinet model pool",
            "physical_acceptance": "pending", "evaluation_ready_episodes": 0}

def prepare_assembly(spec: HouseholdTaskSpec, scene: Path, solve: Path, bindings: list[dict],
                     assets_root: Path, output: Path, *, generation_receipt: Path | None = None,
                     component_context: dict | None = None) -> dict:
    """Each required location must have an explicit source→real-asset binding.

    Binding fields: location_id, source_id, category, model. Actual metadata
    must contain openable links. USD joints/limits/handles are checked later
    against the loaded articulation, never fabricated from configuration.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    admission = None
    if generation_receipt is not None:
        from rummagebench.authoring.tasks.household_dataset import verify_generation_admission
        admission = verify_generation_admission(spec, generation_receipt, solve)
        expected_mapping = admission["measurement"]["candidate_mapping"]
        if {b["location_id"]: b["source_id"] for b in bindings} != expected_mapping:
            raise ValueError("ASSEMBLY_BINDINGS_DIFFER_FROM_ACCEPTED_GENERATION_CANDIDATES")

    inspection = inspect_export(scene, solve)
    if inspection["ambiguous_source_objects"]:
        raise ValueError("AMBIGUOUS_SOURCE_MAPPING")
    for b in bindings:
        if set(b) != {"location_id", "source_id", "category", "model"} or not all(
                isinstance(b[k], str) and re.fullmatch(r"[a-zA-Z0-9_]+", b[k]) for k in ("category", "model")):
            raise ValueError("UNSUPPORTED_INTERACTIVE_ASSET_BINDING")
    by_location = {b["location_id"]: b for b in bindings}
    wanted = {s.location_id for s in spec.locations}
    if set(by_location) != wanted or len(by_location) != len(bindings):
        raise ValueError("INCOMPLETE_OR_DUPLICATE_CANDIDATE_BINDINGS")
    source_ids = [b["source_id"] for b in bindings]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("COUPLED_REVEAL_CANDIDATES_REQUIRE_DIFFERENT_SEARCH_MODEL")
    metadata = read_solve_state(solve)
    confounds = []
    similar = {"tablefork", "tablespoon", "tableknife"} if spec.target.category in {"tablefork", "tablespoon"} else {spec.target.category}
    for key, source in metadata["objects"].items():
        if "Semantics.HandheldItem" in source["tags"]:
            cat = native_category(source)
            if cat is None or cat in similar:
                confounds.append({"source_id": key, "category": cat, "reason": "unknown_or_task_category_ambiguous"})
    extra_storage = []
    for key, source in metadata["objects"].items():
        if "Semantics.Storage" not in source["tags"] or key in source_ids:
            continue
        seen, frontier = set(), [key]
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(r["parent"] for r in metadata["objects"][node]["relations"])
        if any("Semantics.Kitchen" in metadata["objects"][node]["tags"] for node in seen):
            extra_storage.append(key)
    if extra_storage:
        # A closed, unmapped plausible cupboard would silently enlarge the
        # search set while being a noninteractive mesh. This minimum family
        # cannot certify that extra locations are impossible from metadata.
        raise ValueError(f"UNBOUND_KITCHEN_STORAGE_SEARCH_SET_UNRESOLVED:{extra_storage}")
    stage = Usd.Stage.Open(str(scene.resolve()))
    box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    replacement, external = {}, {}
    ledgers = [inspection["dependency_ledger"]]
    opaque_assets = []
    for loc in spec.locations:
        b = by_location[loc.location_id]
        if b["source_id"] not in inspection["source_to_usd"]:
            raise ValueError(f"CANDIDATE_SOURCE_NOT_EXPORTED:{b['source_id']}")
        source = metadata["objects"][b["source_id"]]
        ancestors, frontier = set(), [b["source_id"]]
        while frontier:
            node = frontier.pop()
            if node in ancestors:
                continue
            ancestors.add(node)
            frontier.extend(r["parent"] for r in metadata["objects"][node]["relations"])
        if "Semantics.Storage" not in source["tags"] or not any(
                "Semantics.Kitchen" in metadata["objects"][node]["tags"] for node in ancestors):
            # Room membership can also be inherited via intermediate support
            # relations, but ambiguous/unsupported mappings must be explicit.
            raise ValueError("CANDIDATE_KITCHEN_STORAGE_SEMANTICS_UNRESOLVED")
        meta_path = assets_root / "objects" / b["category"] / b["model"] / "misc/metadata.json"
        asset_meta = json.loads(meta_path.read_text())
        links = [name for name, tags in asset_meta.get("link_tags", {}).items() if "openable" in tags]
        usd = assets_root / "objects" / b["category"] / b["model"] / "usd" / f"{b['model']}.usd"
        if not usd.is_file():
            usd = usd.with_name(f"{b['model']}.encrypted.usd")
        if not links or not usd.is_file():
            raise ValueError("REAL_INTERACTIVE_ASSET_OR_OPENABLE_METADATA_MISSING")
        external[str(meta_path.resolve())] = sha256_path(meta_path)
        external[str(usd.resolve())] = sha256_path(usd)
        if usd.name.endswith(".encrypted.usd"):
            # Enumerate every local model resource conservatively. The opaque
            # reference graph remains explicitly pending until OG decrypts it.
            for resource in sorted(usd.parent.parent.rglob('*')):
                if resource.is_file():
                    external[str(resource.resolve())] = sha256_path(resource)
            opaque_assets.append({"usd": str(usd.resolve()), "resource_root": str(usd.parent.parent.resolve()),
                                  "status": "pending_runtime_decrypted_dependency_graph"})
        else:
            ledgers.append(usd_dependency_ledger(usd))
        prim_path = inspection["source_to_usd"][b["source_id"]]
        prim = stage.GetPrimAtPath(prim_path)
        bbox = box_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
        local_box = box_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        dims = (np.array(local_box.GetMax()) - np.array(local_box.GetMin())) * np.abs(
            np.array(Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).GetScale()))
        scale = dims / np.asarray(asset_meta["bbox_size"], float)
        if not np.isfinite(scale).all() or np.any(scale < .5) or np.any(scale > 2.):
            raise ValueError("INTERACTIVE_REPLACEMENT_SCALE_UNSUPPORTED")
        rotation = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).GetRotation().GetQuat()
        quat = [*rotation.GetImaginary(), rotation.GetReal()]
        bbox_center=((lo+hi)/2).tolist()
        root_position=replacement_root_position(bbox_center,scale,quat,asset_meta)
        replacement[prim_path] = {"name": "interactive_" + loc.location_id, "category": b["category"],
            "source_id": b["source_id"], "location_id": loc.location_id, "source_usd_prim": prim_path,
            "role": "articulated_asset", "openable_metadata_links": links,
            "joint_limits": "pending_runtime_USD_query", "interaction_parts": "pending_runtime_handle_query",
            "skill_bindings": ["OPEN", "CLOSE", "PLACE"], "reset_state": {"open": False},
            "source_bbox_center_world":bbox_center,
            "pose_binding":"DatasetObject_bbox_center_to_root_using_scaled_native_offset",
            "og_config": {"type": "DatasetObject", "name": "interactive_" + loc.location_id,
                          "category": b["category"], "model": b["model"], "fixed_base": True,
                          "scale": scale.tolist(), "position":root_position, "orientation": quat},
            "replacement_support_and_collision": "pending"}
    output.mkdir(parents=True, exist_ok=False)
    assets_dir = output / "objects"; assets_dir.mkdir()
    objects, lighting, artifacts = [], [], {}
    source_layer = stage.Flatten()
    xforms = UsdGeom.XformCache()
    # Preserve every mesh, including unmapped architecture/decorations.
    for i, prim_path in enumerate(inspection["all_mesh_paths"]):
        ancestor = next((p for p in replacement if prim_path == p or prim_path.startswith(p + "/")), None)
        if ancestor is not None:
            if replacement[ancestor] not in objects:
                objects.append(replacement[ancestor])
            continue
        prim = stage.GetPrimAtPath(prim_path)
        matrix = xforms.GetLocalToWorldTransform(prim)
        transform = Gf.Transform(matrix)
        scale, translation, rotation = transform.GetScale(), transform.GetTranslation(), transform.GetRotation().GetQuat()
        # Reject sheared/mirrored transforms; do not silently deform geometry.
        reconstructed = (Gf.Matrix4d(1.).SetScale(scale) * Gf.Matrix4d(1.).SetRotate(Gf.Rotation(rotation))
                         * Gf.Matrix4d(1.).SetTranslate(translation))
        if not np.allclose(np.asarray(matrix), np.asarray(reconstructed), atol=1e-6) or min(scale) <= 0:
            raise ValueError(f"USD_NONRIGID_TRANSFORM_UNSUPPORTED:{prim_path}")
        source_id, component = source_mesh_owner(inspection,prim_path)
        tags = [] if source_id is None else metadata["objects"][source_id]["tags"]
        # Only explicitly tagged single-mesh handheld objects may be dynamic.
        # Grouped multi-mesh bodies need a rigid grouping transform, not one
        # independent rigid body per fragment. Reject rather than break them.
        movable = "Semantics.HandheldItem" in tags
        if movable and sum(p == inspection["source_to_usd"][source_id] or
                           p.startswith(inspection["source_to_usd"][source_id] + "/")
                           for p in inspection["all_mesh_paths"]) != 1:
            raise ValueError("MOVABLE_MULTIMESH_GROUPING_UNSUPPORTED")
        name = f"ig_background_{i:05d}"
        asset_path = assets_dir / f"{name}.usda"
        dst = Usd.Stage.CreateNew(str(asset_path))
        root = UsdGeom.Xform.Define(dst, "/World").GetPrim(); dst.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(dst, "Z"); UsdGeom.SetStageMetersPerUnit(dst, 1.)
        base = UsdGeom.Xform.Define(dst, "/World/base_link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.RigidBodyAPI(base).CreateKinematicEnabledAttr(not movable)
        UsdGeom.Xform.Define(dst, "/World/base_link/geometry")
        geom_path = "/World/base_link/geometry/mesh"
        Sdf.CopySpec(source_layer, prim.GetPath(), dst.GetRootLayer(), Sdf.Path(geom_path))
        geom = dst.GetPrimAtPath(geom_path)
        UsdGeom.Xformable(geom).ClearXformOpOrder()
        # OG hides collision meshes by assigning purpose=guide. A single
        # render+collision mesh makes the entire source background invisible.
        # Retain an exact visible copy and a separate exact physics copy.
        if geom.HasAPI(UsdPhysics.CollisionAPI): geom.RemoveAPI(UsdPhysics.CollisionAPI)
        if geom.HasAPI(UsdPhysics.MeshCollisionAPI): geom.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        UsdGeom.Imageable(geom).CreatePurposeAttr('default')
        copy_effective_materials(stage, prim, dst, geom, source_layer)
        UsdGeom.Xform.Define(dst, '/World/base_link/collisions')
        collision_path = '/World/base_link/collisions/mesh'
        Sdf.CopySpec(dst.GetRootLayer(), Sdf.Path(geom_path), dst.GetRootLayer(), Sdf.Path(collision_path))
        collider = dst.GetPrimAtPath(collision_path)
        UsdGeom.Imageable(collider).CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(collider)
        UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('convexHull' if movable else 'none')
        dst.GetRootLayer().Save()
        rel = str(asset_path.relative_to(output)); artifacts[rel] = sha256_path(asset_path)
        source_id, component = source_mesh_owner(inspection,prim_path)
        tags = [] if source_id is None else metadata["objects"][source_id]["tags"]
        inferred_category = None if source_id is None else native_category(metadata["objects"][source_id])
        category = (inferred_category or ("countertop" if "Semantics.KitchenCounter" in tags else
                    "breakfast_table" if "Semantics.Table" in tags else
                    "shelf" if "Semantics.Storage" in tags else "infinigen_background"))
        objects.append({"name": name, "source_id": source_id, "source_usd_prim": prim_path,
                        "source_component":component,
                        "role": "movable_rigid_body" if movable else "static_background", "source_tags": tags,
                        "category": category, "skill_bindings": ["PLACE"] if category != "infinigen_background" else [],
                        "reset_state": "frozen_native_transform",
                        "og_config": {"type": "USDObject", "name": name, "category": category,
                                      "usd_path": rel, "fixed_base": not movable, "scale": list(scale),
                                      "position": list(translation), "orientation": [*rotation.GetImaginary(), rotation.GetReal()]}})
    lights_path = output / "lighting.usda"
    lights_stage = Usd.Stage.CreateNew(str(lights_path))
    lights_root = UsdGeom.Xform.Define(lights_stage, "/World").GetPrim(); lights_stage.SetDefaultPrim(lights_root)
    for i, prim in enumerate(stage.Traverse()):
        if prim.HasAPI(UsdLux.LightAPI):
            dest = Sdf.Path(f"/World/light_{i}")
            Sdf.CopySpec(source_layer, prim.GetPath(), lights_stage.GetRootLayer(), dest)
            light_prim = lights_stage.GetPrimAtPath(dest)
            xf = UsdGeom.Xformable(light_prim); xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(xforms.GetLocalToWorldTransform(prim))
            normalize_asset_attributes(prim, light_prim)
            lighting.append(str(dest))
    lights_stage.GetRootLayer().Save(); artifacts["lighting.usda"] = sha256_path(lights_path)
    for ref in artifacts:
        ledgers.append(usd_dependency_ledger(output / ref))
    for ledger in ledgers:
        verify_usd_dependency_ledger(ledger)
    manifest = {"version": 2, "status": "pending_simulation", "scene_id": spec.base_layout_id,
                "source": {"upstream_commit": spec.upstream_commit, "scene_usd": str(scene.resolve()),
                           "scene_sha256": inspection["scene_sha256"], "solve_state": str(solve.resolve()),
                           "solve_sha256": inspection["solve_state_sha256"], "license": "BSD-3-Clause", "interactive_asset_license": "BEHAVIOR assets; local only, never redistributed"},
                "objects": objects, "lighting_usd": "lighting.usda", "lighting_prims": lighting,
                "bundle_artifacts": artifacts, "external_asset_sources": external,
                "usd_dependency_ledgers": ledgers, "opaque_external_assets": opaque_assets,
                "dependency_integrity": "supported_local_graphs_hashed; opaque_graphs_pending" if opaque_assets else "complete_supported_local_graphs",
                "candidate_bindings": [replacement[p] for p in sorted(replacement)],
                "source_to_usd": inspection["source_to_usd"], "appearance_request": spec.appearance,
                "source_component_paths": inspection["source_component_paths"],
                "target_asset": spec.target.model_dump(), "family_id": spec.family_id,
                "generation_admission": admission,
                "robot_config_ref": spec.robot_config_ref, "evaluation_ready_episodes": 0,
                "target_category_confounds": confounds, "semantic_labels": "factory_evidence_pending_visual_check",
                "pending_checks": ["real_joints_limits_handles", "native_occupancy", "background_stability",
                                   "replacement_support", "door_sweep", "temporary_place_space", "visibility_and_observe",
                                   "robot_skills", "target_insertion_background_match", "reset", "search_cost", "appearance_images",
                                   "all_additional_counter_drawer_search_locations_classified"]}
    if component_context is not None:
        if admission is not None or component_context.get('complete_leaf_binding') != 'pending':
            raise ValueError('COMPONENT_IMPORT_PREFLIGHT_CANNOT_GRANT_FAMILY_ADMISSION')
        manifest['component_context'] = component_context
        manifest['pending_checks'] += ['complete_kitchen_leaf_binding', 'coupled_reveal_measurement', 'open_cabinet_visibility']
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest))
    atomic_write_json(output / "assembly.json", manifest)
    return manifest


def runtime_articulation_checks(backend, manifest: dict) -> list[str]:
    """Consume real loaded articulation, never metadata labels as proof."""
    failures = []
    for item in manifest["candidate_bindings"]:
        info = backend.articulation_info(item["name"])
        if info is None:
            failures.append(f"ARTICULATION_MISSING:{item['name']}")
            continue
        movable = [j for j in info.joints if j.joint_type in ("revolute", "prismatic")]
        if not movable or any(j.limits is None or not np.isfinite(j.limits).all()
                              or j.limits[0] >= j.limits[1] for j in movable):
            failures.append(f"REAL_MOVABLE_JOINT_LIMITS_MISSING:{item['name']}")
            continue
        # Use the existing easy_v1 interaction protocol. Most native assets
        # have handles integrated into the moving link, with no handle link.
        # Never fabricate an annotation or ground OPEN at the entity root.
        from rummagebench.feasibility.interaction_target import articulated_interaction_targets
        try:
            targets=articulated_interaction_targets(item['name'],backend)
        except (RuntimeError,ValueError,AttributeError):
            targets=[]
        if not targets or any(t.pose is None or not np.isfinite(t.pose.position).all() or
                              not np.isfinite(t.pose.orientation).all() for t in targets):
            failures.append(f"REAL_INTERACTION_INTERFACE_UNRESOLVED:{item['name']}")
        else:
            evidence=getattr(backend,'_findingbench_interaction_interfaces',{})
            evidence[item['name']]={'annotated_handle_link':info.handle_link,
                'protocol':'existing_easy_v1_handle_or_moving_link_interface',
                'targets':[t.to_dict() for t in targets]}
            backend._findingbench_interaction_interfaces=evidence
        if item.get('source_bbox_center_world') is not None:
            bbox=backend.entity_aabb(item['name'])
            if (bbox is None or np.asarray(bbox).shape!=(2,3) or not np.isfinite(np.asarray(bbox)).all()
                    or not np.allclose(np.mean(np.asarray(bbox),axis=0),item['source_bbox_center_world'],atol=.02,rtol=0)):
                failures.append(f"REPLACEMENT_LIVE_BBOX_CENTER_MISMATCH:{item['name']}")
    return failures


def restore_source_static_colliders(obj, expected_names):
    """SDK import hook, before play: retain source triangles on kinematic bodies.

    OG replaces very thin imported meshes with bounding cubes automatically.
    That can seal an entire room. Restore our explicitly authored triangle
    collisions through its existing link API, then verify every collider.
    Dynamic props and BEHAVIOR articulated assets are outside this hook.
    """
    if obj.name not in expected_names:
        return None
    from pxr import UsdGeom, UsdPhysics
    paths=[]
    for link in obj.links.values():
        if (not link.prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or link.prim.GetAttribute('physics:kinematicEnabled').Get() is not True):
            raise ValueError('SOURCE_STATIC_COLLIDER_MUST_REMAIN_KINEMATIC')
        meshes=list(link.collision_meshes.values())
        if not meshes or any(not m.prim.IsA(UsdGeom.Mesh) for m in meshes):
            raise ValueError('SOURCE_STATIC_TRIANGLE_MESH_COLLIDER_REQUIRED')
        link.set_collision_approximation('none')
        for mesh in meshes:
            if UsdPhysics.MeshCollisionAPI(mesh.prim).GetApproximationAttr().Get() != 'none':
                raise ValueError('SOURCE_STATIC_COLLIDER_RESTORE_FAILED')
            paths.append(str(mesh.prim.GetPath()))
    if not paths: raise ValueError('SOURCE_STATIC_COLLIDER_MISSING')
    return {'entity':obj.name,'colliders':paths,'approximation':'none','before_first_play':True}


def install_runtime_lighting(env, manifest_path: str) -> None:
    """Called only from an already running OG environment; never during CPU import."""
    from rummagebench.sim.omnigibson.env_factory import repo_root
    import omnigibson as og

    path = Path(manifest_path)
    if not path.is_absolute():
        path = repo_root() / path
    manifest = load_assembly_manifest(path)
    from rummagebench.authoring.tasks.infinigen_probe import assembly_floor_map
    env._findingbench_assembly_manifest = manifest
    env._findingbench_floor_z = assembly_floor_map(manifest,path).floor_z
    pxr = og.lazy.pxr
    lighting = pxr.Usd.Stage.Open(str(_checked_reference(path.parent, manifest["lighting_usd"])))
    scene_path = str(env.scene.prim_path)
    destination = scene_path + "/findingbench_lighting"
    with og.sim.editing_usd():
        og.sim.stage.DefinePrim(destination, "Xform")
        for prim in lighting.GetDefaultPrim().GetChildren():
            pxr.Sdf.CopySpec(lighting.GetRootLayer(), prim.GetPath(), og.sim.stage.GetRootLayer(),
                             pxr.Sdf.Path(destination + "/" + prim.GetName()))
        photometry=normalize_infinigen_point_photometry(og.sim.stage,destination,manifest['source']['upstream_commit'])
        env._findingbench_appearance_evidence = apply_light_appearance(og.sim.stage, destination, manifest["appearance_request"])
        env._findingbench_appearance_evidence['source_photometry']=photometry
        if manifest["appearance_request"].get("wall_color"):
            from rummagebench.authoring.tasks.household_appearance import apply_wall_palette
            from rummagebench.sim.omnigibson.entity_resolver import resolve_object
            walls=[str(resolve_object(env.scene,item["name"]).prim.GetPath()) for item in manifest["objects"]
                   if item.get("source_component")=="wall" and item["role"]=="static_background"]
            wall=apply_wall_palette(og.sim.stage,walls,manifest["appearance_request"]["wall_color"])
            env._findingbench_appearance_evidence["wall_palette"]=wall
            if wall["status"]=="applied":
                env._findingbench_appearance_evidence["material_requests_pending"].remove("wall_color")


def scene_slots_from_assembly(manifest_path: Path):
    """CPU furniture/slot proposals, retaining the existing SceneSlots contract.

    Inside dimensions are metadata envelopes, not verified cavities. Every
    slot explicitly requires runtime geometry, occupancy and OPEN probing.
    """
    import math
    from rummagebench.authoring.tasks.slots import Furniture, SceneSlots, Slot, SlotGeometry

    doc = load_assembly_manifest(manifest_path)
    furniture, slots = [], []
    for item in doc["candidate_bindings"]:
        cfg = item["og_config"]
        meta_path = next(p for p in doc["external_asset_sources"] if
                         f"/{cfg['category']}/{cfg['model']}/misc/metadata.json" in p)
        native = json.loads(Path(meta_path).read_text())["bbox_size"]
        size = tuple(float(x * y) for x, y in zip(native, cfg["scale"]))
        x, y, z, w = cfg["orientation"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
        position = tuple(item.get('source_bbox_center_world',cfg['position']))
        furniture.append(Furniture(item["name"], cfg["category"], cfg["model"], "kitchen_0", "kitchen",
                                   position, yaw, True, size))
        if doc.get('native_region_catalog'):
            regions=[r for r in doc['native_region_bindings'] if r['name']==item['name']]
            if not regions:raise ValueError('NATIVE_BODY_HAS_NO_REGION_NO_PRUNING')
            for region in regions:
                box=np.asarray(region['native_region']['visual_aabb'],dtype=float)
                if box.shape!=(2,3) or not np.isfinite(box).all() or np.any(box[1]<=box[0]):
                    raise ValueError('NATIVE_REGION_GEOMETRY_INVALID')
                ext=tuple(float(v) for v in box[1]-box[0])
                slots.append(Slot(region['slot_id'],doc['scene_id'],'kitchen_0','kitchen',item['name'],
                    cfg['category'],cfg['model'],'inside',region['slot_type'],region['fillable_link'],True,True,
                    SlotGeometry(ext,float(np.prod(ext)),ext[:2],float(box[0,2]),
                        'actual_native_region_world_aabb_proposal_only; exact_volume_fit_pending_probe'),
                    position,yaw,'actual_fillable_link; requires_open_is_probe_hypothesis'))
            continue
        ext = tuple(max(0., v - .04) for v in size)
        slots.append(Slot(f"{doc['scene_id']}/{item['name']}/interior", doc["scene_id"], "kitchen_0", "kitchen",
                          item["name"], cfg["category"], cfg["model"], "inside", "cabinet_interior", None, True, True,
                          SlotGeometry(ext, float(np.prod(ext)), ext[:2], position[2] - size[2]/2 + .02,
                                       "infinigen_replacement_metadata_pending_cavity_measurement"),
                          position, yaw, "explicit_real_asset_binding_pending_probe"))
    return SceneSlots(doc["scene_id"], 1,
        {"source_kind": "infinigen", "assembly_manifest": str(Path(manifest_path).resolve()),
         "source_manifest_sha256": sha256_path(manifest_path), "source_geometry": "pending_simulation"},
        {"kitchen_0": "kitchen"}, slots, furniture,
        native_objects={"kitchen_0": [o["category"] for o in doc["objects"] if o["role"] == "static_background"]})


def normalize_infinigen_point_photometry(stage, root_path, upstream_commit):
    """Undo ONLY the pinned exporter's legacy Omniverse wattage adjustment.

    b11700e export.py::adjust_wattages multiplies POINT energy by
    20/(4*pi) * 1000/(4*pi*r**2) * 100 before Blender's USD conversion.
    Its historical 1/100 unit assumption does not hold for this metre-stage,
    normalized-light import. Invert that explicit multiplier once on copied
    source lights; preserve native Blender USD photometry, geometry and the
    separately seeded appearance request. No per-image brightness fitting.
    """
    import math
    from pxr import Usd,UsdLux
    from rummagebench.authoring.tasks.infinigen import UPSTREAM_COMMIT
    if upstream_commit!=UPSTREAM_COMMIT:
        raise ValueError('POINT_PHOTOMETRY_REQUIRES_PINNED_EXPORTER')
    result=[]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdLux.SphereLight):continue
        light=UsdLux.SphereLight(prim)
        radius=light.GetRadiusAttr().Get();intensity=light.GetIntensityAttr().Get()
        if (not prim.GetAttribute('userProperties:blender:data_name').Get()
                or light.GetNormalizeAttr().Get() is not True
                or radius is None or not math.isfinite(radius) or radius<=0
                or intensity is None or not math.isfinite(intensity) or intensity<0):
            raise ValueError('PINNED_POINT_LIGHT_PHOTOMETRY_UNRESOLVED')
        marker=prim.GetAttribute('findingbench:pointPhotometryNormalized')
        if marker and marker.Get():raise ValueError('POINT_PHOTOMETRY_DOUBLE_NORMALIZATION')
        factor=20/(4*math.pi)*1000/(4*math.pi*radius**2)*100
        normalized=float(intensity)/factor
        if not math.isfinite(normalized):raise ValueError('POINT_PHOTOMETRY_INVALID')
        light.GetIntensityAttr().Set(normalized)
        from pxr import Sdf
        prim.CreateAttribute('findingbench:pointPhotometryNormalized',Sdf.ValueTypeNames.Bool).Set(True)
        result.append({'prim':str(prim.GetPath()),'source_intensity':float(intensity),'radius_m':float(radius),
                       'removed_legacy_multiplier':factor,'normalized_intensity':normalized})
    return {'version':1,'contract':'pinned_legacy_export_compensation_removed_once',
            'upstream_commit':upstream_commit,'point_lights':result,
            'geometry_or_physics_changed':False,'image_acceptance':'pending'}


def apply_light_appearance(stage, root_path: str, params: dict) -> dict:
    """Real OpenUSD lighting edits, CPU-testable; materials stay explicit pending.

    Only light prims are edited. Unsupported texture/palette requests are
    returned, never silently treated as applied. Images and renderer settings
    remain a separate runtime check.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux
    import math
    multiplier = float(params.get("light_intensity_multiplier", 1.))
    ambient = float(params.get("ambient_multiplier", 1.))
    temperature = float(params.get("light_color_temperature_k", 5000.))
    yaw = float(params.get("light_direction_yaw_deg", 0.))
    if (not all(math.isfinite(v) for v in (multiplier, ambient, temperature, yaw))
            or multiplier <= 0 or ambient <= 0 or not 1000 <= temperature <= 12000):
        raise ValueError("INVALID_LIGHT_APPEARANCE_PARAMETERS")
    root = stage.GetPrimAtPath(root_path)
    applied = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        api = UsdLux.LightAPI(prim)
        value = api.GetIntensityAttr().Get()
        api.CreateIntensityAttr().Set(float(1. if value is None else value) *
                                     (ambient if prim.IsA(UsdLux.DomeLight) else multiplier))
        api.CreateEnableColorTemperatureAttr().Set(True)
        api.CreateColorTemperatureAttr().Set(temperature)
        xf = UsdGeom.Xformable(prim)
        matrix = xf.GetLocalTransformation()
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(matrix * Gf.Matrix4d(1.).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)))
        applied.append(str(prim.GetPath()))
    if not applied:
        raise ValueError("EXPORTED_LIGHTS_MISSING")
    pending = [k for k in ("wall_color", "support_texture_swap", "container_texture_swap") if params.get(k)]
    return {"lighting": "applied", "light_prims": applied, "parameters": params,
            "material_requests_pending": pending, "renderer_and_image_acceptance": "pending"}


def background_records(backend, names: list[str]) -> dict:
    """Actual per-entity pose, velocities/joints/state and geometry for matching."""
    from rummagebench.authoring.tasks.native_occupancy import plain, checked_box
    native = {obj.name: obj for obj in backend._env.scene.objects}
    return {name: {"pose": plain(backend.entity_pose(name)),
                   "aabb": checked_box(backend.entity_aabb(name)),
                   "state": plain(native[name].dump_state(serialized=False))} for name in sorted(names)}


def freeze_family_background(backend, scenarios: list, manifest_path: Path, output: Path) -> dict:
    """GPU stage: freeze once with a parked target, restore before each member.

    Does not certify visibility, skills or search costs. Any change to a
    background entity is rejected exactly, with finite placement retries in
    the existing builder. A failure never drops or reweights a candidate.
    """
    from rummagebench.authoring.tasks.native_occupancy import background_mismatches, capture_native_occupancy, require_known_slots
    from rummagebench.sim.omnigibson.state_io import save_snapshot
    from copy import deepcopy
    from rummagebench.authoring.tasks.household_state import (scenario_state_identity, runtime_topology,
        reconstruct_checkpoint, merge_checkpoint_objects)

    manifest = load_assembly_manifest(manifest_path)
    location_bindings=manifest.get('native_region_bindings',manifest['candidate_bindings'])
    if len(scenarios) != len(location_bindings):
        raise ValueError("INCOMPLETE_FAMILY_SCENARIOS")
    first = scenarios[0]
    native_regions=bool(manifest.get('native_region_catalog'))
    wanted = {(b['name'],b.get('fillable_link') if native_regions else None) for b in location_bindings}
    actual = set()
    for scenario in scenarios:
        if (scenario.scene != first.scene or scenario.robot != first.robot or scenario.anchors != first.anchors
                or scenario.objects != first.objects or scenario.initial_states != first.initial_states
                or scenario.target != first.target or len(scenario.objects) != 1 or len(scenario.placements) != 1):
            raise ValueError("FAMILY_GEOMETRY_OR_TOPOLOGY_DIFFERS")
        placement = scenario.placements[0]
        if placement.entity != first.target.entity or placement.relation != "inside":
            raise ValueError("UNSUPPORTED_FAMILY_TARGET_PLACEMENT")
        actual.add((placement.receptacle,placement.link if native_regions else None))
    if actual != wanted:
        raise ValueError("INCOMPLETE_OR_DUPLICATE_FAMILY_PLACEMENTS")
    output.mkdir(parents=True, exist_ok=False)
    names = [o["name"] for o in manifest["objects"]]
    frozen = {}
    report = {"status": "pending_simulation_acceptance", "evaluation_ready_episodes": 0,
              "version": 2, "gpu_executed": True,
              "assembly_manifest_sha256": sha256_path(manifest_path), "members": [],
              "pending_checks": list(manifest["pending_checks"])}
    try:
        def before_import(b):
            for item in manifest["candidate_bindings"]:
                if not b.set_open(item["name"], False):
                    raise ValueError("INITIAL_BACKGROUND_CLOSE_FAILED")
            b.settle(30); b.validate_physics_state()
            native_input=deepcopy(b.dump_state())
            frozen['native_structured_input']=deepcopy(b._sim.deserialize(deepcopy(native_input))[0])
            reconstruct_checkpoint(b,native_input)
            slots = scene_slots_from_assembly(manifest_path)
            occupancy, _ = capture_native_occupancy(b, slots, source_scene_path=manifest_path)
            require_known_slots(occupancy, [s.slot_id for s in slots.slots])
            atomic_write_json(output / "native_inventory.json", occupancy)
            frozen["records"] = background_records(b, names)
            boxes = np.array([r["aabb"] for r in frozen["records"].values()])
            # Far outside actual native geometry; same parking pose for every
            # member, chosen before reading the hidden target placement.
            frozen["parking"] = (boxes[:, 1, :].max(axis=0) + 5.).tolist()
            return {"background_entities": len(names)}
        def before_place(b):
            mismatch = background_mismatches(frozen["records"], background_records(b, names))
            # Live task import advances physics in the SDK. Explicitly
            # reconstruct the original native inputs with the parked target
            # before insertion; record all import changes, never hide them.
            frozen['import_changes_restored']=mismatch
            complete=merge_checkpoint_objects(frozen['native_structured_input'],
                b._sim.dump_state(serialized=False),first.target.entity)
            input_state=b._sim.serialize(complete)
            frozen['restore_recipe']=reconstruct_checkpoint(b,input_state)
            mismatch = background_mismatches(frozen["records"], background_records(b, names))
            if mismatch:
                raise ValueError(f"BACKGROUND_DISTURBED_BY_TARGET_IMPORT:{mismatch}")
            frozen["snapshot"] = {'sim':deepcopy(input_state),'commanded_pose':None}
            save_snapshot(output / "background_state.pt", frozen["snapshot"]["sim"],
                          {"assembly_manifest_sha256": report["assembly_manifest_sha256"],
                           'restore_recipe':frozen['restore_recipe']})
            atomic_write_json(output / "background_records.json", frozen["records"])
            return {"state_sha256": sha256_path(output / "background_state.pt"),
                    'target_import_changes_restored':frozen['import_changes_restored'],
                    'restore_recipe':frozen['restore_recipe'],
                    "records_sha256": sha256_path(output / "background_records.json")}
        backend.setup(first, native_scene_capture=before_import,
                      task_seed_pose=lambda b, s: frozen["parking"], pre_placement_capture=before_place)
        for scenario in scenarios:
            # Restore full, identical topology, including the parked target.
            # Relation resampling happens only for the target after restore.
            reconstruct_checkpoint(backend,frozen['snapshot']['sim'],
                expected_world_hash=frozen['restore_recipe']['observed_world_state_sha256'])
            backend._scenario = scenario
            mismatch = background_mismatches(frozen["records"], background_records(backend, names))
            if mismatch:
                raise ValueError(f"FROZEN_BACKGROUND_RESTORE_MISMATCH:{mismatch}")
            build = backend._apply_placements(scenario, attempts=3)
            backend.teleport_robot(scenario.anchors[scenario.robot.init_anchor])
            backend.settle(15)
            backend.validate_physics_state()
            mismatch = background_mismatches(frozen["records"], background_records(backend, names))
            if mismatch:
                raise ValueError(f"TARGET_INSERTION_DISTURBED_BACKGROUND:{scenario.id}:{mismatch}")
            # Preserve native INPUT A for each member; dumping the observed
            # background B and using B as the next input drifts on this SDK.
            current=backend._sim.serialize(merge_checkpoint_objects(frozen['native_structured_input'],
                backend._sim.dump_state(serialized=False),first.target.entity))
            recipe=reconstruct_checkpoint(backend,current)
            mismatch=background_mismatches(frozen['records'],background_records(backend,names))
            placement=scenario.placements[0]
            from rummagebench.sim.omnigibson.probe_support import relation_holds,link_membership
            if (mismatch or not relation_holds(backend,first.target.entity,placement.receptacle,'inside')
                    or placement.link is not None and not link_membership(backend,placement.receptacle,first.target.entity,placement.link)):
                raise ValueError(f'MEMBER_RECONSTRUCTION_RELATION_OR_BACKGROUND_MISMATCH:{scenario.id}:{mismatch}')
            for _ in range(2):
                reconstruct_checkpoint(backend,current,expected_world_hash=recipe['observed_world_state_sha256'])
                if background_mismatches(frozen["records"], background_records(backend, names)):
                    raise ValueError(f"MEMBER_RESTORE_MISMATCH:{scenario.id}")
            state_ref = f"states/{scenario.id}.pt"
            records_ref = "background_records.json"
            member = {"version": 2, "status": "frozen_member", "scenario_id": scenario.id,
                      "scenario_sha256": scenario_state_identity(scenario),
                      "assembly_manifest_sha256": report["assembly_manifest_sha256"],
                      "runtime_topology": runtime_topology(backend), "parking": frozen["parking"],
                      "state": state_ref, "background_records": records_ref, "builder": build,
                      'restore_recipe':recipe}
            save_snapshot(output / state_ref, current,
                          {"assembly_manifest_sha256": member["assembly_manifest_sha256"],
                           "scenario_sha256": member["scenario_sha256"],'restore_recipe':recipe})
            member["artifacts"] = {ref: sha256_path(output / ref) for ref in (state_ref, records_ref)}
            member["manifest_sha256"] = sha256_bytes(canonical_json(member))
            atomic_write_json(output / "members" / f"{scenario.id}.json", member)
            report["members"].append({"scenario_id": scenario.id, "background_match": True,
                                      "restore_repeats": 2, "builder": build,
                                      "member_manifest": f"members/{scenario.id}.json"})
        report["background_and_members"] = "frozen_and_restore_checked"
        report['background_restore_recipe']=frozen['restore_recipe']
        report['target_import_changes_restored']=frozen['import_changes_restored']
        report["artifacts"] = {str(p.relative_to(output)): sha256_path(p)
                               for p in sorted(output.rglob("*")) if p.is_file()}
        report["receipt_sha256"] = sha256_bytes(canonical_json(report))
        atomic_write_json(output / "family_preflight.json", report)
        return report
    except Exception as exc:
        report.update(status="rejected_family_preflight", reason=f"{type(exc).__name__}: {exc}")
        atomic_write_json(output / "family_preflight.json", report)
        raise


def normalize_asset_attributes(source_prim, destination_prim) -> None:
    """Anchor external resources before moving their USD properties to new files."""
    from pxr import Sdf, Usd

    for attr in source_prim.GetAttributes():
        if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
            continue
        times = [Usd.TimeCode.Default(), *attr.GetTimeSamples()]
        for time in times:
            value = attr.Get(time)
            if value is None:
                continue
            array = attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray
            result = []
            for asset in (list(value) if array else [value]):
                if not asset.path:
                    result.append(asset); continue
                if not asset.resolvedPath or not Path(asset.resolvedPath).is_file():
                    raise ValueError(f"USD_DEPENDENCY_CANNOT_ANCHOR:{attr.GetPath()}")
                result.append(Sdf.AssetPath(str(Path(asset.resolvedPath).resolve())))
            destination_prim.GetAttribute(attr.GetName()).Set(result if array else result[0], time)


def copy_effective_materials(source_stage, source_mesh, destination_stage, destination_mesh, flattened) -> None:
    """Preserve inherited/collection bindings by binding effective materials explicitly.

    Shader networks must be contained by their material. External shader
    connections are rejected rather than left pointing at missing prims.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    copied = {}
    for source in Usd.PrimRange(source_mesh):
        if source != source_mesh and not source.IsA(UsdGeom.Subset):
            continue
        dest_path = source.GetPath().ReplacePrefix(source_mesh.GetPath(), destination_mesh.GetPath())
        dest = destination_stage.GetPrimAtPath(dest_path)
        for rel in list(dest.GetRelationships()):
            if rel.GetName().startswith('material:binding'):
                dest.RemoveProperty(rel.GetName())
        for purpose in ('', 'preview', 'full'):
            material, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial(materialPurpose=purpose)
            if not material:
                continue
            source_path = material.GetPath()
            if str(source_path) not in copied:
                UsdGeom.Scope.Define(destination_stage, '/World/Looks')
                path = Sdf.Path('/World/Looks/material_' + sha256_bytes(str(source_path).encode())[:16])
                Sdf.CopySpec(flattened, source_path, destination_stage.GetRootLayer(), path)
                for prim in Usd.PrimRange(destination_stage.GetPrimAtPath(path)):
                    original_path = prim.GetPath().ReplacePrefix(path, source_path)
                    original = source_stage.GetPrimAtPath(original_path)
                    normalize_asset_attributes(original, prim)
                    for attr in prim.GetAttributes():
                        connections = attr.GetConnections()
                        if any(not target.HasPrefix(source_path) and not target.HasPrefix(path) for target in connections):
                            raise ValueError('USD_EXTERNAL_SHADER_NETWORK_UNSUPPORTED')
                        if connections:
                            attr.SetConnections([target.ReplacePrefix(source_path, path) for target in connections])
                copied[str(source_path)] = path
            bound = UsdShade.Material(destination_stage.GetPrimAtPath(copied[str(source_path)]))
            UsdShade.MaterialBindingAPI.Apply(dest).Bind(bound, materialPurpose=purpose)
