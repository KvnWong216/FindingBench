"""Recover real upstream cabinet cells before Blender joins their meshes.

This instrumentation adds face provenance only. It neither samples geometry
nor invents joints. Drawer cavities remain separate search locations even
when the current skill binding cannot address them separately.
"""
from __future__ import annotations

import hashlib

import numpy as np

from rummagebench.authoring.tasks.plan import canonical_json

CELL_ATTRIBUTE = "findingbench_cabinet_cell"
PART_ATTRIBUTE = "findingbench_cabinet_part"


def _mark_faces(obj, cell, part):
    if obj.type != "MESH" or not obj.data.polygons:
        raise ValueError("CABINET_COMPONENT_MESH_MISSING")
    for name, value in ((CELL_ATTRIBUTE, cell), (PART_ATTRIBUTE, part)):
        if obj.data.attributes.get(name) is not None:
            raise ValueError("CABINET_COMPONENT_PROVENANCE_ALREADY_PRESENT")
        attribute = obj.data.attributes.new(name, "INT", "FACE")
        attribute.data.foreach_set("value", np.full(len(obj.data.polygons), value, dtype=np.int32))


class CabinetProvenance:
    """Scoped wrapper of the pinned factory's actual component API."""

    def __init__(self, factory_class):
        self.factory_class = factory_class
        self.original = factory_class.create_cabinet_components
        self.records = {}
        self.installed = False

    def install(self):
        if self.installed:
            raise ValueError("CABINET_PROVENANCE_WRAPPER_ALREADY_INSTALLED")
        registry = self

        def components(factory, *args, **kwargs):
            result = registry.original(factory, *args, **kwargs)
            # The original call has already chosen type, widths and layers.
            # Never consume RNG or replace its asset creation functions.
            for index, (frame, kind, *attachments) in enumerate(result):
                cell = len(registry.records) + 1
                _mark_faces(frame, cell, 0)
                record = {"cell_id": cell, "factory_seed": int(factory.factory_seed),
                          "cell_index": index, "kind": kind, "attachments": []}
                if kind == "door":
                    right, left, params = attachments[0]
                    _mark_faces(right, cell, 1)
                    _mark_faces(left, cell, 2)
                    record["attachments"] = [{"part_id": i + 1, "hinge_local": list(pos),
                                               "initial_angle": float(params["door_open_angle"])}
                                              for i, pos in enumerate(params["door_hinge_pos"])]
                elif kind == "drawer":
                    for layer, (drawer, params) in enumerate(attachments[0]):
                        _mark_faces(drawer, cell, layer + 1)
                        record["attachments"].append({"part_id": layer + 1,
                            "hinge_local": list(params["drawer_hinge_pos"]),
                            "interior_dimensions_proposal_m": [float(params["drawer_depth"]),
                                float(params["drawer_board_width"]), float(params["drawer_board_height"])]})
                elif kind != "none":
                    raise ValueError("UNSUPPORTED_REAL_CABINET_COMPONENT_KIND")
                registry.records[cell] = record
            return result

        self.factory_class.create_cabinet_components = components
        self.installed = True
        return self

    def restore(self):
        if self.installed:
            self.factory_class.create_cabinet_components = self.original
            self.installed = False


def mesh_geometry_digest(mesh):
    """Indexed geometry and UV evidence; ignores only added provenance."""
    payload = {"vertices": [list(v.co) for v in mesh.vertices],
               "faces": [list(p.vertices) for p in mesh.polygons],
               "material_indices": [int(p.material_index) for p in mesh.polygons],
               "uv_layers": [[list(d.uv) for d in layer.data] for layer in mesh.uv_layers]}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def census_mesh(obj, registry: CabinetProvenance, source_id: str):
    """Read face ownership from the real final joined mesh, in world space."""
    import bpy

    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh(preserve_all_data_layers=True,
                             depsgraph=bpy.context.evaluated_depsgraph_get())
    try:
        ca, pa = mesh.attributes.get(CELL_ATTRIBUTE), mesh.attributes.get(PART_ATTRIBUTE)
        if (ca is None or pa is None or ca.domain != "FACE" or pa.domain != "FACE"
                or ca.data_type != "INT" or pa.data_type != "INT"):
            raise ValueError(f"NESTED_CABINET_PROVENANCE_LOST:{source_id}")
        ownership = {}
        for face in mesh.polygons:
            cell, part = ca.data[face.index].value, pa.data[face.index].value
            if cell == 0:
                if part != 0:
                    raise ValueError("CABINET_BACKGROUND_PART_WITHOUT_CELL")
                continue  # Real countertop added after the tagged cabinet.
            if cell not in registry.records:
                raise ValueError("CABINET_FACE_REFERENCES_UNKNOWN_COMPONENT")
            ownership.setdefault(cell, {}).setdefault(part, []).append(face.index)
        cells, locations = [], []
        for cell, parts in sorted(ownership.items()):
            record = registry.records[cell]
            expected = {0, *(a["part_id"] for a in record["attachments"])}
            if set(parts) != expected:
                raise ValueError(f"CABINET_COMPONENT_FACE_COVERAGE_MISMATCH:{source_id}:{cell}")
            indices = sorted(i for faces in parts.values() for i in faces)
            points = np.asarray([evaluated.matrix_world @ mesh.vertices[v].co
                                 for i in indices for v in mesh.polygons[i].vertices], dtype=float)
            if not np.isfinite(points).all():
                raise ValueError("CABINET_COMPONENT_GEOMETRY_NONFINITE")
            key = f"{source_id}__cell_{record['cell_index']}_{cell}"
            cells.append(dict(record, source_id=source_id, component_id=key,
                aabb=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                face_indices=indices, attachment_faces={str(p): v for p, v in parts.items()},
                articulation="static_mesh_requires_real_interactive_asset_binding"))
            if record["kind"] == "door":
                locations.append({"location_id": key, "component_id": key, "source_id": source_id,
                                  "slot_type": "cabinet_interior", "coupled_source_component": False})
            elif record["kind"] == "drawer":
                for attachment in record["attachments"]:
                    locations.append({"location_id": key + f"__drawer_{attachment['part_id']}",
                        "component_id": key, "source_id": source_id, "slot_type": "drawer",
                        "part_id": attachment["part_id"],
                        "coupled_source_component": len(record["attachments"]) > 1})
        if not cells:
            raise ValueError(f"CABINET_COMPONENT_CENSUS_EMPTY:{source_id}")
        return {"source_id": source_id, "blender_name": obj.name,
                "geometry_sha256": mesh_geometry_digest(mesh), "face_count": len(mesh.polygons),
                "background_face_count": sum(ca.data[p.index].value == 0 for p in mesh.polygons),
                "cells": cells, "locations": locations, "geometry_modified": False}
    finally:
        evaluated.to_mesh_clear()


def complete_cabinet_census(raw, registry):
    """Traverse actual source relations, including KitchenSpace/Island assets."""
    import bpy
    from rummagebench.authoring.tasks.infinigen import normalize_solver_tag

    objects = raw["objs"]
    roots = []
    for key, source in objects.items():
        seen, queue = set(), [key]
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            if node not in objects:
                raise ValueError("CABINET_CENSUS_SOURCE_RELATION_UNRESOLVED")
            seen.add(node)
            queue.extend(r["target_name"] for r in objects[node].get("relations", []))
        tags = {normalize_solver_tag(t) for t in source.get("tags", [])}
        in_kitchen = any("Semantics.Kitchen" in
            {normalize_solver_tag(t) for t in objects[n].get("tags", [])} for n in seen)
        if in_kitchen and tags & {"Semantics.Storage", "Semantics.KitchenCounter"}:
            roots.append(key)
    result, failures = [], []
    for key in sorted(roots):
        try:
            obj = bpy.data.objects.get(objects[key].get("obj") or "")
            if obj is None or obj.type != "MESH":
                raise ValueError("CABINET_SOURCE_REAL_MESH_MISSING")
            result.append(census_mesh(obj, registry, key))
        except ValueError as exc:
            failures.append({"source_id": key, "reason": str(exc)})
    return {"version": 1, "scope": "all_actual_kitchen_cabinet_cells_and_drawers",
            "status": "source_complete_pending_binding" if roots and not failures else "pending",
            "source_roots": sorted(roots), "sources": result, "failures": failures,
            "search_location_count": sum(len(r["locations"]) for r in result),
            "physical_acceptance": "pending", "evaluation_ready_episodes": 0}
