"""CPU-testable wall shader edits and image QC, with explicit limited scope.

No geometry, collision, object identity or task state is edited. Only a
material cloned for an explicitly mapped room wall may change. Unsupported
shader/texture requests remain pending; numeric image QC is not semantic review.
"""
from __future__ import annotations

import numpy as np

from rummagebench.authoring.tasks.plan import canonical_json,sha256_bytes

WALL_PALETTE_LINEAR_RGB = {"neutral_gray":[.65,.65,.65],"warm_beige":[.76,.70,.60],"cool_white":[.85,.88,.92]}


def geometry_fingerprint(stage, roots) -> str:
    from pxr import Usd,UsdGeom
    records = {}
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not (prim.IsA(UsdGeom.Boundable) or prim.IsA(UsdGeom.Xformable)):
                continue
            attrs = {a.GetName():{"default":str(a.Get()),"samples":{str(t):str(a.Get(t)) for t in a.GetTimeSamples()}}
                     for a in prim.GetAttributes() if a.GetName().startswith(("xform","physics:","physx"))
                     or a.GetName() in ("points","faceVertexIndices","faceVertexCounts","extent","size","radius","height")}
            records[str(prim.GetPath())]=attrs
    return sha256_bytes(canonical_json(records))


def apply_wall_palette(stage, wall_roots: list[str], palette: str) -> dict:
    from pxr import Sdf,Usd,UsdGeom,UsdShade,Gf
    from rummagebench.sim.omnigibson.infinigen_adapter import normalize_asset_attributes
    if palette not in WALL_PALETTE_LINEAR_RGB:
        raise ValueError("UNAPPROVED_WALL_PALETTE")
    roots = sorted(set(wall_roots))
    if len(roots)!=len(wall_roots) or any(not stage.GetPrimAtPath(p) for p in roots):
        raise ValueError("EXPLICIT_WALL_ROOTS_REQUIRED")
    before = geometry_fingerprint(stage,roots)
    applied, pending = [], []
    for root_path in roots:
        mesh_prims = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)) if p.IsA(UsdGeom.Mesh)]
        for mesh in mesh_prims:
            # Include subset-bound materials; never leave a subset overriding
            # a tinted parent with the unchanged inherited material.
            surfaces = [mesh,*[p for p in mesh.GetChildren() if p.IsA(UsdGeom.Subset)]]
            for surface in surfaces:
                material,_=UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()
                if not material:
                    pending.append({"prim":str(surface.GetPath()),"reason":"WALL_EFFECTIVE_MATERIAL_MISSING"})
                    continue
                source = material.GetPath()
                shaders = [p for p in Usd.PrimRange(material.GetPrim()) if p.IsA(UsdShade.Shader)]
                # Only the tested standalone USDPreviewSurface contract is
                # currently supported. An MDL identifier is not guessed.
                preview = [p for p in shaders if UsdShade.Shader(p).GetIdAttr().Get()=="UsdPreviewSurface"]
                supported={"UsdPreviewSurface","UsdUVTexture","UsdPrimvarReader_float2","UsdTransform2d"}
                if len(preview)!=1 or any(UsdShade.Shader(p).GetIdAttr().Get() not in supported for p in shaders):
                    pending.append({"prim":str(surface.GetPath()),"reason":"WALL_SHADER_NETWORK_UNSUPPORTED"})
                    continue
                destination = Sdf.Path(root_path+"/findingbench_appearance/material_"+sha256_bytes(str(surface.GetPath()).encode())[:16])
                UsdGeom.Scope.Define(stage,root_path+"/findingbench_appearance")
                # Copy only this composed material network. Flattening the
                # entire live simulator stage would unnecessarily duplicate
                # every robot/asset layer just to tint one wall.
                originals=list(Usd.PrimRange(material.GetPrim()))
                for original in originals:
                    target=original.GetPath().ReplacePrefix(source,destination)
                    prim=stage.DefinePrim(target,original.GetTypeName())
                    for attr in original.GetAttributes():
                        copy=prim.CreateAttribute(attr.GetName(),attr.GetTypeName(),custom=attr.IsCustom(),variability=attr.GetVariability())
                        value=attr.Get()
                        if value is not None:copy.Set(value)
                        for time in attr.GetTimeSamples():copy.Set(attr.Get(time),time)
                        connections=attr.GetConnections()
                        if any(not p.HasPrefix(source) for p in connections):
                            raise ValueError("WALL_SHADER_CONNECTION_OUTSIDE_PRIVATE_MATERIAL")
                        if connections:copy.SetConnections([p.ReplacePrefix(source,destination) for p in connections])
                    for rel in original.GetRelationships():
                        targets=rel.GetTargets()
                        if any(not p.HasPrefix(source) for p in targets):
                            raise ValueError("WALL_MATERIAL_RELATION_OUTSIDE_PRIVATE_NETWORK")
                        prim.CreateRelationship(rel.GetName(),custom=rel.IsCustom()).SetTargets([p.ReplacePrefix(source,destination) for p in targets])
                    normalize_asset_attributes(original,prim)
                shader_path=preview[0].GetPath().ReplacePrefix(source,destination)
                shader=UsdShade.Shader(stage.GetPrimAtPath(shader_path))
                diffuse=shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f)
                diffuse.DisconnectSource()
                diffuse.Set(Gf.Vec3f(*WALL_PALETTE_LINEAR_RGB[palette]))
                UsdShade.MaterialBindingAPI.Apply(surface).Bind(UsdShade.Material(stage.GetPrimAtPath(destination)))
                applied.append({"prim":str(surface.GetPath()),"private_material":str(destination),"shader":str(shader_path)})
    after=geometry_fingerprint(stage,roots)
    if before!=after:
        raise ValueError("APPEARANCE_CHANGED_GEOMETRY_OR_COLLISION")
    return {"status":"applied" if applied and not pending else "pending","palette":palette,
            "linear_rgb":WALL_PALETTE_LINEAR_RGB[palette],"applied":applied,"pending":pending,
            "geometry_before_sha256":before,"geometry_after_sha256":after,
            "image_acceptance":"pending_gpu","scope":"explicit_room_wall_private_preview_shader_only"}


def image_quality(rgb, expected_shape) -> dict:
    pixels=np.asarray(rgb)
    if pixels.dtype!=np.uint8 or tuple(pixels.shape)!=tuple(expected_shape) or pixels.ndim!=3 or pixels.shape[2]!=3:
        return {"status":"failed","rejections":["IMAGE_RGB_SHAPE_OR_DTYPE_INVALID"],"semantic_review":"pending"}
    # Conservative numeric QC only. Textures/semantic correctness, head-camera
    # protocol and visible target leakage are checked separately.
    luma=pixels.astype(float) @ np.array([.2126,.7152,.0722])
    dark=float((luma<=2.).mean());clipped=float((luma>=253.).mean());spread=float(np.percentile(luma,95)-np.percentile(luma,5))
    failures=[]
    if dark>.95:failures.append("IMAGE_ALMOST_ALL_BLACK")
    if clipped>.95:failures.append("IMAGE_ALMOST_ALL_CLIPPED")
    if spread<8.:failures.append("IMAGE_INSUFFICIENT_LUMINANCE_RANGE")
    return {"status":"failed" if failures else "passed_numeric_qc","rejections":failures,
            "shape":list(pixels.shape),"dark_fraction":dark,"clipped_fraction":clipped,"luminance_range_p5_p95":spread,
            "semantic_review":"pending","scope":"rgb_numeric_quality_only"}
