"""Resolve RAB's published resource namespace without changing host paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys


class AssetNamespace:
    def __init__(self,mounts):
        self.mounts=sorted(((str(k).rstrip('/'),Path(v).resolve()) for k,v in mounts.items()),
                           key=lambda item:-len(item[0]))

    @classmethod
    def from_runtime(cls,runtime):
        from robotagentbench.namespace import command
        # Reuse upstream's complete namespace plan; do not start a container.
        args=command(Path(runtime).resolve(),'rab-resource-plan','plan')
        mounts={}
        for i,arg in enumerate(args):
            if arg!='--mount':continue
            fields=dict(p.split('=',1) for p in args[i+1].split(',') if '=' in p)
            mounts[fields['dst']]=fields['src']
        room_assets=Path(mounts['/home/lingjun/robot_agent_arena/api-protocol/assets'])
        mounts['D:/00_Sale/TableBasse/TableBasse/PBR_Textures']=(
            room_assets/'house_scene_l1/assets_living_room/coffee_table/PBR_Textures')
        for path in sys.path:
            material_root=Path(path)/'isaacsim/kit/mdl/core/Base'
            if (material_root/'OmniGlass.mdl').is_file():
                mounts['/home/lingjun/robot_agent_arena/.venv/lib/python3.11/site-packages/isaacsim/kit/mdl/core/Base']=material_root
                break
        return cls(mounts)

    def resolve(self,name,origin=None):
        if not name:raise ValueError('EMPTY_RAB_ASSET_REFERENCE')
        for logical,physical in self.mounts:
            if name==logical or name.startswith(logical+'/'):
                result=physical/name[len(logical):].lstrip('/')
                if not result.is_file():raise FileNotFoundError('RAB_NAMESPACE_RESOURCE_MISSING:'+name)
                return result.resolve()
        if name.startswith('/'):
            result=Path(name)
        else:
            if origin is None:raise ValueError('RELATIVE_RAB_REFERENCE_WITHOUT_ORIGIN')
            result=Path(origin).parent/name
        if not result.is_file():raise FileNotFoundError('RAB_ASSET_REFERENCE_MISSING:'+name)
        return result.resolve()


class LayerBundle:
    """Relocate only resource references; retain physical/visual USD opinions."""
    def __init__(self,namespace,output):
        self.namespace=namespace;self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.layers={};self.sources={};self.builtin_materials=set()

    def clone(self,path):
        from pxr import Sdf,UsdUtils,Ar
        path=Path(path).resolve(strict=True)
        if path in self.layers:return self.layers[path]
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        # Equal bytes at different source directories can resolve references
        # differently. Include the source identity in the output key.
        key=hashlib.sha256((str(path)+digest).encode()).hexdigest()
        target=self.output/(key+'.usdc')
        source=Sdf.Layer.FindOrOpen(str(path))
        if source is None:raise ValueError('RAB_USD_LAYER_UNREADABLE:'+str(path))
        layer=Sdf.Layer.CreateAnonymous('relocated.usdc')
        layer.TransferContent(source)
        self.layers[path]=target
        self.sources[str(path)]=digest
        def rewrite(reference):
            if not reference:return reference
            # The published flattened layouts retain Unreal source identifiers
            # in assetInfo and info:unreal:sourceAsset. Keep that provenance.
            if reference.startswith('/Game/') or (
                    reference.startswith('/Users/dylan/Documents/Codex/')
                    and ('/kitchen_demo_export/Assets/' in reference
                         or '/kitchen_level_export/Assets/' in reference)):
                return reference
            if Ar.IsPackageRelativePath(reference):
                package,member=Ar.SplitPackageRelativePathOuter(reference)
                resolved=self.namespace.resolve(package,path)
                return Ar.JoinPackageRelativePath(str(resolved),member)
            try:resolved=self.namespace.resolve(reference,path)
            except FileNotFoundError:
                if reference.endswith('.mdl') and '/' not in reference:
                    self.builtin_materials.add(reference);return reference
                raise
            if resolved.suffix.lower() in ('.usd','.usda','.usdc'):
                return str(self.clone(resolved))
            return str(resolved)
        try:
            UsdUtils.ModifyAssetPaths(layer,rewrite)
            temporary=target.with_suffix('.tmp.usdc')
            if not layer.Export(str(temporary)):
                raise ValueError('RAB_LAYER_EXPORT_FAILED:'+str(path))
            temporary.replace(target)
        except Exception:
            self.layers.pop(path,None)
            self.sources.pop(str(path),None)
            raise
        return target

    def receipt(self):
        return dict(source_layers_sha256=self.sources,
                    relocated_layers={str(k):str(v) for k,v in self.layers.items()},
                    sdk_material_modules=sorted(self.builtin_materials),
                    scope='Resource references relocated; no original layers modified')


def joints(stage):
    from pxr import UsdPhysics
    result=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        result.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),
            enabled=prim.GetAttribute('physics:jointEnabled').Get() is not False,
            axis=prim.GetAttribute('physics:axis').Get(),
            lower=prim.GetAttribute('physics:lowerLimit').Get(),
            upper=prim.GetAttribute('physics:upperLimit').Get(),
            body0=list(map(str,prim.GetRelationship('physics:body0').GetTargets())),
            body1=list(map(str,prim.GetRelationship('physics:body1').GetTargets()))))
    return result
