from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))
from rab_assets import AssetNamespace,LayerBundle,joints


def test_namespace_uses_longest_binding_and_rejects_missing_resources(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir();(b/'item.usda').write_text('native')
    ns=AssetNamespace({'/old':a,'/old/special':b})
    assert ns.resolve('/old/special/item.usda')==b/'item.usda'
    with pytest.raises(FileNotFoundError):ns.resolve('/old/special/missing.usda')


def test_usd_relocation_preserves_joint_and_source_bytes(tmp_path):
    from pxr import Sdf,Usd,UsdGeom,UsdPhysics
    assets=tmp_path/'assets';assets.mkdir();source=assets/'source.usda';child=assets/'mechanism.usda'
    stage=Usd.Stage.CreateNew(str(child));root=UsdGeom.Xform.Define(stage,'/Asset').GetPrim();stage.SetDefaultPrim(root)
    body=UsdGeom.Xform.Define(stage,'/Asset/Drawer').GetPrim();UsdPhysics.RigidBodyAPI.Apply(body)
    slide=UsdPhysics.PrismaticJoint.Define(stage,'/Asset/Slide');slide.CreateAxisAttr('X')
    slide.CreateLowerLimitAttr(0);slide.CreateUpperLimitAttr(.3);slide.CreateBody1Rel().SetTargets(['/Asset/Drawer'])
    stage.GetRootLayer().Save()
    stage=Usd.Stage.CreateNew(str(source));stage.DefinePrim('/World/Box').GetReferences().AddReference('/legacy/mechanism.usda');stage.GetRootLayer().Save()
    original=source.read_bytes(),child.read_bytes()
    bundle=LayerBundle(AssetNamespace({'/legacy':assets}),tmp_path/'relocated')
    clone=Usd.Stage.Open(str(bundle.clone(source)))
    found=joints(clone)
    assert len(found)==1 and found[0]['type']=='PhysicsPrismaticJoint'
    assert found[0]['body1']==['/World/Box/Drawer'] and found[0]['upper']==pytest.approx(.3)
    assert clone.GetPrimAtPath('/World/Box/Drawer').HasAPI(UsdPhysics.RigidBodyAPI)
    assert (source.read_bytes(),child.read_bytes())==original
    assert len(bundle.receipt()['source_layers_sha256'])==2


def test_packaged_texture_reference_keeps_member_and_relocates_outer_package(tmp_path):
    from pxr import Sdf,Ar
    resources=tmp_path/'resources';resources.mkdir()
    package=resources/'model.usdz';package.write_bytes(b'package identity retained')
    source=resources/'input.usda';layer=Sdf.Layer.CreateNew(str(source))
    prim=Sdf.CreatePrimInLayer(layer,'/Material')
    attr=Sdf.AttributeSpec(prim,'texture',Sdf.ValueTypeNames.Asset)
    attr.default=Sdf.AssetPath('/legacy/model.usdz[texture.png]');layer.Save()
    bundle=LayerBundle(AssetNamespace({'/legacy':resources}),tmp_path/'clone')
    clone=Sdf.Layer.FindOrOpen(str(bundle.clone(source)))
    actual=clone.GetAttributeAtPath('/Material.texture').default.path
    assert Ar.SplitPackageRelativePathOuter(actual)==(str(package),'texture.png')
