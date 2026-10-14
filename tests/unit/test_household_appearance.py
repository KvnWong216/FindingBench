"""Real standalone OpenUSD edits, synthetic materials/images only; no render."""
import numpy as np
import pytest

from rummagebench.authoring.tasks.household_appearance import apply_wall_palette, image_quality, geometry_fingerprint,WALL_PALETTE_LINEAR_RGB


def test_wall_palette_clones_shared_texture_network_without_changing_table_or_geometry(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Sdf,Gf
    stage=Usd.Stage.CreateNew(str(tmp_path/'synthetic_TEST_ONLY.usda'))
    mesh_roots=['/wall','/table']
    for root in mesh_roots:
        UsdGeom.Xform.Define(stage,root)
        mesh=UsdGeom.Mesh.Define(stage,root+'/mesh')
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    material=UsdShade.Material.Define(stage,'/Looks/shared')
    shader=UsdShade.Shader.Define(stage,'/Looks/shared/preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.1,.2,.3))
    texture=UsdShade.Shader.Define(stage,'/Looks/shared/texture');texture.CreateIdAttr('UsdUVTexture')
    path=tmp_path/'unit_texture.png';path.write_bytes(b'synthetic file only; no rendered data')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('unit_texture.png'))
    texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(),'rgb')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    for root in mesh_roots:UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root)).Bind(material)
    before=geometry_fingerprint(stage,mesh_roots)
    report=apply_wall_palette(stage,['/wall'],'warm_beige')
    assert report['status']=='applied' and report['geometry_before_sha256']==report['geometry_after_sha256']
    assert geometry_fingerprint(stage,mesh_roots)==before
    wall,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/wall/mesh')).ComputeBoundMaterial()
    table,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/table/mesh')).ComputeBoundMaterial()
    assert wall.GetPath()!=material.GetPath() and table.GetPath()==material.GetPath()
    new_shader=UsdShade.Shader(stage.GetPrimAtPath(report['applied'][0]['shader']))
    assert list(new_shader.GetInput('diffuseColor').Get())==pytest.approx(WALL_PALETTE_LINEAR_RGB['warm_beige'])
    assert not new_shader.GetInput('diffuseColor').HasConnectedSource()
    assert shader.GetInput('diffuseColor').HasConnectedSource()  # original shared table network retained
    cloned_texture=UsdShade.Shader(stage.GetPrimAtPath(str(wall.GetPath())+'/texture'))
    assert cloned_texture.GetInput('file').Get().resolvedPath==str(path)
    assert apply_wall_palette(stage,['/wall'],'warm_beige')['status']=='applied'


def test_unsupported_shader_is_pending_and_does_not_claim_applied(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdShade
    stage=Usd.Stage.CreateInMemory();mesh=UsdGeom.Mesh.Define(stage,'/wall/mesh')
    material=UsdShade.Material.Define(stage,'/Looks/unsupported');shader=UsdShade.Shader.Define(stage,'/Looks/unsupported/mdl')
    shader.CreateIdAttr('UnsupportedSyntheticShader')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    report=apply_wall_palette(stage,['/wall'],'cool_white')
    assert report['status']=='pending' and report['pending'][0]['reason']=='WALL_SHADER_NETWORK_UNSUPPORTED'
    assert report['applied']==[]
    assert apply_wall_palette(stage,[],'cool_white')['status']=='pending'


@pytest.mark.parametrize('value,reason',[(0,'IMAGE_ALMOST_ALL_BLACK'),(255,'IMAGE_ALMOST_ALL_CLIPPED'),(80,'IMAGE_INSUFFICIENT_LUMINANCE_RANGE')])
def test_numeric_qc_rejects_blank_images(value,reason):
    report=image_quality(np.full((20,30,3),value,np.uint8),(20,30,3))
    assert report['status']=='failed' and reason in report['rejections']


def test_numeric_qc_is_not_a_semantic_or_camera_certification():
    image=np.tile(np.arange(30,dtype=np.uint8)[None,:,None]*6,(20,1,3))
    report=image_quality(image,(20,30,3))
    assert report['status']=='passed_numeric_qc' and report['semantic_review']=='pending'
    assert image_quality(image.astype(float),(20,30,3))['status']=='failed'
    assert image_quality(image,(40,30,3))['status']=='failed'
