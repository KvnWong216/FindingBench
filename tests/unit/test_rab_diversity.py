import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))
from rab_diversity import geometry


def test_furniture_motion_does_not_create_a_new_house_geometry_group():
    from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf
    stage=Usd.Stage.CreateInMemory()
    def triangle(path):
        mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
        return mesh
    triangle('/World/Room/Wall_0/Mesh')
    table=triangle('/World/Assets/CoffeeTable/Mesh')
    shader=UsdShade.Shader.Define(stage,'/World/Looks/Paint/Shader')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1,0,0))
    first=geometry(stage)
    table.AddTranslateOp().Set((2,0,0))
    moved=geometry(stage)
    assert first['architectural_geometry_signature']==moved['architectural_geometry_signature']
    assert first['architectural_world_geometry_signature']==moved['architectural_world_geometry_signature']
    assert first['furniture_and_architecture_placement_signature']!=moved['furniture_and_architecture_placement_signature']
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(0,1,0))
    tinted=geometry(stage)
    assert tinted['material_opinion_signature']!=moved['material_opinion_signature']
    assert tinted['architectural_geometry_signature']==moved['architectural_geometry_signature']


def test_discovery_includes_raw_packaged_models_outside_registered_wrappers(tmp_path):
    from rab_diversity import discover_entries
    files={
        'libraries/current_assets/all_assets/wallet/asset.usda':tmp_path/'wallet.usda',
        'profiles/room/assets/house_scene_l1/demo_sim/usdz/umbrella.usdz':tmp_path/'umbrella.usdz',
        'profiles/room/assets/house_scene_l1/demo_sim/scene_room_baseline.usda':tmp_path/'scene.usda',
    }
    for value in files.values():value.write_text('resource')
    entries,inventory=discover_entries(dict(resources={k:str(v) for k,v in files.items()}))
    assert len(entries)==2
    assert any(r['id']=='wallet' and r['kind']=='registered_wrapper' for r in entries)
    assert any(r['kind']=='additional_source_model' and 'umbrella' in r['logical'] for r in entries)
    assert len(inventory)==2  # scene sources inventoried, not counted as portable models
