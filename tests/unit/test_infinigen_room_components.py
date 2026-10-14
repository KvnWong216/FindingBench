"""Actual OpenUSD sibling part mapping over explicitly synthetic scene assets."""
import pytest

from test_infinigen_pipeline import usd_bundle, make_spec


def test_sibling_floor_and_wall_keep_room_identity_and_floor_sampling(usd_bundle,tmp_path):
    from pxr import Usd,UsdGeom
    from rummagebench.authoring.tasks.infinigen import inspect_export,source_mesh_owner
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.infinigen_probe import assembly_floor_map
    scene,solve,assets,bindings=usd_bundle
    stage=Usd.Stage.Open(str(scene))
    for part,z in [('floor',0.),('wall',2.)]:
        mesh=UsdGeom.Mesh.Define(stage,'/World/kitchen_'+part)
        mesh.CreatePointsAttr([(-4,-4,z),(4,-4,z),(4,4,z),(-4,4,z)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    # Not an exact known room component; must not acquire fabricated ownership.
    unrelated=UsdGeom.Mesh.Define(stage,'/World/kitchen_floor_fake')
    unrelated.CreatePointsAttr([(0,0,-1),(1,0,-1),(0,1,-1)])
    unrelated.CreateFaceVertexCountsAttr([3]);unrelated.CreateFaceVertexIndicesAttr([0,1,2])
    stage.GetRootLayer().Save()
    inspection=inspect_export(scene,solve)
    assert inspection['source_component_paths']['kitchen']=={'floor':'/World/kitchen_floor','wall':'/World/kitchen_wall'}
    assert source_mesh_owner(inspection,'/World/kitchen_wall')==('kitchen','wall')
    assert source_mesh_owner(inspection,'/World/kitchen_floor_fake')==(None,None)
    manifest=prepare_assembly(make_spec(),scene,solve,bindings,assets,tmp_path/'assembly')
    assert any(o['source_component']=='wall' and o['source_id']=='kitchen' for o in manifest['objects'] if o['role']=='static_background')
    room=assembly_floor_map(manifest,tmp_path/'assembly/assembly.json')
    assert room.floor_z==pytest.approx(0.) and room.free_points('kitchen_0',.4,.25)
