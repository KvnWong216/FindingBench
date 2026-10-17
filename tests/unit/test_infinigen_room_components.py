"""Actual OpenUSD sibling part mapping over explicitly synthetic scene assets."""
import pytest

from test_infinigen_pipeline import usd_bundle, make_spec


def test_static_import_hook_retains_real_mesh_collision_and_refuses_dynamic_body():
    """Real USD, explicit unit-only link fixture; no simulator acceptance."""
    from types import SimpleNamespace
    from pxr import Usd,UsdGeom,UsdPhysics
    from rummagebench.sim.omnigibson.infinigen_adapter import restore_source_static_colliders
    stage=Usd.Stage.CreateInMemory()
    body=UsdGeom.Xform.Define(stage,'/unit_body').GetPrim()
    api=UsdPhysics.RigidBodyAPI.Apply(body);api.CreateKinematicEnabledAttr(True)
    mesh=UsdGeom.Mesh.Define(stage,'/unit_body/unit_mesh').GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    collision=UsdPhysics.MeshCollisionAPI.Apply(mesh);collision.CreateApproximationAttr('boundingCube')
    link=SimpleNamespace(prim=body,collision_meshes={'mesh':SimpleNamespace(prim=mesh)},
        set_collision_approximation=lambda value:collision.GetApproximationAttr().Set(value))
    obj=SimpleNamespace(name='unit_background',links={'base_link':link})
    assert restore_source_static_colliders(obj,{'other'}) is None
    assert collision.GetApproximationAttr().Get()=='boundingCube'
    report=restore_source_static_colliders(obj,{'unit_background'})
    assert collision.GetApproximationAttr().Get()=='none' and report['before_first_play']
    api.GetKinematicEnabledAttr().Set(False)
    with pytest.raises(ValueError,match='MUST_REMAIN_KINEMATIC'):
        restore_source_static_colliders(obj,{'unit_background'})


def test_runtime_lighting_uses_sdk_edit_context_and_real_usd_light(tmp_path,monkeypatch):
    """Real USD light edits with a unit-only SDK synchronization guard."""
    from contextlib import contextmanager
    from types import SimpleNamespace
    import sys
    import pxr
    from pxr import Usd,UsdGeom,UsdLux
    from rummagebench.sim.omnigibson import infinigen_adapter as adapter
    from rummagebench.authoring.tasks import infinigen_probe
    light_path=tmp_path/'lighting.usda'
    lighting=Usd.Stage.CreateNew(str(light_path));root=UsdGeom.Xform.Define(lighting,'/World').GetPrim()
    lighting.SetDefaultPrim(root)
    light=UsdLux.DistantLight.Define(lighting,'/World/unit_light');light.CreateIntensityAttr(100.)
    lighting.GetRootLayer().Save()
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/scene_0')
    guard={'active':False,'entries':0}
    class GuardedStage:
        def DefinePrim(self,*a):
            assert guard['active'],'USD write outside SDK context'
            return stage.DefinePrim(*a)
        def __getattr__(self,name):return getattr(stage,name)
    @contextmanager
    def editing():
        guard.update(active=True,entries=guard['entries']+1)
        try:yield
        finally:guard['active']=False
    og=SimpleNamespace(sim=SimpleNamespace(stage=GuardedStage(),editing_usd=editing),lazy=SimpleNamespace(pxr=pxr))
    monkeypatch.setitem(sys.modules,'omnigibson',og)
    from rummagebench.authoring.tasks.household import UPSTREAM_COMMIT
    manifest={'lighting_usd':'lighting.usda','appearance_request':{'light_intensity_multiplier':2.},
              'source':{'upstream_commit':UPSTREAM_COMMIT}}
    monkeypatch.setattr(adapter,'load_assembly_manifest',lambda *a:manifest)
    monkeypatch.setattr(infinigen_probe,'assembly_floor_map',lambda *a:SimpleNamespace(floor_z=.1))
    env=SimpleNamespace(scene=SimpleNamespace(prim_path='/World/scene_0'))
    adapter.install_runtime_lighting(env,str(tmp_path/'unit_assembly.json'))
    actual=UsdLux.DistantLight(stage.GetPrimAtPath('/World/scene_0/findingbench_lighting/unit_light'))
    assert actual.GetIntensityAttr().Get()==200.
    assert guard['entries']==1 and guard['active'] is False
    assert env._findingbench_appearance_evidence['lighting']=='applied'


def test_background_retains_visible_source_mesh_separate_from_collision_guide(usd_bundle,tmp_path):
    from pxr import Usd,UsdGeom,UsdPhysics
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    scene,solve,assets,bindings=usd_bundle
    output=tmp_path/'assembly'
    manifest=prepare_assembly(make_spec(),scene,solve,bindings,assets,output)
    backgrounds=[o for o in manifest['objects'] if o['role']=='static_background']
    assert backgrounds
    for item in backgrounds:
        stage=Usd.Stage.Open(str(output/item['og_config']['usd_path']))
        visual=stage.GetPrimAtPath('/World/base_link/geometry/mesh')
        collider=stage.GetPrimAtPath('/World/base_link/collisions/mesh')
        assert not visual.HasAPI(UsdPhysics.CollisionAPI) and collider.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(visual).GetPurposeAttr().Get()=='default'
        assert UsdGeom.Imageable(collider).GetPurposeAttr().Get()=='guide'
        for attribute in ('points','faceVertexCounts','faceVertexIndices','primvars:st'):
            assert visual.GetAttribute(attribute).Get()==collider.GetAttribute(attribute).Get()


@pytest.mark.parametrize('meshed_solver_name',[False,True])
def test_sibling_floor_and_wall_keep_room_identity_and_floor_sampling(usd_bundle,tmp_path,meshed_solver_name):
    import json
    from pxr import Usd,UsdGeom
    from rummagebench.authoring.tasks.infinigen import inspect_export,source_mesh_owner
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.infinigen_probe import assembly_floor_map
    scene,solve,assets,bindings=usd_bundle
    if meshed_solver_name:
        raw=json.loads(solve.read_text());raw['objs']['kitchen']['obj']='kitchen.meshed'
        solve.write_text(json.dumps(raw))
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
def test_actual_floor_component_uses_slab_top_not_bottom(usd_bundle,tmp_path):
    from pxr import Usd,UsdGeom
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.infinigen_probe import assembly_floor_map
    from test_infinigen_pipeline import make_spec
    scene,solve,assets,bindings=usd_bundle
    stage=Usd.Stage.Open(str(scene));floor=UsdGeom.Mesh.Define(stage,'/World/kitchen_floor')
    floor.CreatePointsAttr([(0,0,0),(5,0,0),(5,5,0),(0,5,0),(0,0,.1),(5,0,.1),(5,5,.1),(0,5,.1)])
    floor.CreateFaceVertexCountsAttr([4,4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3,4,5,6,7])
    stage.GetRootLayer().Save()
    folder=tmp_path/'slab_fixture_only';manifest=prepare_assembly(make_spec(),scene,solve,bindings,assets,folder)
    assert assembly_floor_map(manifest,folder/'assembly.json').floor_z==pytest.approx(.1)


def test_unmapped_whole_room_skirting_does_not_fill_navigation_interior(usd_bundle,tmp_path):
    """Real USD parsing of unit geometry, never an accepted house."""
    from pxr import Usd,UsdGeom
    from rummagebench.sim.omnigibson.infinigen_adapter import prepare_assembly
    from rummagebench.authoring.tasks.infinigen_probe import assembly_floor_map
    from test_infinigen_pipeline import make_spec
    scene,solve,assets,bindings=usd_bundle
    stage=Usd.Stage.Open(str(scene))
    floor=UsdGeom.Mesh.Define(stage,'/World/kitchen_floor')
    floor.CreatePointsAttr([(-4,-4,0),(4,-4,0),(4,4,0),(-4,4,0)])
    floor.CreateFaceVertexCountsAttr([4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3])
    trim=UsdGeom.Mesh.Define(stage,'/World/unmapped_skirting')
    trim.CreatePointsAttr([(-4,-4,.05),(4,-4,.05),(4,4,.05),(-4,4,.05),
                           (-4,-4,.2),(4,-4,.2),(4,4,.2),(-4,4,.2)])
    trim.CreateFaceVertexCountsAttr([4]*4)
    trim.CreateFaceVertexIndicesAttr([0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
    stage.GetRootLayer().Save()
    folder=tmp_path/'unit_skirting';m=prepare_assembly(make_spec(),scene,solve,bindings,assets,folder)
    item=next(o for o in m['objects'] if 'unmapped_skirting' in o.get('source_usd_prim',''))
    assert item['source_id'] is None and item['role']=='static_background'
    room=assembly_floor_map(m,folder/'assembly.json')
    assert room.free_points('kitchen_0',.4,.25)
    assert max(clearance for clearance,_ in room.clearest_points(3))>.5
