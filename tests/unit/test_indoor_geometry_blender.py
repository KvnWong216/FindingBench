"""Synthetic Blender rotation regression, CPU only; no production assets."""
import importlib.util
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from rummagebench.authoring.layout.indoor_geometry import native_layout_pose, rigid_rotation_update


@unittest.skipUnless(importlib.util.find_spec('bpy'), 'requires the installed CPU bpy environment')
class NativeBlenderRotation(unittest.TestCase):
    def test_repeated_euler_updates_keep_actual_blender_scale(self):
        import bpy
        from mathutils import Quaternion
        mesh=bpy.data.meshes.new('unit_only_mesh')
        mesh.from_pydata([[0,0,0],[.1,0,0],[0,.2,0]],[],[[0,1,2]])
        obj=bpy.data.objects.new('unit_only_rigid',mesh)
        bpy.context.collection.objects.link(obj)
        axis=np.array([1.,2.,3.]);angle=.031
        try:
            for _ in range(1000):
                bpy.context.view_layer.update()
                q=rigid_rotation_update(np.asarray(obj.matrix_world,float),axis,angle)
                obj.rotation_mode='XYZ'
                obj.rotation_euler=Quaternion([q[3],*q[:3]]).to_euler('XYZ')
                bpy.context.view_layer.update()
                native_layout_pose(np.asarray(obj.matrix_world,float))
                self.assertEqual(list(obj.scale),[1.,1.,1.])
            expected=Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle*1000).as_matrix()
            np.testing.assert_allclose(np.asarray(obj.matrix_world)[:3,:3],expected,atol=2e-5,rtol=0)
        finally:
            bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.meshes.remove(mesh)


if __name__=='__main__':unittest.main()
