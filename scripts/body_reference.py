"""Generate robot-only orthographic projections from loaded visual meshes."""
def write_body_reference(env, output):
    import json, math
    from pathlib import Path
    import numpy as np
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    from rummagebench.sim.omnigibson.raw_instance import _array
    b=env._backend;robot=b._robot
    pos,quat=b.robot_pose();R=Rotation.from_quat(quat).as_matrix();pos=np.asarray(pos)
    polygons=[];skipped=[]
    for name,link in robot.links.items():
        for mesh_name,mesh in link.visual_meshes.items():
            try:
                points=np.asarray(_array(mesh.points),float)
                T=np.asarray(_array(mesh.scaled_transform),float)
                world=points@T[:3,:3].T+T[:3,3]
                local=(world-pos)@R
                if len(local)>=3:polygons.append(local)
            except (AttributeError,ValueError,RuntimeError) as error:
                skipped.append(dict(link=name,mesh=mesh_name,error=str(error)))
    if not polygons:raise RuntimeError('Cannot generate body reference: no visual mesh vertices')
    cp,_=b._panorama_camera.faces[0][0].get_position_orientation()
    camera=(np.asarray(_array(cp))-pos)@R
    half=env._session._base_half_extent
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for vertices in polygons:
        for ax,indices in zip(axes,[(1,0),(0,2)]):
            pts=vertices[:,indices].copy()
            if indices==(1,0):pts[:,0]*=-1
            try:
                pts=pts[ConvexHull(pts).vertices];ax.fill(pts[:,0],pts[:,1],color='#7d9ba8',alpha=.45,lw=.2)
            except Exception:pass
    axes[0].add_patch(Rectangle((-half,-half),half*2,half*2,fill=False,color='#c36e12',lw=2,label='Executor footprint'))
    axes[0].add_patch(Circle((0,0),math.sqrt(2)*half,fill=False,color='#c36e12',ls='--',label='Full-turn sweep (no margin)'))
    axes[0].scatter([-camera[1]],[camera[0]],color='#145ec4',label='Actual camera origin')
    axes[0].set(xlabel='Robot right (m)',ylabel='Robot forward (m)',title='Top view: loaded robot visual meshes')
    axes[0].legend(fontsize=8)
    axes[1].scatter([camera[0]],[camera[2]],color='#145ec4')
    axes[1].set(xlabel='Robot forward (m)',ylabel='Robot up (m)',title='Side view: loaded robot visual meshes')
    for ax in axes:ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Measured body reference — initial posture',fontsize=18)
    fig.text(.5,.025,f'Camera in base frame: {camera.round(3).tolist()} m | Executor: {2*half:.2f} × {2*half:.2f} m; obstacle margin 0.03 m\nOrthographic mesh projections, not a scene camera. No environment geometry or route is shown.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.1,1,.94]);out=Path(output)
    fig.savefig(out/'body_reference.png',dpi=140);plt.close(fig)
    data=dict(camera_in_base_m=camera.tolist(),executor_half_extent_m=half,obstacle_margin_m=.03,
              visual_mesh_count=len(polygons),skipped_meshes=skipped,posture='initial',
              projection='convex projection per visual mesh; not a collision or reachability certificate')
    (out/'body_reference.json').write_text(json.dumps(data,indent=2)+'\n')
