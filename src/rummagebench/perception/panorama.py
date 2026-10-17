"""Pure RGB/depth/ID reprojection for one rigid panoramic camera.

Camera coordinates are optical: +x right, +y down, +z forward. Public RGB
and evaluator-only range, IDs and rays always use the SAME nearest sample.
No interpolation of instance IDs or depths across object boundaries.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from rummagebench.perception.camera_geometry import default_intrinsics


def spherical_rays(width, height):
    lon = ((np.arange(width) + .5) / width - .5) * 2 * np.pi
    lat = (.5 - (np.arange(height) + .5) / height) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    return np.stack((np.cos(lat)*np.sin(lon), -np.sin(lat),
                     np.cos(lat)*np.cos(lon)), axis=-1)


def perspective_view(frame, yaw, pitch=0, width=224, height=224, hfov=90):
    if frame.meta.get('projection') != 'equirectangular':
        raise ValueError('VIEW requires a panoramic frame')
    if not np.isfinite([yaw, pitch]).all() or abs(yaw)>180 or abs(pitch)>90:
        raise ValueError('invalid view angles')
    K = default_intrinsics(width, height, hfov)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack(((u-K[0,2])/K[0,0], (v-K[1,2])/K[1,1], np.ones_like(u)), -1)
    R = (Rotation.from_euler('y', -yaw, degrees=True).as_matrix() @
         Rotation.from_euler('x', pitch, degrees=True).as_matrix())
    rays = rays @ R.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lon, lat = np.arctan2(rays[...,0], rays[...,2]), np.arcsin(-rays[...,1])
    x = np.floor((lon/(2*np.pi)+.5)*frame.image_width).astype(int) % frame.image_width
    y = np.clip(np.floor((.5-lat/np.pi)*frame.image_height).astype(int), 0, frame.image_height-1)
    meta = dict(frame.meta, projection='perspective', view_yaw=yaw, view_pitch=pitch)
    meta['pixel_rays'] = np.asarray(frame.meta['pixel_rays'])[y,x].copy()
    return replace(frame, rgb=frame.rgb[y,x].copy(), depth=frame.depth[y,x].copy(),
                   instance_segmentation=frame.instance_segmentation[y,x].copy(),
                   camera_intrinsics=K, image_width=width, image_height=height, meta=meta)


def compose_cube(faces, width, height):
    """Six fixed co-located pinhole frames -> one equirectangular frame.

    No world simulation steps may occur between face captures. Extrinsics
    refer to optical coordinates. Re-label face-local IDs into one namespace.
    """
    if len(faces) != 6:
        raise ValueError('six cube faces required')
    reference = faces[0]
    T = np.asarray(reference.camera_extrinsics)
    if not all(np.allclose(f.camera_extrinsics[:3,3], T[:3,3], atol=1e-6) for f in faces):
        raise ValueError('panoramic faces must share one optical centre')
    rays = spherical_rays(width, height)
    world_rays = rays @ T[:3,:3].T
    chosen = np.full((height,width), -1, dtype=int)
    best = np.full((height,width), -np.inf)
    projected = []
    for i,f in enumerate(faces):
        if not f.meta.get('visual_grounding_supported'):
            raise RuntimeError('incomplete panoramic renderer modalities')
        local = world_rays @ f.camera_extrinsics[:3,:3]
        z = local[...,2]
        K = f.camera_intrinsics
        with np.errstate(divide='ignore', invalid='ignore'):
            xf = K[0,0]*local[...,0]/z+K[0,2]
            yf = K[1,1]*local[...,1]/z+K[1,2]
        valid = (z>0)&(xf>=-.5)&(xf<f.image_width-.5)&(yf>=-.5)&(yf<f.image_height-.5)
        pick = valid & (z>best)
        chosen[pick], best[pick] = i, z[pick]
        projected.append((local, xf, yf))
    if (chosen<0).any():
        raise RuntimeError(f'panorama has {int((chosen<0).sum())} uncovered pixels; '
                           f'intrinsics={[f.camera_intrinsics.tolist() for f in faces]}')
    rgb = np.empty((height,width,3), np.uint8)
    depth = np.empty((height,width), float)
    ids = np.zeros((height,width), np.int64)
    sampled_rays = np.empty((height,width,3), float)
    labels, next_id = {}, 1
    for i,f in enumerate(faces):
        mask = chosen==i
        _,xf,yf = projected[i]
        x = np.clip(np.rint(xf[mask]).astype(int),0,f.image_width-1)
        y = np.clip(np.rint(yf[mask]).astype(int),0,f.image_height-1)
        rgb[mask] = f.rgb[y,x]
        K = f.camera_intrinsics
        ray = np.stack(((x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones(len(x))), -1)
        norm = np.linalg.norm(ray,axis=-1)
        depth[mask] = f.depth[y,x] * (norm if f.depth_convention=='z_depth' else 1)
        sampled_rays[mask] = (ray/norm[:,None]) @ f.camera_extrinsics[:3,:3].T @ T[:3,:3]
        values = f.instance_segmentation[y,x]
        remapped = np.zeros(len(x),np.int64)
        for old in np.unique(values):
            if old in (0, 0xFFFFFFFF): continue
            label = f.meta['instance_labels'].get(str(old))
            if label is None: continue
            remapped[values==old] = next_id
            labels[str(next_id)] = label
            next_id += 1
        ids[mask] = remapped
    return replace(reference, rgb=rgb, depth=depth, instance_segmentation=ids,
                   camera_intrinsics=np.eye(3), image_width=width, image_height=height,
                   depth_convention='euclidean_range', meta=dict(
                       projection='equirectangular', pixel_rays=sampled_rays,
                       instance_labels=labels, visual_grounding_supported=True,
                       bridge='segmentation', calibration_verified=False))
