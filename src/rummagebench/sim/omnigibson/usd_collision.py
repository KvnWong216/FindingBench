"""USD collision geometry export: world bodies -> coal collision geometries.

Fidelity chain (recorded per body, never silent):

    existing physics collider (mesh BVH / exact primitive)
    -> world-aligned AABB primitive

The robot is excluded by the caller: its collision geometry comes from the
exported URDF (URDF <collision>), never from visual meshes.

Collider transforms carry SCALE: BEHAVIOR colliders are unit meshes /
primitives scaled by their xform (e.g. a cabinet panel = unit cube x
[0.017, 0.46, 0.04]). The world matrix (USD row-vector convention,
p_world = p_local @ L + t) is polar-decomposed into L = P @ R: the
symmetric stretch P (scale + shear, constant for a collider because parent
motion is rigid) is BAKED into the geometry, and the body pose carries only
the rigid part (R, t).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rummagebench.feasibility.fcl_compat import collision_backend

from rummagebench.sim.base import WorldCollisionObject

logger = logging.getLogger(__name__)

_MAX_TRIANGLE_BUDGET = 60000


def _coal():
    return collision_backend()


def iter_prims(root_prim):
    yield root_prim
    for child in root_prim.GetAllChildren():
        yield from iter_prims(child)


def collect_entity_collision(obj, geom_cache: dict[str, Any]) -> list[WorldCollisionObject]:
    """All collision bodies of one scene object, with current world poses."""
    from pxr import Usd, UsdGeom, UsdPhysics

    coal = _coal()
    prim = obj.prim
    category = getattr(obj, "category", "")

    bodies: list[WorldCollisionObject] = []
    # link decomposition: rigid bodies when present, else the object root
    link_prims = [p for p in iter_prims(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not link_prims:
        link_prims = [prim]

    def usd_matrix(p):
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array([[float(m[i][j]) for j in range(4)] for i in range(4)])

    for link_prim in link_prims:
        link_name = link_prim.GetName()
        is_root = link_prim == prim
        # collider-to-link transforms are static USD data, but the link's
        # WORLD pose lives in the physics engine: USD xforms of simulated
        # bodies are stale (verified: a dropped object kept its spawn pose,
        # opened drawers their closed pose)
        M_link_usd = usd_matrix(link_prim)
        M_link = physics_link_matrix(obj, link_name, M_link_usd)
        usd_to_physics = np.linalg.inv(M_link_usd) @ M_link
        colliders = [
            p for p in iter_prims(link_prim)
            if p.HasAPI(UsdPhysics.CollisionAPI) and p != link_prim
        ] if link_prim != prim else [
            p for p in iter_prims(link_prim) if p.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not colliders:
            continue
        for collider in colliders:
            enabled = collider.GetAttribute("physics:collisionEnabled")
            if enabled and enabled.Get() is False:
                continue
            path = str(collider.GetPath())
            M = usd_matrix(collider) @ usd_to_physics
            stretch, rot, pos = decompose_world_transform(M)
            # the stretch is constant per collider (rigid parent motion); key
            # on it anyway so a rescaled object can never reuse stale geometry
            key = (path, tuple(np.round(stretch, 6).ravel()))
            cached = geom_cache.get(key)
            if cached is None:
                cached = _build_geometry(collider, stretch, coal)
                if cached[0] is None:
                    continue
                geom_cache[key] = cached
            geometry, approximation, local_pts = cached
            quat = _quat_from_matrix(rot)
            aabb = None
            if local_pts is not None and len(local_pts):
                world = local_pts @ rot.T + pos
                aabb = ([float(v) for v in world.min(0)], [float(v) for v in world.max(0)])

            bodies.append(
                WorldCollisionObject(
                    entity=obj.name,
                    link=None if is_root and len(link_prims) == 1 else link_name,
                    geometry=geometry,
                    pose=_pose_from(pos, quat),
                    category=category,
                    approximation=approximation or "aabb_primitive",
                    aabb=aabb,
                    source_prim=path,
                    physics_approximation=(str(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() or "none")
                                           if collider.HasAPI(UsdPhysics.MeshCollisionAPI) else "none"),
                )
            )
    return bodies


def physics_link_matrix(obj, link_name: str, M_link_usd: np.ndarray) -> np.ndarray:
    """World matrix of a link with its rigid pose from the physics engine
    and its stretch (object scale) from USD. Falls back to USD when the
    object exposes no such link (static, non-simulated prims)."""
    link = (getattr(obj, "links", None) or {}).get(link_name)
    if link is None:
        return M_link_usd
    try:
        pos, quat = link.get_position_orientation()
    except Exception:
        return M_link_usd
    pos = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos, dtype=float)
    quat = np.asarray(quat.detach().cpu() if hasattr(quat, "detach") else quat, dtype=float)
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        return M_link_usd
    return compose_link_matrix(M_link_usd, pos, quat)


def compose_link_matrix(M_link_usd: np.ndarray, pos, quat_xyzw) -> np.ndarray:
    """Replace the rigid part of a (possibly scaled) USD link matrix by a
    physics pose, keeping the link's stretch: [[P @ R_row, 0], [t, 1]]."""
    from scipy.spatial.transform import Rotation

    P, _, _ = decompose_world_transform(M_link_usd)
    M = np.eye(4)
    M[:3, :3] = P @ Rotation.from_quat(quat_xyzw).as_matrix().T
    M[3, :3] = pos
    return M


def _pose_from(pos: np.ndarray, quat_xyzw: list[float]):
    from rummagebench.feasibility.ik_solver import Pose

    return Pose.from_lists([float(v) for v in pos], quat_xyzw)


# ---------------------------------------------------------------- pure math
# (numpy only; unit-tested without the simulator)

def decompose_world_transform(M: np.ndarray):
    """USD row-vector world matrix -> (P, R, t) with p_world = (p @ P) @ R.T + t.

    Polar decomposition of the linear part L (p_world = p_local @ L + t):
    L = P @ R_row, P symmetric positive (scale + shear, local frame),
    R_row orthonormal. Returned R is the column-convention rotation
    (R = R_row.T) so callers rotate with ``x @ R.T`` / ``R @ x``.
    """
    M = np.asarray(M, dtype=float)
    L, t = M[:3, :3], M[3, :3].copy()
    U, S, Vt = np.linalg.svd(L)
    R_row = U @ Vt
    if np.linalg.det(R_row) < 0:  # mirrored collider: keep a proper rotation
        U[:, -1] *= -1
        S[-1] *= -1
        R_row = U @ Vt
    P = U @ np.diag(S) @ U.T
    return P, R_row.T, t


def bake_points(points: np.ndarray, stretch: np.ndarray) -> np.ndarray:
    """Local collider points with the collider's stretch applied."""
    return np.asarray(points, dtype=float) @ stretch


def box_corners(half: np.ndarray) -> np.ndarray:
    hx, hy, hz = (float(v) for v in half)
    return np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)])


_BOX_TRIS = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def _is_diagonal(P: np.ndarray, tol: float = 1e-6) -> bool:
    return bool(np.all(np.abs(P - np.diag(np.diag(P))) <= tol * max(1.0, np.abs(P).max())))


def obb_in_parent(local_pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> dict:
    """Oriented box bounding (stretched) collider points, in the parent
    frame where p_parent = R @ p + t: {center, half, rpy} (URDF fixed-axis
    roll-pitch-yaw). Exact for box colliders (unit cube x scale)."""
    from scipy.spatial.transform import Rotation

    pts = np.asarray(local_pts, dtype=float)
    lo, hi = pts.min(0), pts.max(0)
    center = R @ ((lo + hi) / 2.0) + np.asarray(t, dtype=float)
    return {"center": [float(v) for v in center],
            "half": [float(v) for v in (hi - lo) / 2.0],
            "rpy": [float(v) for v in Rotation.from_matrix(R).as_euler("xyz")]}


def _quat_from_matrix(R: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return [float(v) for v in Rotation.from_matrix(R).as_quat()]


# ------------------------------------------------------------ USD -> coal

def _bvh(points: np.ndarray, tris, coal):
    # Coal 3 renamed the double-precision vector wrapper from Vec3f to Vec3s.
    vector_type = getattr(coal, "StdVec_Vec3s", None) or coal.StdVec_Vec3f
    verts_std = vector_type()
    for p in points:
        verts_std.append(np.array([float(p[0]), float(p[1]), float(p[2])]))
    tris_std = coal.StdVec_Triangle()
    for a, b, c in tris:
        tris_std.append(coal.Triangle(int(a), int(b), int(c)))
    bvh = coal.BVHModelOBB()
    bvh.beginModel(len(verts_std), len(tris_std))
    bvh.addSubModel(verts_std, tris_std)
    bvh.endModel()
    return bvh


def primitive_geometry(kind: str, params: dict, stretch: np.ndarray, coal):
    """coal geometry for a scaled USD primitive, its approximation level and
    its local (stretched) bounding points. Exact whenever the stretched
    shape is still the same primitive; otherwise a stated approximation."""
    P = np.asarray(stretch, dtype=float)
    diag = np.diag(P)
    if kind == "Cube":
        half = np.full(3, float(params.get("size", 1.0)) / 2.0)
        corners = bake_points(box_corners(half), P)
        if _is_diagonal(P):
            h = np.abs(half * diag)
            # coal.Box takes FULL side lengths (halfSide = x / 2)
            return (coal.Box(*(float(2.0 * v) for v in h)), "physics_collider_box",
                    box_corners(h))
        return _bvh(corners, _BOX_TRIS, coal), "physics_collider_box_sheared", corners
    if kind == "Sphere":
        r = float(params.get("radius", 0.1))
        ext = np.abs(r * diag)
        if _is_diagonal(P) and np.allclose(ext, ext[0], rtol=1e-6):
            return coal.Sphere(float(ext[0])), "physics_collider_sphere", box_corners(ext)
        if _is_diagonal(P) and hasattr(coal, "Ellipsoid"):
            return (coal.Ellipsoid(*(float(v) for v in ext)), "physics_collider_ellipsoid",
                    box_corners(ext))
        corners = bake_points(box_corners(np.full(3, r)), P)
        return _bvh(corners, _BOX_TRIS, coal), "scaled_sphere_bounding_box", corners
    if kind == "Cylinder":
        r = float(params.get("radius", 0.1))
        h = float(params.get("height", 0.2))
        axis = str(params.get("axis", "Z")).upper()
        if axis == "Z" and _is_diagonal(P) and np.isclose(abs(diag[0]), abs(diag[1]), rtol=1e-6):
            rr, hh = abs(r * diag[0]), abs(h * diag[2])
            return (coal.Cylinder(float(rr), float(hh)), "physics_collider_cylinder",
                    box_corners(np.array([rr, rr, hh / 2.0])))
        half = {"X": [h / 2, r, r], "Y": [r, h / 2, r]}.get(axis, [r, r, h / 2])
        corners = bake_points(box_corners(np.array(half)), P)
        return _bvh(corners, _BOX_TRIS, coal), "scaled_cylinder_bounding_box", corners
    raise ValueError(f"unsupported primitive {kind}")


def _build_geometry(collider, stretch, coal):
    """(coal geometry, approximation level, local stretched points) for one
    USD collider, or (None, None, None)."""
    from pxr import UsdGeom

    tname = collider.GetTypeName()
    try:
        if tname == "Cube":
            size = float(UsdGeom.Cube(collider).GetSizeAttr().Get() or 1.0)
            return primitive_geometry("Cube", {"size": size}, stretch, coal)
        if tname == "Sphere":
            r = float(UsdGeom.Sphere(collider).GetRadiusAttr().Get() or 0.1)
            return primitive_geometry("Sphere", {"radius": r}, stretch, coal)
        if tname == "Cylinder":
            cyl = UsdGeom.Cylinder(collider)
            return primitive_geometry("Cylinder", {
                "radius": float(cyl.GetRadiusAttr().Get() or 0.1),
                "height": float(cyl.GetHeightAttr().Get() or 0.2),
                "axis": str(cyl.GetAxisAttr().Get() or "Z")}, stretch, coal)
        if tname == "Mesh":
            return _mesh_geometry(UsdGeom.Mesh(collider), stretch, coal)
    except Exception as e:
        logger.warning("exact collider build failed for %s: %s", collider.GetPath(), e)
    # fallback: stretched local bounding box of the collider
    try:
        from pxr import Usd

        rng = UsdGeom.Boundable(collider).ComputeLocalBound(
            Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
        lo, hi = np.array(rng.min, dtype=float), np.array(rng.max, dtype=float)
        corners = bake_points(box_corners((hi - lo) / 2.0) + (lo + hi) / 2.0, stretch)
        return _bvh(corners, _BOX_TRIS, coal), "local_bound_box", corners
    except Exception as e:
        logger.warning("bounding-box fallback failed for %s: %s", collider.GetPath(), e)
        return None, None, None


def mesh_triangles(counts, idx) -> list[tuple[int, int, int]]:
    tris = []
    i = 0
    for n in counts:
        for k in range(1, n - 1):
            tris.append((idx[i], idx[i + k], idx[i + k + 1]))
        i += n
    return tris


def mesh_navigation_aabbs(points, counts, indices, matrix, floor_z):
    """Conservative bounds of real wall faces clipped to the base z band.

    A merged room's outer AABB fills its interior. Triangle envelopes retain
    every intersecting face without declaring the empty room an obstacle.
    Only convex planar faces are supported; uncertain tessellation rejects.
    Output z is relative to the verified floor, as the public footprint band.
    """
    points,matrix=np.asarray(points,float),np.asarray(matrix,float)
    if (points.ndim!=2 or points.shape[1]!=3 or not len(points) or matrix.shape!=(4,4)
            or not np.isfinite(points).all() or not np.isfinite(matrix).all() or not np.isfinite(floor_z)
            or not np.allclose(matrix[:,3],[0.,0.,0.,1.],atol=1e-10,rtol=0)
            or sum(counts)!=len(indices) or any(type(n) is not int or n<3 for n in counts)
            or any(type(i) is not int or not 0<=i<len(points) for i in indices)):
        raise ValueError('NAV_ARCHITECTURE_MESH_INVALID')
    points=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
    if all(n==3 for n in counts):
        # Triangles are already convex and planar, including collapsed ones.
        # Clip all their vertices/edge intersections in bulk. Real exported
        # trim meshes contain many thousands of faces and are queried by MOVE.
        triangles=points[np.asarray(indices,dtype=int).reshape(-1,3)]
        lower,upper=floor_z+.05,floor_z+.5
        triangles=triangles[(triangles[:,:,2].max(1)>=lower)&(triangles[:,:,2].min(1)<=upper)]
        if not len(triangles):return []
        vertices=[triangles];masks=[(triangles[:,:,2]>=lower)&(triangles[:,:,2]<=upper)]
        end=np.roll(triangles,-1,axis=1);dz=end[:,:,2]-triangles[:,:,2]
        for height in (lower,upper):
            fraction=np.divide(height-triangles[:,:,2],dz,out=np.zeros_like(dz),where=dz!=0)
            vertices.append(triangles+(end-triangles)*fraction[:,:,None])
            masks.append((dz!=0)&(fraction>=0)&(fraction<=1))
        vertices=np.concatenate(vertices,axis=1);mask=np.concatenate(masks,axis=1)
        if not mask.any(1).all():raise ValueError('NAV_TRIANGLE_CLIPPING_UNRESOLVED')
        lo=np.where(mask[:,:,None],vertices,np.inf).min(1)
        hi=np.where(mask[:,:,None],vertices,-np.inf).max(1)
        lo[:,2]-=floor_z;hi[:,2]-=floor_z
        return list(zip(lo.tolist(),hi.tolist()))
    boxes=[];offset=0
    def clip(poly,height,keep_above):
        out=[]
        for a,b in zip(poly,[*poly[1:],poly[0]]):
            inside_a=a[2]>=height if keep_above else a[2]<=height
            inside_b=b[2]>=height if keep_above else b[2]<=height
            if inside_a:out.append(a)
            if inside_a!=inside_b:out.append(a+(b-a)*((height-a[2])/(b[2]-a[2])))
        return out
    for count in counts:
        face=points[indices[offset:offset+count]];offset+=count
        if count==3 and (face[:,2].max()<floor_z+.05 or face[:,2].min()>floor_z+.5):
            continue
        normals=np.cross(face[1:-1]-face[0],face[2:]-face[0])
        lengths=np.linalg.norm(normals,axis=1)
        if not (lengths>1e-10).any():
            # Real exported boolean wall meshes include collapsed triangles.
            # Retain a conservative clipped face envelope, including point
            # and line geometry, instead of dropping its collision evidence.
            lo,hi=face.min(axis=0),face.max(axis=0)
            lo[2]=max(lo[2],floor_z+.05);hi[2]=min(hi[2],floor_z+.5)
            if lo[2]<=hi[2]:
                lo[2]-=floor_z;hi[2]-=floor_z
                boxes.append((lo.tolist(),hi.tolist()))
            continue
        normal=normals[np.argmax(lengths)]/lengths.max()
        edge=np.roll(face,-1,axis=0)-face
        turns=np.cross(edge,np.roll(edge,-1,axis=0))@normal
        if ((face-face[0])@normal).max()>1e-5 or abs(((face-face[0])@normal).min())>1e-5 or (turns< -1e-8).any():
            raise ValueError('NAV_ARCHITECTURE_FACE_TESSELLATION_UNSUPPORTED')
        for i in range(1,count-1):
            poly=clip(list(face[[0,i,i+1]]),floor_z+.05,True)
            if poly:poly=clip(poly,floor_z+.5,False)
            if poly:
                local=np.asarray(poly);local[:,2]-=floor_z
                boxes.append((local.min(0).tolist(),local.max(0).tolist()))
    return boxes


def collider_navigation_aabbs(obj):
    """Conservative bounds per enabled collider, using live physics poses.

    A single dataset object can contain disconnected paving / wall pieces.
    Its whole-object AABB fills the empty rooms between those pieces.
    Keep each actual collider separate without dropping categories or faces.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    boxes = []
    for prim in iter_prims(obj.prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        body = prim
        while body != obj.prim and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        base = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0.), float)
        correction = np.linalg.inv(base) @ physics_link_matrix(obj, body.GetName(), base)
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.), float) @ correction
        else:
            bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = np.asarray(bound.min), np.asarray(bound.max)
            points = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            matrix = correction
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError('NATIVE_NAVIGATION_COLLIDER_GEOMETRY_MISSING')
        world = points @ matrix[:3,:3] + matrix[3,:3]
        if not np.isfinite(world).all():
            raise ValueError('NATIVE_NAVIGATION_COLLIDER_NONFINITE')
        boxes.append((world.min(0).tolist(), world.max(0).tolist()))
    return boxes


def static_architecture_navigation_aabbs(obj, floor_z):
    """Use enabled, kinematic triangle-mesh colliders and their live frames.

    Includes unmapped source geometry such as whole-house skirting boards;
    their outer AABB spans empty rooms just like a joined wall mesh does.
    No factory/category inference or omitted collision faces is needed.
    """
    from pxr import UsdGeom,UsdPhysics
    boxes=[];found=False
    for prim in iter_prims(obj.prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:
            continue
        found=True
        approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if not prim.IsA(UsdGeom.Mesh) or approximation not in (None,'none'):
            raise ValueError('NAV_ARCHITECTURE_REQUIRES_ACTUAL_TRIANGLE_COLLIDER')
        body=prim
        while body!=obj.prim and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if body.HasAPI(UsdPhysics.RigidBodyAPI) and body.GetAttribute('physics:kinematicEnabled').Get() is not True:
            raise ValueError('NAV_ARCHITECTURE_DYNAMIC_COLLIDER_UNSUPPORTED')
        base=np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0.),float)
        actual=physics_link_matrix(obj,body.GetName(),base)
        world=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.),float)@np.linalg.inv(base)@actual
        mesh=UsdGeom.Mesh(prim)
        boxes.extend(mesh_navigation_aabbs(mesh.GetPointsAttr().Get(),list(mesh.GetFaceVertexCountsAttr().Get()),
                                          list(mesh.GetFaceVertexIndicesAttr().Get()),world,floor_z))
    if not found:raise ValueError('NAV_ARCHITECTURE_COLLIDERS_UNAVAILABLE')
    return boxes


def mesh_geometry_from_points(pts: np.ndarray, tris, physx_approximation: str, coal):
    """coal geometry for a (stretched) collider mesh.

    PhysX simulates a ``convexHull`` collider as the SOLID hull of its
    points: exported as a coal Convex (a body fully inside collides). Every
    other approximation (convexDecomposition of concave parts such as an
    open drawer box, triangle meshes, ...) stays an exact surface BVH — a
    single hull would fill the drawer and make its contents unreachable.
    """
    bvh = _bvh(pts, tris, coal)
    if physx_approximation == "convexHull":
        try:
            bvh.buildConvexHull(True, "Qt")
            if bvh.convex is not None:
                return bvh.convex, "physics_convex_hull"
        except Exception as e:  # degenerate (flat) hull: keep the surface
            logger.debug("convex hull build failed: %s", e)
    return bvh, "physics_mesh_bvh"


def _mesh_geometry(mesh, stretch, coal):
    """Exact collider mesh (existing physics collision), stretched, within a
    triangle budget; above it the caller's fallback applies."""
    from pxr import UsdPhysics

    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    idx = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    points = mesh.GetPointsAttr().Get()
    if not counts or not idx or points is None:
        return None, None, None
    tris = mesh_triangles(counts, idx)
    if len(tris) > _MAX_TRIANGLE_BUDGET:
        return None, None, None
    pts = bake_points(np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points]),
                      stretch)
    approx = "none"
    prim = mesh.GetPrim()
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        approx = str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
    geometry, level = mesh_geometry_from_points(pts, tris, approx, coal)
    return geometry, level, pts


def compute_link_aabb(link_prim):
    """World-frame AABB (lo, hi) of a link's collider bounds.

    Accepts either a pxr Usd.Prim or an OmniGibson prim wrapper (.prim)."""
    from pxr import Usd, UsdGeom

    prim = getattr(link_prim, "prim", link_prim)

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    try:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return ([float(v) for v in rng.min], [float(v) for v in rng.max])
    except Exception as e:
        logger.warning("link aabb failed for %s: %s", getattr(prim, "GetPath", lambda: "?")(), e)
        return None
