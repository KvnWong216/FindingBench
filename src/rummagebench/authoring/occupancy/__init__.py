"""Occupancy-guided initialization (§17-§22).

Occupancy is a PROPOSAL/INITIALIZATION representation over conservative voxel
proxies. It is never the final physics validator: coarse overlap means
"needs refinement", and final certification always uses real collision
geometry inside the simulator (revision §7).
"""
from rummagebench.authoring.occupancy.grid import (
    ContainerMask,
    OccupancyGrid,
    SupportHeightGrid,
)
from rummagebench.authoring.occupancy.lift import lift_to_pose
from rummagebench.authoring.occupancy.packing import (
    pack_in_container,
    pack_on_support,
    propose_cover,
)
from rummagebench.authoring.occupancy.proxy import (
    cache_key,
    load_proxy,
    proxy_cache_path,
    save_proxy,
)
from rummagebench.authoring.occupancy.voxelize import (
    aabb_proxy,
    voxelize_dims,
)

__all__ = [
    "OccupancyGrid", "SupportHeightGrid", "ContainerMask", "lift_to_pose",
    "pack_on_support", "pack_in_container", "propose_cover", "aabb_proxy",
    "voxelize_dims", "cache_key", "proxy_cache_path", "load_proxy",
    "save_proxy",
]
