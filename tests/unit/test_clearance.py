import numpy as np
import pytest
from rummagebench.sim.omnigibson.clearance import warp_points


def test_wall_warp_fixes_outer_bounds_and_height():
    points=np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]],float)
    out=warp_points(points,0,[0,1,3,4],[0,.5,3.5,4])
    assert np.allclose(out[:,0],[0,.5,2,3.5,4])
    assert np.array_equal(out[:,1:],points[:,1:])
    assert np.array_equal(points[:,0],np.arange(5))


def test_invalid_warp_rejected():
    with pytest.raises(ValueError):warp_points([[1,2,3]],0,[0,1,2],[0,3,2])
    with pytest.raises(ValueError):warp_points([[1,2,3]],0,[0,1],[0,2])
