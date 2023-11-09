import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
`p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return (hull.find_simplex(p)>=0)
