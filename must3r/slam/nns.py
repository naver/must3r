# Copyright (C) 2025-present Naver Corporation. All rights reserved.
from scipy.spatial import KDTree
import numpy as np
from functools import partial

from must3r.slam.tools import get_quadrant_id, ravel3d, to_np


def get_searcher(method, isquadrant=False):
    if 'quadrant_x' in method and not isquadrant:
        out = QuandrantSearcher(method)
    elif "kdtree-scipy" in method:
        out = KDTree_scipy()
    elif method == 'none':
        out = None
    else:
        raise ValueError(f"Unknown searcher method {method}")

    return out


class Base_NN():
    """
    Base NN class, that can add observations to the search struct and query points from it
    IO is torch Tensors
    """

    def __init__(self, subsamp=None):
        self.subsamp = subsamp

    def add_pts(self, pts, **kw):  # add [N, 3] db pts
        # Add points to the existing search structure
        raise NotImplementedError("Overload this function for your needs")

    def query(self, pts, **kw):  # [N, 3] query pts
        # Query 3D points
        raise NotImplementedError("Overload this function for your needs")


class KDTree_scipy(Base_NN):
    """
    Simple KDTree from scipy
    """

    def __init__(self):
        super().__init__()
        self.all_points = []
        self.kdtree = None

    def add_pts(self, pts, **kw):
        if len(self.all_points) == 0:
            self.all_points = ravel3d(pts)
        else:
            self.all_points = np.concatenate([self.all_points, ravel3d(pts)])
        self.kdtree = KDTree(self.all_points)

    def query(self, pts, **kw):
        if self.kdtree is None:
            distances = np.full(pts.shape[0], np.inf)
        else:
            distances, indices = self.kdtree.query(ravel3d(pts), k=1, workers=4)
        return distances


class QuandrantSearcher(Base_NN):
    """ 
    Split the view direction space in quadrants to account for ray directions (e.g. visibility) while querying
    Simply one query struct per quadrant
    """

    def __init__(self, method):
        super().__init__()
        # split the rotation sphere into 2N**2 regular quadrants
        self.quadrant_divider = int(method.split('quadrant_x')[-1].split('-')[0])

        self.all_points = [[] for _ in range(2 * self.quadrant_divider**2)]
        self.search_structs = [get_searcher(method, isquadrant=True) for _ in range(2 * self.quadrant_divider**2)]
        self.get_quadrant_id = partial(get_quadrant_id, quadrant_divider=self.quadrant_divider)

    def add_pts(self, pts, cam_center, **kw):
        quadrant_id = self.get_quadrant_id(to_np(pts - cam_center[None]))
        for quad in np.unique(quadrant_id):
            idx = quadrant_id == quad
            self.search_structs[quad].add_pts(pts[idx])

    def query(self, pts, cam_center, **kw):
        quadrant_id = self.get_quadrant_id(to_np(pts - cam_center[None]))
        dists = np.zeros(pts.shape[0])
        for quad in np.unique(quadrant_id):
            idx = quadrant_id == quad
            dists[idx] = self.search_structs[quad].query(pts[idx])
        return dists
