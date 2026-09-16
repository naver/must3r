# Copyright (C) 2025-present Naver Corporation. All rights reserved.
from scipy.spatial import KDTree
import numpy as np
from functools import partial
try:
    from pykdtree.kdtree import KDTree as pyKDTree
    has_pykdtree = True
except ImportError:
    has_pykdtree = False

from must3r.slam.tools import get_quadrant_id, ravel3d, to_np


def get_searcher(method, isquadrant=False):
    if 'quadrant_x' in method and not isquadrant:
        out = QuandrantSearcher(method)
    elif "kdtree-scipy" in method:
        out = KDTree_scipy()
    elif "pykdtree" in method:
        out = KDTree_pykdtree()
    elif method == 'none':
        out = None
    else:
        raise ValueError(f"Unknown searcher method {method}")

    return out


class Base_NN():
    """
    Base NN class, that can add observations to the search struct and query points from it
    IO is torch Tensors or numpy arrays
    """

    def __init__(self, subsamp=None):
        self.subsamp = subsamp

    def add_pts(self, pts, **kw):  # add [N, 3] db pts
        # Add points to the existing search structure
        raise NotImplementedError("Overload this function for your needs")

    def query(self, pts, **kw):  # [N, 3] query pts
        # Query 3D points
        raise NotImplementedError("Overload this function for your needs")


class _LazyKDTree(Base_NN):
    """
    KDTree with lazy rebuild: the tree is only (re)built on the first query after new points were added
    """

    def __init__(self):
        super().__init__()
        self._points_list = []
        self.kdtree = None
        self._dirty = False

    def _build_tree(self, all_points):
        raise NotImplementedError("Overload this function for your needs")

    def _query_tree(self, pts):
        raise NotImplementedError("Overload this function for your needs")

    def __getstate__(self):
        # the tree itself is not picklable (pykdtree) and is cheap to rebuild from the points
        state = self.__dict__.copy()
        state['kdtree'] = None
        state['_dirty'] = len(self._points_list) > 0
        return state

    def add_pts(self, pts, **kw):
        self._points_list.append(ravel3d(pts))
        self._dirty = True

    def _rebuild(self):
        if self._dirty:
            all_points = np.concatenate(self._points_list) if len(self._points_list) > 1 else self._points_list[0]
            self.kdtree = self._build_tree(all_points)
            self._dirty = False

    def query(self, pts, **kw):
        self._rebuild()
        pts = ravel3d(pts)
        if self.kdtree is None:
            distances = np.full(pts.shape[0], np.inf)
        else:
            distances, _ = self._query_tree(pts)
        return distances


class KDTree_scipy(_LazyKDTree):
    """
    Simple KDTree from scipy
    """

    def _build_tree(self, all_points):
        return KDTree(all_points)

    def _query_tree(self, pts):
        return self.kdtree.query(pts, k=1, workers=4)


class KDTree_pykdtree(_LazyKDTree):
    """
    Simple KDTree from pykdtree (supports float32, faster than scipy)
    """

    def __init__(self):
        assert has_pykdtree, "Needs: pip install pykdtree"
        super().__init__()

    def _build_tree(self, all_points):
        return pyKDTree(all_points)

    def _query_tree(self, pts):
        return self.kdtree.query(pts, k=1)


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
