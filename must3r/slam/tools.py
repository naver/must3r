# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np


def ravel3d(x): return x.view(-1, 3).cpu().numpy()
def to_np(x): return x.cpu().numpy()


def get_quadrant_id(rays, quadrant_divider=4, eps=1e-5):
    # Simply split the rotation sphere into 2N**2 regularly spaced quadrants
    # Input is ray directions in 3D
    # turn them into spherical coordinates and quantize the theta/phis to get a 2D coord
    # raveled into a single output index value
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(eps)

    # Spherical coordinates (r=1)
    thetas = np.arccos(rays[:, -1]) / np.pi  # acos(z)/pi in [0,1]
    phis = np.arctan2(rays[:, 1], rays[:, 0]) / np.pi  # atan(y,x)/pi in [-1,1]

    # Clip to prevent floating point errors
    thetas = thetas.clip(eps, 1 - eps)
    phis = phis.clip(-1 + eps, 1 - eps)

    # Quantize
    theta_idx = np.floor(thetas * quadrant_divider).astype(int)  # in [0, quadrant_divider]
    phis_idx = np.floor(phis * quadrant_divider).astype(int) + quadrant_divider  # in [0, 2*quadrant_divider]

    # turn the 2D quadrant coordinates into a 1D index
    quadrant_index = theta_idx + phis_idx * quadrant_divider

    return quadrant_index.astype(int)


