# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import os.path as osp
import cv2
import numpy as np

from dust3r.utils.image import imread_cv2

from must3r.datasets.base.must3r_base_dataset import MUSt3RBaseDataset
from must3r.datasets.base.tuple_maker import select_tuple_from_pairs

import must3r.tools.path_to_dust3r  # noqa
from dust3r.datasets.scannetpp import ScanNetpp as DUSt3R_ScanNetpp  # noqa


class ScanNetpp(DUSt3R_ScanNetpp, MUSt3RBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, split='train', **kwargs)
        self.is_metric_scale = True
        self.pairs_per_image = [set() for _ in range(len(self.images))]
        for idx1, idx2 in self.pairs:
            self.pairs_per_image[idx1].add(idx2)
            self.pairs_per_image[idx2].add(idx1)

    def _load_view(self, idx, view_idx, resolution, rng):
        scene_id = self.sceneids[view_idx]
        scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

        intrinsics = self.intrinsics[view_idx]
        camera_pose = self.trajectories[view_idx]
        basename = self.images[view_idx]

        # Load RGB image
        rgb_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
        # Load depthmap
        depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000
        depthmap[~np.isfinite(depthmap)] = 0  # invalid

        rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsics.astype(np.float32),
            dataset='ScanNet++',
            label=self.scenes[scene_id] + '_' + basename,
            instance=f'{str(idx)}_{str(view_idx)}',
        )

    def _get_views(self, idx, resolution, memory_num_views, rng):
        idx1, idx2 = self.pairs[idx]
        def get_pairs(view_idx): return self.pairs_per_image[view_idx]
        def get_view(view_idx, rng): return self._load_view(idx, view_idx, resolution, rng)
        views = select_tuple_from_pairs(get_pairs, get_view, self.num_views, memory_num_views, rng, idx1, idx2)
        return views
