# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import os.path as osp
import cv2
import numpy as np

from dust3r.utils.image import imread_cv2

from must3r.datasets.base.must3r_base_dataset import MUSt3RBaseDataset
from must3r.datasets.base.tuple_maker import select_tuple_from_360_scene

import must3r.tools.path_to_dust3r  # noqa
from dust3r.datasets.co3d import Co3d as DUSt3R_Co3d  # noqa


class Co3d(DUSt3R_Co3d, MUSt3RBaseDataset):
    def __init__(self, *args, num_views, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self.num_images_per_object = 100
        self.invalidate = {scene: [False for _ in range(self.num_images_per_object)] for scene in self.scene_list}

    def __len__(self):
        return len(self.scenes) * self.num_images_per_object

    def _load_view(self, view_idx, obj, instance, resolution, rng, mask_bg):
        impath = self._get_impath(obj, instance, view_idx)
        depthpath = self._get_depthpath(obj, instance, view_idx)

        # load camera params
        metadata_path = self._get_metadatapath(obj, instance, view_idx)
        input_metadata = np.load(metadata_path)
        camera_pose = input_metadata['camera_pose'].astype(np.float32)
        intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

        # load image and depth
        rgb_image = imread_cv2(impath)
        depthmap = self._read_depthmap(depthpath, input_metadata)

        if mask_bg:
            # load object mask
            maskpath = self._get_maskpath(obj, instance, view_idx)
            maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            maskmap = (maskmap / 255.0) > 0.1

            # update the depthmap with mask
            depthmap *= maskmap

        rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

        view = dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose,
            camera_intrinsics=intrinsics,
            dataset=self.dataset_label,
            label=osp.join(obj, instance),
            instance=osp.split(impath)[1],
        )
        return view

    def _get_views(self, idx, resolution, memory_num_views, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx = idx % self.num_images_per_object

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
        nimg_per_scene = min(len(image_pool), self.num_images_per_object)

        def is_valid_getter(view_idx):
            return view_idx < len(image_pool) and not self.invalidate[obj, instance][view_idx]

        def is_valid_check(view, view_idx):
            view_valid = (view['depthmap'] > 0.0).sum() > 0
            if not view_valid:
                # problem, invalidate image
                self.invalidate[obj, instance][view_idx] = True
            return view_valid

        def get_view(view_idx, rng): return self._load_view(image_pool[view_idx], obj, instance, resolution, rng,
                                                            mask_bg)
        views = select_tuple_from_360_scene(is_valid_getter, is_valid_check, get_view,
                                            nimg_per_scene, self.num_views, rng, im1_idx)
        return views
