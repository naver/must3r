# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import must3r.tools.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, rgb, auto_cam_size


def viz_dataset(dataset, sampler=None):
    iterable = np.random.permutation(len(dataset)) if sampler is None else sampler
    for idx in iterable:
        views = dataset[idx]
        assert len(views) == dataset.num_views
        unique_views = set()
        for view_idx in range(dataset.num_views):
            v_name = view_name(views[view_idx])
            print(v_name)
            unique_views.add(v_name)
        print(f'{len(unique_views)} unique views')
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in range(dataset.num_views)]
        cam_size = max(auto_cam_size(poses), 0.001)
        memory_num_views = views[0]['memory_num_views']
        print(f'{memory_num_views} memory views')
        for view_idx in range(dataset.num_views):
            v_idx = view_idx / (dataset.num_views - 1)
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            if view_idx < memory_num_views:
                color = (0, 0, v_idx * 255)
            else:
                color = (v_idx * 255, (1 - v_idx) * 255, 0)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=color,
                           image=colors,
                           cam_size=cam_size)
        viz.show()
