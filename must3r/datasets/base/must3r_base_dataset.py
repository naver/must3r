# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import torch

import must3r.tools.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import (BaseStereoViewDataset, is_good_type, transpose_to_landscape,
                                                           view_name)
from dust3r.datasets.base.easy_dataset import EasyDataset, CatDataset, MulDataset, ResizedDataset
from dust3r.datasets.base.batched_sampler import BatchedRandomSampler as DUSt3R_BatchedRandomSampler
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


class BatchedRandomSampler(DUSt3R_BatchedRandomSampler):

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        pool_size = self.pool_size if isinstance(self.pool_size, list) else [self.pool_size]
        idxs = []
        for pool_size in pool_size:
            n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
            if isinstance(pool_size, tuple):
                feat_idxs = rng.integers(*pool_size, size=n_batches)
            else:
                feat_idxs = rng.integers(pool_size, size=n_batches)
            feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
            feat_idxs = feat_idxs.ravel()[:self.total_size]
            idxs.append(feat_idxs)

        # put them together
        idxs = np.c_[sample_idxs, *idxs]  # shape = (total_size, n_feats)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size - 1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank * size_per_proc: (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)


class EasyDataset_MUSt3R(EasyDataset):
    def __add__(self, other):
        return CatDataset_MUSt3R([self, other])

    def __rmul__(self, factor):
        return MulDataset_MUSt3R(factor, self)

    def __rmatmul__(self, factor):
        return ResizedDataset_MUSt3R(factor, self)

    def make_sampler(self, batch_size, shuffle=True, world_size=1, rank=0, drop_last=True):
        if not (shuffle):
            raise NotImplementedError()  # cannot deal yet
        num_of_aspect_ratios = len(self._resolutions)
        min_memory_num_views = self.min_memory_num_views
        max_memory_num_views = self.max_memory_num_views
        return BatchedRandomSampler(self, batch_size, [num_of_aspect_ratios, (min_memory_num_views, max_memory_num_views + 1)], world_size=world_size, rank=rank, drop_last=drop_last)


class CatDataset_MUSt3R(CatDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.datasets[0].min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.datasets[0].max_memory_num_views

    def __getitem__(self, idx):
        other = None
        if isinstance(idx, tuple):
            other = idx[1:]
            idx = idx[0]

        if not (0 <= idx < len(self)):
            raise IndexError()

        db_idx = np.searchsorted(self._cum_sizes, idx, 'right')
        dataset = self.datasets[db_idx]
        new_idx = idx - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if other is not None:
            new_idx = (new_idx, *other)
        return dataset[new_idx]


class MulDataset_MUSt3R(MulDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.dataset.min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.dataset.max_memory_num_views

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self.dataset[idx[0] // self.multiplicator, *idx[1:]]
        else:
            return self.dataset[idx // self.multiplicator]


class ResizedDataset_MUSt3R(ResizedDataset, EasyDataset_MUSt3R):

    @property
    def min_memory_num_views(self):
        return self.dataset.min_memory_num_views

    @property
    def max_memory_num_views(self):
        return self.dataset.max_memory_num_views

    def __getitem__(self, idx):
        assert hasattr(self, '_idxs_mapping'), 'You need to call dataset.set_epoch() to use ResizedDataset.__getitem__()'
        if isinstance(idx, tuple):
            return self.dataset[self._idxs_mapping[idx[0]], *idx[1:]]
        else:
            return self.dataset[self._idxs_mapping[idx]]


class MUSt3RBaseDataset(BaseStereoViewDataset, EasyDataset_MUSt3R):
    def __init__(self, *args, num_views, min_memory_num_views, max_memory_num_views, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this
        self.num_views = num_views
        self.min_memory_num_views = min_memory_num_views
        self.max_memory_num_views = max_memory_num_views

    def _get_views(self, idx, resolution, memory_num_views, rng):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, memory_num_views = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            memory_num_views = self.num_views

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, memory_num_views, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']
            view['memory_num_views'] = memory_num_views
            view['is_metric_scale'] = self.is_metric_scale
            # Pixels for which depth is fundamentally undefined
            view['sky_mask'] = (view['depthmap'] < 0)

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views
