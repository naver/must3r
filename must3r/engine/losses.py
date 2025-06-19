# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from must3r.tools.geometry import apply_log_to_norm, normalize_pointcloud
import must3r.tools.path_to_dust3r  # noqa
from dust3r.utils.geometry import geotrf
from dust3r.losses import Criterion, L21, MultiLoss, Sum  # noqa


class Regr3D (Criterion, MultiLoss):
    def __init__(self, criterion, norm_mode='?avg_dis', sky_loss_value=2, loss_in_log=False):
        super().__init__(criterion)
        self.loss_in_log = loss_in_log
        if norm_mode.startswith('?'):
            # use the same scale factor as ground-truth for predictions in metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.sky_loss_value = sky_loss_value

    def get_all_pts3d(self, gt, pred, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        device = pred['pts3d'].device

        gt_c2w = [b['camera_pose'] for b in gt]
        gt_c2w = torch.stack(gt_c2w, dim=1).to(device)  # B, nimgs, 4, 4
        gt_w2c = torch.linalg.inv(gt_c2w)

        in_camera0 = gt_w2c[:, 0]

        gt_pts3d = [b['pts3d'] for b in gt]
        gt_pts3d = torch.stack(gt_pts3d, dim=1).to(device)  # B, nimgs, H, W, 3

        gt_pts3d_local = geotrf(gt_w2c, gt_pts3d)  # B, nimgs, H, W, 3
        gt_pts = geotrf(in_camera0, gt_pts3d)  # B, nimgs, H, W, 3

        valid = [b['valid_mask'] for b in gt]
        valid = torch.stack(valid, dim=1).to(device).clone()

        is_metric_scale = gt[0]['is_metric_scale'].to(device).clone()

        sky_mask = [b['sky_mask'] for b in gt]
        sky_mask = torch.stack(sky_mask, dim=1).to(device).clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis_g = gt_pts.norm(dim=-1)  # (B, nimgs, H, W)
            dis_l = gt_pts3d_local.norm(dim=-1)  # (B, nimgs, H, W)
            valid_g = valid & (dis_g <= dist_clip)
            valid_l = valid & (dis_l <= dist_clip)
        else:
            valid_g = valid
            valid_l = valid

        pr_pts = pred['pts3d'].clone()
        if 'pts3d_local' in pred:
            pr_pts_local = pred['pts3d_local'].clone()
        else:
            pr_pts_local = None

        if not self.norm_all:
            mask = ~is_metric_scale
        else:
            mask = torch.ones_like(is_metric_scale)

        # normalize 3d points
        if self.norm_mode and mask.any():
            pr_pts[mask], norm_factor_pred = normalize_pointcloud(pr_pts[mask], None, self.norm_mode, valid[mask], None,
                                                                  ret_factor=True)
            if pr_pts_local is not None:
                pr_pts_local[mask] = pr_pts_local[mask] / norm_factor_pred

        if self.norm_mode:
            gt_pts, norm_factor = normalize_pointcloud(gt_pts, None, self.norm_mode, valid, None, ret_factor=True)
            gt_pts3d_local = gt_pts3d_local / norm_factor
            pr_pts[~mask] = pr_pts[~mask] / norm_factor[~mask]
            if pr_pts_local is not None:
                pr_pts_local[~mask] = pr_pts_local[~mask] / norm_factor[~mask]

        # return sky segmentation, making sure they don't include any labelled 3d points
        sky_g = sky_mask & (~valid_g)
        sky_l = sky_mask & (~valid_l)
        return gt_pts, gt_pts3d_local, pr_pts, pr_pts_local, valid_g, valid_l, sky_g, sky_l, {}

    def compute_loss(self, gt, pred, **kw):
        gt_pts, gt_pts3d_local, pred_pts, pred_pts_local, mask_g, mask_l, sky_g, sky_l, monitoring = \
            self.get_all_pts3d(gt, pred, **kw)

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # add the sky pixel as "valid" pixels...
            mask_g = mask_g | sky_g
            mask_l = mask_l | sky_l

        # loss on pts3d global
        gt_pts = gt_pts[mask_g]
        if self.loss_in_log:
            gt_pts = apply_log_to_norm(gt_pts, dim=-1)
            pred_pts = apply_log_to_norm(pred_pts, dim=-1)
        pred_pts_m = pred_pts[mask_g]

        l1 = self.criterion(pred_pts_m, gt_pts)

        # loss on pts3d local
        if pred_pts_local is not None:
            pred_pts_local = pred_pts_local[mask_l]
            gt_pts3d_local = gt_pts3d_local[mask_l]
            if self.loss_in_log and self.loss_in_log != 'before':
                gt_pts3d_local = apply_log_to_norm(gt_pts3d_local, dim=-1)
                pred_pts_local = apply_log_to_norm(pred_pts_local, dim=-1)
            l2 = self.criterion(pred_pts_local, gt_pts3d_local)
        else:
            l2 = None

        if self.sky_loss_value > 0:
            assert self.criterion.reduction == 'none', 'sky_loss_value should be 0 if no conf loss'
            # ... but force the loss to be high there
            l1 = torch.where(sky_g[mask_g], self.sky_loss_value, l1)
            if l2 is not None:
                l2 = torch.where(sky_l[mask_l], self.sky_loss_value, l2)

        self_name = type(self).__name__
        details = {self_name + '_pts3d': float(l1.mean())}
        if l2 is not None:
            details[self_name + '_pts3d_local'] = float(l2.mean())
        return Sum((l1, mask_g), (l2, mask_l)), (details | monitoring)


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: low impact parameter?
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt, pred, **kw):
        # compute per-pixel loss
        ((loss_g, msk_g), (loss_l, msk_l)), details = self.pixel_loss(gt, pred, **kw)

        # weight by confidence
        if 'conf' not in pred:
            # not an actual conf loss, so do nothing
            conf_loss_g = loss_g.mean() if loss_g.numel() > 0 else 0
            if loss_l is not None:
                conf_loss_l = loss_l.mean() if loss_l.numel() > 0 else 0
            else:
                conf_loss_l = 0
            details_conf = dict(conf_loss_g=float(conf_loss_g), **details)
            if loss_l is not None:
                details_conf['conf_loss_l'] = float(conf_loss_l)
            return conf_loss_g + conf_loss_l, details_conf
        else:
            # compute conf loss for global point and local pointmap separately, then sum
            conf_pred = pred['conf'][msk_g]
            conf_g, log_conf_g = self.get_conf_log(conf_pred)
            conf_loss_g = loss_g * conf_g - self.alpha * log_conf_g
            # average + nan protection (in case of no valid pixels at all)
            conf_loss_g = conf_loss_g.mean() if conf_loss_g.numel() > 0 else 0

            if loss_l is not None:
                conf_l, log_conf_l = self.get_conf_log(pred['conf'][msk_l])
                conf_loss_l = loss_l * conf_l - self.alpha * log_conf_l
                conf_loss_l = conf_loss_l.mean() if conf_loss_l.numel() > 0 else 0
            else:
                conf_loss_l = 0
            details_conf = dict(conf_loss_g=float(conf_loss_g), **details)
            if loss_l is not None:
                details_conf['conf_loss_l'] = float(conf_loss_l)
            return conf_loss_g + conf_loss_l, details_conf
