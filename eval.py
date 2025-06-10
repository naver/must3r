#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from must3r.model import *
from must3r.model.blocks.attention import toggle_memory_efficient_attention
from must3r.engine.inference import inference, postprocess, concat_preds
from must3r.tools.geometry import apply_log_to_norm
from must3r.datasets import *  # noqa

import must3r.tools.path_to_dust3r  # noqa
from dust3r.losses import L21
from dust3r.utils.geometry import geotrf
torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('MUSt3R eval', add_help=False)
    parser.add_argument('--output', default=None)

    # model and criterion
    parser.add_argument('--encoder', default=None, type=str)
    parser.add_argument('--decoder', default=None)

    parser.add_argument('--init_num_views', default=2, type=int,
                        help="number of views to use when initializing the memory")
    parser.add_argument('--batch_num_views', default=1, type=int,
                        help="number of views to use at once when updating the memory")
    parser.add_argument('--max_batch_size', default=None, type=int,
                        help="max batch size for encoder/renderer")

    parser.add_argument('--render_once', action='store_true', default=False)

    parser.add_argument('--loss_in_log', action='store_true', default=False,
                        help="apply loss in log")
    parser.add_argument('--chkpt', required=True, type=str, help="path to weights")

    parser.add_argument('--eval_memory_num_views', default=None, nargs='+', type=int,
                        help="number of views to use when updating the memory")

    parser.add_argument('--verbose', action='store_true', default=False)
    # dataset
    parser.add_argument('--dataset',
                        required=True,
                        type=str, help="test set")
    parser.add_argument('--num_workers', default=8, type=int,
                        help="max batch size for encoder/renderer")

    parser.add_argument('--batch_size', default=8, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    return parser


if __name__ == "__main__":
    device = 'cuda'
    toggle_memory_efficient_attention(True)
    parser = get_args_parser()
    args = parser.parse_args()

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    criterion = L21
    print('Loading pretrained: ', args.chkpt)
    encoder, decoder = load_model(
        args.chkpt, encoder=args.encoder, decoder=args.decoder, device='cuda')
    pointmaps_activation = get_pointmaps_activation(decoder)
    dataset = eval(args.dataset)
    dataset.set_epoch(0)

    num_views_all = len(dataset[0])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        if args.eval_memory_num_views is None:
            num_views_dec_all = list(range(args.init_num_views, num_views_all + 1))
        else:
            num_views_dec_all = args.eval_memory_num_views

        for num_views_dec in num_views_dec_all:
            losses_firstpass = [[] for _ in range(num_views_all)]  # loss for each image, seen and unseen
            losses_imgs = [[] for _ in range(num_views_all)]  # loss for each image, seen and unseen
            losses_all = []
            for views in tqdm(dataloader):
                assert len(views) == num_views_all

                # DATA PREPARATION
                imgs = [b['img'] for b in views]
                imgs = torch.stack(imgs, dim=1).to(device)
                B, _, three, H, W, = imgs.shape

                true_shape = [b['true_shape'] for b in views]
                true_shape = torch.stack(true_shape, dim=1).to(device)

                gt_c2w = [b['camera_pose'] for b in views]
                gt_c2w = torch.stack(gt_c2w, dim=1).to(device)  # B, nimgs, 4, 4
                gt_w2c = torch.linalg.inv(gt_c2w)

                in_camera0 = gt_w2c[:, 0]

                gt_pts = [b['pts3d'] for b in views]
                gt_pts = torch.stack(gt_pts, dim=1).to(device)
                gt_pts = geotrf(in_camera0, gt_pts)  # B, nimgs, H, W, 3

                if args.loss_in_log:
                    gt_pts_log = apply_log_to_norm(gt_pts, dim=-1)

                gt_valid = [b['valid_mask'] for b in views]  # B, H, W
                gt_valid = torch.stack(gt_valid, dim=1).to(device)  # B, nimgs, H, W

                mem_batches = [min(args.init_num_views, num_views_dec)]
                while (sum_b := sum(mem_batches)) != num_views_dec:
                    size_b = min(args.batch_num_views, num_views_dec - sum_b)
                    mem_batches.append(size_b)

                if args.render_once:
                    to_render = list(range(num_views_dec, num_views_all))
                else:
                    to_render = None
                x_out_0, x_out = inference(encoder, decoder, imgs, true_shape, mem_batches,
                                           verbose=args.verbose, max_bs=args.max_batch_size,
                                           to_render=to_render)
                x_out_0 = postprocess(x_out_0, pointmaps_activation=pointmaps_activation)
                x_out = postprocess(x_out, pointmaps_activation=pointmaps_activation)
                if to_render is not None:
                    x_out = concat_preds(x_out_0, x_out)

                x_out_0, x_out = x_out_0['pts3d'], x_out['pts3d']
                if x_out_0 is not None:
                    # apply the loss
                    x_out_0_v = x_out_0.view(B, num_views_dec, H, W, three)
                    for b in range(B):
                        for i in range(num_views_dec):
                            loss_i = criterion(gt_pts[b, i][gt_valid[b, i]], x_out_0_v[b, i][gt_valid[b, i]])
                            losses_firstpass[i].append(loss_i.cpu())

                # apply the loss
                x_out_v = x_out.view(B, num_views_all, H, W, three)
                for b in range(B):
                    for i in range(num_views_all):
                        loss_i = criterion(gt_pts[b, i][gt_valid[b, i]], x_out_v[b, i][gt_valid[b, i]])
                        losses_imgs[i].append(loss_i.cpu())
                for b in range(B):
                    loss_value = criterion(gt_pts[b][gt_valid[b]], x_out_v[b][gt_valid[b]])
                    losses_all.append(loss_value.cpu())

            result_str = f'{num_views_dec=}\n'
            if len(losses_firstpass[0]) > 0:
                for i in range(num_views_dec):
                    result_str += (f'first pass {i} - mean = {np.mean(losses_firstpass[i])}, '
                                   f'median = {np.median(losses_firstpass[i])}\n')
            for i in range(num_views_all):
                result_str += f'{i} - mean = {np.mean(losses_imgs[i])}, median = {np.median(losses_imgs[i])}\n'
            result_str += f'global - mean = {np.mean(losses_all)}, median = {np.median(losses_all)}\n'

            print(result_str)
            if args.output is not None:
                with open(args.output, 'a') as fid:
                    fid.write(result_str)
