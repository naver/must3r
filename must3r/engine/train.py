# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from pathlib import Path
from typing import Sized
from itertools import chain

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from must3r.datasets import *
from must3r.model import *  # noqa: F401, needed when loading the model
from must3r.model.blocks.attention import toggle_memory_efficient_attention
import must3r.engine.optimizer as optim
from must3r.engine.inference import inference, concat_preds, postprocess
from must3r.engine.losses import *
import must3r.engine.io as checkpoints

import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
import croco.utils.misc as dist
from croco.utils.misc import MetricLogger, SmoothedValue


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion

    parser.add_argument('--encoder', default="Dust3rEncoder()", type=str, help="dust3r encoder init")
    parser.add_argument('--decoder', default="CausalMUSt3R()", help='decoder init')

    parser.add_argument('--memory_num_views', default=10, type=int,
                        help="max number of views to use when updating the memory")
    parser.add_argument('--memory_batch_views', default=None, type=int,
                        help="max number of views to use when updating the memory")
    parser.add_argument('--min_memory_num_views', default=2, type=int,
                        help="min number of views to use when updating the memory")

    parser.add_argument('--causal', action='store_true', default=False, help="update the memory in a single forward")
    parser.add_argument('--ignore_dataloader_memory_num_views', action='store_true', default=False)

    parser_render = parser.add_mutually_exclusive_group()
    parser_render.add_argument('--render_once', action='store_true', default=False)
    parser_render.add_argument('--disable_render', action='store_true', default=False)
    parser.add_argument('--max_render_count', default=None, type=int)

    parser.add_argument('--finetune_encoder', default=False, action='store_true', help="Also finetune dust3r's encoder")
    parser.add_argument('--loss_in_log', action='store_true', default=False)
    parser.add_argument('--criterion',
                        default="ConfLoss(Regr3D(L21, norm_mode='?avg_dis', sky_loss_value=2, loss_in_log=args.loss_in_log), alpha=0.2)",
                        type=str, help="loss")

    parser_chkpt = parser.add_mutually_exclusive_group()
    parser_chkpt.add_argument('--dust3r_chkpt', default=None, type=str, help="path to dust3r encoder weights")
    parser_chkpt.add_argument('--croco_chkpt', default=None, type=str, help="path to croco decoder weights")
    parser_chkpt.add_argument('--chkpt', default=None, type=str, help="optional path to decoder weights")

    # dataset
    parser.add_argument('--dataset', required=True, type=str, help="training set")

    # training
    parser.add_argument('--seed', default=777, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=2, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=2, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--max_batch_size', default=None, type=int)

    parser.add_argument('--epochs', default=20, type=int, help="Maximum number of epochs for the scheduler")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=6, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--warmup_lr', type=float, default=0., help='lr at the start of warm-up')

    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False,
                        help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument('--use_memory_efficient_attention', action='store_true',
                        help='use flash attention or xformers mem_eff_attention.')
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--disable_tf32", action='store_true', default=False,
                        help="set cudnn.benchmark = False")

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--nodist', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser


def build_dataset(args, dataset=None):
    if dataset is None:
        dataset = getattr(args, 'dataset')

    print(f'Building Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_mem=True,
                             shuffle=True,
                             drop_last=True)

    print(f"dataset length: ", len(loader))
    return loader


def select_batch(device, args, rng, memory_num_views, progress, imgs, true_shape, nimgs):
    to_skip = 0
    to_render = None

    if args.memory_num_views < nimgs:
        # in this scenario, we will update part of the memory in no_grad
        # we allow more and more images to be no_grad in a curriculum way
        memory_num_views = 1
        max_views = math.ceil(args.memory_num_views + progress * (nimgs - args.memory_num_views))
        max_views = min(max_views, nimgs)
        # choose how many images to no_grad
        to_skip = rng.choice(max_views - args.min_memory_num_views + 1)
        if to_skip < args.min_memory_num_views:
            # let's not split the intialization
            to_skip = 0
            memory_num_views = args.min_memory_num_views

        max_n_imgs = min(to_skip + memory_num_views + args.memory_num_views, max_views)
        imgs = imgs[:, :max_n_imgs].contiguous()
        true_shape = true_shape[:, :max_n_imgs].contiguous()

        number_unseen = max_n_imgs - (to_skip + memory_num_views)
        if args.render_once:
            # render only unseen images
            if number_unseen > 0:
                to_render = torch.randperm(number_unseen, device=device) + to_skip + memory_num_views
            else:
                to_render = []
        else:
            # render half unseen, half random images
            to_render = torch.randperm(number_unseen, device=device) + to_skip + memory_num_views
            to_render = to_render[:math.ceil(args.memory_num_views / 2)]

            n_selected = len(to_render)
            to_render = torch.concatenate([to_render,
                                           torch.randperm((to_skip + memory_num_views), device=device)[:(args.memory_num_views - n_selected)]])
    elif args.render_once:
        # render only unseen images
        to_render = list(range(memory_num_views, nimgs))

    to_skip_batches = []
    mem_batches = []
    if args.memory_batch_views is not None:
        if not args.causal:
            # will process multiple images at once
            # will pick a random number of images to process each time
            if to_skip > 0:
                assert to_skip >= args.min_memory_num_views
                while (sum_b := sum(to_skip_batches)) != to_skip:
                    size_b = rng.choice(min(args.memory_batch_views, to_skip)) + 1
                    size_b = min(size_b, to_skip - sum_b)
                    to_skip_batches.append(size_b)
            while (sum_b := sum(mem_batches)) != memory_num_views:
                size_b = rng.choice(min(args.memory_batch_views, memory_num_views)) + 1
                size_b = min(size_b, memory_num_views - sum_b)
                mem_batches.append(size_b)
        else:
            # will process multiple images at once, maximum memory_batch_views
            if to_skip > 0:
                assert to_skip >= args.min_memory_num_views
                while (sum_b := sum(to_skip_batches)) != to_skip:
                    size_b = min(args.memory_batch_views, to_skip - sum_b)
                    to_skip_batches.append(size_b)
            while (sum_b := sum(mem_batches)) != memory_num_views:
                size_b = min(args.memory_batch_views, memory_num_views - sum_b)
                mem_batches.append(size_b)
    else:
        # process it dust3r like, one image at a time, except for initialization
        if not args.causal:
            if to_skip > 0:
                assert to_skip >= args.min_memory_num_views
                to_skip_batches = [args.min_memory_num_views] + \
                    [1 for _ in range(to_skip - args.min_memory_num_views)]
                mem_batches = [1 for _ in range(memory_num_views)]
            else:
                mem_batches = [args.min_memory_num_views] + \
                    [1 for _ in range(memory_num_views - args.min_memory_num_views)]
        else:
            if to_skip > 0:
                assert to_skip >= args.min_memory_num_views
                to_skip_batches = [to_skip]
            else:
                mem_batches = [memory_num_views]

    return imgs, true_shape, memory_num_views, to_skip, to_render, to_skip_batches, mem_batches


def train(args):
    assert os.environ.get('MKL_NUM_THREADS') == '1', 'otherwise inefficient'
    assert os.environ.get('NUMEXPR_NUM_THREADS') == '1', 'otherwise inefficient'
    assert os.environ.get('OMP_NUM_THREADS') == '1', 'otherwise inefficient'

    dist.init_distributed_mode(args)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
    torch.backends.cudnn.allow_tf32 = not args.disable_tf32

    toggle_memory_efficient_attention(enabled=args.use_memory_efficient_attention)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    last_ckpt_fname = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.dataset))
    start_time = time.time()
    data_loader_train = build_dataset(args)

    # model
    print('Loading encoder: {:s}'.format(args.encoder))
    encoder = eval(args.encoder)
    print('Loading decoder: {:s}'.format(args.decoder))
    decoder = eval(args.decoder)

    print(f'>> Creating criterion')
    criterion = eval(args.criterion)

    encoder.to(device)
    decoder.to(device)
    encoder_without_ddp = encoder
    decoder_without_ddp = decoder
    print("encoder = %s" % str(encoder_without_ddp))
    print("decoder = %s" % str(decoder_without_ddp))

    if args.chkpt and last_ckpt_fname is None:
        print('Loading pretrained: ', args.chkpt)
        ckpt = torch.load(args.chkpt, map_location=device, weights_only=False)
        print(encoder.load_state_dict(ckpt['encoder'], strict=False))
        print(decoder.load_state_dict(ckpt['decoder'], strict=False))
        del ckpt  # in case it occupies memory
    elif args.dust3r_chkpt is not None and last_ckpt_fname is None:
        # load dust3r encoder
        print('Loading pretrained: ', args.dust3r_chkpt)
        ckpt = torch.load(args.dust3r_chkpt, map_location=device, weights_only=False)
        encoder.from_dust3r(ckpt['model'])
        decoder.from_dust3r(ckpt['model'])
    elif args.croco_chkpt is not None and last_ckpt_fname is None:
        # load croco decoder
        print('Loading pretrained: ', args.croco_chkpt)
        ckpt = torch.load(args.croco_chkpt, map_location=device, weights_only=False)
        encoder.from_croco(ckpt['model'])
        decoder.from_croco(ckpt['model'])
    elif last_ckpt_fname is None:
        print('from scratch')

    eff_batch_size = args.batch_size * args.accum_iter * dist.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(
            encoder, device_ids=[args.gpu], find_unused_parameters=False, static_graph=False, broadcast_buffers=True)
        encoder_without_ddp = encoder.module

        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder, device_ids=[args.gpu], find_unused_parameters=False, static_graph=False, broadcast_buffers=True)
        decoder_without_ddp = decoder.module

    # following timm: set wd as 0 for bias and norm layers*
    param_groups = []
    if args.finetune_encoder:
        param_groups += optim.get_parameter_groups(encoder_without_ddp, 0, args.weight_decay)
    param_groups += optim.get_parameter_groups(decoder_without_ddp, encoder_without_ddp.depth, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats):
        if dist.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname):
        checkpoints.save_model(args=args, encoder=encoder_without_ddp, decoder=decoder_without_ddp,
                               optimizer=optimizer, loss_scaler=loss_scaler,
                               epoch=epoch, fname=fname)

    checkpoints.load_model(args=args, chkpt_path=last_ckpt_fname, encoder=encoder_without_ddp,
                           decoder=decoder_without_ddp, optimizer=optimizer,
                           loss_scaler=loss_scaler)
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    args.pointmaps_activation = get_pointmaps_activation(decoder_without_ddp)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_stats = train_one_epoch(
            encoder, decoder, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

        write_log_stats(epoch, train_stats)

        # Save the 'last' checkpoint
        if epoch >= args.start_epoch:
            save_model(epoch, 'last')
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch, str(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, encoder=encoder_without_ddp, decoder=decoder_without_ddp)


def save_final_model(args, epoch, encoder, decoder):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'

    to_save = {
        'args': args,
        'encoder': encoder if isinstance(encoder, dict) else encoder.state_dict(),
        'decoder': decoder if isinstance(decoder, dict) else decoder.state_dict(),
        'epoch': epoch
    }
    print(f'>> Saving model to {checkpoint_path} ...')
    dist.save_on_master(to_save, checkpoint_path)


def train_one_epoch(encoder: torch.nn.Module, decoder: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == (not args.disable_tf32)

    # torch.set_anomaly_enabled(True)
    encoder.train(args.finetune_encoder)
    decoder.train(True)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    # fix the seed
    seed = args.seed + epoch * dist.get_world_size() + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=args.seed + epoch)

    dtype = get_dtype(args)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        progress = epoch_f / args.epochs

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            dist.adjust_learning_rate(optimizer, epoch_f, args)

        imgs = [b['img'] for b in batch]
        imgs = torch.stack(imgs, dim=1).to(device)  # B, nimgs, 3, H, W
        B, nimgs, three, H, W = imgs.shape

        true_shape = [b['true_shape'] for b in batch]
        true_shape = torch.stack(true_shape, dim=1).to(device)  # B, nimgs, 3, H, W

        if args.ignore_dataloader_memory_num_views:  # similar to the CVPR implementation: extra images may not overlap with the keyframes
            memory_num_views = rng.choice(args.memory_num_views - args.min_memory_num_views + 1) \
                + args.min_memory_num_views
        else:
            memory_num_views = int(batch[0]['memory_num_views'][0])
        imgs, true_shape, memory_num_views, to_skip, to_render, to_skip_batches, mem_batches = select_batch(
            device, args, rng, memory_num_views, progress, imgs, true_shape, nimgs)

        mem_batches = to_skip_batches + mem_batches

        finetune_encoder = args.finetune_encoder
        if args.max_render_count is not None:
            if to_render is None:
                to_render = list(range(nimgs))
            to_render = rng.choice(to_render, size=args.max_render_count, replace=False)
        if args.disable_render:
            to_render = []
        with torch.autocast("cuda", dtype=dtype):
            x_out_0, x_out = inference(encoder, decoder, imgs, true_shape, mem_batches,
                                       train_decoder_skip=len(to_skip_batches),
                                       max_bs=args.max_batch_size,
                                       to_render=to_render, encoder_requires_grad=finetune_encoder)
        with torch.autocast("cuda", dtype=torch.float32):
            x_out_0 = postprocess(x_out_0, pointmaps_activation=args.pointmaps_activation)
            x_out = postprocess(x_out, pointmaps_activation=args.pointmaps_activation)

            b0 = batch[to_skip:(to_skip + memory_num_views)]
            if to_render is None:
                br = batch
            else:
                br = [batch[i] for i in to_render]
            gt = b0 + br
            x_out = concat_preds(x_out_0, x_out)

            loss, loss_details = criterion(gt, x_out)
            loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        if args.finetune_encoder:
            parameters_chain = chain(encoder.parameters(), decoder.parameters())
        else:
            parameters_chain = decoder.parameters()
        loss_scaler(loss, optimizer, parameters=parameters_chain,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = dist.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
