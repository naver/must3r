#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
#
# --------------------------------------------------------
# MUSt3R demo executable for exporting reconstructions
# --------------------------------------------------------
import os
import torch
import argparse
import pickle

from must3r.model import *
from must3r.model.blocks.layers import MEMORY_MODES
from must3r.model.blocks.attention import toggle_memory_efficient_attention
from must3r.demo.gradio import get_reconstructed_scene, get_3D_model_from_scene

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")

    parser.add_argument("--image_dir", required=True, type=str, help="image dir")
    parser.add_argument("--output", required=True, type=str, help="output dir")

    parser.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser.add_argument("--encoder", type=str, default=None, help="encoder class instantiation")
    parser.add_argument("--decoder", type=str, default=None, help="decoder class instantiation")
    parser.add_argument("--memory_mode", type=str, default=None, choices=MEMORY_MODES,
                        help="decoder memory_mode override")

    parser.add_argument("--retrieval", type=str, help="path to the retrieval weights", default=None)

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--amp", type=str, default=False)

    parser.add_argument("--execution_mode", type=str, default="linseq",
                        choices=["linseq", "retrieval", "vidseq", "vidslam"])

    parser.add_argument("--max_bs", type=int, default=1)
    parser.add_argument("--num_refinements_iterations", type=int, default=0)
    parser.add_argument('--render_once', action='store_true', default=False, help="skip the final rendering step")

    # linseq / retrieval params
    parser.add_argument("--num_mem_imgs", type=int, default=50)

    # vidseq /vidslam params
    parser.add_argument("--local_context_size", type=int, default=0)
    # vidseq params
    parser.add_argument("--keyframe_interval", type=int, default=3)
    # vidslam params
    parser.add_argument("--subsample", type=int, default=2)
    parser.add_argument("--min_conf_keyframe", type=float, default=1.5)
    parser.add_argument("--keyframe_overlap_thr", type=float, default=0.05)
    parser.add_argument("--overlap_percentile", type=float, default=85)

    # viz params
    parser.add_argument("--cam_size", type=float, default=0.05)
    parser.add_argument("--camera_conf_thr", type=float, default=0.0)

    parser.add_argument("--file_type", type=str, default="glb", choices=["glb", "ply"])
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    toggle_memory_efficient_attention(enabled=True)

    images = sorted([os.path.join(args.image_dir, f)
                     for f in os.listdir(args.image_dir)
                     if os.path.isfile(os.path.join(args.image_dir, f))])
    os.makedirs(args.output, exist_ok=True)

    weights_path = args.weights
    model = load_model(weights_path, encoder=args.encoder, decoder=args.decoder, device=args.device,
                       img_size=args.image_size, memory_mode=args.memory_mode)
    num_mem_imgs = min(args.num_mem_imgs, len(images))

    min_conf_thr = 1.05
    cam_size = args.cam_size
    execution_mode = args.execution_mode
    assert execution_mode != "retrieval" or args.retrieval is not None, "You need to provide --retrieval for execution_mode==retrieval"

    camera_conf_thr = args.camera_conf_thr
    num_refinements_iterations = args.num_refinements_iterations
    scene, outfile = get_reconstructed_scene(outdir=args.output, viser_server=None, should_save_glb=False, model=model,
                                             retrieval=args.retrieval, device=args.device,
                                             verbose=True, image_size=args.image_size, amp=args.amp,
                                             filelist=images, min_conf_thr=min_conf_thr,
                                             as_pointcloud=True, transparent_cams=False, local_pointmaps=False,
                                             cam_size=cam_size, num_mem_images=num_mem_imgs, max_bs=args.max_bs,
                                             render_once=args.render_once, camera_conf_thr=camera_conf_thr,
                                             num_refinements_iterations=num_refinements_iterations,
                                             execution_mode=execution_mode,
                                             vidseq_local_context_size=args.local_context_size, keyframe_interval=args.keyframe_interval,
                                             slam_local_context_size=args.local_context_size,
                                             subsample=args.subsample, min_conf_keyframe=args.min_conf_keyframe,
                                             keyframe_overlap_thr=args.keyframe_overlap_thr, overlap_percentile=args.overlap_percentile
                                             )
    threshold_list = [6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, min_conf_thr]
    for thr in threshold_list:
        try:
            outfile = get_3D_model_from_scene(outdir=args.output, verbose=True, scene=scene, min_conf_thr=thr,
                                              as_pointcloud=True, transparent_cams=False, cam_size=cam_size,
                                              filename=f'scene_{thr}.{args.file_type}')
        except Exception as e:
            continue
    with open(os.path.join(args.output, f'scene.pkl'), 'wb') as f:
        pickle.dump(scene, f)
