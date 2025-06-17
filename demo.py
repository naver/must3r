#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
#
# --------------------------------------------------------
# MUSt3R gradio/viser demo executable
# --------------------------------------------------------
import os
import torch
import tempfile

from must3r.model import *
from must3r.demo.gradio import get_args_parser, main_demo
from must3r.model.blocks.attention import has_xformers, toggle_memory_efficient_attention

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    toggle_memory_efficient_attention(enabled=has_xformers)

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    weights_path = args.weights
    model = load_model(weights_path, encoder=args.encoder, decoder=args.decoder, device=args.device,
                       img_size=args.image_size, memory_mode=args.memory_mode, verbose=args.verbose)

    # must3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if args.verbose:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.retrieval, args.device, args.image_size,
                  server_name, args.server_port, verbose=args.verbose, amp=args.amp, with_viser=args.viser,
                  allow_local_files=args.allow_local_files)
