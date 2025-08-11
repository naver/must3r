#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
#
# --------------------------------------------------------
# MUSt3R gradio/viser demo executable
# --------------------------------------------------------
import torch
from must3r.demo.gradio import main
import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


if __name__ == '__main__':
    main()