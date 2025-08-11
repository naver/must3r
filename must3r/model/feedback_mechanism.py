# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from typing import Optional, List, Callable

import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa
from croco.models.blocks import Mlp


def create_feedback_layers(embed_dim, depth, feedback_type):
    if feedback_type == 'single_mlp':
        feedback_layer = Mlp(embed_dim, hidden_features=4 * embed_dim, out_features=embed_dim)
        feedback_norm = nn.LayerNorm(embed_dim)
    elif feedback_type == 'single_linear':
        feedback_layer = nn.Linear(embed_dim, out_features=embed_dim)
        feedback_norm = nn.LayerNorm(embed_dim)
    else:
        assert not feedback_type
        feedback_layer = None
        feedback_norm = None

    return feedback_layer, feedback_norm


def init_feedback_layers(feedback_type, feedback_layer):
    # init as zeros so that it's inactive at the start
    if feedback_layer is not None:
        if feedback_type == 'single_mlp':
            nn.init.constant_(feedback_layer.fc2.bias, 0)
            nn.init.constant_(feedback_layer.fc2.weight, 0)
        elif feedback_type == 'single_linear':
            nn.init.constant_(feedback_layer.bias, 0)
            nn.init.constant_(feedback_layer.weight, 0)
        else:
            raise ValueError(f"Unknown {feedback_type=}")


def run_feedback_layers(
    feedback_layer: Optional[Callable],
    feedback_norm: Optional[Callable],
    mem: List[torch.Tensor]
) -> List[torch.Tensor]:
    # nothing to do ?
    if feedback_layer is None:
        return mem
    blk, blk_ln = feedback_layer, feedback_norm
    offset = blk(blk_ln(mem[-1]))
    new_mem = [
        memi + offset for memi in mem[:-1]
    ]
    new_mem.append(mem[-1])
    return new_mem
