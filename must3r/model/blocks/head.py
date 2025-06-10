# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch.nn as nn
from enum import Enum
from must3r.tools.image import unpatchify
from must3r.tools.geometry import apply_exp_to_norm


class ActivationType(Enum):
    NORM_EXP = "norm_exp"
    LINEAR = "linear"


def apply_activation(xyz, activation):
    if isinstance(activation, str):
        activation = ActivationType(activation)
    if activation == ActivationType.NORM_EXP:
        return apply_exp_to_norm(xyz, dim=-1)
    elif activation == ActivationType.LINEAR:
        return xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        x = head(decout, (H, W))
        return x

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return head(decout, (W, H)).swapaxes(1, 2)

        # batch is a mix of both portrait & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = head(selout(is_portrait), (W, H)).swapaxes(1, 2)

        x = l_result.new(B, *l_result.shape[1:])
        x[is_landscape] = l_result
        x[is_portrait] = p_result
        return x

    return wrapper_yes if activate else wrapper_no


class LinearHead(nn.Module):
    def __init__(self, embed_dim, output_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, feats, img_shape):
        x = self.proj(feats[-1])
        x = unpatchify(x, self.patch_size, img_shape).permute(0, 2, 3, 1)
        return x
