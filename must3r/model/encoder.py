# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from must3r.model.blocks.pos_embed import get_pos_embed
from must3r.model.blocks.layers import BaseTransformer, Block

import must3r.tools.path_to_dust3r  # noqa
from dust3r.patch_embed import get_patch_embed


class Dust3rEncoder(BaseTransformer):
    def __init__(self,
                 img_size=(224, 224),           # input image size
                 patch_size=16,          # patch_size
                 embed_dim=1024,      # encoder feature dimension
                 depth=24,           # encoder depth
                 num_heads=16,       # encoder number of heads in the transformer block
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 patch_embed='PatchEmbedDust3R',
                 pos_embed='RoPE100'):
        super(Dust3rEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.set_patch_embed(patch_embed, img_size, patch_size, embed_dim)

        self.max_seq_len = max(img_size) // patch_size
        self.grid_size = self.patch_embed.grid_size
        self.rope = get_pos_embed(pos_embed)

        self.blocks_enc = nn.ModuleList([
            Block(embed_dim, num_heads, pos_embed=self.rope, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_enc = norm_layer(embed_dim)
        self.initialize_weights()

    def set_patch_embed(self, patch_embed_name='PatchEmbedDust3R', img_size=224, patch_size=16, patch_embed_dim=768):
        self.patch_size = patch_size
        assert self.embed_dim == patch_embed_dim
        self.patch_embed = get_patch_embed(patch_embed_name, img_size, patch_size, patch_embed_dim)
        self.grid_size = self.patch_embed.grid_size

    @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, img, true_shape):
        x, pos = self.patch_embed(img, true_shape=true_shape)
        for blk in self.blocks_enc:
            x = blk(x, pos)
        x = self.norm_enc(x)
        return x, pos

    def from_dust3r(self, state_dict, verbose=True):
        state_dict = {k.replace('enc_blocks', 'blocks_enc').replace(
            'enc_norm', 'norm_enc'): v for k, v in state_dict.items()}
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        if verbose:
            print(incompatible_keys)
        assert len(incompatible_keys.missing_keys) == 0
        return incompatible_keys

    def from_croco(self, state_dict, verbose=True):
        # same format
        return self.from_dust3r(state_dict, verbose=verbose)
