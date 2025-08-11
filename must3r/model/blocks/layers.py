# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from must3r.model.blocks.attention import Attention, CachedCrossAttention
import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa
from croco.models.blocks import Mlp, DropPath

MEMORY_MODES = ['norm_y', 'kv', 'raw']


class BaseTransformer(nn.Module):
    def initialize_weights(self):
        # linears and layer norms
        self.apply(self._init_weights)
        self.apply(self._init_override)

    def _init_override(self, m):
        init_weight_override_fun = getattr(m, "_init_weight_override", None)
        if callable(init_weight_override_fun):
            init_weight_override_fun()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    def __init__(self, dim, num_heads, pos_embed=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # SA
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, xpos=None):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CachedDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, pos_embed=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, memory_mode="norm_y"):
        super().__init__()
        assert memory_mode in MEMORY_MODES
        self.memory_mode = memory_mode

        # SA
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)

        # CA
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = CachedCrossAttention(dim, pos_embed=None, num_heads=num_heads, qkv_bias=qkv_bias,
                                               attn_drop=attn_drop, proj_drop=drop)

        # MLP
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def prepare_y(self, y):
        if self.memory_mode == 'raw':
            return y
        y_ = self.norm_y(y)
        if self.memory_mode == 'norm_y':
            return y_.to(y.dtype)
        k, v = self.cross_attn.prepare_kv(y_, y_)
        return torch.concatenate([k, v], dim=-1)

    def forward(self, x, y, xpos=None, ypos=None, ca_attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y) if self.memory_mode == 'raw' else y
        if self.memory_mode == 'kv':
            key, value = torch.split(y_, x.shape[-1], dim=-1)
        else:
            key, value = self.cross_attn.prepare_kv(y_, y_)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), key, value, xpos, ypos, ca_attn_mask))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x
