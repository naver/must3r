# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn

try:
    import xformers.ops
    has_xformers = True
except Exception as e:
    has_xformers = False

try:
    from torch.nn.functional import scaled_dot_product_attention  # noqa
    has_scaled_dot_product_attention = True
except Exception as e:
    has_scaled_dot_product_attention = False


_use_memory_efficient_attention = False


def toggle_memory_efficient_attention(enabled: bool = True):
    global _use_memory_efficient_attention
    _use_memory_efficient_attention = enabled


def is_memory_efficient_attention_enabled():
    return _use_memory_efficient_attention


class CoreAttention (nn.Module):
    def __init__(self, pos_embed=None, attn_drop=0.):
        super().__init__()
        self.pos_embed = pos_embed
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_val = attn_drop

    def attention(self, q, k, v, qpos=None, kpos=None, attn_mask=None):
        B, H, Nq, D = q.shape
        C = D * H
        assert H == self.num_heads

        if self.pos_embed is not None:
            q = self.pos_embed(q, qpos)
            k = self.pos_embed(k, kpos)

        if is_memory_efficient_attention_enabled() and (attn_mask is None or attn_mask.dtype != torch.bool):
            assert has_xformers
            # print('using xformers')
            # q, k, v are batch, num_heads, seqlen, K
            # Supported formats for inputs/outputs:
            # [batch, seqlen, num_heads, K]
            # [batch, seqlen, K] (Legacy format)
            # with (batch, seqlen, num_heads, K), need to use contiguous() or something's wrong with the stride for bwd
            # q, k, v = map(lambda val: val.transpose(1, 2).contiguous(), (q, k, v))
            # x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None).reshape(B, Nq, C)
            # the second format is more natural for croco
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
            if k.dtype != v.dtype:
                k = k.to(v.dtype)
            assert attn_mask is None or attn_mask.dtype == v.dtype  # because casting it here will mess up stride
            q, k, v = map(lambda val: val.reshape(B * self.num_heads, -1, D), (q, k, v))
            x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, p=self.attn_drop_val)
            x = x.reshape(B, self.num_heads, -1, D).transpose(1, 2).reshape(B, -1, C)
        elif has_scaled_dot_product_attention and (attn_mask is None or attn_mask.dtype == torch.bool):
            if q.dtype != v.dtype:
                q = q.to(v.dtype)
            if k.dtype != v.dtype:
                k = k.to(v.dtype)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                                 dropout_p=self.attn_drop_val)
            x = x.transpose(1, 2).reshape(B, Nq, C)
        else:
            assert attn_mask is None
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return x


class Attention(CoreAttention):
    def __init__(self, dim, pos_embed=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qkln=False):
        super().__init__(pos_embed=pos_embed, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xpos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv.select(2, i) for i in range(3)]
        x = self.attention(q, k, v, xpos, xpos)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(CoreAttention):
    def __init__(self, dim, pos_embed=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(pos_embed=pos_embed, attn_drop=attn_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, qpos, kpos, attn_mask=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x = self.attention(q, k, v, qpos, kpos, attn_mask=attn_mask)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CachedCrossAttention(CrossAttention):
    def __init__(self, dim, pos_embed=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(dim=dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias,
                         attn_drop=attn_drop, proj_drop=proj_drop)

    def prepare_kv(self, key, value):
        k = self.projk(key)
        v = self.projv(value)
        return k, v

    def forward(self, query, key, value, qpos, kpos, attn_mask=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = key.reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = value.reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = self.attention(q, k, v, qpos, kpos, attn_mask=attn_mask)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
