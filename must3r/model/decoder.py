# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from must3r.model.blocks import get_current_dtype
from must3r.model.blocks.layers import BaseTransformer, CachedDecoderBlock, MEMORY_MODES
from must3r.model.blocks.head import ActivationType, LinearHead, transpose_to_landscape
from must3r.model.blocks.dropout import MemoryDropoutSelector, TemporaryMemoryDropoutSelector
from must3r.model.blocks.pos_embed import get_pos_embed
from must3r.model.feedback_mechanism import create_feedback_layers, init_feedback_layers, run_feedback_layers


class MUSt3R(BaseTransformer):
    """
    inference class
    """

    def __init__(self,
                 img_size=(224, 224),           # input image size
                 enc_embed_dim=1024,      # encoder feature dimension
                 patch_size=16,          # encoder patch_size
                 embed_dim=768,      # decoder feature dimension
                 output_dim=1792,      # 16*16*7
                 depth=12,           # decoder depth
                 num_heads=12,       # decoder number of heads in the transformer block
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,      # activation layer in the mlp
                 pos_embed='RoPE100',
                 landscape_only=True,
                 head='Linear',
                 feedback_type=None,
                 memory_mode="norm_y",  # 3 choices, norm_y, kv and raw
                 pointmaps_activation=ActivationType.NORM_EXP,
                 block_type=CachedDecoderBlock,
                 ** kv):
        super(MUSt3R, self).__init__()
        self.pointmaps_activation = pointmaps_activation
        self._init_projector(enc_embed_dim, embed_dim)
        self._init_pos_embed(img_size, patch_size, embed_dim, num_heads, pos_embed)
        self._init_blocks(block_type, embed_dim, depth, num_heads, mlp_ratio, norm_layer, act_layer,
                          memory_mode=memory_mode)
        self._init_feedback_mechanism(embed_dim, depth, feedback_type)
        self._init_head(enc_embed_dim, patch_size, embed_dim, output_dim, depth, norm_layer, landscape_only, head)
        self.initialize_weights()
        init_feedback_layers(self.feedback_type, self.feedback_layer)

    def _init_projector(self, enc_embed_dim, embed_dim):
        self.feat_embed_enc_to_dec = nn.Linear(enc_embed_dim, embed_dim, bias=True)
        self.image2_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.image2_embed, std=.02)

    def _init_pos_embed(self, img_size, patch_size, embed_dim, num_heads, pos_embed):
        self.max_seq_len = max(img_size) // patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.rope = get_pos_embed(pos_embed)

    def _init_blocks(self, block_type, embed_dim, depth, num_heads, mlp_ratio, norm_layer, act_layer, memory_mode):
        if isinstance(block_type, str):
            block_type = eval(block_type)
        self.depth = depth
        self.embed_dim = embed_dim
        self.memory_mode = memory_mode
        self.attn_num_heads = num_heads
        self.blocks_dec = nn.ModuleList([
            block_type(embed_dim, num_heads, self.rope, mlp_ratio, qkv_bias=True,
                       norm_layer=norm_layer, act_layer=act_layer, memory_mode=memory_mode)
            for i in range(depth)])

    def _init_feedback_mechanism(self, embed_dim, depth, feedback_type):
        self.feedback_type = feedback_type
        self.feedback_layer, self.feedback_norm = create_feedback_layers(embed_dim, depth, feedback_type)

    def _init_head(self, enc_embed_dim, patch_size, embed_dim, output_dim, depth, norm_layer, landscape_only, head):
        self.norm_dec = norm_layer(embed_dim)
        if head == 'Linear':
            self.head_dec = LinearHead(embed_dim, output_dim, patch_size)
        else:
            raise ValueError(f'invalid head {head}')
        self._head_wrapper = transpose_to_landscape(self.head_dec, activate=landscape_only)

    def from_dust3r(self, state_dict, verbose=True, load_head=False):
        state_dict = {k.replace('dec_blocks.', 'blocks_dec.').replace(
            'decoder_embed.', 'feat_embed_enc_to_dec.').replace(
            'dec_norm.', 'norm_dec.'): v for k, v in state_dict.items()}
        if load_head:
            state_dict = {k.replace('downstream_head.proj.', 'head_dec.proj.'): v for k, v in state_dict.items()}
        incompatible_keys = self.load_state_dict(state_dict, strict=False)
        if verbose:
            print(incompatible_keys)
        return incompatible_keys

    def from_croco(self, state_dict, verbose=True):
        # same format
        return self.from_dust3r(state_dict, verbose=verbose)

    def set_freeze(self, freeze='none'):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'not_head': [self.feat_embed_enc_to_dec, self.image2_embed, self.blocks_dec,
                         self.feedback_layer, self.feedback_norm],
        }
        for module in to_be_frozen[freeze]:
            try:
                for n, param in module.named_parameters():
                    param.requires_grad = False
            except AttributeError:
                # module is directly a parameter
                module.requires_grad = False

    def change_memory_mode(self, memory_mode="norm_y"):
        assert memory_mode in MEMORY_MODES
        for blk in self.blocks_dec:
            blk.memory_mode = memory_mode
        self.memory_mode = memory_mode

    def make_mem_mask(self, nimgs, N, Nm, device):
        if isinstance(nimgs, list):
            assert isinstance(N, list)
            tokens_images = [nimg * Ni for nimg, Ni in zip(nimgs, N)]

            Nt = sum(tokens_images)
            mem_masks = [torch.ones((nimg, Nm + Nt), dtype=torch.bool, device=device) for nimg in nimgs]
            offset = 0
            for i, (nimg, Ni) in enumerate(zip(nimgs, N)):
                for j in range(nimg):
                    mem_masks[i][j, Nm + offset + (j * Ni):Nm + offset + ((j + 1) * Ni)] = 0
                offset += nimg * Ni
            return mem_masks
        else:
            mem_mask = torch.ones((1, N), dtype=torch.bool, device=device)
            mem_mask = mem_mask.repeat(nimgs, 1)  # nimgs, N
            mem_mask = torch.block_diag(*mem_mask).view(nimgs, -1)  # nimgs, nimgs * N
            mem_mask = torch.concatenate([torch.zeros((nimgs, Nm), dtype=mem_mask.dtype, device=device),
                                          mem_mask], dim=1)  # nimgs, Nm + nimgs * N
            mem_mask = ~mem_mask
            return mem_mask

    def _get_empty_memory(self, device, current_dtype, B, mem_D):
        current_mem = [torch.zeros((B, 0, mem_D), dtype=current_dtype, device=device) for _ in range(self.depth)]
        current_mem_labels = torch.zeros((B, 0), dtype=torch.int64, device=device)
        mem_nimgs = 0
        mem_protected_imgs = 0
        mem_protected_tokens = 0
        return current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens

    def _compute_prediction_head(self, true_shape, B, nimgs, feats):
        feats[-1] = self.norm_dec(feats[-1])
        decout = feats
        with torch.autocast("cuda", dtype=torch.float32):
            decout = [tok.float() for tok in decout]
            x = self._head_wrapper(decout, true_shape.view(B * nimgs, *true_shape.shape[2:]))
            x = x.view(B, nimgs, *x.shape[1:])
        return x

    def forward_list(self, x, pos, true_shape, current_mem=None, render=False, return_feats=False):
        # forward_list is called at inference when dealing with multiple aspect ratios or limited batch size
        x = x.copy()  # to be able to make views without changing the parent list
        pos = pos.copy()
        true_shape.copy()

        x_shapes = []
        device = x[0].device
        xdtype = x[0].dtype

        current_dtype = get_current_dtype(xdtype)
        feats = []
        for i in range(len(x)):
            B, nimg, Ni, Denc = x[i].shape
            xi_v = x[i].view(B * nimg, Ni, Denc)
            feats.append([xi_v])
            x[i] = self.feat_embed_enc_to_dec(xi_v).view(B, nimg, Ni, -1)
            if current_mem is None and i == 0:
                # initialization
                x[i][:, 1:] = x[i][:, 1:] + self.image2_embed.to(current_dtype)
            else:
                x[i] = x[i] + self.image2_embed.to(current_dtype)  # not the reference image / memory

            x_shapes.append(x[i].shape)

            D = x[i].shape[-1]
            x[i] = x[i].view(B * nimg, Ni, D)
            pos[i] = pos[i].view(B * nimg, Ni, 2)

        B = x_shapes[0][0]
        D = x_shapes[0][-1]
        mem_D = 2 * D if self.memory_mode == "kv" else D
        nimgs = [x_shapesi[1] for x_shapesi in x_shapes]
        N = [x_shapesi[2] for x_shapesi in x_shapes]
        if current_mem is None:
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = \
                self._get_empty_memory(device, current_dtype, B, mem_D)
        else:
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = current_mem

        mem = []
        Nm = current_mem[0].shape[1]
        if not render and (Nm > 0 or sum(nimgs) > 1):
            # when updating the memory, do not let an image do CA with its own tokens
            # ignore this rule when initializing from only one image
            mem_mask = self.make_mem_mask(nimgs, N, Nm, device)
        else:
            mem_mask = None

        new_mem = []
        for blk, current_mem_blk in zip(self.blocks_dec, current_mem):
            if not render:
                # update the memory for this layer
                x_cat = [xi.view(B, -1, D) for xi in x]
                x_cat = torch.concatenate(x_cat, dim=1)
                new_mem.append(x_cat)
                mem_i = torch.concatenate([current_mem_blk, blk.prepare_y(x_cat)], dim=1)
            else:
                mem_i = current_mem_blk

            # mem is B, Nmi, D
            # we need B*nimgs, Nmi, D for CA
            if mem_mask is not None:
                mem_l = [mem_i.unsqueeze(1).expand(-1, nimgs[i], -1, -1)[:, mem_mask[i]].reshape(B * nimgs[i], -1, mem_D)
                         for i in range(len(nimgs))]
            else:
                Nmi = mem_i.shape[1]
                mem_l = [mem_i.unsqueeze(1).expand(-1, nimgs[i], -1, -1).reshape(B * nimgs[i], Nmi, mem_D)
                         for i in range(len(nimgs))]

            # apply decoder
            for i in range(len(x)):
                x[i] = blk(x[i], mem_l[i], pos[i], None)
                feats[i].append(x[i])

        if not render:
            new_mem = run_feedback_layers(self.feedback_layer, self.feedback_norm, new_mem)
            mem = []
            for i in range(len(new_mem)):
                new_mem_i = self.blocks_dec[i].prepare_y(new_mem[i])
                mem.append(torch.concatenate([current_mem[i], new_mem_i], dim=1))

            new_labels = []
            offset = 0
            for i, (nimg, Ni) in enumerate(zip(nimgs, N)):
                new_labels_i = torch.arange(nimg, dtype=current_mem_labels.dtype, device=current_mem_labels.device)
                new_labels_i = new_labels_i.view(1, nimg, 1).repeat(B, 1, Ni).view(B, nimg * Ni)
                new_labels_i = new_labels_i + mem_nimgs + offset
                new_labels.append(new_labels_i)
                offset += nimg
            new_labels = torch.concatenate(new_labels, dim=1)
            mem_labels = torch.concatenate([current_mem_labels, new_labels], dim=1)
            mem_nimgs = mem_nimgs + sum(nimgs)
            out = (mem, mem_labels, mem_nimgs, mem_nimgs, mem_labels.shape[1])
        else:
            out = (current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens)

        # apply prediction head
        for i in range(len(x)):
            x[i] = self._compute_prediction_head(true_shape[i], B, nimgs[i], feats[i])
        if return_feats:
            # return memory, pointmaps, feats
            feats = [[feats[i][j].view(B, nimgs[i], *feats[i][j].shape[1:]) for j in range(len(feats[i]))]
                     for i in range(len(feats))]
            return out, x, feats
        else:
            # return memory, pointmaps
            return out, x

    def forward(self, x, pos, true_shape, current_mem=None, render=False, return_feats=False):
        if isinstance(x, list):
            # multiple ar in this batch
            return self.forward_list(x, pos, true_shape, current_mem, render)

        current_dtype = get_current_dtype(x.dtype)
        B, nimgs, N, Denc = x.shape
        feats = [x.view(B * nimgs, N, Denc)]
        x = self.feat_embed_enc_to_dec(feats[0]).view(B, nimgs, N, -1)
        B, nimgs, N, D = x.shape
        mem_D = 2 * D if self.memory_mode == "kv" else D
        assert not render or current_mem is not None

        if current_mem is None:
            # initialization
            x[:, 1:] = x[:, 1:] + self.image2_embed.to(current_dtype)
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = \
                self._get_empty_memory(x.device, current_dtype, B, mem_D)
        else:
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = current_mem
            x = x + self.image2_embed.to(current_dtype)  # not the reference image / memory
        x = x.view(B * nimgs, N, D)
        pos = pos.view(B * nimgs, N, 2)

        mem = []
        Nm = current_mem[0].shape[1]
        if not render and (Nm > 0 or nimgs > 1):
            # when updating the memory, do not let an image do CA with its own tokens
            # ignore this rule when initializing from only one image
            mem_mask = self.make_mem_mask(nimgs, N, Nm, x.device)
        else:
            mem_mask = None

        new_mem = []
        for blk, current_mem_blk in zip(self.blocks_dec, current_mem):
            if not render:
                # update the memory for this layer
                xmem = x.view(B, nimgs * N, D)
                new_mem.append(xmem)
                mem_i = torch.concatenate([current_mem_blk, blk.prepare_y(xmem)], dim=1)
            else:
                mem_i = current_mem_blk

            # mem is B, Nmi, D
            # we need B*nimgs, Nmi, D for CA
            if mem_mask is not None:
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1)[:, mem_mask].reshape(
                    B * nimgs, Nm + ((nimgs - 1)) * N, mem_D)
            else:
                Nmi = mem_i.shape[1]
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1).reshape(B * nimgs, Nmi, mem_D)

            # apply decoder
            x = blk(x, mem_i, pos, None)
            feats.append(x)

        if not render:
            # assert (Nm + nimgs * N) == mem[0].shape[1]
            new_mem = run_feedback_layers(self.feedback_layer, self.feedback_norm, new_mem)

            mem = []
            for i in range(len(new_mem)):
                new_mem_i = self.blocks_dec[i].prepare_y(new_mem[i])
                mem.append(torch.concatenate([current_mem[i], new_mem_i], dim=1))

            new_labels = torch.arange(nimgs, dtype=current_mem_labels.dtype, device=current_mem_labels.device).view(
                1, nimgs, 1).repeat(B, 1, N).view(B, N * nimgs) + mem_nimgs
            mem_labels = torch.concatenate([current_mem_labels, new_labels], dim=1)

            mem_nimgs = mem_nimgs + nimgs
            out = (mem, mem_labels, mem_nimgs, mem_nimgs, mem_labels.shape[1])
        else:
            out = (current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens)

        # apply prediction head
        x = self._compute_prediction_head(true_shape, B, nimgs, feats)

        if return_feats:
            # return memory, pointmaps, feats
            feats = [feats[i].view(B, nimgs, *feats[i].shape[1:]) for i in range(len(feats))]
            return out, x, feats
        else:
            # return memory, pointmaps
            return out, x


class CausalMUSt3R(MUSt3R):
    """
    Training class
    """

    def __init__(self,
                 protected_imgs=1,
                 mem_dropout=0.0,
                 dropout_mode='temporary',
                 use_xformers_mask=False,
                 use_mem_mask=False, **kv):
        super().__init__(**kv)
        self._init_dropout(protected_imgs, mem_dropout, dropout_mode)
        self.use_xformers_mask = use_xformers_mask
        self.use_mem_mask = use_mem_mask

    def _init_dropout(self, protected_imgs, mem_dropout, dropout_mode):
        self.protected_imgs = protected_imgs
        self.dropout_mode = dropout_mode
        if dropout_mode == 'permanent':
            self.mem_dropout = MemoryDropoutSelector(mem_dropout)
        elif dropout_mode == 'temporary':
            self.mem_dropout = TemporaryMemoryDropoutSelector(mem_dropout)
        else:
            raise ValueError(f'Invalid dropout mode = {dropout_mode}')

    def make_mem_mask(self, nimgs, N, Nm, device):
        mem_mask = torch.ones((1, N), dtype=torch.bool, device=device)
        mem_mask = mem_mask.repeat(nimgs, 1)  # nimgs, N
        mem_mask = torch.block_diag(*mem_mask).view(nimgs, -1)  # nimgs, nimgs * N
        mem_mask = torch.concatenate([torch.zeros((nimgs, Nm), dtype=mem_mask.dtype, device=device),
                                      mem_mask], dim=1)  # nimgs, Nm + nimgs * N
        mem_mask = ~mem_mask
        return mem_mask

    def make_attn_mask(self, x, B, nimgs, N, mem_nimgs, Nm, mem_not_sel, mem_labels, mem_mask):
        idx = torch.arange(nimgs, device=x.device).view(1, nimgs, 1) + mem_nimgs
        idx = idx.expand(B, -1, mem_labels.shape[-1])  # B, nimgs, Nmem

        mem_labels_view = mem_labels.view(B, 1, -1).expand(-1, nimgs, -1)  # B, nimgs, Nmem
        # do not attend tokens from the same image
        attn_mask = mem_labels_view != idx  # B, nimgs, Nmem

        # only attend tokens of the previous images
        if Nm == 0:  # exception for initialization, let the first image do CA with the second image
            idx = idx.clone()
            idx[:, 0] = idx[:, 0] + 2  # idx for img 0 will become 2
        attn_mask = attn_mask & (mem_labels_view < idx)

        if mem_not_sel is not None:
            # mask dropped out tokens
            for i in range(len(mem_not_sel) - 1):
                mem_not_sel_c = mem_not_sel[i]  # Nmem_out
                mem_not_sel_c = mem_not_sel_c.unsqueeze(0).expand(B, -1)
                attn_mask[:, i] = attn_mask[:, i].scatter(
                    dim=-1, index=mem_not_sel_c, src=torch.zeros_like(mem_not_sel_c, dtype=torch.bool))

        if mem_mask is not None:
            # use mem_mask on attn_mask
            mem_mask_attn = mem_mask.view(1, nimgs, Nm + nimgs * N)
            mem_mask_attn = mem_mask_attn.expand(B, -1, -1)
            attn_mask = attn_mask[mem_mask_attn]

        attn_mask = attn_mask.view(B, nimgs, 1, 1, -1)
        attn_mask = attn_mask.repeat(1, 1, self.attn_num_heads, N, 1)
        attn_mask = attn_mask.reshape(B * nimgs, self.attn_num_heads, N, -1)

        if self.use_xformers_mask:
            current_dtype = get_current_dtype(x.dtype)
            # xformers mask is in an additive mask in float
            # -torch.inf for ignored values, 0 for values we keep
            # you need to ensure memory is aligned by slicing a bigger tensor
            attn_mask = attn_mask.reshape(B * nimgs * self.attn_num_heads, N, -1)
            last_dim = attn_mask.shape[-1]
            last_dim = (last_dim + 7) // 8 * 8
            attn_mask_float = torch.full((B * nimgs * self.attn_num_heads, N, last_dim),
                                         -torch.inf, dtype=current_dtype, device=x.device
                                         )[:, :, :attn_mask.shape[-1]]
            attn_mask_float[attn_mask] = 0
            attn_mask = attn_mask_float
        return attn_mask

    def forward(self, x, pos, true_shape, current_mem=None, render=False, return_feats=False):
        current_dtype = get_current_dtype(x.dtype)
        # project encoder features to the correct dimension
        B, nimgs, N, Denc = x.shape
        feats = [x.view(B * nimgs, N, Denc)]
        x = self.feat_embed_enc_to_dec(feats[0]).view(B, nimgs, N, -1)
        B, nimgs, N, D = x.shape
        mem_D = 2 * D if self.memory_mode == "kv" else D
        # render=True means we do not update the memory
        assert not render or current_mem is not None

        if current_mem is None:
            # initialization
            x[:, 1:] = x[:, 1:] + self.image2_embed.to(current_dtype)
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = \
                self._get_empty_memory(x.device, current_dtype, B, mem_D)
        else:
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = current_mem
            x = x + self.image2_embed.to(current_dtype)  # not the reference image / memory

        # protected tokens will not be dropped out
        if not render:
            current_mem_protected_imgs = mem_protected_imgs
            mem_protected_imgs = min(self.protected_imgs, current_mem_protected_imgs + nimgs)
            mem_protected_tokens = mem_protected_tokens + (mem_protected_imgs - current_mem_protected_imgs) * N

        x = x.view(B * nimgs, N, D)
        pos = pos.view(B * nimgs, N, 2)

        Nm = current_mem[0].shape[1]  # number of memory tokens at the previous step

        mem_sel = None
        mem_not_sel = None
        active_mem = current_mem
        if not render and self.mem_dropout.p > 0.0:
            # random token dropout, efficient for training
            mem_sel, mem_not_sel = self.mem_dropout(Nm, nimgs, N, protected=mem_protected_tokens, device=x.device)
        elif render and self.mem_dropout.p > 0.0 and self.dropout_mode == 'temporary':
            new_mem_tokens = 0
            mem_sel, mem_not_sel = self.mem_dropout(Nm, 1, new_mem_tokens, protected=mem_protected_tokens,
                                                    device=x.device)

            # dropout mem here
            active_mem = [mem_i[:, mem_sel[0]] for mem_i in current_mem]
            mem_sel, mem_not_sel = None, None
            Nm = active_mem[0].shape[1]  # number of memory tokens at the previous step

        if not render:
            # prepare labels for the new memory tokens
            new_labels = torch.arange(nimgs, dtype=current_mem_labels.dtype, device=current_mem_labels.device).view(
                1, nimgs, 1).repeat(B, 1, N).view(B, N * nimgs) + mem_nimgs
            mem_labels = torch.concatenate([current_mem_labels, new_labels], dim=1)
        else:
            mem_labels = current_mem_labels

        if mem_sel is not None and self.dropout_mode == 'permanent':
            # select the new memory labels after dropout
            mem_labels_out = mem_labels[:, mem_sel[-1]]
        else:
            mem_labels_out = mem_labels

        mem_mask = None
        attn_mask = None
        if not render and (Nm > 0 or nimgs > 1):
            # when updating the memory, do not let an image do CA with its own tokens
            # ignore this rule when initializing from only one image
            if self.use_mem_mask:
                # physically remove the self attending memory tokens
                mem_mask = self.make_mem_mask(nimgs, N, Nm, x.device)
            # create mask for the cross attention
            attn_mask = self.make_attn_mask(x, B, nimgs, N, mem_nimgs, Nm, mem_not_sel, mem_labels, mem_mask)

        new_mem = []
        for blk, current_mem_blk in zip(self.blocks_dec, active_mem):
            if not render:
                # update the memory for this layer
                xmem = x.view(B, nimgs * N, D)
                new_mem.append(xmem)
                mem_i = torch.concatenate([current_mem_blk, blk.prepare_y(xmem)], dim=1)
            else:
                mem_i = current_mem_blk

            # mem is B, Nmi, D
            # we need B*nimgs, Nmi, D for CA
            if mem_mask is not None:
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1)
                mem_i = mem_i[:, mem_mask]
                mem_i = mem_i.reshape(B * nimgs, Nm + ((nimgs - 1)) * N, mem_D)
            else:
                Nmi = mem_i.shape[1]
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1).reshape(B * nimgs, Nmi, mem_D)

            # apply decoder
            x = blk(x, mem_i, pos, None, ca_attn_mask=attn_mask)
            feats.append(x)

        if not render:
            new_mem = run_feedback_layers(self.feedback_layer, self.feedback_norm, new_mem)
            mem = []
            for i in range(len(new_mem)):
                new_mem_i = self.blocks_dec[i].prepare_y(new_mem[i])
                mem.append(torch.concatenate([current_mem[i], new_mem_i], dim=1))
            if mem_sel is not None and self.dropout_mode == 'permanent':
                mem = [mem_i[:, mem_sel[-1]] for mem_i in mem]
            mem_nimgs = mem_nimgs + nimgs
            out = (mem, mem_labels_out, mem_nimgs, mem_protected_imgs, mem_protected_tokens)
        else:
            out = (current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens)

        # apply prediction head
        x = self._compute_prediction_head(true_shape, B, nimgs, feats)

        if return_feats:
            # return memory, pointmaps, feats
            feats = [feats[i].view(B, nimgs, *feats[i].shape[1:]) for i in range(len(feats))]
            return out, x, feats
        else:
            # return memory, pointmaps
            return out, x


if __name__ == '__main__':
    from must3r.model.blocks.attention import toggle_memory_efficient_attention
    from must3r.model.encoder import Dust3rEncoder
    import must3r.tools.path_to_dust3r  # noqa
    import dust3r.utils.path_to_croco  # noqa
    from croco.models.blocks import PositionGetter
    toggle_memory_efficient_attention(enabled=True)

    enc = Dust3rEncoder(img_size=(224, 224), patch_embed='PatchEmbedDust3R').to('cuda')

    dec = CausalMUSt3R(img_size=(224, 224), mem_dropout=0.00, feedback_type='single_mlp', use_xformers_mask=False,
                       dropout_mode='temporary', memory_mode='norm_y', use_mem_mask=True).to('cuda')
    # dec = MUSt3R(img_size=(224, 224), feedback_type='single_mlp').to('cuda')
    MB = 1024.0 * 1024.0

    BS = 2
    device = 'cuda'

    # true_shape = [[[512, 384], [384, 512]], [[512, 336]]]
    # x = [torch.randn((BS, 2, 3, 384, 512)).to('cuda'), torch.randn((BS, 1, 3, 336, 512)).to(device)]
    # true_shape = [torch.tensor(true_shape[0], dtype=torch.int64, device=device).repeat(BS, 1, 1),
    #               torch.tensor(true_shape[1], dtype=torch.int64, device=device).repeat(BS, 1, 1)]
    true_shape = [[224, 224], [224, 224], [224, 224]]
    true_shape = torch.tensor(true_shape, dtype=torch.int64, device=device).repeat(BS, 1, 1)
    x = torch.randn((BS, 3, 3, 224, 224)).to(device)
    nimg = 3
    from contextlib import nullcontext
    with torch.cuda.amp.autocast(dtype=torch.float16):  # nullcontext():
        with torch.no_grad():
            x, pos = enc(x.view(BS * nimg, 3, 224, 224), true_shape.view(BS * nimg, 2))
            x = x.view(BS, nimg, *x.shape[1:])
            pos = pos.view(BS, nimg, *pos.shape[1:])
            true_shape = true_shape.view(BS, nimg, 2)
            o1 = None
            for i in range(10):
                o1, _ = dec(x, pos, true_shape, o1)
                try:
                    print(f'{i+1} - {o1[0][0].shape}')
                except Exception as e:
                    print(f'{i+1} - {o1[0][0][0].shape}')
                memory = torch.cuda.max_memory_allocated() / MB
                print(memory)

            for i in range(10):
                o1, _ = dec(x, pos, true_shape, o1, render=True)
                try:
                    print(f'{i+1} - {o1[0][0].shape}')
                except Exception as e:
                    print(f'{i+1} - {o1[0][0][0].shape}')
                memory = torch.cuda.max_memory_allocated() / MB
                print(memory)
