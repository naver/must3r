# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import re
from .encoder import Dust3rEncoder  # noqa
from .decoder import *  # noqa
from .blocks.head import ActivationType, apply_activation  # noqa


def get_pointmaps_activation(decoder, verbose=True):
    try:
        pointmaps_activation = decoder.pointmaps_activation
    except Exception as e:
        pointmaps_activation = ActivationType.NORM_EXP
    if verbose:
        print(f'pointmaps_activation set to {pointmaps_activation}')
    return pointmaps_activation


def get_dtype(amp):
    if amp == "fp16":
        dtype = torch.float16
    elif amp == "bf16":
        assert torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16
    else:
        assert not amp
        dtype = torch.float32
    return dtype


def load_model(chkpt_path, encoder=None, decoder=None, device='cuda', img_size=None, memory_mode=None, verbose=True):
    ckpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)

    encoder_args = encoder or ckpt['args'].encoder
    decoder_args = decoder or convert_decoder_args(ckpt['args'].decoder)
    if img_size is not None:
        encoder_args = set_image_size_in_args(encoder_args, img_size, verbose=verbose)
        decoder_args = set_image_size_in_args(decoder_args, img_size, verbose=verbose)
    encoder = eval(encoder_args)
    decoder = eval(decoder_args)
    if memory_mode is not None:
        decoder.change_memory_mode(memory_mode)

    encoder.load_state_dict(ckpt['encoder'], strict=True)
    decoder.load_state_dict(ckpt['decoder'], strict=True)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def convert_decoder_args(decoder_args):
    dec_corresp_dict = {'CausalMUSt3R': 'MUSt3R',
                        'landscape_only=True': "landscape_only=False",
                        }

    decoder_args = decoder_args.replace(' ', '')
    for k, v in dec_corresp_dict.items():
        decoder_args = decoder_args.replace(k, v)
    if 'landscape_only=False' not in decoder_args:
        decoder_args = decoder_args[:-1] + ",landscape_only=False)"
    return decoder_args


def set_image_size_in_args(model_args, img_size, verbose=True):
    model_args = model_args.replace(' ', '')

    match_size = re.search(r'img_size=\((\d+),(\d+)\)', model_args)
    if not match_size:
        raise ValueError("No image_size tuple found in model args")
    h, w = map(int, match_size.groups())
    assert h == w
    if verbose:
        print(f"image_size {h} -> {img_size}")

    match_adaptative_pos_embed = re.search(r"pos_embed='([A-Za-z]+)(\d+)\_(\d+):(\d+)'", model_args)
    if match_adaptative_pos_embed:
        prefix, freq, base_size, new_size = match_adaptative_pos_embed.groups()
        freq, base_size, new_size = map(int, (freq, base_size, new_size))
        pos_embed_is_arg = True
    else:
        match_bare_pos_embed = re.search(r"pos_embed='([A-Za-z]+)(\d+)'", model_args)
        if match_bare_pos_embed:
            prefix, freq = match_bare_pos_embed.groups()
            freq = int(freq)
            pos_embed_is_arg = True
        else:
            # default value
            prefix, freq = "RoPE", 100
            pos_embed_is_arg = False
        base_size = new_size = h

    if verbose:
        print(f"Parsed pos_embed: {prefix}{freq}, base size = {base_size}")

    if img_size != h:
        model_args = model_args.replace(f'img_size=({h},{h})', f'img_size=({img_size},{img_size})')
    if img_size != new_size:
        new_pos_embed = f"{prefix}{freq}_{base_size}:{img_size}"
        if pos_embed_is_arg:
            model_args = re.sub(
                r"(pos_embed=')(?:[A-Za-z]+\d+(?:_\d+:\d+)?)(')",
                rf"\1{new_pos_embed}\2",
                model_args)
        else:
            model_args = model_args[:-1] + ",pos_embed='" + new_pos_embed + "')"
    return model_args
