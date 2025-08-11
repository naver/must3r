# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa
from croco.models.pos_embed import RoPE2D  # noqa


def get_pos_embed(pos_embed_name):
    # adaptative frequencies
    F0 = 1.0  # default
    assert pos_embed_name.startswith('RoPE')
    if '_' in pos_embed_name:
        """ Adapting pose embeddings for higher-resolution.
        if pos_embed_name == 'RoPE100_224:512':
            => frequencies are now going to behave in [0,512] like they behaved in [0,224] before
        """
        pos_embed_name, resolutions = pos_embed_name.split('_')
        old_grid, new_grid = resolutions.split(':')
        F0 = float(old_grid) / float(new_grid)
        print(f'>> Using adaptive frequencies: {F0=}={old_grid}/{new_grid}')
    freq = float(pos_embed_name[len('RoPE'):])
    block_pos_embed = RoPE2D(freq, F0=F0)
    return block_pos_embed
