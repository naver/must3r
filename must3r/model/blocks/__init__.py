# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch


def get_current_dtype(default_dtype, verbose=False):
    current_dtype = default_dtype
    try:
        if torch.is_autocast_cpu_enabled():
            current_dtype = torch.get_autocast_cpu_dtype()
        elif torch.is_autocast_enabled():
            current_dtype = torch.get_autocast_gpu_dtype()
    except Exception as e:
        pass
    if verbose:
        print(current_dtype)
    return current_dtype
