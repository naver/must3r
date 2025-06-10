# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import os
import PIL.Image
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as tvf


def unpatchify(x, patch_size, true_shape):
    B = x.shape[0]
    H, W = true_shape
    x = x.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    x = F.pixel_shuffle(x, patch_size)  # B,channels,H,W
    return x


ratios_resolutions = {
    224: {1.0: [224, 224]},
    336: {1.0: [336, 336]},
    384: {4 / 3: [384, 288], 3 / 2: [384, 256], 2 / 1: [384, 192], 3 / 1: [384, 128]},
    448: {1.0: [448, 448]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]},
    768: {4 / 3: [768, 576], 3 / 2: [768, 512], 16 / 9: [768, 432], 2 / 1: [768, 384], 16 / 5: [768, 240]},
}


def get_HW_resolution(H, W, maxdim, patchsize=16):
    if isinstance(maxdim, int):
        assert maxdim in ratios_resolutions, f"Error, {maxdim=} not implemented yet."
    ratios_resolutions_maxdim = maxdim if isinstance(maxdim, dict) else ratios_resolutions[maxdim]
    mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
    ratio = W / H
    ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
    islandscape = (W >= H)
    if islandscape:
        diff = np.abs(ratio - ref_ratios)
    else:
        diff = np.abs(ratio - (1 / ref_ratios))
    selkey = ref_ratios[np.argmin(diff)]
    res = ratios_resolutions_maxdim[selkey]
    # check patchsize and make sure output resolution is a multiple of patchsize
    if isinstance(patchsize, tuple):
        assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
            patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
        assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
        patchsize = patchsize[0]
    if isinstance(maxdim, int):
        assert max(res) == maxdim
    assert min(res) in mindims
    return res[::-1] if islandscape else res  # return HW


def get_resize_function(maxdim, patch_size, H, W, is_mask=False):
    resolutions_dict = maxdim if isinstance(maxdim, dict) else ratios_resolutions[maxdim]
    if [max(H, W), min(H, W)] in resolutions_dict.values():
        return lambda x: x, np.eye(3), np.eye(3)
    else:
        target_HW = get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

        ratio = W / H
        target_ratio = target_HW[1] / target_HW[0]
        to_orig_crop = np.eye(3)
        to_rescaled_crop = np.eye(3)
        if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
            crop_W = W
            crop_H = H
        elif ratio - target_ratio < 0:
            crop_W = W
            crop_H = int(W / target_ratio)
            to_orig_crop[1, 2] = (H - crop_H) / 2.0
            to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
        else:
            crop_W = int(H * target_ratio)
            crop_H = H
            to_orig_crop[0, 2] = (W - crop_W) / 2.0
            to_rescaled_crop[0, 2] = - (W - crop_W) / 2.0

        crop_op = tvf.CenterCrop([crop_H, crop_W])

        if is_mask:
            resize_op = tvf.Resize(size=target_HW, interpolation=tvf.InterpolationMode.NEAREST_EXACT)
        else:
            resize_op = tvf.Resize(size=target_HW)
        to_orig_resize = np.array([[crop_W / target_HW[1], 0, 0],
                                   [0, crop_H / target_HW[0], 0],
                                   [0, 0, 1]])
        to_rescaled_resize = np.array([[target_HW[1] / crop_W, 0, 0],
                                       [0, target_HW[0] / crop_H, 0],
                                       [0, 0, 1]])

        op = tvf.Compose([crop_op, resize_op])

        return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize


def is_image_extension_known_by_pil(file_path):
    """
    Returns True if the file has a “known” image extension according to PIL.
    Does NOT open the file—it only inspects the extension.
    """
    _, ext = os.path.splitext(file_path)
    valids_exts = PIL.Image.registered_extensions()
    return ext.lower() in valids_exts


def is_valid_pil_image_file(file_path):
    """
    First checks extension, then tries to open/verify the file.
    """
    if not is_image_extension_known_by_pil(file_path):
        return False

    try:
        with PIL.Image.open(file_path) as img:
            img.verify()     # Verify that it’s not truncated/corrupt
        return True
    except (PIL.UnidentifiedImageError, IOError):
        return False
