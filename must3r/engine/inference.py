# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from contextlib import nullcontext
import numpy as np
import itertools
from tqdm import tqdm
import roma
from collections import deque
import math

from must3r.model import ActivationType, apply_activation
import must3r.tools.path_to_dust3r  # noqa
from dust3r.post_process import estimate_focal_knowing_depth


@torch.autocast("cuda", dtype=torch.float32)
def postprocess(pointmaps, pointmaps_activation=ActivationType.NORM_EXP, compute_cam=False):
    out = {}
    channels = pointmaps.shape[-1]
    out['pts3d'] = pointmaps[..., :3]
    out['pts3d'] = apply_activation(out['pts3d'], activation=pointmaps_activation)
    if channels >= 6:
        out['pts3d_local'] = pointmaps[..., 3:6]
        out['pts3d_local'] = apply_activation(out['pts3d_local'], activation=pointmaps_activation)
    if channels == 4 or channels == 7:
        out['conf'] = 1.0 + pointmaps[..., -1].exp()

    if compute_cam:
        batch_dims = out['pts3d'].shape[:-3]
        num_batch_dims = len(batch_dims)
        H, W = out['conf'].shape[-2:]
        pp = torch.tensor((W / 2, H / 2), device=out['pts3d'].device)
        focal = estimate_focal_knowing_depth(out['pts3d_local'].reshape(math.prod(batch_dims), H, W, 3), pp,
                                             focal_mode='weiszfeld')
        out['focal'] = focal.reshape(*batch_dims)

        R, T = roma.rigid_points_registration(
            out['pts3d_local'].reshape(*batch_dims, -1, 3),
            out['pts3d'].reshape(*batch_dims, -1, 3),
            weights=out['conf'].reshape(*batch_dims, -1) - 1.0, compute_scaling=False)

        c2w = torch.eye(4, device=out['pts3d'].device)
        c2w = c2w.view(*([1] * num_batch_dims), 4, 4).repeat(*batch_dims, 1, 1)
        c2w[..., :3, :3] = R
        c2w[..., :3, 3] = T.view(*batch_dims, 3)
        out['c2w'] = c2w
    return out


def split_list(lst, split_size):
    return [lst[i:i + split_size] for i in range(0, len(lst), split_size)]


def split_list_of_tensors(tensor, max_bs):
    tensor_splits = []
    for s in tensor:
        if isinstance(s, list):
            tensor_splits.extend(split_list(s, max_bs))
        else:
            tensor_splits.extend(torch.split(s, max_bs))
    return tensor_splits


def stack_views(true_shape, values, max_bs=None):
    # first figure out what the unique aspect ratios are
    unique_true_shape, inverse_indices = torch.unique(true_shape, dim=0, return_inverse=True)

    # we group the values that share the same AR
    true_shape_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    index_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    value_stacks = [
        [[] for _ in range(unique_true_shape.shape[0])]
        for _ in range(len(values))
    ]

    for i in range(true_shape.shape[0]):
        true_shape_stacks[inverse_indices[i]].append(true_shape[i])
        index_stacks[inverse_indices[i]].append(i)

        for j in range(len(values)):
            value_stacks[j][inverse_indices[i]].append(values[j][i])

    # regroup all None values together (these typically are missing encoder features that'll be recomputed later)
    for i in range(len(true_shape_stacks)):
        # get a mask for each type of value
        none_mask = [[vl == None for vl in v[i]]
                     for v in value_stacks
                     ]
        # apply "or" on all the different types of values
        none_mask = [any([v[j] for v in none_mask]) for j in range(len(true_shape_stacks[i]))]
        if not any(none_mask) or all(none_mask):
            # there was no None or all were None skip
            continue
        not_none_mask = [not x for x in none_mask]

        def get_filtered_list(l, local_mask):
            return [v for v, m in zip(l, local_mask) if m]
        true_shape_stacks.append(get_filtered_list(true_shape_stacks[i], none_mask))
        true_shape_stacks[i] = get_filtered_list(true_shape_stacks[i], not_none_mask)

        index_stacks.append(get_filtered_list(index_stacks[i], none_mask))
        index_stacks[i] = get_filtered_list(index_stacks[i], not_none_mask)

        for j in range(len(value_stacks)):
            value_stacks[j].append(get_filtered_list(value_stacks[j][i], none_mask))
            value_stacks[j][i] = get_filtered_list(value_stacks[j][i], not_none_mask)

    # stack tensors
    true_shape_stacks = [torch.stack(true_shape_stack, dim=0) for true_shape_stack in true_shape_stacks]
    value_stacks = [
        [torch.stack(v, dim=0) if None not in v else v for v in value_stack]
        for value_stack in value_stacks
    ]

    # split all sub-tensors in blocks of max_size = max_bs
    if max_bs is not None:
        true_shape_stacks = split_list_of_tensors(true_shape_stacks, max_bs)

        index_stacks = [torch.tensor(s) for s in index_stacks]
        index_stacks = split_list_of_tensors(index_stacks, max_bs)
        index_stacks = [s.tolist() for s in index_stacks]

        value_stacks = [
            split_list_of_tensors(value_stack, max_bs)
            for value_stack in value_stacks
        ]

    # some cleaning, replace list of None by a single None
    for value_stack in value_stacks:
        for j in range(len(value_stack)):
            if isinstance(value_stack[j], list):
                if None in value_stack[j]:
                    value_stack[j] = None

    return true_shape_stacks, index_stacks, *value_stacks


@torch.no_grad()
def encoder_multi_ar(encoder, imgs, true_shape, verbose=False, max_bs=None, device=None, preserve_gpu_mem=False):
    # forward through dust3r encoder
    if verbose:
        print(f'running encoder')
    nimgs = true_shape.shape[0]
    device = device or true_shape.device
    outdevice = device if not preserve_gpu_mem else "cpu"
    true_shape_stacks, index_stacks, imgs_stacks = stack_views(true_shape, [imgs], max_bs=max_bs)
    x, pos = [None for _ in range(nimgs)], [None for _ in range(nimgs)]

    pbar = tqdm(zip(imgs_stacks, true_shape_stacks, index_stacks),
                disable=not verbose, total=len(imgs_stacks))
    for imgs_stack, true_shape_stack, index_stack in pbar:
        nimgs_stack = imgs_stack.shape[0]
        # encode all images (concat them in the batch dimension for efficiency)
        x_stack, pos_stack = encoder(imgs_stack.to(device), true_shape_stack.to(device))
        for i in range(nimgs_stack):
            x[index_stack[i]] = x_stack[i].to(outdevice)
            pos[index_stack[i]] = pos_stack[i].to(outdevice)

        try:
            pbar.set_postfix({'Mem_r': str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB",
                              'Mem_a': str(int(torch.cuda.max_memory_allocated(device) / (1024 ** 2))) + " MB"})
        except Exception as e:
            pass
    return x, pos


@torch.no_grad()
def inference_multi_ar_batch(encoder, decoder, imgs, true_shape, mem=None, verbose=False,
                             encoder_precomputed_features=None,
                             preserve_gpu_mem=False, post_process_function=lambda x: {'pts3d': x}, device=None,
                             render=False, viser_server=None):
    device = device or true_shape.device
    outdevice = device if not preserve_gpu_mem else "cpu"
    if encoder_precomputed_features is None:
        # already stacked
        x, pos = [], []
        for i in range(len(imgs)):
            xi, posi = encoder(imgs[i].to(device), true_shape[i].to(device))
            x.append(xi)
            pos.append(posi)
    else:
        x, pos = encoder_precomputed_features

    x = [v.unsqueeze(0).to(device) for v in x]
    pos = [v.unsqueeze(0).to(device) for v in pos]
    true_shape = [v.unsqueeze(0).to(device) for v in true_shape]
    imgs = [v.unsqueeze(0).to(device) for v in imgs]

    mem, pointmaps_0 = decoder(x, pos, true_shape, mem, render=render)
    pointmaps_0_pp = []
    for pointmaps_0_i in pointmaps_0:
        pointmaps_0_i = pointmaps_0_i.squeeze(0)
        if post_process_function is not None:
            pointmaps_0_i = post_process_function(pointmaps_0_i)
            pointmaps_0_i = {k: v.to(outdevice) for k, v in pointmaps_0_i.items()}
        else:
            pointmaps_0_i = pointmaps_0_i.to(outdevice)

        pointmaps_0_pp.append(pointmaps_0_i)

    return mem, pointmaps_0_pp


def _remove_from_mem(mem_values, mem_labels, idx):
    to_keep_mask = mem_labels != idx
    B, _, D = mem_values[0].shape
    mem_values = [
        mem_value[to_keep_mask].view(B, -1, D)
        for mem_value in mem_values
    ]
    mem_labels = mem_labels[to_keep_mask].view(B, -1)
    return mem_values, mem_labels


def _restore_label_in_mem(mem_labels, old_idx_to_restore, new_idx_to_remove):
    mask = mem_labels == new_idx_to_remove
    mem_labels[mask] = old_idx_to_restore
    return mem_labels


def _update_in_mem(old_values, new_values, old_labels, new_labels, old_idx, new_idx):
    old_mask = old_labels == old_idx
    new_mask = new_labels == new_idx

    for k in range(len(old_values)):  # iterate over mem_vals
        old_values[k][old_mask] = new_values[k][new_mask]
    return old_values


@torch.no_grad()
def inference_video_multi_ar(encoder, decoder, imgs, true_shape, mem_batches,
                             verbose=False, max_bs=None, encoder_precomputed_features=None,
                             preserve_gpu_mem=False, post_process_function=lambda x: {'pts3d': x}, device=None,
                             return_mem=False, viser_server=None, num_refinements_iterations=0, local_context_size=25,
                             is_keyframe_function=lambda id, res, scene_state: (id % 3 == 0),
                             scene_state=None, scene_state_update_function=lambda res, scene_state: scene_state):
    true_shape = torch.stack(true_shape, dim=0)
    nimgs = true_shape.shape[0]
    device = device or true_shape.device
    if encoder_precomputed_features is None:
        x = [None for _ in range(nimgs)]
        pos = [None for _ in range(nimgs)]
    else:
        x, pos = encoder_precomputed_features

    # use the decoder to update the memory
    # we'll also get first pass pointmaps in pointmaps_0
    # not all images have to update the memory
    if verbose:
        print(f'updating memory')
    mem = None
    mem_batches = [0] + np.cumsum(mem_batches).tolist()
    pointmaps_0 = [None for _ in range(mem_batches[-1])]
    img_labels = {}
    keyframes = set()
    img_ids = [torch.tensor(v) for v in range(nimgs)]
    for _ in range(num_refinements_iterations + 1):
        pbar = tqdm(range(len(mem_batches) - 1), disable=not verbose, total=len(mem_batches) - 1)
        working_memory_idx = deque()
        for i in pbar:
            true_shape_i = true_shape[mem_batches[i]:mem_batches[i + 1]]
            imgs_i = imgs[mem_batches[i]:mem_batches[i + 1]]
            img_ids_i = img_ids[mem_batches[i]:mem_batches[i + 1]]

            # find out if we need to compute some encoder features
            x_i = x[mem_batches[i]:mem_batches[i + 1]]
            pos_i = pos[mem_batches[i]:mem_batches[i + 1]]
            if None in x_i or None in pos_i:
                x_i, pos_i = encoder_multi_ar(encoder, imgs_i, true_shape_i,
                                              verbose=False, max_bs=max_bs, device=device)
                x[mem_batches[i]:mem_batches[i + 1]] = x_i
                pos[mem_batches[i]:mem_batches[i + 1]] = pos_i

            true_shape_stacks_i, index_stacks_i, x_stacks_i, pos_stacks_i, imgs_stacks_i = stack_views(true_shape_i, [x_i, pos_i, imgs_i],
                                                                                                       max_bs=max_bs)

            Nmem_before = get_Nmem(mem)
            new_mem, pointmaps_0_i = inference_multi_ar_batch(
                encoder, decoder, imgs_stacks_i, true_shape_stacks_i, mem, verbose=verbose,
                encoder_precomputed_features=(x_stacks_i, pos_stacks_i),
                preserve_gpu_mem=preserve_gpu_mem, post_process_function=post_process_function, device=device,
                viser_server=viser_server
            )
            # unstack
            pointmaps_0_i = unstack_pointmaps(index_stacks_i, pointmaps_0_i)
            pointmaps_0[mem_batches[i]:mem_batches[i + 1]] = pointmaps_0_i

            new_mem = list(new_mem)  # cast tuple to list
            new_labels = sorted(torch.unique(new_mem[1][:, Nmem_before:]))
            new_labels = [int(v) for v in new_labels]
            mem = new_mem
            local_keyframes = []
            if len(img_labels) == 0:  # at initialization, all keyframes (to simplify things a bit)
                for j, img_id_i in enumerate(img_ids_i):
                    img_id_i = int(img_id_i)
                    img_labels[img_id_i] = new_labels[j]
                    working_memory_idx.append(img_id_i)
                    keyframes.add(img_id_i)
                    local_keyframes.append(True)
                    scene_state = scene_state_update_function(pointmaps_0_i[j], scene_state)
            else:
                # for each image, we will run some checks
                for j, img_id_i in enumerate(img_ids_i):
                    img_id_i = int(img_id_i)

                    if img_id_i in img_labels:  # seen before
                        # do not check again
                        # maybe we want to re-check for non_keyframes (it might be slow though ?)
                        is_keyframe = img_id_i in keyframes
                    else:
                        is_keyframe = is_keyframe_function(img_id_i, pointmaps_0_i[j], scene_state)
                    working_memory_idx.append(img_id_i)
                    local_keyframes.append(is_keyframe)
                    if is_keyframe and img_id_i in img_labels:
                        # if keyframe and seen before, it means we should update it (and remove the tokens)
                        old_label_j = img_labels[img_id_i]
                        if old_label_j != 0:  # for now ref img is not updated
                            mem[0] = _update_in_mem(mem[0], mem[0], mem[1], mem[1], old_label_j, new_labels[j])
                        mem[0], mem[1] = _remove_from_mem(mem[0], mem[1], new_labels[j])
                    elif img_id_i in img_labels:
                        # not a keyframe, has been seen before but doesn't stay in memory
                        # let's just relabel it
                        mem[1] = _restore_label_in_mem(mem[1], img_labels[img_id_i], new_labels[j])
                    else:
                        # never seen before, do nothing
                        img_labels[img_id_i] = new_labels[j]
                        if is_keyframe:
                            keyframes.add(img_id_i)
                            scene_state = scene_state_update_function(pointmaps_0_i[j], scene_state)

            if viser_server is not None:
                viser_server.set_views(img_ids_i, imgs_i, pointmaps_0_i, local_keyframes)
            # cleaning
            # remove local frames that are out of the local window
            while len(working_memory_idx) > local_context_size:
                to_remove_id = working_memory_idx.popleft()
                if to_remove_id not in keyframes:
                    mem[0], mem[1] = _remove_from_mem(mem[0], mem[1], img_labels[to_remove_id])

            # restore mem_nimgs
            mem[2] = len(img_labels)

            try:
                pbar.set_postfix({'Mem_r': str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB",
                                  'Mem_a': str(int(torch.cuda.max_memory_allocated(device) / (1024 ** 2))) + " MB",
                                  "keyframe": len(keyframes),
                                  "Nmem": get_Nmem(mem)})
                if preserve_gpu_mem:
                    torch.cuda.empty_cache()
            except Exception as e:
                pass

        # remove all non keyframes from memory to prepare for the new pass
        assert mem is not None
        while len(working_memory_idx) > 0:
            to_remove_id = working_memory_idx.popleft()
            if to_remove_id not in keyframes:
                mem[0], mem[1] = _remove_from_mem(mem[0], mem[1], img_labels[to_remove_id])

        pbar.close()

    if return_mem:
        return mem, pointmaps_0
    else:
        return pointmaps_0


@torch.no_grad()
def inference_multi_ar(encoder, decoder, imgs, img_ids, true_shape, mem_batches,
                       verbose=False, max_bs=None, to_render=None, encoder_precomputed_features=None,
                       precomputed_mem=None, preserve_gpu_mem=False, post_process_function=lambda x: {'pts3d': x},
                       device=None, return_mem=False, viser_server=None, num_refinements_iterations=0):
    true_shape = torch.stack(true_shape, dim=0)
    nimgs = true_shape.shape[0]
    device = device or true_shape.device
    if encoder_precomputed_features is None:
        x = [None for _ in range(nimgs)]
        pos = [None for _ in range(nimgs)]
    else:
        x, pos = encoder_precomputed_features

    if precomputed_mem is None:
        # use the decoder to update the memory
        # we'll also get first pass pointmaps in pointmaps_0
        # not all images have to update the memory
        if verbose:
            print(f'updating memory')
        mem = None
        mem_batches = [0] + np.cumsum(mem_batches).tolist()
        pointmaps_0 = [None for _ in range(mem_batches[-1])]
        img_labels = {}

        for _ in range(num_refinements_iterations + 1):
            pbar = tqdm(range(len(mem_batches) - 1), disable=not verbose, total=len(mem_batches) - 1)
            for i in pbar:
                true_shape_i = true_shape[mem_batches[i]:mem_batches[i + 1]]
                imgs_i = imgs[mem_batches[i]:mem_batches[i + 1]]
                img_ids_i = img_ids[mem_batches[i]:mem_batches[i + 1]]

                # find out if we need to compute some encoder features
                x_i = x[mem_batches[i]:mem_batches[i + 1]]
                pos_i = pos[mem_batches[i]:mem_batches[i + 1]]
                if None in x_i or None in pos_i:
                    x_i, pos_i = encoder_multi_ar(encoder, imgs_i, true_shape_i,
                                                  verbose=False, max_bs=max_bs, device=device)
                    x[mem_batches[i]:mem_batches[i + 1]] = x_i
                    pos[mem_batches[i]:mem_batches[i + 1]] = pos_i

                true_shape_stacks_i, index_stacks_i, x_stacks_i, pos_stacks_i, imgs_stacks_i = stack_views(true_shape_i, [x_i, pos_i, imgs_i],
                                                                                                           max_bs=max_bs)

                if all([int(img_ids_ij) in img_labels for img_ids_ij in img_ids_i]):
                    update_mem = True
                else:
                    update_mem = False

                new_mem, pointmaps_0_i = inference_multi_ar_batch(
                    encoder, decoder, imgs_stacks_i, true_shape_stacks_i, mem, verbose=verbose,
                    encoder_precomputed_features=(x_stacks_i, pos_stacks_i),
                    preserve_gpu_mem=preserve_gpu_mem, post_process_function=post_process_function, device=device,
                    viser_server=viser_server
                )

                Nmem_before = get_Nmem(mem)
                new_labels = sorted(torch.unique(new_mem[1][:, Nmem_before:]))
                if update_mem:
                    # here we update the tokens of the image
                    assert mem is not None
                    for j, img_id_i in enumerate(img_ids_i):
                        old_label_j = img_labels[int(img_id_i)]
                        if old_label_j == 0:
                            continue  # for now ignore ref img
                        old_mask_j = mem[1] == old_label_j  # old mem_labels correspond to this image
                        new_mask_j = new_mem[1] == new_labels[j]  # new mem_labels correspond to this image
                        # assert torch.sum(old_mask_j) > 0
                        # assert torch.sum(new_mask_j) == torch.sum(old_mask_j)
                        for k in range(len(mem[0])):  # iterate over mem_vals
                            mem[0][k][old_mask_j] = new_mem[0][k][new_mask_j]
                    del new_mem
                else:
                    mem = new_mem
                    for j, img_id_i in enumerate(img_ids_i):
                        img_labels[int(img_id_i)] = int(new_labels[j])

                # unstack
                pointmaps_0_i = unstack_pointmaps(index_stacks_i, pointmaps_0_i)
                pointmaps_0[mem_batches[i]:mem_batches[i + 1]] = pointmaps_0_i
                if viser_server is not None:
                    viser_server.set_views(img_ids_i, imgs_i, pointmaps_0_i, [True for _ in range(len(imgs_i))])

                try:
                    pbar.set_postfix({'Mem_r': str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB",
                                      'Mem_a': str(int(torch.cuda.max_memory_allocated(device) / (1024 ** 2))) + " MB",
                                      "Nmem": get_Nmem(mem)})
                    if preserve_gpu_mem:
                        torch.cuda.empty_cache()
                except Exception as e:
                    pass
            pbar.close()
    else:
        pointmaps_0 = None
        mem = precomputed_mem

    if to_render is not None:
        # with to_render, you can select a list of images to render, instead of rendering all of them
        x = [x[v] for v in to_render]
        pos = [pos[v] for v in to_render]
        true_shape = true_shape[to_render].contiguous()
        imgs = [imgs[v] for v in to_render]
        img_ids = [img_ids[v] for v in to_render]
        nimgs = len(x)

    # render pointmaps using the accumulated memory
    assert mem is not None
    Nmem = get_Nmem(mem)
    if verbose:
        print(f"Nmem={Nmem}")

    if nimgs == 0:
        pointmaps = []
        if return_mem:
            return mem, pointmaps_0, pointmaps
        else:
            return pointmaps_0, pointmaps

    if verbose:
        print(f'rendering {nimgs} extra images')
    true_shape_stacks, index_stacks, x_stacks, pos_stacks, imgs_stacks, img_ids_stacks = stack_views(true_shape,
                                                                                                     [x, pos, imgs,
                                                                                                         img_ids],
                                                                                                     max_bs=max_bs)
    pbar = tqdm(zip(x_stacks, pos_stacks, true_shape_stacks, imgs_stacks, img_ids_stacks),
                disable=not verbose, total=len(x_stacks))

    pointmaps_stacks = []
    for x_stack, pos_stack, true_shape_stack, imgs_stack, img_ids_stack in pbar:
        if x_stack is None or pos_stack is None:
            encoder_precomputed_features = None
        else:
            encoder_precomputed_features = ([x_stack], [pos_stack])

        _, pointmaps_stack = inference_multi_ar_batch(
            encoder, decoder, [imgs_stack], [true_shape_stack], mem, verbose=verbose,
            encoder_precomputed_features=encoder_precomputed_features,
            preserve_gpu_mem=preserve_gpu_mem, post_process_function=post_process_function, device=device,
            render=True, viser_server=viser_server
        )

        pointmaps_stacks.append(pointmaps_stack[0])
        if viser_server is not None:
            tmp_pointmaps_unstack = unstack_pointmaps([torch.arange(img_ids_stack.shape[0])], pointmaps_stack)
            for i in range(img_ids_stack.shape[0]):
                viser_server.set_views([img_ids_stack[i]], [imgs_stack[i]], [tmp_pointmaps_unstack[i]])

        try:
            pbar.set_postfix({'Mem_r': str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB",
                              'Mem_a': str(int(torch.cuda.max_memory_allocated(device) / (1024 ** 2))) + " MB"})
        except Exception as e:
            pass

    pointmaps = unstack_pointmaps(index_stacks, pointmaps_stacks)

    if return_mem:
        return mem, pointmaps_0, pointmaps
    else:
        return pointmaps_0, pointmaps


def get_Nmem(mem):
    if mem is None:
        return 0
    mem_labels = mem[1]
    _, Nmem = mem_labels.shape
    return Nmem


def unstack_pointmaps(index_stacks_i, pointmaps_0_i):
    num_elements = max([max(index_stack_i) for index_stack_i in index_stacks_i]) + 1
    pointmaps_0 = [None for _ in range(num_elements)]
    for pointmaps_0_i_stack, index_stack_i in zip(pointmaps_0_i, index_stacks_i):
        out_pointmaps_0_i = {}
        for k, v in pointmaps_0_i_stack.items():
            for j in range(v.shape[0]):
                if j not in out_pointmaps_0_i:
                    out_pointmaps_0_i[j] = {}
                out_pointmaps_0_i[j][k] = v[j]

        for j in out_pointmaps_0_i.keys():
            pointmaps_0[index_stack_i[j]] = out_pointmaps_0_i[j]
    return pointmaps_0


def groupby_consecutive(data):
    """
    identify groups of consecutive numbers
    """
    if not data:
        return []
    # Sort the data to ensure consecutive numbers are adjacent
    data = sorted(data)
    result = []
    # consecutive numbers have the same (value - index)
    for k, g in itertools.groupby(enumerate(data), lambda x: x[1] - x[0]):
        group = list(map(lambda x: x[1], g))
        result.append((group[0], group[-1]))
    return result


def inference_encoder(encoder, imgs, true_shape_view, max_bs=None, requires_grad=False):
    def encoder_get_context():
        return torch.no_grad() if not requires_grad \
            else nullcontext()

    with encoder_get_context():
        # x, pos = encoder_blk(imgs)
        B, nimgs = imgs.shape[:2]
        if max_bs is None or B * nimgs <= max_bs:
            # encode all images (concat them in the batch dimension for efficiency)
            x, pos = encoder(imgs.view(B * nimgs, *imgs.shape[2:]), true_shape_view)
        else:
            # can also do it slice by slice in case all images don't fit at once
            imgs_view = imgs.view(B * nimgs, *imgs.shape[2:])
            x, pos = [], []
            for imgs_view_slice, true_shape_slice in zip(torch.split(imgs_view, max_bs), torch.split(true_shape_view, max_bs)):
                xi, posi = encoder(imgs_view_slice, true_shape_slice)
                x.append(xi)
                pos.append(posi)
            x = torch.concatenate(x)
            pos = torch.concatenate(pos)
        return x.view(B, nimgs, *x.shape[1:]), pos.view(B, nimgs, *pos.shape[1:])
    return x, pos


def inference(encoder, decoder, imgs, true_shape, mem_batches, verbose=False, max_bs=None,
              train_decoder_skip=0, to_render=None, encoder_requires_grad=False):
    # forward through dust3r encoder
    B, nimgs = imgs.shape[:2]
    true_shape_view = true_shape.view(B * nimgs, 2)
    x, pos = inference_encoder(encoder, imgs, true_shape_view, max_bs, encoder_requires_grad)
    _, _, N, D = x.shape

    # use the decoder to update the memory
    # we'll also get first pass pointmaps in pointmaps_0
    # not all images have to update the memory
    mem = None
    mem_batches = [0] + np.cumsum(mem_batches).tolist()
    # when training for a large number of views, we may want to freeze the decoder for the first views
    outshape = None
    for i in range(train_decoder_skip):
        with torch.no_grad():
            xi = x[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
            posi = pos[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
            true_shapei = true_shape[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
            mem, pt_tmp = decoder(xi, posi, true_shapei, mem, render=False)
            if outshape is None:
                outshape = pt_tmp.shape

    pointmaps_0 = []
    for i in range(train_decoder_skip, len(mem_batches) - 1):
        xi = x[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        posi = pos[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        true_shapei = true_shape[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        mem, pointmaps_0i = decoder(xi, posi, true_shapei, mem, render=False)
        if outshape is None:
            outshape = pointmaps_0i.shape

        pointmaps_0.append(pointmaps_0i)

    # concatenate the first pass pointmaps together
    if len(pointmaps_0) > 0:
        # B, mem_batches[-1] - mem_batches[train_decoder_skip], N, D
        pointmaps_0 = torch.concatenate(pointmaps_0, dim=1)
    else:
        pointmaps_0 = torch.empty((B, 0, *outshape[2:]), dtype=x.dtype, device=x.device)

    if to_render is not None:
        # with to_render, you can select a list of images to render, instead of rendering all of them
        x = x[:, to_render].contiguous()
        pos = pos[:, to_render].contiguous()
        true_shape = true_shape[:, to_render].contiguous()
        imgs = imgs[:, to_render].contiguous()
        nimgs = x.shape[1]

    # render pointmaps using the accumulated memory
    assert mem is not None
    mem_vals, mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = mem
    try:
        _, Nmem, Dmem = mem_vals[-1].shape
    except Exception as e:
        _, Nmem, Dmem = mem_vals[0][-1].shape
    if verbose:
        print(f"Nmem={Nmem}")

    if nimgs == 0:
        pointmaps = torch.empty((B, 0, *pointmaps_0.shape[2:]), dtype=x.dtype, device=x.device)
        return pointmaps_0, pointmaps
    elif max_bs is None or B * nimgs <= max_bs:
        # render all images (concat them in the batch dimension for efficiency)
        _, pointmaps = decoder(x, pos, true_shape, mem, render=True)
    else:
        # can also do it slice by slice in case all images don't fit at once
        x_view = x.view(B * nimgs, N, D)
        pos_view = pos.view(B * nimgs, N, 2)
        true_shape_view = true_shape.view(B * nimgs, 2)
        pointmaps = []

        mem_vals = [mem_vals[i].unsqueeze(1).expand(B, nimgs, Nmem, Dmem).reshape(B * nimgs, Nmem, Dmem)
                    for i in range(len(mem_vals))]
        mem_vals_splits = [torch.split(mem_vals[i], max_bs) for i in range(len(mem_vals))]
        mem_labels = mem_labels.unsqueeze(1).expand(B, nimgs, Nmem).reshape(B * nimgs, Nmem)
        mem_labels_splits = torch.split(mem_labels, max_bs)

        for lidx, (x_view_slice, pos_view_slice, true_shape_view_slice) in enumerate(
            zip(torch.split(x_view, max_bs),
                torch.split(pos_view, max_bs),
                torch.split(true_shape_view, max_bs))):
            memi = [m[lidx] for m in mem_vals_splits]
            mem_labelsi = mem_labels_splits[lidx]
            _, xi_out = decoder(x_view_slice.unsqueeze(1),
                                pos_view_slice.unsqueeze(1),
                                true_shape_view_slice.unsqueeze(1),
                                (memi, mem_labelsi, mem_nimgs, mem_protected_imgs, mem_protected_tokens), render=True)
            pointmaps.append(xi_out.squeeze(1))
        pointmaps = torch.concatenate(pointmaps)
        pointmaps = pointmaps.view(B, nimgs, *pointmaps.shape[1:])

    return pointmaps_0, pointmaps


def concat_preds(out0, out):
    for k in out.keys():
        if k in out0:
            out[k] = torch.concatenate([out0[k], out[k]], dim=1)
    return out
