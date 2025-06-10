# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import numpy as np
import datetime
import PIL.Image

from must3r.retrieval.processor import Retriever
from must3r.retrieval.graph import farthest_point_sampling

from must3r.engine.inference import encoder_multi_ar, inference_multi_ar, postprocess, inference_video_multi_ar
from must3r.model import get_pointmaps_activation
from must3r.tools.image import get_resize_function

import must3r.tools.path_to_dust3r  # noqa
from dust3r.viz import rgb
from dust3r.datasets import ImgNorm

from must3r.slam.model import get_overlap_score, choose_keyframe_from_overlap


class SceneState:
    def __init__(self, x_out, imgs, true_shape, focals, cams2world, image_list):
        self.x_out = x_out
        self.imgs = imgs
        self.true_shape = true_shape
        self.focals = focals
        self.cams2world = cams2world
        self.image_list = image_list


class MUSt3R_Retriever(Retriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _preproc(self, enc_ims, device):
        imids = []
        features = []
        with torch.no_grad():
            for i, enc_im in enumerate(enc_ims):
                feat, _, _ = self.model.forward_local(enc_im.to(device))
                feat = feat.flatten(0, 1).cpu()
                imids.append(i * torch.ones_like(feat[:, 0]).to(dtype=torch.int64))
                features.append(feat)
        features = torch.cat(features, dim=0)
        imids = torch.cat(imids, dim=0)
        return features, imids

    def __call__(self, enc, device):
        # build the database
        feat, ids = self._preproc(enc, device)  # turn encoded image into feats for retrieval
        feat = feat.cpu().numpy()
        ids = ids.cpu().numpy()

        asmk_dataset = self.asmk.build_ivf(feat, ids)
        metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)

        scores = np.empty_like(ranked_scores)
        scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores

        return scores


def load_images(folder_content, size, patch_size=16, verbose=True):
    imgs = []
    transform = ImgNorm

    for path in folder_content:
        rgb_image = PIL.Image.open(path).convert('RGB')
        rgb_image.load()
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(size, patch_size, H, W)
        rgb_tensor = resize_func(transform(rgb_image))
        imgs.append(dict(img=rgb_tensor, true_shape=np.int32([rgb_tensor.shape[-2], rgb_tensor.shape[-1]])))
        if verbose:
            print(f' - adding {path} with resolution {W}x{H} --> {rgb_tensor.shape[-1]}x{rgb_tensor.shape[-2]}')
    return imgs


def slam_is_keyframe(subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile, overlap_mode, id, res, scene_state):
    cam_center = res['c2w'][:3, -1]

    res_unsqueeze = {k: v.unsqueeze(0).unsqueeze(0) for k, v in res.items()}
    overlap_score = get_overlap_score(res_unsqueeze,
                                      scene_state,
                                      cam_center=cam_center,
                                      mode=overlap_mode,
                                      kf_x_subsamp=subsample,
                                      min_conf_keyframe=min_conf_keyframe,
                                      percentile=overlap_percentile,
                                      )
    assert not np.isnan(overlap_score)
    return choose_keyframe_from_overlap(overlap_score, keyframe_overlap_thr, overlap_mode)


def slam_update_scene_state(subsample, min_conf_keyframe, res, scene_state):
    cam_center = res['c2w'][:3, -1]
    msk = res['conf'] > min_conf_keyframe

    if subsample:
        msk = msk[::subsample, ::subsample]
        pts = res['pts3d'][::subsample, ::subsample][msk]
    else:
        pts = res['pts3d'][msk]

    scene_state.add_pts(pts, cam_center=cam_center)
    return scene_state


def must3r_inference(model, retrieval, device, image_size, amp,
                     filelist, num_mem_images, max_bs, init_num_images, batch_num_views, render_once,
                     is_sequence, viser_server=None, num_refinements_iterations=0, verbose=True):
    if amp == "fp16":
        dtype = torch.float16
    elif amp == "bf16":
        assert torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16
    else:
        assert not amp
        dtype = torch.float32

    max_bs = None if max_bs == 0 else max_bs
    encoder, decoder = model
    pointmaps_activation = get_pointmaps_activation(decoder, verbose=verbose)
    def post_process_function(x): return postprocess(x, pointmaps_activation=pointmaps_activation, compute_cam=True)

    if verbose:
        print('loading images')
    time_start = datetime.datetime.now()
    views = load_images(filelist, size=image_size, patch_size=encoder.patch_size, verbose=verbose)
    nimgs = len(views)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'loaded in {ellapsed}')
        print('running inference')
    time_start = datetime.datetime.now()
    if viser_server is not None:
        viser_server.reset(nimgs)

    imgs = [b['img'].to('cpu') for b in views]
    true_shape = [torch.from_numpy(b['true_shape']).to('cpu') for b in views]
    true_shape = torch.stack(true_shape, dim=0)
    nimgs = true_shape.shape[0]

    # select keyframes
    if is_sequence or retrieval is None:
        keyframes = np.linspace(0, len(imgs) - 1, num_mem_images, dtype=int).tolist()
        encoder_precomputed_features = None
    else:
        # run encoder
        with torch.autocast("cuda", dtype=dtype):
            x_start, pos_start = encoder_multi_ar(encoder, imgs, true_shape, verbose=verbose, max_bs=max_bs,
                                                  device=device, preserve_gpu_mem=True)
        encoder_precomputed_features = (x_start, pos_start)

        retriever = MUSt3R_Retriever(retrieval, backbone=encoder, verbose=verbose)
        sim_matrix = retriever([xi.unsqueeze(0).float() for xi in x_start], device=device)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()
        anchor_idx, _ = farthest_point_sampling(1 - sim_matrix, N=num_mem_images, dist_thresh=None)
        sim_matrix = sim_matrix[anchor_idx, :][:, anchor_idx]

        diag = np.diag_indices(num_mem_images)
        sim_matrix[diag[0], diag[1]] = 0
        sim_sum = np.sum(sim_matrix, axis=-1)

        keyframes = [np.argmax(sim_sum)]  # start with image that has the highest overlap
        sim_matrix[:, keyframes[0]] = 0  # invalidate column
        while len(keyframes) != num_mem_images:
            # last_keyframe = keyframes[-1]
            # best_next_image = np.argmax(sim_matrix[last_keyframe])
            sim_matrix_sel = sim_matrix[np.array(keyframes)]
            best_next_image = np.unravel_index(np.argmax(sim_matrix_sel),
                                               sim_matrix_sel.shape)[1]  # we need the column index
            keyframes.append(best_next_image)
            sim_matrix[:, best_next_image] = 0
        keyframes = [anchor_idx[k] for k in keyframes]

    not_keyframes = sorted(set(range(nimgs)).difference(set(keyframes)))
    assert (len(keyframes) + len(not_keyframes)) == nimgs
    # reorder images
    views = [views[i] for i in keyframes] + [views[i] for i in not_keyframes]
    imgs = [b['img'].to(device) for b in views]
    true_shape = [torch.from_numpy(b['true_shape']).to(device) for b in views]
    filenames = [filelist[i] for i in keyframes + not_keyframes]
    img_ids = [torch.tensor(v) for v in keyframes + not_keyframes]

    if encoder_precomputed_features is not None:
        x_start, pos_start = encoder_precomputed_features
        x = [x_start[i] for i in keyframes] + [x_start[i] for i in not_keyframes]
        pos = [pos_start[i] for i in keyframes] + [pos_start[i] for i in not_keyframes]
        encoder_precomputed_features = (x, pos)

    mem_batches = [init_num_images]
    while (sum_b := sum(mem_batches)) != max(num_mem_images, init_num_images):
        size_b = min(batch_num_views, num_mem_images - sum_b)
        mem_batches.append(size_b)

    if render_once:
        to_render = list(range(num_mem_images, nimgs))
    else:
        to_render = None

    with torch.autocast("cuda", dtype=dtype):
        x_out_0, x_out = inference_multi_ar(encoder, decoder, imgs, img_ids, true_shape, mem_batches,
                                            max_bs=max_bs, verbose=verbose, to_render=to_render,
                                            encoder_precomputed_features=encoder_precomputed_features,
                                            device=device, preserve_gpu_mem=True,
                                            post_process_function=post_process_function,
                                            viser_server=viser_server,
                                            num_refinements_iterations=num_refinements_iterations)
    if to_render is not None:
        x_out = x_out_0 + x_out

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'inference in {ellapsed}')
        try:
            print(str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB")
        except Exception as e:
            pass

    if viser_server is not None:
        viser_server.reset_cam_visility()
        viser_server.send_message("Finished")

    if verbose:
        print('preparing pointcloud')
    time_start = datetime.datetime.now()
    focals = []
    cams2world = []
    for i in range(nimgs):
        focals.append(float(x_out[i]['focal'].cpu()))
        cams2world.append(x_out[i]['c2w'].cpu())

    # x_out to cpu
    for i in range(len(x_out)):
        for k in x_out[i].keys():
            x_out[i][k] = x_out[i][k].cpu()

    rgbimg = [rgb(imgs[i], true_shape[i]) for i in range(nimgs)]
    scene = SceneState(x_out, rgbimg, true_shape, focals, cams2world, filenames)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'pointcloud prepared in {ellapsed}')
    return scene


def must3r_inference_video(model, device, image_size, amp,
                           filelist, max_bs, init_num_images, batch_num_views,
                           viser_server=None, num_refinements_iterations=0, local_context_size: int = 25,
                           is_keyframe_function=lambda id, res, scene_state: (id % 3 == 0),
                           scene_state=None,
                           scene_state_update_function=lambda res, scene_state: scene_state,
                           verbose=True):
    if amp == "fp16":
        dtype = torch.float16
    elif amp == "bf16":
        assert torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16
    else:
        assert not amp
        dtype = torch.float32

    max_bs = None if max_bs == 0 else max_bs
    encoder, decoder = model
    pointmaps_activation = get_pointmaps_activation(decoder, verbose=verbose)
    def post_process_function(x): return postprocess(x, pointmaps_activation=pointmaps_activation, compute_cam=True)

    if verbose:
        print('loading images')
    time_start = datetime.datetime.now()
    views = load_images(filelist, size=image_size, patch_size=encoder.patch_size, verbose=verbose)
    nimgs = len(views)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'loaded in {ellapsed}')
        print('running inference')
    time_start = datetime.datetime.now()
    if viser_server is not None:
        viser_server.reset(nimgs)

    imgs = [b['img'].to('cpu') for b in views]
    true_shape = [torch.from_numpy(b['true_shape']).to('cpu') for b in views]
    true_shape = torch.stack(true_shape, dim=0)
    nimgs = true_shape.shape[0]

    imgs = [b['img'].to(device) for b in views]
    true_shape = [torch.from_numpy(b['true_shape']).to(device) for b in views]
    filenames = filelist
    # img_ids = [torch.tensor(v) for v in range(nimgs)]

    mem_batches = [init_num_images]
    while (sum_b := sum(mem_batches)) != nimgs:
        size_b = min(batch_num_views, nimgs - sum_b)
        mem_batches.append(size_b)

    with torch.autocast("cuda", dtype=dtype):
        x_out = inference_video_multi_ar(encoder, decoder, imgs, true_shape, mem_batches,
                                         max_bs=max_bs, verbose=verbose,
                                         device=device, preserve_gpu_mem=True,
                                         post_process_function=post_process_function,
                                         viser_server=viser_server,
                                         num_refinements_iterations=num_refinements_iterations,
                                         local_context_size=local_context_size,
                                         is_keyframe_function=is_keyframe_function,
                                         scene_state=scene_state,
                                         scene_state_update_function=scene_state_update_function)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'inference in {ellapsed}')
        try:
            print(str(int(torch.cuda.max_memory_reserved(device) / (1024 ** 2))) + " MB")
        except Exception as e:
            pass

    if viser_server is not None:
        viser_server.reset_cam_visility()
        viser_server.send_message("Finished")

    if verbose:
        print('preparing pointcloud')
    time_start = datetime.datetime.now()
    focals = []
    cams2world = []
    for i in range(nimgs):
        focals.append(float(x_out[i]['focal'].cpu()))
        cams2world.append(x_out[i]['c2w'].cpu())

    # x_out to cpu
    for i in range(len(x_out)):
        for k in x_out[i].keys():
            x_out[i][k] = x_out[i][k].cpu()

    rgbimg = [rgb(imgs[i], true_shape[i]) for i in range(nimgs)]
    scene = SceneState(x_out, rgbimg, true_shape, focals, cams2world, filenames)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'pointcloud prepared in {ellapsed}')
    return scene
