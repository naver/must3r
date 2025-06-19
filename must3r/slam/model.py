# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import torch
import roma
from PIL import Image
from tqdm import tqdm
import pickle as pkl

from must3r.model import *
from must3r.model import load_model
from must3r.engine.inference import postprocess

import must3r.tools.path_to_dust3r  # noqa
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import _resize_pil_image

from .nns import get_searcher
from .tools import laplacian_smoothing, laplacian_smoothing_with_confidence

# Forward and processing
@torch.no_grad()
def forward_must3r(model,
                   input_views,
                   memory,
                   render=False,
                   device='cuda:0'):

    encoder, decoder = model

    true_shapes = []
    encoded_inputs = []
    pos = []
    for input_view in input_views:
        true_shapei = input_view['true_shape'][None]
        encoded_input, posi = encoder(input_view['img'].to(device),
                                      torch.as_tensor(true_shapei).to(device).view(-1, 2))

        true_shapes.append(torch.tensor(true_shapei, device=device))
        encoded_inputs.append(encoded_input[None])
        pos.append(posi[None])

    temp_memory = memory

    # get pred and updated memory
    new_memory, preds = decoder(encoded_inputs,
                                pos,
                                true_shapes,
                                temp_memory,
                                render=render)

    torch.cuda.empty_cache()

    pointmaps_activation = get_pointmaps_activation(decoder, verbose=False)
    out = []
    for pred in preds:
        out.append(postprocess(pred, pointmaps_activation=pointmaps_activation))

    return out, new_memory


def get_overlap_score(res,
                      overlap_tree,
                      cam_center,  # if needed for ori in NN
                      mode='nn',
                      kf_x_subsamp=None,
                      min_conf_keyframe=1.5,
                      percentile=70,  # 50,
                      eps=1e-9,
                      ):
    outscore = 0.
    if mode == 'meanconf':
        outscore = res['conf'].mean()
    elif mode == 'medianconf':
        outscore = res['conf'].median()
    elif 'nn' in mode:
        pts3d = res['pts3d'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp] if kf_x_subsamp else res['pts3d']
        msk = res['conf'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp] if kf_x_subsamp else res['conf']
        msk = msk > min_conf_keyframe
        outscore = 0.
        if msk.sum() > 0:
            dists = overlap_tree.query(pts3d[msk], cam_center=cam_center)
            if 'norm' in mode:
                depths = res['pts3d_local'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp, -1]
                dists /= depths[msk].cpu().numpy() + eps
            # if ended up in an unseen quadrant, put a number to avoid getting a nan from np.percentile
            dists[np.isposinf(dists)] = np.finfo(dists.dtype).max
            outscore = np.percentile(dists, percentile)
    else:
        raise ValueError(f"Unknown overlap score method {mode}")
    return outscore


def prep_imgs(cimg, img_mean, img_std):
    cimg = cimg.permute(0, 2, 3, 1)  # [B, H, W, 3] color is float3 \in [0,1]
    return (cimg * torch.tensor(img_std, device=cimg.device)) + torch.tensor(img_mean, device=cimg.device)


def preproc_frame(img, idx, res=512, transform=ImgNorm):
    img = Image.fromarray(img)
    W1, H1 = img.size
    halfw, halfh = cx, cy = W1 // 2, H1 // 2
    longsize = res
    if res in [224, 336, 448]: 
        longsize = max(W1, H1) / min(W1, H1) * res  # mindim has to be at least 224
    # resize long side to given size
    img = _resize_pil_image(img, longsize)
    # update size after resize
    W, H = img.size
    cx, cy = W // 2, H // 2

    to_orig_focal = W1 / W

    if res in [224, 336, 448]: # hardcoded from specific training runs, could be automatically detected
        halfw = halfh = res // 2  # square crop
    else:
        # make sure we have multiple of 16
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    return dict(img=transform(img)[None], true_shape=np.int32([img.size[::-1]]), idx=idx, instance=str(idx), offset=np.int32([[cx - halfw, cy - halfh]])), to_orig_focal


def choose_keyframe_from_overlap(overlap_score, thr, overlap_mode):
    if 'nn' in overlap_mode:
        outchoice = overlap_score > thr
    else:
        outchoice = overlap_score < thr
    return outchoice


def mean_focal(seq_focals):  # wavg of seq focals
    out = None
    if len(seq_focals['f']):
        focals = np.array(seq_focals['f'])
        confs = np.array(seq_focals['conf'])
        out = (focals * confs / confs.sum()).sum()
    return out


def build_intr(focal, W, H, device, dtype):
    out = torch.eye(3, device=device, dtype=dtype)
    out[0, 0] = out[1, 1] = float(focal)
    out[:2, -1] = torch.tensor([W / 2, H / 2], device=device, dtype=dtype)
    return out


def get_camera_pose(res, seq_focal, HW, is_first_frame=False, rectify=True):
    device = res['pts3d'].device
    B = res['pts3d'].shape[1]

    H, W = HW
    pp = torch.tensor((W / 2, H / 2), device=device)

    focal = estimate_focal_knowing_depth(res['pts3d_local'][0], pp, focal_mode='weiszfeld')
    focal_ratio = 1.
    if seq_focal is not None and rectify:
        focal_ratio = seq_focal / focal[:, None]  

    if is_first_frame:  # first frame defines the origin of the coordinate system
        R = torch.eye(3, device=device).repeat(B, 1, 1)
        T = torch.zeros(3, device=device).repeat(B, 1)
    else:
        pts3d_local = res['pts3d_local'][0].view(B, -1, 3)
        pts3d_local[..., -1] *= focal_ratio

        R, T = roma.rigid_points_registration(pts3d_local, res['pts3d'][0].view(
            B, -1, 3), weights=res['conf'][0].view(B, -1) - 1., compute_scaling=False)
       
    c2w = torch.eye(4, device=device).repeat(B, 1, 1)
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = T
    return c2w, focal


def get_map(ptscolsconfs, confthr):
    allpts = []
    allcols = []
    for pts, cols, confs in ptscolsconfs:
        msk = confs > confthr
        allpts.append(pts[msk])
        allcols.append(cols[0, 0, msk])
    return torch.cat(allpts), torch.cat(allcols)


def postproc_pred(img,
                  res,
                  is_first_frame,
                  seq_focals,
                  fixed_focal=True,
                  overlap_mode='nn-norm',
                  overlap_tree=None,
                  kf_x_subsamp=None,
                  keyframe_overlap_thr=.15,
                  min_conf_keyframe=1.5,
                  overlap_percentile=70,
                  img_mean=[0.5, 0.5, 0.5],
                  img_std=[0.5, 0.5, 0.5]):

    assert res['pts3d'].shape[0] == 1 and res['pts3d'].shape[1] == 1, "Need to implement batching if ever needed, frames should come 1 by 1 here."

    # assumes frames come in 1 by 1
    # recover depth from local pointmap
    depth = res['pts3d_local'][0, 0, ..., -1]  # query view depth
    conf = res['conf'][0, 0]

    # Mask pointmap and colors
    msk = res['conf'] > min_conf_keyframe

    if kf_x_subsamp: # view subsampling to increase frame rate
        msk = msk[0, 0, ::kf_x_subsamp, ::kf_x_subsamp]
        pts = res['pts3d'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp][msk]
    else:
        pts = res['pts3d'][msk]

    cols = prep_imgs(img['img'], img_mean=img_mean, img_std=img_std)[None]

    c2w = None
    seq_focal = mean_focal(seq_focals) if fixed_focal else None
    c2w, focal = get_camera_pose(res, seq_focal, HW=img['true_shape'][0], is_first_frame=is_first_frame)
    c2w = c2w[0]
    cam_center = c2w[:3, -1]

    res['overlap_score'] = get_overlap_score(res,
                                             overlap_tree,
                                             cam_center=cam_center,
                                             mode=overlap_mode,
                                             kf_x_subsamp=kf_x_subsamp,
                                             min_conf_keyframe=min_conf_keyframe,
                                             percentile=overlap_percentile)

    # Check if memory frame
    iskeyframe = is_first_frame or (
        choose_keyframe_from_overlap(res['overlap_score'], keyframe_overlap_thr, overlap_mode)
        and conf.median() > min_conf_keyframe
    )

    allpts = res['pts3d'][0, 0]

    out = (pts,
           allpts, 
           cols.to(torch.float32),
           depth,
           conf,
           focal,
           torch.inverse(c2w),
           cam_center,
           iskeyframe)
    return out


class MUSt3R_Agent():
    """
    Manage focal length, and smoothing operations for each agent independently
    """

    def __init__(self,
                 # a single focal for all sequence (we should be able to handle zoom but I have not GT to check that)
                 fixed_focal=True,
                 # Smoothing terms
                 smooth_focal_changes=False,  # add smoothing term to focal length changes (only when non fixed_focal)
                 img_mean=[0.5, 0.5, 0.5],
                 img_std=[0.5, 0.5, 0.5]
                 ):
        assert fixed_focal or not smooth_focal_changes, "TODO maybe: online focal smoothing when varying focals"
        self.fixed_focal = fixed_focal
        self.smooth_focal_changes = smooth_focal_changes
        self.img_mean = img_mean
        self.img_std = img_std
        self.reset()

    def reset(self):
        self.seq_focals = {'f': [], 'conf': [], 'to_orig': []}

    def get_true_focal(self):
        out = None
        if len(self.seq_focals['f']) != 0:
            if self.fixed_focal:
                assert np.all(np.array(self.seq_focals['to_orig']) == self.seq_focals['to_orig']
                              [0]), "To orig should be constant for a single true focal"
                out = mean_focal(self.seq_focals) * self.seq_focals['to_orig'][0]
            else:
                out = [ff * tt for ff, tt in zip(self.seq_focals['f'], self.seq_focals['to_orig'])]
        return out

    @torch.no_grad()
    def update(self,
               inp,
               pred,
               is_first_frame,
               overlap_mode,
               overlap_tree,
               kf_x_subsamp,
               keyframe_overlap_thr,
               min_conf_keyframe,
               overlap_percentile,
               to_orig_focal):

        selpts3d, pts3d, colors, depth, conf, focal, w2c, cam_center, iskeyframe = postproc_pred(img=inp,
                                                                                                 res=pred,
                                                                                                 is_first_frame=is_first_frame,
                                                                                                 seq_focals=self.seq_focals,
                                                                                                 fixed_focal=self.fixed_focal,
                                                                                                 overlap_mode=overlap_mode,
                                                                                                 overlap_tree=overlap_tree,
                                                                                                 kf_x_subsamp=kf_x_subsamp,
                                                                                                 keyframe_overlap_thr=keyframe_overlap_thr,
                                                                                                 min_conf_keyframe=min_conf_keyframe,
                                                                                                 overlap_percentile=overlap_percentile,
                                                                                                 img_mean=self.img_mean,
                                                                                                 img_std=self.img_std
                                                                                                 )

        self.seq_focals['f'].append(focal[0].cpu().numpy())
        self.seq_focals['to_orig'].append(to_orig_focal)
        self.seq_focals['conf'].append(conf.mean().cpu().numpy() - 1.)
        outfocal = mean_focal(self.seq_focals) if self.fixed_focal else self.seq_focals['f'][-1]
        return selpts3d, pts3d, colors, depth, conf, outfocal, w2c, cam_center, iskeyframe


class SLAM_MUSt3R():
    """
    Main memory manager, will take care of redistributing input images to respective agent, will gather agent's response and update memory accordingly
    You can save/load memory for/from other runs
    """

    def __init__(self, chkpt,
                 res=512,
                 searcher='kdtree-scipy-quadrant_x2',  # 'kdtree-scipy',
                 overlap_mode='nn-norm',  # 'nn',
                 kf_x_subsamp=4,
                 keyframe_overlap_thr=.15,
                 min_conf_keyframe=1.5,
                 overlap_percentile=70.,
                 rerender=False,
                 fixed_focal=True,
                 keep_memory=False,
                 load_memory=None,
                 num_agents=1,
                 device='cuda:0',
                 num_init_frames=2,
                 ):

        self.agents = [MUSt3R_Agent(fixed_focal) for _ in range(num_agents)]
        self.num_init_frames = num_init_frames

        # inference resolution
        self.res = res
        self.device = device

        # Load Model
        self.transform = ImgNorm
        self.model = load_model(chkpt, device=device)

        # params
        self.searcher = searcher
        self.kf_x_subsamp = kf_x_subsamp
        self.keyframe_overlap_thr = keyframe_overlap_thr
        self.min_conf_keyframe = min_conf_keyframe
        self.overlap_percentile = overlap_percentile
        self.overlap_mode = overlap_mode
        self.rerender = rerender  # once sequence is processed, repredict everything from full memory
        self.keep_memory = keep_memory  # save latent_kf+pointmaps for export

        # Loaded Memory
        self.memory_memory = None
        self.memory_map = None
        self.memory_data = []
        self.memory_overlap_tree = None
        
        if load_memory is not None:
            self.load_memory(load_memory)
        
        self.reset()


    def reset(self):
        # Reset data structures to loaded memory if available else full reinit
        # Sequence data
        self.all_poses = []
        self.all_confs = []
        self.all_timestamps = []
        # Memory and keyframes
        self.memory = self.memory_memory
        self.keyframe_pointmaps = self.memory_data
        self.keyframes = []
        # params for overlap, keyframe selection
        if self.memory_overlap_tree is None:
            self.overlap_tree = get_searcher(self.searcher if 'nn' in self.overlap_mode else 'none')
        else:
            self.overlap_tree = self.memory_overlap_tree
        # Re-rendering
        self.all_images = []
        self.all_pts3d = None

        self.mem_was_loaded = self.memory_memory is not None  # reset to loaded memory
        # Reset all agents
        for i in range(len(self.agents)):
            self.agents[i].reset()

    @property
    def num_mem_frames(self):
        return len(self.keyframes)

    def get_true_focals(self):
        # would be better if agid was a tag instead of an int
        return {agid: agent.get_true_focal() for agid, agent in enumerate(self.agents)}

    def write_all_poses(self, path, filtering_mode=None, filtering_steps=5, filtering_alpha=.5, **tolog):        
        print(f"Writing full trajectory in {path}")
        all_poses = torch.stack(self.all_poses).cpu().numpy()
        timestamps = np.stack(self.all_timestamps).astype(int)
        conf = torch.stack(self.all_confs).cpu().numpy()
        focals = self.get_true_focals()

        if filtering_mode is not None:
            if 'laplacian' in filtering_mode:
                trajectory = all_poses[:, :3, -1]
                if 'conf' in filtering_mode:
                    conf_remap = (conf-conf.min())/(conf.max()-conf.min())  # remap [1,inf] in between [0,1]
                    smoothed_trajectory = laplacian_smoothing_with_confidence(
                        trajectory, conf_remap, alpha=filtering_alpha, iterations=filtering_steps)
                else:
                    smoothed_trajectory = laplacian_smoothing(
                        trajectory, alpha=filtering_alpha, iterations=filtering_steps)
                all_poses[:, :3, -1] = smoothed_trajectory
            else:
                raise ValueError(f"Unknown filtering mode {filtering_mode}")

        np.savez(path, poses=all_poses, timestamps=timestamps, confs=conf, focal=focals, **tolog)

    def save_memory(self, output):
        mem = (self.memory, self.keyframe_pointmaps, self.overlap_tree)
        pkl.dump(mem, open(output, 'wb'))

    def load_memory(self, mem_file):
        self.memory_memory, self.memory_data, self.memory_overlap_tree = pkl.load(open(mem_file, 'rb'))
        self.memory = self.memory_memory
        self.keyframe_pointmaps = self.memory_data
        self.overlap_tree = self.memory_overlap_tree
        self.mem_was_loaded = True

    def fetch_memory_map(self, conf_thr):
        if self.mem_was_loaded:
            self.memory_map = get_map(self.memory_data, conf_thr)
            self.mem_was_loaded = False
        return self.memory_map

    @torch.no_grad()
    def rerender_all_frames(self, maxbs=64):
        assert len(self.agents) == 1, "Multiagent rerender to be managed (different focal lengths)"
        if self.rerender:
            B = len(self.all_images)
            all_imgs = {}
            all_preds = []
            keys_of_interest = ['pts3d', 'pts3d_local', 'conf']
            def keys_of_interest_to_cpu(dd): return {k: dd[k].cpu() for k in keys_of_interest}
            for i in tqdm(range(B // maxbs + 1)):
                sel = self.all_images[slice(i * maxbs, (i + 1) * maxbs)]
                if sel == []:
                    continue
                all_imgs['img'] = torch.cat([im['img'] for im in sel])  # [1,B,3,H,W]
                all_imgs['true_shape'] = np.concatenate([im['true_shape'] for im in sel])  # [1,B,3,H,W]
                pred, _ = forward_must3r(self.model,
                                         [all_imgs],
                                         self.memory,
                                         render=True,
                                         device=self.device)
                all_preds.append(keys_of_interest_to_cpu(pred[0]))
            res = {}
            def cat_pred(k): return torch.cat([pp[k] for pp in all_preds], dim=1)
            for kk in keys_of_interest:
                res[kk] = cat_pred(kk)

            focal = mean_focal(self.agents[0].seq_focals)
            c2w, _ = get_camera_pose(res, focal, HW=all_imgs['true_shape'][0], is_first_frame=False)

            self.all_pts3d = res['pts3d']
            self.all_poses = [cc for cc in c2w]

    @torch.no_grad()
    def __call__(self, img, frame_id, cam_id):
        query_view_prep, to_orig_focal = preproc_frame(img, frame_id, res=self.res, transform=self.transform)

        if self.memory is not None and len(self.all_images) < self.num_init_frames:
            # we have not reached the correct initialization, reset memory
            other_init_images = self.all_images
            frame_ids = self.all_timestamps
            self.reset()
            self.all_images = other_init_images.copy()
        else:
            other_init_images = []
            frame_ids = []

        if self.rerender or (len(self.all_images) < self.num_init_frames):
            self.all_images.append(query_view_prep)

        query_images = other_init_images + [query_view_prep]
        frame_ids += [frame_id]
        preds, newmem = forward_must3r(self.model,
                                       query_images,
                                       self.memory,
                                       device=self.device)

        for query_view_prep, pred, frame_id in zip(query_images, preds, frame_ids):
            HW = query_view_prep['true_shape'][0]
            selpts3d, pts3d, colors, depth, conf, focal, w2c, cam_center, iskeyframe = self.agents[cam_id].update(query_view_prep,
                                                                                                                  pred,
                                                                                                                  self.memory is None,  # is first frame
                                                                                                                  overlap_mode=self.overlap_mode,
                                                                                                                  overlap_tree=self.overlap_tree,
                                                                                                                  kf_x_subsamp=self.kf_x_subsamp,
                                                                                                                  keyframe_overlap_thr=self.keyframe_overlap_thr,
                                                                                                                  min_conf_keyframe=self.min_conf_keyframe,
                                                                                                                  overlap_percentile=self.overlap_percentile,
                                                                                                                  to_orig_focal=to_orig_focal)
            self.all_timestamps.append(frame_id)
            self.all_poses.append(w2c.inverse())
            self.all_confs.append(conf.mean())

            if iskeyframe:
                self.memory = newmem
                self.keyframes.append(frame_id)
                if self.overlap_tree is not None:
                    self.overlap_tree.add_pts(selpts3d, cam_center=cam_center)
                if self.keep_memory:
                    self.keyframe_pointmaps.append([pts3d.cpu(), colors.cpu(), conf.cpu()])

        return pts3d.cpu().numpy(), colors.cpu().numpy(), depth, conf, focal, w2c, HW, iskeyframe
