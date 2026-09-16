# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import time
import threading
import contextlib
import traceback
import numpy as np
import torch
import roma
import cv2
from PIL import Image
from tqdm import tqdm
import pickle as pkl

from must3r.model import load_model, get_pointmaps_activation
from must3r.engine.inference import postprocess

import must3r.tools.path_to_dust3r  # noqa
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import _resize_pil_image

from .nns import get_searcher
from .tools import laplacian_smoothing, laplacian_smoothing_with_confidence


def autocast(dtype):
    """torch.autocast context, disabled for float32 (avoids the per-call 'dtype not supported' warning)"""
    return torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)


# Forward and processing
@torch.no_grad()
def forward_must3r(model,
                   input_views,
                   memory,
                   render=False,
                   device='cuda:0',
                   pointmaps_activation=None):

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

    # get pred and updated memory
    new_memory, preds = decoder(encoded_inputs,
                                pos,
                                true_shapes,
                                memory,
                                render=render)

    if pointmaps_activation is None:
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
            # Single GPU->CPU transfer for all points, then pass numpy to searcher
            pts3d_np = pts3d[msk].cpu().numpy()
            cam_center_np = cam_center.cpu().numpy() if torch.is_tensor(cam_center) else cam_center
            dists = overlap_tree.query(pts3d_np, cam_center=cam_center_np)
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


def _crop_half_sizes(res, cx, cy):
    if res in [224, 336, 448]:  # hardcoded from specific training runs, could be automatically detected
        halfw = halfh = res // 2  # square crop
    else:
        # make sure we have multiple of 16
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    return halfw, halfh


def _long_size(res, W1, H1):
    longsize = res
    if res in [224, 336, 448]:
        longsize = max(W1, H1) / min(W1, H1) * res  # mindim has to be at least 224
    return longsize


def preproc_frame(img, idx, res=512, transform=ImgNorm):
    """Reference preprocessing (PIL): resize long side to res, center crop to a multiple of 16"""
    img = Image.fromarray(img)
    W1, H1 = img.size
    img = _resize_pil_image(img, _long_size(res, W1, H1))
    # update size after resize
    W, H = img.size
    cx, cy = W // 2, H // 2
    to_orig_focal = W1 / W

    halfw, halfh = _crop_half_sizes(res, cx, cy)
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    view = dict(img=transform(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=idx,
                instance=str(idx),
                offset=np.int32([[cx - halfw, cy - halfh]]))
    return view, to_orig_focal


def preproc_frame_fast(img, idx, res=512):
    """Same as preproc_frame with ImgNorm but using cv2 instead of PIL (faster for live camera streams)."""
    H1, W1 = img.shape[:2]
    longsize = _long_size(res, W1, H1)

    S = max(W1, H1)
    scaling = longsize / S
    new_W, new_H = int(round(W1 * scaling)), int(round(H1 * scaling))
    interp = cv2.INTER_LANCZOS4 if scaling < 1 else cv2.INTER_CUBIC
    img_resized = cv2.resize(img, (new_W, new_H), interpolation=interp)

    W, H = new_W, new_H
    cx, cy = W // 2, H // 2
    to_orig_focal = W1 / W

    halfw, halfh = _crop_half_sizes(res, cx, cy)
    img_cropped = img_resized[cy - halfh:cy + halfh, cx - halfw:cx + halfw]
    out_H, out_W = img_cropped.shape[:2]

    # ToTensor + Normalize(0.5, 0.5, 0.5) in one shot, avoiding PIL
    img_t = torch.from_numpy(np.ascontiguousarray(img_cropped)).permute(2, 0, 1).to(dtype=torch.float32)
    img_t.div_(255.0).sub_(0.5).div_(0.5)

    view = dict(img=img_t[None],
                true_shape=np.int32([(out_H, out_W)]),
                idx=idx,
                instance=str(idx),
                offset=np.int32([[cx - halfw, cy - halfh]]))
    return view, to_orig_focal


def choose_keyframe_from_overlap(overlap_score, thr, overlap_mode):
    if 'nn' in overlap_mode:
        outchoice = overlap_score > thr
    else:
        outchoice = overlap_score < thr
    return outchoice


def mean_focal(seq_focals):  # wavg of seq focals
    focals = np.array(seq_focals['f'])
    confs = np.array(seq_focals['conf'])
    # 'f' is appended before 'conf' by the live thread, a concurrent reader (MemoryRefresher) may see one more focal
    n = min(len(focals), len(confs))
    if n == 0:
        return None
    focals, confs = focals[:n], confs[:n]
    return (focals * confs / confs.sum()).sum()


def rigid_inverse(c2w):
    """Invert a rigid-body 4x4 transform [R|t; 0 0 0 1] efficiently."""
    w2c = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -(R.T @ t)
    return w2c


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
        pts3d_local = res['pts3d_local'][0].view(B, -1, 3).float()
        pts3d_local[..., -1] *= focal_ratio
        with torch.autocast("cuda", enabled=False):  # registration in full precision
            R, T = roma.rigid_points_registration(pts3d_local, res['pts3d'][0].view(B, -1, 3).float(),
                                                  weights=res['conf'][0].view(B, -1).float() - 1., compute_scaling=False)

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
                  last_kf_id,
                  seq_focals,
                  fixed_focal=True,
                  overlap_mode='nn-norm',
                  overlap_tree=None,
                  kf_x_subsamp=None,
                  keyframe_overlap_thr=.15,
                  min_conf_keyframe=1.5,
                  min_keyframe_spacing=1,
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

    if kf_x_subsamp:  # view subsampling to increase frame rate
        msk = msk[0, 0, ::kf_x_subsamp, ::kf_x_subsamp]
        pts = res['pts3d'][0, 0, ::kf_x_subsamp, ::kf_x_subsamp][msk]
    else:
        pts = res['pts3d'][msk]

    cols = prep_imgs(img['img'], img_mean=img_mean, img_std=img_std)[None]

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
        and (last_kf_id is None or (img['idx'] - last_kf_id) >= min_keyframe_spacing)
        and conf.median() > min_conf_keyframe
    )

    allpts = res['pts3d'][0, 0]

    out = (pts,
           allpts,
           cols.to(torch.float32),
           depth,
           conf,
           focal,
           rigid_inverse(c2w),
           cam_center,
           iskeyframe)
    return out


class GpuGate:
    """
    Two-priority mutex around GPU-heavy sections.
    The live loop uses high(): it waits only for an in-flight low() section to finish.
    Background work uses low(): it only starts when nobody is running and no high-priority
    caller is waiting, i.e. right when the live loop releases the GPU. This is what lets the
    background refresh fill the idle gaps (CPU postproc, GUI, frame grabbing) of the live loop
    without ever being able to starve it.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._busy = False
        self._hi_waiting = 0

    def _release(self):
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    @contextlib.contextmanager
    def high(self):
        with self._cv:
            self._hi_waiting += 1
            while self._busy:
                self._cv.wait()
            self._hi_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextlib.contextmanager
    def low(self):
        with self._cv:
            while self._busy or self._hi_waiting > 0:
                self._cv.wait()
            self._busy = True
        try:
            yield
        finally:
            self._release()


class MemoryRefresher(threading.Thread):
    """
    Background worker that re-renders memory frames against the full current memory.
    It runs FIFO passes over the memory frames (oldest first). Once a pass is done it waits
    until new frames were added to the memory (or the memory was reset) and loops again.
    Each refreshed frame gets a new pointmap/conf and camera pose, the latent memory tokens are left untouched
    (see SLAM_MUSt3R.refresh_memory_frames).

    batch_size: number of memory frames re-rendered in a single forward
    duty: fraction of wall-clock time the refresher is allowed to hold the GPU (0 < duty <= 1),
          after a job of duration t it sleeps t * (1 - duty) / duty
    """

    def __init__(self, slam, batch_size=1, duty=0.5, poll=0.05):
        super().__init__(name='MemoryRefresher', daemon=True)
        assert 0. < duty <= 1., "duty must be in ]0, 1]"
        self.slam = slam
        self.batch_size = max(1, int(batch_size))
        self.duty = duty
        self.poll = poll
        self._stop_event = threading.Event()
        self.num_passes = 0

    def stop(self, join=True, timeout=5.):
        self._stop_event.set()
        if join and self.is_alive():
            self.join(timeout=timeout)

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def _mem_state(self):
        with self.slam.state_lock:
            return len(self.slam.keyframes), self.slam._mem_generation

    def run(self):
        while not self.stopped:
            try:
                n, gen = self._mem_state()
                if n < 2:
                    # nothing to refine against
                    self._stop_event.wait(self.poll)
                    continue

                # One FIFO pass over the frames that are in memory at the start of the pass
                k = 0
                while k < n and not self.stopped:
                    if self._mem_state()[1] != gen:
                        break  # memory was reset, restart from scratch
                    ks = list(range(k, min(k + self.batch_size, n)))
                    t0 = time.perf_counter()
                    self.slam.refresh_memory_frames(ks)
                    dt = time.perf_counter() - t0
                    k += len(ks)
                    if self.duty < 1.:
                        self._stop_event.wait(dt * (1. - self.duty) / self.duty)
                else:
                    self.num_passes += 1

                # Pass done: wait for new frames in memory (or a reset) before looping again
                while not self.stopped:
                    n2, gen2 = self._mem_state()
                    if n2 != n or gen2 != gen:
                        break
                    self._stop_event.wait(self.poll)
            except Exception:
                print("[MemoryRefresher] error, retrying:")
                traceback.print_exc()
                self._stop_event.wait(1.)


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
               last_kf_id,
               overlap_mode,
               overlap_tree,
               kf_x_subsamp,
               keyframe_overlap_thr,
               min_conf_keyframe,
               min_keyframe_spacing,
               overlap_percentile,
               to_orig_focal):

        selpts3d, pts3d, colors, depth, conf, focal, w2c, cam_center, iskeyframe = postproc_pred(img=inp,
                                                                                                 res=pred,
                                                                                                 is_first_frame=is_first_frame,
                                                                                                 last_kf_id=last_kf_id,
                                                                                                 seq_focals=self.seq_focals,
                                                                                                 fixed_focal=self.fixed_focal,
                                                                                                 overlap_mode=overlap_mode,
                                                                                                 overlap_tree=overlap_tree,
                                                                                                 kf_x_subsamp=kf_x_subsamp,
                                                                                                 keyframe_overlap_thr=keyframe_overlap_thr,
                                                                                                 min_conf_keyframe=min_conf_keyframe,
                                                                                                 min_keyframe_spacing=min_keyframe_spacing,
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
                 min_keyframe_spacing=1,
                 overlap_percentile=70.,
                 rerender=False,
                 fixed_focal=True,
                 keep_memory=False,
                 load_memory=None,
                 num_agents=1,
                 device='cuda:0',
                 num_init_frames=2,
                 force_first_keyframes=None,
                 no_model=False,
                 amp=torch.float32,
                 ):

        self.agents = [MUSt3R_Agent(fixed_focal) for _ in range(num_agents)]
        self.num_init_frames = num_init_frames
        self.force_first_keyframes = force_first_keyframes

        # inference resolution
        self.res = res
        self.device = device
        self.amp = amp

        # Load Model
        self.transform = ImgNorm
        self.model = None if no_model else load_model(chkpt, device=device)
        self._pointmaps_activation = get_pointmaps_activation(self.model[1]) if self.model is not None else None

        # params
        self.searcher = searcher
        self.kf_x_subsamp = kf_x_subsamp
        self.keyframe_overlap_thr = keyframe_overlap_thr
        self.min_conf_keyframe = min_conf_keyframe
        self.min_keyframe_spacing = min_keyframe_spacing
        self.overlap_percentile = overlap_percentile
        self.overlap_mode = overlap_mode
        self.rerender = rerender  # once sequence is processed, repredict everything from full memory
        self.keep_memory = keep_memory  # save latent_kf+pointmaps for export

        # Loaded Memory
        self.memory_memory = None
        self.memory_map = None
        self.memory_data = []
        self.memory_overlap_tree = None
        self.memory_keyframe_idx = None  # indices of the keyframes kept in 1st pass

        # Concurrency (live loop vs background memory refresh, see MemoryRefresher)
        self.gpu_gate = GpuGate()  # orders GPU-heavy sections, live loop has priority
        self.state_lock = threading.RLock()  # protects memory / keyframe bookkeeping
        self._mem_generation = 0  # bumped at every reset so background work can detect it
        self.n_refreshed = 0  # total number of memory frames refreshed so far

        if load_memory is not None:
            self.load_memory(load_memory)

        self.reset()

    def reset(self):
        # Reset data structures to loaded memory if available else full reinit
        with self.state_lock:
            self._reset()

    def _reset(self):
        self._mem_generation += 1
        # Sequence data
        self.all_poses = []
        self.all_confs = []
        self.all_timestamps = []
        # Memory and keyframes
        self.memory = self.memory_memory
        self.keyframe_pointmaps = list(self.memory_data)
        self.keyframes = []
        # Per memory frame data needed to refresh it later (aligned with self.keyframes)
        self.keyframe_views = []  # preprocessed input view
        self.keyframe_cam_ids = []  # agent that produced it
        self.keyframe_c2w = {}  # frame_id -> latest c2w estimate
        self.refresh_results = []  # refreshed frames not yet consumed by the caller (see pop_refresh_results)
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
                    conf_remap = (conf - conf.min()) / (conf.max() - conf.min())  # remap [1,inf] in between [0,1]
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
        mem = (self.memory, self.keyframe_pointmaps, self.overlap_tree, self.keyframes)
        pkl.dump(mem, open(output, 'wb'))

    def load_memory(self, mem_file):
        mem = pkl.load(open(mem_file, 'rb'))
        if len(mem) == 3:  # older memory files without the keyframe indices
            mem = (*mem, None)
        self.memory_memory, self.memory_data, self.memory_overlap_tree, self.memory_keyframe_idx = mem
        self.memory = self.memory_memory
        self.keyframe_pointmaps = list(self.memory_data)
        self.overlap_tree = self.memory_overlap_tree
        self.mem_was_loaded = True

    def fetch_memory_map(self, conf_thr):
        if self.mem_was_loaded:
            self.memory_map = get_map(self.memory_data, conf_thr)
            self.mem_was_loaded = False
        return self.memory_map

    # ------------------------------------------------------------------
    # Background memory refresh
    def pop_refresh_results(self):
        """Return (and clear) the list of memory frames refreshed since last call."""
        with self.state_lock:
            out = self.refresh_results
            self.refresh_results = []
        return out

    def _postproc_refresh(self, pred, views, frame_ids, cam_ids, ks, seq_focals, img_mean, img_std):
        results = []
        for j, (view, fid, cid, seq_focal) in enumerate(zip(views, frame_ids, cam_ids, seq_focals)):
            res_j = {kk: pred[kk][:, j:j + 1] for kk in ('pts3d', 'pts3d_local', 'conf')}
            HW = view['true_shape'][0]
            c2w, focal = get_camera_pose(res_j, seq_focal, HW=HW, is_first_frame=False)
            c2w = c2w[0]
            cols = prep_imgs(view['img'], img_mean=img_mean, img_std=img_std)[None].to(torch.float32)
            results.append(dict(frame_id=fid,
                                cam_id=cid,
                                k=ks[j],
                                pts3d=res_j['pts3d'][0, 0].cpu(),
                                colors=cols.cpu(),
                                conf=res_j['conf'][0, 0].cpu(),
                                c2w=c2w,
                                w2c=rigid_inverse(c2w).cpu(),
                                focal=float(seq_focal if seq_focal is not None else focal[0]),
                                HW=HW))
        return results

    @torch.no_grad()
    def refresh_memory_frames(self, ks):
        """
        Re-render memory frames ks (indices in self.keyframes) against the full current memory, then
        atomically update their camera pose (all_poses / keyframe_c2w) and pointmaps. This is a render-only
        forward: the latent memory tokens are left untouched.
        The forward runs in a low-priority GPU section so the live loop is never starved.
        Results are queued for the viewer (see pop_refresh_results). Returns the number of refreshed frames.
        """
        assert self.model is not None
        with self.state_lock:
            gen = self._mem_generation
            snapshot = self.memory
            if snapshot is None or len(self.keyframes) < 2:
                return 0
            ks = [k for k in ks if 0 <= k < len(self.keyframes)]
            if len(ks) == 0:
                return 0
            cam_ids = [self.keyframe_cam_ids[k] for k in ks]
            views = [self.keyframe_views[k] for k in ks]
            frame_ids = [self.keyframes[k] for k in ks]
            seq_focals = [mean_focal(self.agents[c].seq_focals) if self.agents[c].fixed_focal else None
                          for c in cam_ids]
            img_mean, img_std = self.agents[0].img_mean, self.agents[0].img_std

        true_shapes = np.concatenate([v['true_shape'] for v in views])
        assert (true_shapes == true_shapes[0]).all(), "batched refresh requires identical image shapes"
        batch = {'img': torch.cat([v['img'] for v in views]), 'true_shape': true_shapes}

        with self.gpu_gate.low():
            with autocast(self.amp):
                preds, _ = forward_must3r(self.model, [batch], snapshot,
                                          render=True,  # memory tokens untouched
                                          device=self.device,
                                          pointmaps_activation=self._pointmaps_activation)
            results = self._postproc_refresh(preds[0], views, frame_ids, cam_ids, ks, seq_focals, img_mean, img_std)
            torch.cuda.synchronize()  # leave the GPU idle for the live loop

        with self.state_lock:
            if gen != self._mem_generation or self.memory is None:
                return 0  # memory was reset while we were computing, discard
            # Poses and pointmaps
            n_loaded = len(self.memory_data)
            for r in results:
                fid = r['frame_id']
                self.keyframe_c2w[fid] = r['c2w']
                if fid in self.all_timestamps:
                    idx = self.all_timestamps.index(fid)
                    self.all_poses[idx] = r['c2w'].to(self.all_poses[idx].device, self.all_poses[idx].dtype)
                if self.keep_memory and n_loaded + r['k'] < len(self.keyframe_pointmaps):
                    self.keyframe_pointmaps[n_loaded + r['k']] = [r['pts3d'], r['colors'], r['conf']]
            self.refresh_results.extend(results)
            self.n_refreshed += len(results)
        return len(results)

    @torch.no_grad()
    def rerender_all_frames(self, maxbs=64, update_viz=True):
        assert len(self.agents) == 1, "Multiagent rerender to be managed (different focal lengths)"
        if self.rerender:
            B = len(self.all_images)
            all_imgs = {}
            all_preds = []
            keys_of_interest = ['pts3d', 'pts3d_local', 'conf']
            def keys_of_interest_to_cpu(dd): return {k: dd[k].cpu() for k in keys_of_interest}
            n_loaded = len(self.memory_data)
            for i in tqdm(range(B // maxbs + 1)):
                sel = self.all_images[slice(i * maxbs, (i + 1) * maxbs)]
                if sel == []:
                    continue
                all_imgs['img'] = torch.cat([im['img'] for im in sel])  # [1,B,3,H,W]
                all_imgs['true_shape'] = np.concatenate([im['true_shape'] for im in sel])  # [1,B,3,H,W]
                with autocast(self.amp):
                    pred, _ = forward_must3r(self.model,
                                             [all_imgs],
                                             self.memory,
                                             render=True,
                                             device=self.device,
                                             pointmaps_activation=self._pointmaps_activation)
                pred = pred[0]
                all_preds.append(keys_of_interest_to_cpu(pred))
                if update_viz and self.keep_memory:
                    # update pointmaps for vizu, leave memory tokens untouched
                    for fid, qview in enumerate(sel):
                        if qview['idx'] in self.keyframes:
                            to_updt = n_loaded + self.keyframes.index(qview['idx'])
                            self.keyframe_pointmaps[to_updt] = [pred['pts3d'][0, fid].cpu(),
                                                                self.keyframe_pointmaps[to_updt][1],
                                                                pred['conf'][0, fid].cpu()]
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
        # Use fast cv2 path for RGB frames with the default normalization, fall back to PIL otherwise
        if isinstance(img, np.ndarray) and self.transform == ImgNorm:
            query_view_prep, to_orig_focal = preproc_frame_fast(img, frame_id, res=self.res)
        else:
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

        with self.gpu_gate.high(), autocast(self.amp):
            preds, newmem = forward_must3r(self.model,
                                           query_images,
                                           self.memory,
                                           device=self.device,
                                           pointmaps_activation=self._pointmaps_activation)

        mem_updated = False
        # the first frame defines the origin, the following ones (init frames forwarded together) are registered to it
        is_first_frame = self.memory is None
        for query_view_prep, pred, frame_id in zip(query_images, preds, frame_ids):
            HW = query_view_prep['true_shape'][0]
            selpts3d, pts3d, colors, depth, conf, focal, w2c, cam_center, iskeyframe = self.agents[cam_id].update(query_view_prep,
                                                                                                                  pred,
                                                                                                                  is_first_frame,
                                                                                                                  max(self.keyframes) if self.keyframes else None,  # last kf id
                                                                                                                  overlap_mode=self.overlap_mode,
                                                                                                                  overlap_tree=self.overlap_tree,
                                                                                                                  kf_x_subsamp=self.kf_x_subsamp,
                                                                                                                  keyframe_overlap_thr=self.keyframe_overlap_thr,
                                                                                                                  min_conf_keyframe=self.min_conf_keyframe,
                                                                                                                  min_keyframe_spacing=self.min_keyframe_spacing,
                                                                                                                  overlap_percentile=self.overlap_percentile,
                                                                                                                  to_orig_focal=to_orig_focal)
            c2w = rigid_inverse(w2c)
            with self.state_lock:
                self.all_timestamps.append(frame_id)
                self.all_poses.append(c2w)
                self.all_confs.append(conf.mean())

                if iskeyframe or (self.force_first_keyframes is not None
                                  and len(self.all_timestamps) < self.force_first_keyframes):
                    mem_updated = True
                    is_first_frame = False
                    self.keyframes.append(frame_id)
                    self.keyframe_views.append(query_view_prep)
                    self.keyframe_cam_ids.append(cam_id)
                    self.keyframe_c2w[frame_id] = c2w
                    if self.overlap_tree is not None:
                        self.overlap_tree.add_pts(selpts3d, cam_center=cam_center)
                    if self.keep_memory:
                        self.keyframe_pointmaps.append([pts3d.cpu(), colors.cpu(), conf.cpu()])

        if mem_updated:
            with self.state_lock:
                # the background refresher never modifies the tokens, so the forward output is the new memory
                self.memory = newmem

        return pts3d.cpu().numpy(), colors.cpu().numpy(), depth, conf, focal, w2c, HW, iskeyframe
