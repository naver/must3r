#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
"""
MUSt3R online visual odometry demo (open3d GUI)

Examples:
  python slam.py --chkpt checkpoints/must3r_512.pth --input cam:0 --res 512 --subsamp 8 --gui --searcher pykdtree-quadrant_x2 --viz_conf 2.5 --amp fp16 --refresh_memory
  python slam.py --chkpt checkpoints/must3r_512.pth --input /path/to/video.mp4 --res 512 --subsamp 4 --output /path/to/export
"""
import numpy as np
import torch
import cv2
import time
import os
import pickle as pkl
from tqdm import tqdm
import logging as log
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from must3r.model import get_dtype
from must3r.model.blocks.attention import has_xformers, toggle_memory_efficient_attention
from must3r.slam.data import AutoMultiLoader
from must3r.slam.model import SLAM_MUSt3R, MemoryRefresher, rigid_inverse

try:
    o3d.cuda
except AttributeError:
    print('Fallback to open3d.cpu')
    o3d.cuda = o3d.cpu  # workaround for module open3d has no attribute cuda


MB = 1024.**2
camcols = [  # different frustrum colors for each agent
    [.1, .1, .9],  # blue
    [1., .5, 0.],  # orange
    [.5, 0., .5],  # purple
    [0., 1., 1.],  # cyan
]
KEYFRAME_COLOR = [0.1, 0.9, 0.1]
MIN_CONF_DISPLAY = 1.1  # frames with a lower mean confidence are not added to the 3D view

SKIP_EVERY = 1


def grab_frame(camera):
    read = camera.read()
    frame = read[1]
    camid = 0 if len(read) != 3 else read[2]

    for _ in range(SKIP_EVERY - 1):
        camera.grab()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None
    return img, camid


def img2o3d(im):
    res = o3d.cuda.pybind.geometry.Image(im.astype(np.uint8))
    return res


def colorize_depth(depth, mode='grayscale'):
    if depth is None:
        return depth
    colored_depth = None
    if mode == 'grayscale':
        mind, maxd = depth.min(), depth.max()
        depth = 255. * (depth - mind) / (maxd - mind + 1e-9)
        colored_depth = torch.stack([depth, depth, depth], dim=-1)
    elif mode == 'conf':
        colored_depth = depth - 1.0
    else:
        raise ValueError(f"Unknown colorization mode {mode}.")
    return colored_depth.cpu().numpy()


def make_pcd(pts3d, colors):
    """[N, 3] points and colors (torch or numpy) -> open3d tensor PointCloud, None if empty"""
    if pts3d.shape[0] == 0:
        return None
    dtype = o3d.core.float32
    pts3d = pts3d.cpu().numpy() if torch.is_tensor(pts3d) else pts3d
    colors = colors.cpu().numpy() if torch.is_tensor(colors) else colors
    pcd = o3d.cuda.pybind.t.geometry.PointCloud()
    pcd.point.positions = o3d.cuda.pybind.core.Tensor(pts3d, dtype=dtype)
    pcd.point.colors = o3d.cuda.pybind.core.Tensor(colors, dtype=dtype)
    return pcd


def make_frustrum(HW, focal, w2c, color, scale=0.075):
    H, W = HW
    K = np.eye(3)
    K[0, 0] = K[1, 1] = focal
    K[0, -1] = W / 2
    K[1, -1] = H / 2
    w2c = w2c.cpu().numpy() if torch.is_tensor(w2c) else w2c
    frustrum = o3d.geometry.LineSet.create_camera_visualization(W, H, intrinsic=K, extrinsic=w2c, scale=scale)
    frustrum.paint_uniform_color(color)
    return frustrum


# Open3D classes
# Processing
class PipelineModel:
    """Controls IO. Methods run
    in worker threads."""

    def __init__(self,
                 model,
                 camera,
                 update_view,
                 device=None,
                 res=512,
                 show_cameras=True,
                 viz_conf=2.5,  # conf thresh for pts3d viz
                 ):
        """Initialize.
        Args:
            model (SLAM_MUSt3R): slam model, called on every frame
            camera: frame source (see must3r.slam.data)
            update_view (callback): Callback to update display elements for a
                frame.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
            res: maxdim of the images in pixels
            show_camera: display camera locations with the 3D model
            viz_conf: confidence threshold of the displayed 3D points
        """
        self.res = res
        self.show_cameras = show_cameras
        self.viz_conf = viz_conf
        self.update_view = update_view
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)

        self.cv_capture = threading.Condition()  # condition variable
        self.query_view = None
        self.must3r = model
        self.camera = camera
        self.depth_in_color = None
        self.conf_in_color = None

        self.flag_start = False

        self.keyframes_data = []  # [tag, geometry] of all keyframes (pointcloud + frustrum)
        self._kf_data_cursor = 0  # keyframes_data[:cursor] were already sent to the view
        self.keyframe_focals = []
        self.keyframe_confs = []

        self.pcd_frame = None
        self.frustrum = None
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Process')
        self.flag_exit = False

    @property
    def max_points(self):
        return 10 * self.res**2

    def get_pred(self, frame_id, camid):
        return self.must3r(self.query_view, frame_id, camid)

    def wait_if_needed(self, frame_id):
        pass

    def reset(self):
        self.query_view = None
        self.keyframes_data = []
        self._kf_data_cursor = 0
        self.must3r.reset()
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def collect_refreshed_keyframes(self):
        """
        Fetch memory frames refreshed by the background MemoryRefresher and turn them into
        (tag, geometry) pairs replacing the keyframe pointcloud and camera in the viewer.
        Also patches self.keyframes_data so a GUI re-init shows the refreshed version.
        """
        replace_data = []
        refreshed = self.must3r.pop_refresh_results()
        if len(refreshed) == 0:
            return replace_data
        tag2idx = {tag: i for i, (tag, _) in enumerate(self.keyframes_data)}
        for r in refreshed:
            fid = r['frame_id']
            msk = r['conf'] > self.viz_conf
            pcd = make_pcd(r['pts3d'][msk], r['colors'][0, 0, msk])
            frustrum = make_frustrum(r['HW'], r['focal'], r['w2c'], KEYFRAME_COLOR) if self.show_cameras else None
            for tag, geom in ((f'{fid}_kpcd', pcd), (f'{fid}_kfrustrum', frustrum)):
                idx = tag2idx.get(tag, None)
                if idx is None:
                    continue  # never displayed (e.g. memory got reset), nothing to replace
                self.keyframes_data[idx][1] = geom
                if idx < self._kf_data_cursor:
                    replace_data.append((tag, geom))  # already on screen -> swap it
                # else: not yet sent to the view, it will be added with the refreshed geometry
        return replace_data

    def run(self):
        """Run pipeline."""
        frame_id = 0
        t1 = time.perf_counter()
        memory_map = None
        next_frame_future = None
        c2w = None
        while not self.flag_exit:
            if not self.flag_start:
                if self.query_view is not None:
                    # Reset camera and memory
                    self.reset()
                    frame_id = 0
                    next_frame_future = None
                    c2w = None
            else:
                # Use prefetched frame if available, otherwise grab now
                if next_frame_future is not None:
                    self.query_view, camid = next_frame_future.result()
                    next_frame_future = None
                else:
                    self.query_view, camid = grab_frame(self.camera)
                if self.query_view is None:
                    # End of video file, wait for the user to toggle start
                    self.query_view = 'to_reset'
                    continue

                # Prefetch next frame while GPU processes current one
                next_frame_future = self.executor.submit(grab_frame, self.camera)

                # Preproc, Forward, Postproc
                pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = self.get_pred(frame_id, camid)

                # Memory frames refreshed in the background since last iteration: rebuild their geometry
                replace_data = self.collect_refreshed_keyframes()

                # Conf thr
                msk = (conf > self.viz_conf).cpu()
                pts3d = pts3d[msk]
                colors = colors[0, 0, msk]
                if iskeyframe:
                    self.keyframe_focals.append(focal)
                    self.keyframe_confs.append(conf.mean().cpu())
                self.depth_in_color = colorize_depth(depth)
                self.conf_in_color = colorize_depth(conf)
                self.pcd_frame = None
                self.frustrum = None

                if iskeyframe or conf.mean() >= MIN_CONF_DISPLAY:  # when conf is too low, ignore
                    c2w = rigid_inverse(w2c)
                    self.pcd_frame = make_pcd(pts3d, colors)
                    if self.show_cameras:
                        self.frustrum = make_frustrum(HW, focal, w2c,
                                                      KEYFRAME_COLOR if iskeyframe else camcols[camid % len(camcols)])

                    if iskeyframe:
                        # Move Pointmap and camera to keyframes data
                        self.keyframes_data.append([f'{frame_id}_kpcd', self.pcd_frame])
                        self.keyframes_data.append([f'{frame_id}_kfrustrum', self.frustrum])
                        self.pcd_frame = None
                        self.frustrum = None

                t0, t1 = t1, time.perf_counter()
                ms_per_frame = (t1 - t0) * 1000.
                fps = 1000 / ms_per_frame
                max_mem = torch.cuda.max_memory_allocated() / MB
                if frame_id % 60 == 0 and frame_id > 0:
                    print(f"frame_id = {frame_id},"
                          f"\t{fps:0.2f} fps"
                          f"\t{ms_per_frame:0.2f} ms/frame")

                # Prepare memory map if needed (only at first frame)
                memory_map = None
                if frame_id == 0:
                    mmap = self.must3r.fetch_memory_map(self.viz_conf)
                    if mmap is not None:
                        memory_map = make_pcd(*mmap)

                focal_el = self.must3r.get_true_focals()[camid]
                if isinstance(focal_el, list):
                    focal_el = focal_el[-1]
                if focal_el is None:
                    focal_el = focal

                frame_elements = {  # creating the entire window (3 images on the right and pcd on main left)
                    'color': self.query_view,
                    'depth': self.depth_in_color,
                    'conf': self.conf_in_color,
                    'pcd': self.pcd_frame,
                    f'frustrum_{camid}': self.frustrum,
                    'keyframes_data': self.keyframes_data[self._kf_data_cursor:],
                    'replace_data': replace_data,
                    'c2w': c2w.cpu().numpy() if c2w is not None else None,  # last confident pose
                    'mem': max_mem,
                    'fps': fps,
                    'focal': focal_el,
                    'num_mem_frames': self.must3r.num_mem_frames,
                    'memory_map': memory_map,
                }
                self._kf_data_cursor = len(self.keyframes_data)
                self.update_view(frame_elements)
                self.wait_if_needed(frame_id)  # to control FPS while replaying

                frame_id += 1

        self.executor.shutdown()
        print("Shutdown")


class ReplayPipelineModel(PipelineModel):
    """Replays predictions exported with --export_replay instead of running the model"""

    def __init__(self, replay_file, *nargs, device='cuda:0', **kwargs):
        super().__init__(*nargs, device=device, **kwargs)
        self.precomputed_data = pkl.load(open(replay_file, 'rb'))
        self.torch_device = torch.device(device)

    def get_pred(self, frame_id, camid):
        pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = self.precomputed_data[frame_id][1]
        return (pts3d, colors, torch.tensor(depth, device=self.torch_device), torch.tensor(conf, device=self.torch_device),
                focal, torch.tensor(w2c, device=self.torch_device), HW, iskeyframe)

    def wait_if_needed(self, frame_id):
        time.sleep(0.05)


# GUI and render
class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, num_sources=1, **callbacks):
        """Initialize.
        Args:
            vfov (float): Vertical field of view for the 3D scene.
            max_pcd_vertices (int): Maximum point cloud vertices for which memory
                is allocated.
            callbacks (dict of kwargs): Callbacks provided by the controller
                for various operations.
        """

        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "MUSt3R || Online RGB Video Processing", 1620, 1080)
        # Called on window layout (eg: resize)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultUnlit"  # "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.pcd_material.point_size = int(4 * self.window.scaling)

        self.cam_material = o3d.visualization.rendering.MaterialRecord()
        self.cam_material.shader = "unlitLine"
        self.cam_material.line_width = 4

        # 3D scene
        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(
            True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-30, -30, -30],
                                                              [30, 30, 30])
        self.reset_view()  # Initial view
        em = self.window.theme.font_size / 2
        self.fps_panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.fps_panel.preferred_width = int(200 * self.window.scaling)
        self.window.add_child(self.fps_panel)
        self.fps = gui.Label("FPS: N/A")
        self.fps_panel.add_child(self.fps)
        self.mem = gui.Label("Mem: N/A")
        self.fps_panel.add_child(self.mem)
        self.focal = gui.Label("Focal: N/A")
        self.fps_panel.add_child(self.focal)
        self.num_mem_frames = gui.Label("Mem frames: N/A")
        self.fps_panel.add_child(self.num_mem_frames)

        # Options panel
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(400 * self.window.scaling)
        self.window.add_child(self.panel)
        toggles = gui.Horiz(em)
        self.panel.add_child(toggles)

        self.flag_followcam = True
        self.toggle_followcam = gui.ToggleSwitch("Follow Cam")
        self.toggle_followcam.is_on = True
        self.toggle_followcam.set_on_clicked(
            callbacks['on_toggle_followcam'])  # callback
        toggles.add_child(self.toggle_followcam)

        self.flag_start = False
        self.toggle_start = gui.ToggleSwitch("Start/Stop")
        self.toggle_start.is_on = False
        self.toggle_start.set_on_clicked(
            callbacks['on_toggle_start'])  # callback
        toggles.add_child(self.toggle_start)

        view_buttons = gui.Horiz(em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()  # for centering
        reset_view = gui.Button("Reset View")
        reset_view.set_on_clicked(self.reset_view)  # callback
        view_buttons.add_child(reset_view)

        self.current_view_viz = 0
        self.num_sources = num_sources
        if self.num_sources > 1:
            self.current_view = gui.Button("Next agent")
            self.current_view.set_on_clicked(
                self.next_view)  # callback
            view_buttons.add_child(self.current_view)

        view_buttons.add_stretch()  # for centering
        self.video_size = (int(240 * self.window.scaling),
                           int(320 * self.window.scaling), 3)

        # Camera stream
        self.show_color = gui.CollapsableVert("Video stream")
        self.show_color.set_is_open(True)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)

        # Show pred depth
        self.show_depth = gui.CollapsableVert("Predicted Depth")
        self.show_depth.set_is_open(True)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)

        # Show pred conf
        self.show_conf = gui.CollapsableVert("Predicted Confidence")
        self.show_conf.set_is_open(True)
        self.panel.add_child(self.show_conf)
        self.conf_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_conf.add_child(self.conf_video)

        self.status_message = gui.Label("")
        self.panel.add_child(self.status_message)

        self.flag_exit = False
        self.flag_gui_init = False
        self.flag_normals = False

    def next_view(self):
        self.current_view_viz = (self.current_view_viz + 1) % self.num_sources

    def update(self, frame_elements):
        """Update visualization with point cloud and images. Must run in main
        thread since this makes GUI calls.
        Args:
            frame_elements: dict {element_type: geometry element}.
                Dictionary of element types to geometry elements to be updated
                in the GUI:
                    'pcd': point cloud,
                    'color': rgb image (3 channel, uint8),
                    'depth': depth image (uint8),
                    'status_message': message
        """
        if not self.flag_gui_init:
            self.pcdview.scene.clear_geometry()
            # Set dummy point cloud to allocate graphics memory
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })

            # initialize camera
            self.pcd_material.shader = "normals" if self.flag_normals else "defaultUnlit"  # "defaultLit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.pcdview.scene.add_geometry('memory_map', dummy_pcd, self.pcd_material)

            self.flag_gui_init = True

        update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                        rendering.Scene.UPDATE_COLORS_FLAG |
                        (rendering.Scene.UPDATE_NORMALS_FLAG
                            if self.flag_normals else 0))

        def add_or_update_if_needed(tag, data):
            if data is not None:
                always_remove = ['frustrum', 'memory_map']
                for toremove in always_remove:
                    if toremove in tag and self.pcdview.scene.has_geometry(tag):
                        self.pcdview.scene.remove_geometry(tag)
                if self.pcdview.scene.has_geometry(tag):
                    self.pcdview.scene.scene.update_geometry(tag, data, update_flags)
                else:
                    material = self.cam_material if 'frustrum' in tag else self.pcd_material
                    self.pcdview.scene.add_geometry(tag, data, material)

        # Load memory map if present
        if frame_elements.get('memory_map', None) is not None:
            add_or_update_if_needed('memory_map', frame_elements['memory_map'])

        # update scene and cameras
        update_cam = False
        for kk in frame_elements:
            if 'frustrum' in kk:
                update_cam = int(kk.split('_')[1]) == self.current_view_viz
                add_or_update_if_needed(kk, frame_elements[kk])

        for kf_key, kf_data in frame_elements['keyframes_data']:
            add_or_update_if_needed(kf_key, kf_data)

        # Memory frames refreshed in the background: drop the old geometry, add the new one
        for tag, data in frame_elements.get('replace_data', []):
            if self.pcdview.scene.has_geometry(tag):
                self.pcdview.scene.remove_geometry(tag)
            if data is not None:
                material = self.cam_material if 'frustrum' in tag else self.pcd_material
                self.pcdview.scene.add_geometry(tag, data, material)

        if update_cam:
            # Update current pointcloud, color and depth images
            add_or_update_if_needed('pcd', frame_elements['pcd'])
            if self.show_color.get_is_open() and 'color' in frame_elements:
                self.color_video.update_image(img2o3d(frame_elements['color']))
            if self.show_depth.get_is_open() and frame_elements.get('depth', None) is not None:
                self.depth_video.update_image(img2o3d(frame_elements['depth']))
            if self.show_conf.get_is_open() and frame_elements.get('conf', None) is not None:
                self.conf_video.update_image(img2o3d(frame_elements['conf']))
            if 'focal' in frame_elements:
                self.focal.text = "Focal: " + f"{frame_elements['focal']:0.2f}"

            if self.flag_followcam and frame_elements.get('c2w', None) is not None:
                self.reset_view(pose=frame_elements['c2w'])

        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]
        if 'fps' in frame_elements:
            self.fps.text = "FPS: " + f"{frame_elements['fps']:0.2f}"
        if 'mem' in frame_elements:
            self.mem.text = "Mem: " + str(int(frame_elements["mem"])) + " MB"
        if 'num_mem_frames' in frame_elements:
            self.num_mem_frames.text = f"Mem frames: {frame_elements['num_mem_frames']}"

        self.pcdview.force_redraw()

    def reset_view(self, pose=None):
        """Callback to reset point cloud view to init or cam pose if given"""
        if pose is None:
            self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
            self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 0, -2.], [0, -1, 0])
        else:
            Rp = pose[:3, :3].T
            center = pose[:3, -1]  # look at the view camera center
            eye = center + np.array([[0, -.6, -1.5]]) @ Rp  # put GUI camera behind view and slightly above
            up = np.array([[0, -1, 0]]) @ Rp  # same orientation as input image
            self.pcdview.scene.camera.look_at(center, eye[0], up[0])

    def on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        """Callback on window initialize / resize"""
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context,
                                                    self.panel.Constraints())
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width,
                                    frame.y, panel_size.width,
                                    panel_size.height)

        fps_size = self.fps_panel.calc_preferred_size(layout_context,
                                                      self.fps_panel.Constraints())
        self.fps_panel.frame = gui.Rect(0,
                                        frame.y, fps_size.width,
                                        fps_size.height)


# Overall Controller
class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, args, camera):
        kwargs = dict(device=args.device,
                      res=args.res,
                      show_cameras=not args.hide_cameras,
                      viz_conf=args.viz_conf)
        if args.replay is not None:
            self.pipeline_model = ReplayPipelineModel(args.replay, args.model, camera, self.update_view, **kwargs)
        else:
            self.pipeline_model = PipelineModel(args.model, camera, self.update_view, **kwargs)
        self.pipeline_view = PipelineView(
            max_pcd_vertices=self.pipeline_model.max_points,
            num_sources=len(args.input),
            on_window_close=self.on_window_close,
            on_toggle_followcam=self.on_toggle_followcam,
            on_toggle_start=self.on_toggle_start)

        threading.Thread(name='PipelineModel',
                         target=self.pipeline_model.run).start()

        time.sleep(1)
        gui.Application.instance.run()

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread.
        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    def on_toggle_followcam(self, is_enabled):
        """Callback to toggle display of normals"""
        self.pipeline_view.flag_followcam = is_enabled

    def on_toggle_start(self, is_enabled):
        """Callback to start/stop MUSt3r"""
        self.pipeline_model.flag_start = is_enabled
        self.pipeline_view.flag_start = is_enabled
        self.pipeline_view.flag_gui_init = False

    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window


# MAIN
def main():
    global SKIP_EVERY
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chkpt', required=True, help='Path to checkpoint.')
    parser.add_argument('--device', default='cuda:0', help='Device to run on (e.g. "cpu" or "cuda:0").')
    parser.add_argument('--input', default='cam:0', nargs='+',
                        help="Media to load (can be paths to videos or webcam indices like 'cam:0 cam:1').")
    parser.add_argument('--image_string', default=None, type=str,
                        help="In the case of an image collection, string to identify image files.")
    parser.add_argument('--load_memory', default=None, type=str, help="Load memory from another run.")
    parser.add_argument('--output', default=None, type=str, help="Output directory to write predictions")

    # Processing related opts
    parser.add_argument('--res', default=224, choices=[224, 512],
                        type=int, help="Image resolution that works for the model used.")
    parser.add_argument('--skip_every', default=1, type=int, help="Subsample input by skipping frames.")
    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False,
                        help="Use Automatic Mixed Precision, fp16 might be unstable")
    parser.add_argument('--rerender', action='store_true', default=False, help="Rerender all frames at the end.")
    parser.add_argument('--rerender_bs', default=64, type=int, help="Re-rendering batch size")
    parser.add_argument('--filter', action='store_true', default=False,
                        help="Minimal Laplacian filtering after rerender.")
    parser.add_argument('--refresh_memory', action='store_true', default=False,
                        help="Background thread re-rendering memory frames (pointmaps and camera poses, latent "
                             "memory untouched) against the full memory in the GPU idle time of the live loop "
                             "(FIFO passes, loops again whenever new frames were added to memory).")
    parser.add_argument('--refresh_bs', default=1, type=int, help="Memory frames re-rendered per refresh forward")
    parser.add_argument('--refresh_duty', default=0.5, type=float,
                        help="Max fraction of wall-clock time the refresher may occupy the GPU (]0,1])")

    # Replay options
    parser.add_argument('--export_replay', default=None, type=str,
                        help="Export predictions (no gui) for a later replay with --replay")
    parser.add_argument('--replay', default=None, type=str, help="Replay exported predictions in the gui")
    parser.add_argument('--same_kf', action='store_true', default=False,
                        help="With --load_memory, keyframes are chosen w.r.t. indices in loaded memory.")

    # Hyperparams
    parser.add_argument('--searcher', default="kdtree-scipy-quadrant_x2", type=str,
                        help="Method for overlap prediction (e.g. kdtree-scipy-quadrant_x2, pykdtree-quadrant_x2)")
    parser.add_argument('--overlap_mode', default="nn-norm", type=str,
                        help="How to estimate overlap")  # overlap_mode='nn-norm', #'nn',
    parser.add_argument('--subsamp', default=2, type=int)
    # .15, # 3., # .15, ##3., #.05 for retrieval, # 2. - 3. for meanconf / median conf
    parser.add_argument('--keyframe_overlap_thr', default=.1, type=float,
                        help="At least this overlap to add incoming image in memory")
    parser.add_argument('--min_conf_keyframe', default=1.2, type=float, help="Ignore 3D points below this confidence.")
    parser.add_argument('--min_keyframe_spacing', default=1, type=int,
                        help="Minimal spacing (in frames) between selected keyframes to lower memory.")
    parser.add_argument('--overlap_percentile', default=85., type=float,
                        help="Percentile of image distances to compute overlap")
    parser.add_argument('--varying_focals', action='store_true', default=False,
                        help="Focals may vary along sequence (e.g. zoom-in/out).")

    parser.add_argument('--force_first_keyframes', default=None, type=int,
                        help="Force the first N frames to be keyframes")
    parser.add_argument('--num_init_frames', default=2, type=int)

    # GUI related opts
    parser.add_argument('--viz_conf', default=4., type=float, help="Conf threshold for pts3d vizu")
    parser.add_argument('--gui', action='store_true', default=False, help="Show predictions in GUI")
    parser.add_argument('--hide_cameras', action='store_true', default=False)

    args = parser.parse_args()

    toggle_memory_efficient_attention(has_xformers)
    SKIP_EVERY = args.skip_every
    amp = get_dtype(args.amp)
    args.model = SLAM_MUSt3R(chkpt=args.chkpt,
                             res=args.res,
                             kf_x_subsamp=args.subsamp,
                             searcher=args.searcher,
                             overlap_mode=args.overlap_mode,
                             keyframe_overlap_thr=args.keyframe_overlap_thr,
                             min_conf_keyframe=args.min_conf_keyframe,
                             min_keyframe_spacing=args.min_keyframe_spacing,
                             overlap_percentile=args.overlap_percentile,
                             rerender=args.rerender,
                             keep_memory=args.output is not None,
                             load_memory=args.load_memory,
                             fixed_focal=not args.varying_focals,
                             num_agents=len(args.input),
                             device=args.device,
                             num_init_frames=args.num_init_frames,
                             force_first_keyframes=args.force_first_keyframes,
                             no_model=args.replay is not None,
                             amp=amp,
                             )

    # Prepare Camera Stream
    CAMERA = AutoMultiLoader(args.input, args.image_string, res=args.res)

    # Background refresh of memory frames
    refresher = None
    if args.refresh_memory and args.replay is None:
        refresher = MemoryRefresher(args.model, batch_size=args.refresh_bs, duty=args.refresh_duty)
        refresher.start()

    # prepare output
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    if args.gui:
        # Main GUI
        PipelineController(args, CAMERA)
        tolog = {}
    else:
        # Only write output
        assert args.output is not None or args.export_replay is not None, "You should define an output folder"
        print(f"Start processing sequence of {len(CAMERA)} frames")
        frame, cam_id = grab_frame(CAMERA)
        start = time.perf_counter()
        imgHWs = [frame.shape[:2]]
        all_frames_data = []
        for frame_id in tqdm(range(len(CAMERA) // SKIP_EVERY)):
            pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = args.model(frame, frame_id * SKIP_EVERY, cam_id)
            if args.same_kf and args.model.memory_keyframe_idx is not None:
                iskeyframe = frame_id * SKIP_EVERY in args.model.memory_keyframe_idx
            if args.export_replay is not None:
                all_frames_data.append((time.perf_counter() - start,
                                        (pts3d, colors, depth.cpu().numpy(), conf.cpu().numpy(), focal,
                                         w2c.cpu().numpy(), HW, iskeyframe)))  # (timestamp, data)
            frame, cam_id = grab_frame(CAMERA)
            if frame is None:
                break
            imgHWs.append(frame.shape[:2])

        if args.export_replay is not None:
            print(f"Dumping replay data as {args.export_replay}")
            pkl.dump(all_frames_data, open(args.export_replay, 'wb'))

        if refresher is not None:
            refresher.stop()
            print(f"Memory refresher: {args.model.n_refreshed} frame refreshes in {refresher.num_passes} passes")

        # Re-render if activated
        if args.rerender:
            args.model.rerender_all_frames(maxbs=args.rerender_bs)

        # Logging FPS and GPU mem usage
        wallclock_time = time.perf_counter() - start
        fps = (len(CAMERA) // SKIP_EVERY) / wallclock_time
        gpumem = torch.cuda.max_memory_allocated() / MB
        print(f"Done @{fps}fps on average using {gpumem}MB GPU Memory")
        tolog = {'fps': fps,
                 'gpumem': gpumem,
                 'imgHWs': imgHWs,
                 }

    if refresher is not None:
        refresher.stop()

    if args.output is not None:
        # Write full trajectory
        if not args.filter:
            args.model.write_all_poses(os.path.join(args.output, 'all_poses.npz'), **tolog)
        else:
            # Postprocessing
            filtering_mode = 'laplacian'
            filtering_alpha = .1
            filtering_steps = 256
            outfile = os.path.join(args.output,
                                   f"all_poses{filtering_mode}_{filtering_steps}-steps_{filtering_alpha}-alpha.npz")
            args.model.write_all_poses(outfile,
                                       filtering_mode=filtering_mode,
                                       filtering_steps=filtering_steps,
                                       filtering_alpha=filtering_alpha, **tolog)

        # Export memory for later use
        outname = os.path.join(args.output, "memory.pkl")
        count = 0
        while args.load_memory == outname:  # make sure you do not overwrite loaded memory file
            outname = os.path.join(args.output, f"memory_{count}.pkl")
            count += 1
        print(f"Dumping memory as {outname}")
        args.model.save_memory(outname)
