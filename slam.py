#!/usr/bin/env python3
# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np
import torch
import cv2
import time
import os
import time
from tqdm import tqdm
import logging as log
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from must3r.model.blocks.attention import has_xformers, toggle_memory_efficient_attention
from must3r.slam.data import AutoMultiLoader
from must3r.slam.model import SLAM_MUSt3R

try:
    o3d.cuda
except AttributeError as e:
    print('Fallback to open3d.cpu')
    o3d.cuda = o3d.cpu  # workaround for module open3d has no attribute cuda


MB = 1024.**2
camcols = [  # different frustrum colors for each agent
    [.1, .1, .9],  # blue
    [1., .5, 0.],  # orange
    [.5, 0., .5],  # purple
    [0., 1., 1.],  # cyan
]

SKIP_EVERY = 1


def grab_frame(camera):
    read = camera.read()
    frame = read[1]
    camid = 0 if len(read) != 3 else read[2]

    for _ in range(SKIP_EVERY - 1):
        camera.read()

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


# Open3D classes
# Processing
class PipelineModel:
    """Controls IO. Methods run
    in worker threads."""

    def __init__(self,
                 update_view,
                 device=None,
                 res=512,
                 show_cameras=True,
                 chunk=-1,  # -1 means no chunking
                 chunking_overlap=4,
                 viz_conf=2.5,  # conf thresh for pts3d viz
                 ):
        """Initialize.
        Args:
            update_view (callback): Callback to update display elements for a
                frame.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
            res: maxdim of the images in pixels
            show_camera: display camera locations with the 3D model
            chunk: chunk size for keyframe chunking (split sequence memory to 
                 redefine origin as the frame number augments since MUSt3R can hardly go above 50 keyframes)
            chunking_overlap : when creating a new memory chunk, how many images of the previous one should be used
        """
        self.chunk = chunk
        self.chunking_overlap = chunking_overlap
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
        self.must3r = args.model
        self.depth_in_color = None

        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.flag_start = False

        self.keyframes_data = []
        self.keyframe_focals = []
        self.keyframe_confs = []

        self.pcd_frame = None
        self.rgbd_frame = None
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Process')
        self.flag_exit = False

        self.cache = {}

    @property
    def max_points(self):
        return 10 * self.res**2

    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        cam_centers = []
        while not self.flag_exit:
            if not self.flag_start:
                if self.query_view is not None:
                    # Reset camera and memory
                    self.query_view = None
                    self.keyframes_data = []
                    self.must3r.reset()
                    frame_id = 0
                    cam_centers = []
                    CAMERA.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.query_view, camid = grab_frame(CAMERA)
                if self.query_view is None:
                    # print("End of video file, waiting...")
                    continue

                # Preproc, Forward, Postproc
                pts3d, colors, depth, conf, focal, w2c, HW, iskeyframe = self.must3r(self.query_view, frame_id, camid)
                c2w = w2c.inverse()
                cam_centers.append(c2w[:3, -1])
                # Conf thr
                msk = conf > self.viz_conf
                pts3d = pts3d[msk.cpu()]
                colors = colors[0, 0, msk.cpu()]
                if iskeyframe:
                    self.keyframe_focals.append(focal)
                    self.keyframe_confs.append(conf.mean().cpu())
                self.depth_in_color = colorize_depth(depth)
                self.conf_in_color = colorize_depth(conf)
                dtype = o3d.core.float32
                self.pcd_frame = None
                self.frustrum = None
                if pts3d.shape[0] != 0:
                    self.pcd_frame = o3d.cuda.pybind.t.geometry.PointCloud()
                    self.pcd_frame.point.positions = o3d.cuda.pybind.core.Tensor(pts3d, dtype=dtype)
                    self.pcd_frame.point.colors = o3d.cuda.pybind.core.Tensor(colors, dtype=dtype)

                if self.show_cameras:
                    H, W = HW
                    K = np.eye(3)
                    K[0, 0] = K[1, 1] = focal
                    K[0, -1] = W / 2
                    K[1, -1] = H / 2
                    self.frustrum = o3d.geometry.LineSet.create_camera_visualization(
                        W, H, intrinsic=K, extrinsic=w2c.cpu().numpy(), scale=0.075)
                    self.frustrum.paint_uniform_color([0.1, 0.9, 0.1] if iskeyframe else camcols[camid % len(camcols)])

                if iskeyframe:
                    # Move Pointmap and camera to keyframes data
                    self.keyframes_data.append([f'{frame_id}_kpcd', self.pcd_frame])
                    self.keyframes_data.append([f'{frame_id}_kfrustrum', self.frustrum])
                    self.pcd_frame = None
                    self.frustrum = None

                t0, t1 = t1, time.perf_counter()
                ms_per_frame = (t1 - t0) * 1000.
                # ms_per_frame = (t1-t0)*1000./60
                fps = 1000 / ms_per_frame
                max_mem = torch.cuda.max_memory_allocated() / MB
                if frame_id % 60 == 0 and frame_id > 0:
                    t0, t1 = t1, time.perf_counter()
                    print(f"frame_id = {frame_id},"
                          f"\t{fps:0.2f} fps"
                          f"\t{ms_per_frame:0.2f} ms/frame")

                # Prepare camera centers to display trajectory
                tempcamc = torch.stack(cam_centers).cpu().numpy()
                camc_frame = o3d.cuda.pybind.t.geometry.PointCloud()
                camc_frame.point.positions = o3d.cuda.pybind.core.Tensor(tempcamc, dtype=dtype)
                camc_frame.point.colors = o3d.cuda.pybind.core.Tensor(np.zeros_like(tempcamc), dtype=dtype)

                # Prepare memory map if needed
                memory_map = self.must3r.fetch_memory_map(self.viz_conf)
                if memory_map is not None:
                    mempts, memcols = memory_map
                    memory_map = o3d.cuda.pybind.t.geometry.PointCloud()
                    memory_map.point.positions = o3d.cuda.pybind.core.Tensor(mempts.cpu().numpy(), dtype=dtype)
                    memory_map.point.colors = o3d.cuda.pybind.core.Tensor(memcols.cpu().numpy(), dtype=dtype)
                focal_el = self.must3r.get_true_focals()[camid]
                if isinstance(focal_el, list):
                    focal_el = focal_el[-1]
                frame_elements = {  # creating the entire window (3 images on the right and pcd on main left)
                    'color': self.query_view,
                    'depth': self.depth_in_color,
                    'conf': self.conf_in_color,
                    'pcd': self.pcd_frame,
                    'cam_centers': camc_frame,
                    f'frustrum_{camid}': self.frustrum,
                    'keyframes_data': self.keyframes_data,
                    'c2w': c2w.cpu().numpy(),
                    'mem': max_mem,
                    'fps': fps,
                    'focal': focal_el,
                    'num_mem_frames': self.must3r.num_mem_frames,
                    'memory_map': memory_map,
                }
                self.update_view(frame_elements)

                frame_id += 1

        self.executor.shutdown()
        print("Shutdown")

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

        # Show pred depth
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
            self.pcdview.scene.add_geometry('cam_centers', dummy_pcd, self.pcd_material)
            self.pcdview.scene.add_geometry('memory_map', dummy_pcd, self.pcd_material)

            self.flag_gui_init = True

        update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                        rendering.Scene.UPDATE_COLORS_FLAG |
                        (rendering.Scene.UPDATE_NORMALS_FLAG
                            if self.flag_normals else 0))

        def add_or_update_if_needed(tag, data):
            if data is not None:
                always_remove = ['frustrum', 'cam_centers', 'memory_map']
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

        add_or_update_if_needed('cam_centers', frame_elements['cam_centers'])

        for kf_key, kf_data in frame_elements['keyframes_data']:
            add_or_update_if_needed(kf_key, kf_data)
            # remove item from rendering queue
            frame_elements['keyframes_data'] = None

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

            if self.flag_followcam:
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

    def __init__(self, args):
        self.pipeline_model = PipelineModel(self.update_view,
                                            device=args.device,
                                            res=args.res,
                                            show_cameras=not args.hide_cameras,
                                            viz_conf=args.viz_conf,
                                            # chunk = args.chunk,
                                            # chunking_overlap = args.chunking_overlap,
                                            )
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
if __name__ == "__main__":
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
    parser.add_argument('--rerender', action='store_true', default=False, help="Rerender all frames at the end.")
    parser.add_argument('--rerender_bs', default=64, type=int, help="Re-rendering batch size")

    # Hyperparams
    parser.add_argument('--searcher', default="kdtree-scipy-quadrant_x2", type=str,
                        help="Method for overlap prediction")  # searcher='kdtree-scipy-quadrant_x2', # 'kdtree-scipy',
    parser.add_argument('--overlap_mode', default="nn-norm", type=str,
                        help="How to estimate overlap")  # overlap_mode='nn-norm', #'nn',
    parser.add_argument('--subsamp', default=2, type=int)
    # .15, # 3., # .15, ##3., #.05 for retrieval, # 2. - 3. for meanconf / median conf
    parser.add_argument('--keyframe_overlap_thr', default=.1, type=float,
                        help="At least this overlap to add incoming image in memory")
    parser.add_argument('--min_conf_keyframe', default=1.2, type=float, help="Ignore 3D points below this confidence.")
    parser.add_argument('--overlap_percentile', default=85., type=float,
                        help="Percentile of image distances to compute overlap")
    parser.add_argument('--filter', action='store_true', default=False, help="Try different filtering setups")
    parser.add_argument('--varying_focals', action='store_true', default=False,
                        help="Focals may vary along sequence (e.g. zoom-in/out).")

    parser.add_argument('--force_first_keyframes', default=None, type=int)
    parser.add_argument('--num_init_frames', default=1, type=int)

    # GUI related opts
    parser.add_argument('--viz_conf', default=4., type=float, help="Conf threshold for pts3d vizu")
    parser.add_argument('--gui', action='store_true', default=False, help="Show predictions in GUI")
    parser.add_argument('--hide_cameras', action='store_true', default=False)

    args = parser.parse_args()

    toggle_memory_efficient_attention(has_xformers)
    SKIP_EVERY = args.skip_every
    args.model = SLAM_MUSt3R(chkpt=args.chkpt,
                             res=args.res,
                             kf_x_subsamp=args.subsamp,
                             searcher=args.searcher,
                             overlap_mode=args.overlap_mode,
                             keyframe_overlap_thr=args.keyframe_overlap_thr,
                             min_conf_keyframe=args.min_conf_keyframe,
                             overlap_percentile=args.overlap_percentile,
                             rerender=args.rerender,
                             keep_memory=args.output is not None,
                             load_memory=args.load_memory,
                             fixed_focal=not args.varying_focals,
                             num_agents=len(args.input),
                             device=args.device,
                             num_init_frames=args.num_init_frames,
                             )

    # Prepare Camera Stream
    CAMERA = AutoMultiLoader(args.input, args.image_string)

    # prepare output
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    if args.gui:
        # Main GUI
        PipelineController(args)
        tolog = {}
    else:
        # Only write output
        assert args.output is not None, "You should define an output folder"
        print(f"Start processing sequence of {len(CAMERA)} frames")
        frame, cam_id = grab_frame(CAMERA)
        start = time.time()
        imgHWs = [frame.shape[:2]]
        for frame_id in tqdm(range(len(CAMERA) // SKIP_EVERY)):
            out = args.model(frame, frame_id * SKIP_EVERY, cam_id)
            frame, cam_id = grab_frame(CAMERA)
            if frame is not None:
                imgHWs.append(frame.shape[:2])

        # Re-render if activated
        if args.rerender:
            args.model.rerender_all_frames(maxbs=args.rerender_bs)

        # Logging FPS and GPU mem usage
        wallclock_time = time.time() - start
        fps = (len(CAMERA) // SKIP_EVERY) / wallclock_time
        gpumem = torch.cuda.max_memory_allocated() / MB
        print(f"Done @{fps}fps on average using {gpumem}MB GPU Memory")
        tolog = {'fps': fps,
                 'gpumem': gpumem,
                 'imgHWs': imgHWs,
                 }

    if args.output is not None:
        # Write full trajectory
        if not args.filter:
            args.model.write_all_poses(os.path.join(args.output, 'all_poses.npz'), **tolog)
        else:
            # Postprocessing
            filtering_modes = [None, 'laplacian', 'laplacian_conf']
            filtering_alphas = [.1]  
            all_filtering_steps = [256]

            for filtering_mode in filtering_modes:
                for filtering_alpha in filtering_alphas:
                    for filtering_steps in all_filtering_steps:
                        tag = "all_poses"
                        if filtering_mode is not None:
                            tag += f"{filtering_mode}_{filtering_steps}-steps_{filtering_alpha}-alpha"
                        outfile = os.path.join(args.output, tag + '.npz')
                        args.model.write_all_poses(outfile,
                                                   filtering_mode=filtering_mode if filtering_alpha is not None else None,
                                                   filtering_steps=filtering_steps,
                                                   filtering_alpha=filtering_alpha, **tolog)
                        if filtering_mode is None:
                            continue
                    if filtering_mode is None:
                        continue

        # Export memory for later use
        outname = os.path.join(args.output, "memory.pkl")
        count = 0
        while args.load_memory == outname:  # make sure you do not overwrite loaded memory file
            outname = os.path.join(args.output, f"memory_{count}.pkl")
        print(f"Dumping memory as {outname}")
        args.model.save_memory(outname)
