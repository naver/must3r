# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import numpy as np

import must3r.tools.path_to_dust3r  # noqa
from dust3r.utils.geometry import geotrf
try:
    import viser
    import viser.transforms as viser_tf
    viser_enabled = True
except ImportError:
    viser_enabled = False


def get_pointcloud_key(frame_id):
    return f"/frames/t{frame_id}/point_cloud"


def get_camera_key(frame_id):
    return f"/frames/t{frame_id}/frustum"


def colorize_grayscale(depth: np.ndarray):
    mind, maxd = depth.min(), depth.max()
    depth = (depth - mind) / (maxd - mind + 1e-9)
    return np.stack([depth, depth, depth], axis=-1)


class ViserWrapper():
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, label: str | None = None, verbose: bool = True) -> None:
        self.server = viser.ViserServer(host, port, label, verbose)

        self.server.scene.set_up_direction('-y')

        self.num_imgs = 0
        self.progress_bar = self.server.gui.add_progress_bar(
            value=100
        )

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            """For each client that connects, create GUI elements for adjusting the
            near/far clipping planes."""
            client.camera.near = 0.01
            client.camera.far = 100.0
            camera_slider = client.gui.add_multi_slider(
                "Camera Near/far", min=0.01, max=1000.0, step=0.001, initial_value=(client.camera.near, client.camera.far), order=0
            )

            @camera_slider.on_update
            def _(_) -> None:
                near, far = camera_slider.value
                client.camera.near = near
                client.camera.far = far

        self.gui_point_size = self.server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.001, initial_value=0.01
        )
        self.frustum_scale = self.server.gui.add_slider(
            "Camera size", min=0.01, max=1.0, step=0.01, initial_value=0.05
        )
        self.confidence_threshold = self.server.gui.add_slider(
            "Confidence", min=1.0, max=10.0, step=0.1, initial_value=3.0
        )
        self.max_points_per_frame = self.server.gui.add_slider(
            "Max Points", min=0, max=250_000, step=1000, initial_value=20_000
        )
        self.local_pointmap = self.server.gui.add_checkbox(
            "Local pointmaps", initial_value=True
        )
        self.follow_cam = self.server.gui.add_checkbox(
            "Follow Cam", initial_value=False
        )
        self.keyframes_only = self.server.gui.add_checkbox(
            "Keyframes Only", initial_value=True
        )
        self.hide_images = self.server.gui.add_checkbox(
            "Hide Images", initial_value=False, hint="Hide the images in the camera frustum in the scene"
        )
        self.hide_images_gui = self.server.gui.add_checkbox(
            "Hide Predictions", initial_value=False, hint="Hide the rgb,depth,conf images"
        )

        empty_img = np.array([[[0, 0, 0]]])
        self.rgb = self.server.gui.add_image(empty_img, label="RGB", jpeg_quality=80, visible=False)
        self.depth = self.server.gui.add_image(empty_img, label="Depth", jpeg_quality=80, visible=False)
        self.conf = self.server.gui.add_image(empty_img, label="Confidence", jpeg_quality=80, visible=False)

        self.point_nodes: dict[str, viser.PointCloudHandle] = {}
        self.camera_nodes: dict[str, viser.CameraFrustumHandle] = {}
        self.pointmaps: dict[str, dict] = {}
        self.images: dict[str, dict] = {}

        @self.gui_point_size.on_update
        def _(_) -> None:
            for frame_id in list(self.point_nodes.keys()):
                self.point_nodes[frame_id].point_size = self.gui_point_size.value

        @self.frustum_scale.on_update
        def _(_) -> None:
            for frame_id in list(self.camera_nodes.keys()):
                self.camera_nodes[frame_id].scale = self.frustum_scale.value

        @self.confidence_threshold.on_update
        def _(_) -> None:
            for frame_id in list(self.point_nodes.keys()):
                self.make_point_cloud(frame_id)

        @self.local_pointmap.on_update
        def _(_) -> None:
            for frame_id in list(self.point_nodes.keys()):
                self.make_point_cloud(frame_id)

        @self.follow_cam.on_update
        def _(_) -> None:
            self.reset_cam_visility()

        @self.keyframes_only.on_update
        def _(_) -> None:
            self.reset_point_cloud_visility()

        @self.hide_images.on_update
        def _(_) -> None:
            for frame_id in list(self.camera_nodes.keys()):
                self.make_camera_frustum(frame_id)

        @self.hide_images_gui.on_update
        def _(_) -> None:
            self.set_images_gui_visibility()

        @self.max_points_per_frame.on_update
        def _(_) -> None:
            for frame_id in list(self.point_nodes.keys()):
                self.make_point_cloud(frame_id)

    @property
    def address(self):
        return f"{self.server.get_host()}:{self.server.get_port()}"

    def set_images_gui_visibility(self):
        if len(self.rgb.image) > 0:
            self.rgb.visible = not self.hide_images_gui.value
        if len(self.depth.image) > 0:
            self.depth.visible = not self.hide_images_gui.value
        if len(self.conf.image) > 0:
            self.conf.visible = not self.hide_images_gui.value

    def reset_cam_visility(self):
        for frame_id in list(self.camera_nodes.keys()):
            if not self.camera_nodes[frame_id].visible:
                self.camera_nodes[frame_id].visible = True

    def reset_point_cloud_visility(self):
        for frame_id in list(self.point_nodes.keys()):
            self.point_nodes[frame_id].visible = self.pointmaps[frame_id]['is_keyframe'] or not self.keyframes_only.value

    def reset(self, num_imgs):
        self.progress_bar.value = 0
        self.num_imgs = num_imgs

        for frame_id in self.point_nodes.keys():
            self.server.scene.remove_by_name(get_pointcloud_key(frame_id))
        self.point_nodes = {}

        for frame_id in self.camera_nodes.keys():
            self.server.scene.remove_by_name(get_camera_key(frame_id))
        self.camera_nodes = {}

        self.pointmaps = {}

    def make_point_cloud(self, frame_id):
        mask = self.pointmaps[frame_id]['conf'] >= self.confidence_threshold.value
        points = self.pointmaps[frame_id]['pts3d_local'] if self.local_pointmap.value else self.pointmaps[frame_id]['pts3d']
        points = points[mask]
        colors = self.pointmaps[frame_id]['rgb'][mask]
        is_keyframe = self.pointmaps[frame_id]['is_keyframe']

        num_point = points.shape[0]
        if self.max_points_per_frame.value > 0 and num_point > self.max_points_per_frame.value:
            submask = np.linspace(0, num_point - 1, self.max_points_per_frame.value, dtype=np.int64)
            points = points[submask]
            colors = colors[submask]

        self.point_nodes[frame_id] = \
            self.server.scene.add_point_cloud(
                name=get_pointcloud_key(frame_id),
                points=points,
                colors=colors,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
                visible=is_keyframe or not self.keyframes_only.value
        )

    def make_camera_frustum(self, frame_id):
        fov = self.images[frame_id]['fov']
        aspect = self.images[frame_id]['aspect']
        c2w = self.images[frame_id]['c2w']
        color = self.images[frame_id]['color']
        img = self.images[frame_id]['img'] if not self.hide_images.value else None

        self.camera_nodes[frame_id] = self.server.scene.add_camera_frustum(
            get_camera_key(frame_id),
            fov=fov,
            aspect=aspect,
            scale=self.frustum_scale.value,
            image=img,
            wxyz=viser_tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            color=color
        )

    def set_views(self, frame_ids, rgbs, pointmaps, is_keyframe=None):
        if len(frame_ids) == 0:
            return

        for i, frame_id in enumerate(frame_ids):
            frame_id = str(frame_id)
            img = rgbs[i]
            img = (img * 0.5) + 0.5  # unnormalize image
            img = img.cpu().numpy().transpose(1, 2, 0)

            if is_keyframe is None and frame_id in self.pointmaps:
                is_keyframe_i = self.pointmaps[frame_id]['is_keyframe']
            elif is_keyframe is None:
                is_keyframe_i = False
            else:
                is_keyframe_i = is_keyframe[i]
            c2w = pointmaps[i]['c2w'].cpu().numpy()
            self.pointmaps[frame_id] = {
                'pts3d': pointmaps[i]['pts3d'].cpu().numpy().reshape(-1, 3),
                'pts3d_local': geotrf(c2w, pointmaps[i]['pts3d_local'].cpu().numpy().reshape(-1, 3)),
                'conf': pointmaps[i]['conf'].cpu().numpy().ravel(),
                'rgb': img.reshape(-1, 3),
                'is_keyframe': bool(is_keyframe_i)
            }
            self.make_point_cloud(frame_id)

            focal = float(pointmaps[i]['focal'].cpu())
            H, W = img.shape[:2]
            fov = 2 * np.arctan2(H / 2, focal)
            aspect = W / H
            color = (20, 20, 20) if not is_keyframe_i else (20, 200, 20)

            self.images[frame_id] = {
                'fov': fov,
                'aspect': aspect,
                'c2w': c2w,
                'color': color,
                'img': img
            }
            self.make_camera_frustum(frame_id)

            self.progress_bar.value = int(100 * len(self.pointmaps) / self.num_imgs)

        # only do this for the last one, we guarantee that c2w has a value with the early exit check
        self.set_images_gui_visibility()
        if not self.hide_images_gui.value:
            self.rgb.image = img
            self.depth.image = colorize_grayscale(pointmaps[-1]['pts3d_local'].cpu().numpy()[..., 2])
            self.conf.image = colorize_grayscale(pointmaps[-1]['conf'].cpu().numpy())
        if self.follow_cam.value:
            self.reset_cam_visility()
            self.camera_nodes[frame_id].visible = False
            for client in self.server.get_clients().values():
                with client.atomic():
                    client.camera.wxyz = viser_tf.SO3.from_matrix(c2w[:3, :3]).wxyz
                    client.camera.position = c2w[:3, 3]

    def send_message(self, message):
        for client in self.server.get_clients().values():
            client.add_notification(
                title="Gradio Update",
                body=message,
                loading=False,
                with_close_button=True,
                auto_close=False,
            )
