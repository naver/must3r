# Copyright (C) 2025-present Naver Corporation. All rights reserved.
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import datetime
from scipy.spatial.transform import Rotation

from must3r.demo.viser import ViserWrapper
from must3r.demo.inference import *
from must3r.tools.image import is_valid_pil_image_file
from must3r.slam.model import get_searcher

import must3r.tools.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import geotrf
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes

from must3r.model.blocks.layers import MEMORY_MODES

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
except ImportError:
    pass


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 384, 224, 336, 448, 768],
                        help="image size: 224, 336, 448 are square images and others support multiple aspect ratios")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default=None)

    parser.add_argument("--encoder", type=str, default=None, help="encoder class instantiation")
    parser.add_argument("--decoder", type=str, default=None, help="decoder class instantiation")
    parser.add_argument("--memory_mode", type=str, default=None, choices=MEMORY_MODES,
                        help="decoder memory_mode override")

    parser.add_argument("--retrieval", type=str, help="path to the retrieval weights", default=None)

    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument('-q', '--silent', '--quiet', action='store_false', dest='verbose')

    parser.add_argument("--viser", action='store_true', default=False)
    parser.add_argument('--amp', choices=[False, "bf16", "fp16"], default=False,
                        help="Use Automatic Mixed Precision, fp16 might be unstable")
    parser.add_argument("--allow_local_files", action='store_true', default=False)
    return parser


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world,
                                 cam_size=0.05, cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, verbose=True,
                                 filename='scene.glb', camera_mask=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    if camera_mask is not None:
        assert len(imgs) == len(camera_mask)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if camera_mask is not None and not camera_mask[i]:
            continue
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, filename)
    if verbose:
        print('(exporting 3D scene to', outfile, ')')
        assert as_pointcloud

    if filename.endswith('ply'):
        if verbose:
            print('WARNING: export to ply - cameras will be ignore')
        pct.export(file_obj=outfile, file_type='ply')
    else:
        scene.export(file_obj=outfile)
    return outfile


@torch.no_grad()
def get_3D_model_from_scene(outdir, verbose, scene, min_conf_thr=3.0, as_pointcloud=False,
                            transparent_cams=False, local_pointmaps=False, cam_size=0.05, camera_conf_thr=0.0,
                            filename='scene.glb'):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None

    # get optimized values from scene
    x_out, imgs = scene.x_out, scene.imgs
    focals, cams2world = scene.focals, scene.cams2world
    nimgs = len(imgs)

    # 3D pointcloud from depthmap, poses and intrinsics
    if local_pointmaps:
        pts3d = [geotrf(cams2world[i], x_out[i]['pts3d_local'].cpu()) for i in range(nimgs)]
    else:
        pts3d = [x_out[i]['pts3d'].cpu() for i in range(nimgs)]
    msk = [(x_out[i]['conf'] >= min_conf_thr).cpu() for i in range(nimgs)]
    camera_mask = [(x_out[i]['conf'].median() >= camera_conf_thr).cpu() for i in range(nimgs)]
    return _convert_scene_output_to_glb(outdir, imgs, pts3d, msk, focals, cams2world,
                                        as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, verbose=verbose,
                                        filename=filename, camera_mask=camera_mask)


@torch.no_grad()
def get_reconstructed_scene(outdir, viser_server, model, retrieval, device, verbose, image_size, amp,
                            filelist, max_bs, num_refinements_iterations,  # main params
                            execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile,  # execution params
                            min_conf_thr, as_pointcloud, transparent_cams, local_pointmaps, cam_size, camera_conf_thr=0.0,  # output params
                            loaded_files=""
                            ):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    filelist = filelist or loaded_files.split("\n")
    if execution_mode == "vidseq" or execution_mode == "vidslam":
        if execution_mode == "vidseq":
            local_context_size = vidseq_local_context_size
            def is_keyframe_function(id, res, scene_state): return (id % keyframe_interval == 0)
            scene_state = None
            def scene_state_update_function(res, scene_state): return scene_state
        elif execution_mode == "vidslam":
            local_context_size = slam_local_context_size
            overlap_mode = "nn-norm"
            is_keyframe_function = functools.partial(
                slam_is_keyframe, subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile, overlap_mode)
            scene_state = get_searcher("kdtree-scipy-quadrant_x2")
            scene_state_update_function = functools.partial(slam_update_scene_state, subsample, min_conf_keyframe)
        else:
            raise ValueError(f"Invalid {execution_mode=}")
        scene = must3r_inference_video(model, device, image_size, amp, filelist, max_bs, init_num_images=2, batch_num_views=1,
                                       viser_server=viser_server, num_refinements_iterations=num_refinements_iterations,
                                       local_context_size=local_context_size, is_keyframe_function=is_keyframe_function,
                                       scene_state=scene_state, scene_state_update_function=scene_state_update_function,
                                       verbose=verbose)
    else:
        is_sequence = (execution_mode == "linseq")
        scene = must3r_inference(model, retrieval, device, image_size, amp, filelist,
                                 num_mem_images, max_bs, init_num_images=2, batch_num_views=1, render_once=render_once,
                                 is_sequence=is_sequence, viser_server=viser_server,
                                 num_refinements_iterations=num_refinements_iterations,
                                 verbose=verbose)
    if verbose:
        print('preparing pointcloud')
    time_start = datetime.datetime.now()
    outfile = get_3D_model_from_scene(outdir, verbose, scene, min_conf_thr, as_pointcloud, transparent_cams,
                                      local_pointmaps, cam_size, camera_conf_thr=camera_conf_thr)

    ellapsed = (datetime.datetime.now() - time_start)
    if verbose:
        print(f'pointcloud prepared in {ellapsed}')

    return scene, outfile


def load_local_files(inputfiles, textinput, execution_mode,
                     num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile):

    if textinput is not None and textinput:
        files = os.listdir(textinput)
        files = [os.path.join(textinput, f) for f in files]
        files = [f for f in files if is_valid_pil_image_file(f)]
        files = sorted(files)
    inputfiles = gradio.File(value=None, file_count="multiple",
                             file_types=list(PIL.Image.registered_extensions().keys()))
    loaded_files = gradio.TextArea(interactive=False, value="\n".join(files), visible=True)

    return inputfiles, loaded_files, *set_execution_params(files, execution_mode,
                                                           num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile)


def upload_files(inputfiles, loaded_files, execution_mode,
                 num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile):
    if inputfiles is not None:
        loaded_files = gradio.TextArea(value="", interactive=False, visible=False)
        valid_files = [f for f in inputfiles if is_valid_pil_image_file(f)]
        inputfiles_component = gradio.File(value=valid_files, file_count="multiple",
                                           file_types=list(PIL.Image.registered_extensions().keys()))
    elif loaded_files:
        inputfiles = loaded_files.split("\n")
        loaded_files = gradio.TextArea(interactive=False, value=loaded_files, visible=True)
        inputfiles_component = gradio.File(value=None, file_count="multiple",
                                           file_types=list(PIL.Image.registered_extensions().keys()))
    else:
        loaded_files = gradio.TextArea(value="", interactive=False, visible=False)
        inputfiles_component = gradio.File(value=None, file_count="multiple",
                                           file_types=list(PIL.Image.registered_extensions().keys()))

    return inputfiles_component, loaded_files, *set_execution_params(inputfiles, execution_mode,
                                                                     num_mem_images, render_once, vidseq_local_context_size,
                                                                     keyframe_interval, slam_local_context_size, slam_subsample,
                                                                     min_conf_keyframe, keyframe_overlap_thr, overlap_percentile)


def change_execution_mode(inputfiles, loaded_files, execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile):
    if inputfiles is not None:
        files = inputfiles
    elif loaded_files:
        files = loaded_files.split("\n")
    else:
        files = None
    return set_execution_mode(files, execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile)


def set_execution_params(inputfiles, execution_mode,
                         num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile):
    num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile = set_execution_mode(
        inputfiles, execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile)
    return num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile


def set_execution_mode(inputfiles, execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile):
    # linseq or retrieval
    num_mem_images = gradio.Slider(label="Number of memory images", value=num_mem_images,
                                   minimum=num_mem_images, maximum=num_mem_images, step=1, visible=False)
    render_once = gradio.Checkbox(value=render_once, label="Render once", visible=False)

    # vidseq
    vidseq_local_context_size = gradio.Slider(label="Local context size", value=vidseq_local_context_size,
                                              minimum=vidseq_local_context_size, maximum=vidseq_local_context_size, step=1, visible=False)
    keyframe_interval = gradio.Slider(label="Keyframe Interval", value=keyframe_interval,
                                      minimum=keyframe_interval, maximum=keyframe_interval, step=1, visible=False)

    # vidslam
    slam_local_context_size = gradio.Slider(label="Local context size", value=slam_local_context_size,
                                            minimum=slam_local_context_size, maximum=slam_local_context_size, step=1, visible=False)
    slam_subsample = gradio.Slider(label="subsample", value=slam_subsample,
                                   minimum=1, maximum=8, step=1, visible=False)
    min_conf_keyframe = gradio.Slider(label="min conf keyframe", value=min_conf_keyframe,
                                      minimum=1.0, maximum=3.0, step=0.1, visible=False)
    keyframe_overlap_thr = gradio.Slider(label="keyframe overlap thr", value=keyframe_overlap_thr,
                                         minimum=0.01, maximum=0.3, step=0.01, visible=False)
    overlap_percentile = gradio.Slider(label="overlap percentile", value=overlap_percentile,
                                       minimum=10, maximum=100, step=1, visible=False)

    if inputfiles is None or len(inputfiles) == 0:
        return num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile

    num_files = len(inputfiles)

    if execution_mode in ["linseq", "retrieval"]:
        current_num_mem_images = num_mem_images.constructor_args["value"] \
            if num_mem_images.constructor_args["value"] > 0 else min(num_files, 50)
        current_num_mem_images = min(num_files, current_num_mem_images)

        num_mem_images = gradio.Slider(label="Number of memory images", value=current_num_mem_images,
                                       minimum=1, maximum=num_files, step=1, visible=True)
        render_once = gradio.Checkbox(value=render_once.constructor_args["value"], label="Render once", visible=True)
    elif execution_mode == "vidseq":
        curr_vidseq_local_context_size = vidseq_local_context_size.constructor_args["value"] \
            if vidseq_local_context_size.constructor_args["value"] > 0 else min(num_files, 25)
        curr_vidseq_local_context_size = min(num_files, curr_vidseq_local_context_size)
        vidseq_local_context_size = gradio.Slider(label="Local context size", value=curr_vidseq_local_context_size,
                                                  minimum=0, maximum=num_files, step=1, visible=True)

        curr_keyframe_interval = keyframe_interval.constructor_args["value"] \
            if keyframe_interval.constructor_args["value"] > 0 else min(num_files, 3)
        curr_keyframe_interval = min(num_files, curr_keyframe_interval)
        keyframe_interval = gradio.Slider(label="Keyframe Interval", value=curr_keyframe_interval,
                                          minimum=1, maximum=num_files, step=1, visible=True)
    else:
        # vidslam
        curr_slam_local_context_size = slam_local_context_size.constructor_args["value"] \
            if slam_local_context_size.constructor_args["value"] > 0 else 0
        curr_slam_local_context_size = min(num_files, curr_slam_local_context_size)
        slam_local_context_size = gradio.Slider(label="Local context size", value=curr_slam_local_context_size,
                                                minimum=0, maximum=num_files, step=1, visible=True)

        slam_subsample = gradio.Slider(label="subsample", value=slam_subsample.constructor_args["value"],
                                       minimum=1, maximum=8, step=1, visible=True)
        min_conf_keyframe = gradio.Slider(label="min conf keyframe", value=min_conf_keyframe.constructor_args["value"],
                                          minimum=1.0, maximum=3.0, step=0.1, visible=True)
        keyframe_overlap_thr = gradio.Slider(label="keyframe overlap thr", value=keyframe_overlap_thr.constructor_args["value"],
                                             minimum=0.01, maximum=0.3, step=0.01, visible=True)
        overlap_percentile = gradio.Slider(label="overlap percentile", value=overlap_percentile.constructor_args["value"],
                                           minimum=10, maximum=100, step=1, visible=True)

    return num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile


def main_demo(tmpdirname, model, retrieval, device, image_size, server_name, server_port,
              verbose=True, amp=False, with_viser=False, allow_local_files=False):
    if with_viser:
        viser_server = ViserWrapper(host=server_name)
    else:
        viser_server = None

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, viser_server, model,
                                  retrieval, device, verbose, image_size, amp)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, verbose)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MUSt3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)

        available_modes = [("sequence: linspace", "linseq"),
                           ("sequence: slam keyframes", "vidslam"),
                           ("sequence: local context and linspace keyframes", "vidseq"),]
        if retrieval:
            available_modes.append(("unordered: retrieval", "retrieval"))

        gradio.HTML('<h2 style="text-align: center;">MUSt3R Demo</h2>')
        with gradio.Column():
            with gradio.Tab("upload"):
                inputfiles = gradio.File(file_count="multiple",
                                         file_types=list(PIL.Image.registered_extensions().keys()))
            with gradio.Tab("local_path", visible=allow_local_files):
                textinput = gradio.Textbox(label="Path to a local directory")
                load_files = gradio.Button("Load")
                loaded_files = gradio.TextArea(value="", interactive=False, visible=False)

            # inference options
            with gradio.Row():
                with gradio.Column():
                    num_refinements_iterations = gradio.Slider(label="Number of refinement iterations", value=0,
                                                               minimum=0, maximum=100, step=1, visible=True)
                    max_bs = gradio.Number(value=1, minimum=0, maximum=100_000, step=1,
                                           label="Maximum batch size", visible=True)
                with gradio.Column():
                    execution_mode = gradio.Dropdown(available_modes,
                                                     value='vidslam', label="Mode",
                                                     info="Define how to run MUSt3R",
                                                     interactive=True)

                    # linseq or retrieval
                    num_mem_images = gradio.Slider(label="Number of memory images", value=0,
                                                   minimum=0, maximum=0, step=1, visible=False)
                    render_once = gradio.Checkbox(value=False, label="Render once", visible=False)

                    # vidseq
                    vidseq_local_context_size = gradio.Slider(label="Local context size", value=0,
                                                              minimum=0, maximum=0, step=1, visible=False)
                    keyframe_interval = gradio.Slider(label="Keyframe Interval", value=0,
                                                      minimum=0, maximum=0, step=1, visible=False)

                    # vidslam
                    # also uses local_context_size
                    slam_local_context_size = gradio.Slider(label="Local context size", value=0,
                                                            minimum=0, maximum=0, step=1, visible=False)
                    slam_subsample = gradio.Slider(label="subsample", value=2,
                                                   minimum=1, maximum=8, step=1, visible=False)
                    min_conf_keyframe = gradio.Slider(label="min conf keyframe", value=1.5,
                                                      minimum=1.0, maximum=3.0, step=0.1, visible=False)
                    keyframe_overlap_thr = gradio.Slider(label="keyframe overlap thr", value=0.05,
                                                         minimum=0.01, maximum=0.3, step=0.01, visible=False)
                    overlap_percentile = gradio.Slider(label="overlap percentile", value=85,
                                                       minimum=10, maximum=100, step=1, visible=False)
            run_btn = gradio.Button("Run")

            # visualization options
            with gradio.Row():
                with gradio.Column():
                    # adjust the confidence threshold
                    min_conf_thr = gradio.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1)
                    camera_conf_thr = gradio.Slider(label="camera_conf_thr", value=1.5,
                                                    minimum=1.0, maximum=20, step=0.1)
                    # adjust the camera size in the output pointcloud
                    cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)

                with gradio.Column():
                    as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                    transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                    local_pointmaps = gradio.Checkbox(value=False, label="viz local pointmaps pointcloud")

            outmodel = gradio.Model3D()

            # events
            inputfiles.change(upload_files,
                              inputs=[inputfiles, loaded_files, execution_mode, num_mem_images,
                                      render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile],
                              outputs=[inputfiles, loaded_files, num_mem_images, render_once,
                                       vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile])
            if allow_local_files:
                load_files.click(fn=load_local_files,
                                 inputs=[inputfiles, textinput, execution_mode, num_mem_images,
                                         render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile],
                                 outputs=[inputfiles, loaded_files, num_mem_images, render_once,
                                          vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile])
            execution_mode.change(change_execution_mode,
                                  inputs=[inputfiles, loaded_files, execution_mode, num_mem_images, render_once,
                                          vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile],
                                  outputs=[num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile])

            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, max_bs, num_refinements_iterations,
                                  execution_mode, num_mem_images, render_once, vidseq_local_context_size, keyframe_interval, slam_local_context_size, slam_subsample, min_conf_keyframe, keyframe_overlap_thr, overlap_percentile,
                                  min_conf_thr, as_pointcloud, transparent_cams, local_pointmaps, cam_size, camera_conf_thr, loaded_files],
                          outputs=[scene, outmodel])

            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud,
                                         transparent_cams, local_pointmaps, cam_size, camera_conf_thr],
                                 outputs=outmodel)
            camera_conf_thr.release(fn=model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, local_pointmaps,
                                            cam_size, camera_conf_thr],
                                    outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, local_pointmaps,
                                    cam_size, camera_conf_thr],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams, local_pointmaps,
                                         cam_size, camera_conf_thr],
                                 outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams,
                                            local_pointmaps, cam_size, camera_conf_thr],
                                    outputs=outmodel)
            local_pointmaps.change(model_from_scene_fun,
                                   inputs=[scene, min_conf_thr, as_pointcloud, transparent_cams,
                                           local_pointmaps, cam_size, camera_conf_thr],
                                   outputs=outmodel)
    demo.launch(share=False, server_name=server_name, server_port=server_port)
