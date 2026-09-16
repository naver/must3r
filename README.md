![logo](assets/must3r.jpg)

Official implementation of `MUSt3R: Multi-view Network for Stereo 3D Reconstruction`  
[[Project page](https://europe.naverlabs.com/research/publications/must3r-multi-view-network-for-stereo-3d-reconstruction/)], [[MUSt3R arxiv](https://arxiv.org/abs/2503.01661)]  

![examples](assets/examples.jpg)
![overview](assets/overview.jpg)

```bibtex
@inproceedings{must3r_cvpr25,
      title={MUSt3R: Multi-view Network for Stereo 3D Reconstruction}, 
      author={Yohann Cabon and Lucas Stoffl and Leonid Antsfeld and Gabriela Csurka and Boris Chidlovskii and Jerome Revaud and Vincent Leroy},
      booktitle = {CVPR},
      year = {2025}
}

@misc{must3r_arxiv25,
      title={MUSt3R: Multi-view Network for Stereo 3D Reconstruction}, 
      author={Yohann Cabon and Lucas Stoffl and Leonid Antsfeld and Gabriela Csurka and Boris Chidlovskii and Jerome Revaud and Vincent Leroy},
      year={2025},
      eprint={2503.01661},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Table of Contents

- [Table of Contents](#table-of-contents)
- [License](#license)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
- [Demo](#demo)
  - [Offline Gradio (+viser) Demo](#offline-gradio-viser-demo)
  - [Online Visual Odometry Demo (open3d)](#online-visual-odometry-demo-open3d)
- [Training](#training)
  - [Notes](#notes)
  - [Hyperparameters](#hyperparameters)

## License
MUSt3R is released under the MUSt3R Non-Commercial License. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for more information.  
[NOTICE](NOTICE) also contains information about the datasets used to train the checkpoints. The mapfree dataset in particular, which was used to train all models, has a very restrictive license.

## Get Started

MUSt3R extends the DUSt3R architecture through several modifications: making it symmetric and enabling online predictions of the camera pose and 3D structure of a collection of images by using a multi-layer memory mechanism.

### Installation

#### using setup.py
```bash
micromamba create -n must3r python=3.11 cmake=3.14.0
micromamba activate must3r 
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 # use the correct version of cuda for your system

# (recommended) if you can, install xFormers for memory-efficient attention
pip3 install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126
pip3 install must3r@git+https://github.com/naver/must3r.git
# pip3 install must3r[optional]@git+https://github.com/naver/must3r.git # adds pillow-heif
# pip3 install --no-build-isolation must3r[curope]@git+https://github.com/naver/must3r.git # adds curope
# pip3 install --no-build-isolation must3r[all]@git+https://github.com/naver/must3r.git # adds all optional dependencies
```

#### development (no installation)

```bash
micromamba create -n must3r python=3.11 cmake=3.14.0
micromamba activate must3r 
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126 # use the correct version of cuda for your system

# (recommended) if you can, install xFormers for memory-efficient attention
pip3 install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu126

git clone --recursive https://github.com/naver/must3r.git
cd must3r
# if you have already cloned must3r:
# git submodule update --init --recursive

pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
pip install -r requirements.txt

# install asmk
pip install faiss-cpu  # or the officially supported way (not tested): micromamba install -c pytorch faiss-cpu=1.11.0  # faiss-gpu=1.11.0 
mkdir build
cd build
git clone https://github.com/jenicek/asmk.git
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .
cd ../..

# Optional step: MUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd dust3r/croco/models/curope/
pip install .
cd ../../../../
```

### Checkpoints
We provide several pre-trained models. For these checkpoints, make sure to agree to the license of all the training datasets we used, in addition to [MUSt3R License](LICENSE). For more information, check [NOTICE](NOTICE).

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`MUSt3R_224_cvpr.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_cvpr.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`MUSt3R_512_cvpr.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_cvpr.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`MUSt3R_512.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |

`MUSt3R_224_cvpr` and `MUSt3R_512_cvpr` are the same checkpoints that we evaluated for CVPR.  
`MUSt3R_512` was finetuned from `MUSt3R_512_cvpr` with updated hyperparamers (20 views instead of 10, bf16, less token dropout, also see training section) and additional datasets (higher resolution version of [ARKitScenes](https://github.com/apple/ARKitScenes) from the depth upsample subset, updated [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) to v2, added back [Virtual Kitti 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/), some scenes from [Hypersim](https://github.com/apple/ml-hypersim), and generated scenes with [InfiniGen](https://github.com/princeton-vl/infinigen)). It outperforms `MUSt3R_512_cvpr` in most of the evaluations (see [the updated evaluations](assets/evals.md)).

We also provide both the trainingfree.pth and codebook.pkl files necessary to run image retrieval. `MUSt3R_512_cvpr` and `MUSt3R_512` share the same encoder so there's only one set of files for both of them.  
[`MUSt3R_224_retrieval_trainingfree.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_retrieval_trainingfree.pth)  
[`MUSt3R_224_retrieval_codebook.pkl`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_retrieval_codebook.pkl)  

[`MUSt3R_512_retrieval_trainingfree.pth`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth)  
[`MUSt3R_512_retrieval_codebook.pkl`](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_codebook.pkl)  

```
MD5 checksums (https://download.europe.naverlabs.com/ComputerVision/MUSt3R/checksums.txt):

ac176abd2b2c3bc5f2aea664d82e9ffa  MUSt3R_224_cvpr.pth
2a82597c3317efac40657d4f881c71f0  MUSt3R_224_retrieval_trainingfree.pth
e675ec36c7c40d512ef321fdd289bdbe  MUSt3R_224_retrieval_codebook.pkl

43808705f381a8724aafcd27c88ece35  MUSt3R_512_cvpr.pth
8854f948a8674fb1740258c1872f80dc  MUSt3R_512.pth
f7c133906bcfd4fe6ee157a9ffa85a23  MUSt3R_512_retrieval_trainingfree.pth
1125d80b9de940de2655d19b3ff78bb5  MUSt3R_512_retrieval_codebook.pkl
```

## Demo

### Offline Gradio (+viser) Demo

By default, `demo.py` will open a gradio instance on localhost:7860. If you launch the demo with `--viser`, it will also lauch a viser instance on localhost:8080. Load the images with gradio, hit run and visualize the reconstruction as it's being made in the viser tab.

> [!NOTE]
> `demo.py` is installed as `must3r_demo` (or `must3r_demo.exe`) when must3r is installed to `site-packages`. 

```bash
python demo.py --weights /path/to/MUSt3R_512.pth --retrieval /path/to/MUSt3R_512_retrieval_trainingfree.pth --image_size 512 --viser --embed_viser

# use --amp bf16 if your gpu supports
# Use --local_network to make it accessible on the local network, or --server_name to specify the url manually
# Use --server_port to change the port, by default it will search for an available port starting at 7860
# Use --device to use a different device, by default it's "cuda"
# --allow_local_files adds a second tab to load images from a local directory
# --viser is used to launch the viser server at the same time as gradio (for real-time updates). 
# Two options:
# 1) use --embed_viser to replace the gradio "Model3D" component with the embedded viser page (recommended) 
# 2) Open a new tab and access the viser url, typically http://localhost:8080/.
# Note: only one instance of viser is launched so all clients will see the same reconstructions.
# Viser's viewer will try to target a fixed framerate and lower the quality when the framerate is low.
# To disable this behaviour, open viser with http://localhost:8080/?fixedDpr=1 (recommended)

# other examples
# 512 resolution bf16, allow local files
python demo.py --weights /path/to/MUSt3R_512.pth --retrieval /path/to/MUSt3R_512_retrieval_trainingfree.pth --image_size 512 --amp bf16 --viser --embed_viser --allow_local_files

# 224 resolution, fp16, allow local files
python3 demo.py --weights /path/to/MUSt3R_224_cvpr.pth --retrieval /path/to/MUSt3R_224_retrieval_trainingfree.pth --image_size 224 --viser --embed_viser --allow_local_files --amp fp16
# 768 resolution (will use interpolated positional embeddings)
python demo.py --weights /path/to/MUSt3R_512.pth --retrieval /path/to/MUSt3R_512_retrieval_trainingfree.pth --image_size 768 --amp bf16 --viser --embed_viser
```

> [!Note]
>  IMPORTANT: Explanation of the demo parameters

1) select images:  
     - you can upload images using the gradio.File component.
     - if you use `--allow_local_files`, a second tab will appear: `local_path`. In this tab, you can paste a directory path from your local machine and hit `load` to quickly select all the images inside this directory (not recursive).

2) select global parameters
     - "Number of refinement iterations" increase it to 1 or 2 to do multiple pass on the keyframes (useful for loop closure)  
     - "Maximum batch size" -> If you are using a small gpu or you have a lot of images, put 1 to limit the vram usage (IMPORTANT).  

3) select the inference algorithm:  
  There are 4 Modes implemented. You'll have to select the mode that is most suited to your data:  
> [!Note]
>  If your images are unordered, then you **HAVE TO** select `unordered: retrieval` (only available with the --retrieval option). 

- 3.a. for an unordered collection of images, change it to "unordered: retrieval"  
     - Select the Number of memory images (also called keyframes). For this one, I would say to put as many as possible but to not go above 300, the more images there are, the slower/more memory hungry it'll be.
     - Leave "Render once" toggled OFF. You can toggle it ON if "Number of refinement iterations" > 0. 
- 3.b. for a simple video sequence, you can select "sequence: linspace"
     - Same parameters as 3.a.
- 3.c. for a longer video sequence : "sequence: slam keyframes"  (selected by default).
     - Default parameters should be good enough. You can increase subsample to 4 for slam at higher resolutions (>= 512)
     - The online version is equivalent to "sequence: slam keyframes" with local_context=0.
- 3.d. for a longer video sequence, if the KDTree/slam is too slow for you : "sequence: local context and linspace keyframes".
    - You can also keep default parameters.

4) Hit "Run". Wait for the process the load all images and switch to the viser tab to see the reconstruction, or wait until the full reconstruction appears in gradio.

### Online Visual Odometry Demo (open3d)

> [!NOTE]
> `slam.py` is installed as `must3r_slam` (or `must3r_slam.exe`) when must3r is installed to `site-packages`. 

```bash
# examples
# slam demo from a webcam (512 model), fp16 and background refresh of the memory frames
python slam.py --chkpt /path/to/MUSt3R_512.pth --input cam:0 --res 512 --subsamp 8 --gui --searcher pykdtree-quadrant_x2 --viz_conf 2.5 --amp fp16 --refresh_memory

# slam demo from a directory of images (512 model)
python slam.py \
	--chkpt /path/to/MUSt3R_512.pth \
	--input "/path_to/TUM_RGBD/rgbd_dataset_freiburg1_xyz/rgb" \  # can be a folder of video frames, a video file, or a webcam: cam:0
	--res 512 --subsamp 8 --gui --searcher pykdtree-quadrant_x2 --viz_conf 2.5 --amp fp16 --refresh_memory

# slam demo without a gui (it will write final memory state <memory.pkl> and camera trajectory <all_poses.npz>, optionally rerendered with --rerender)
python slam.py --chkpt /path/to/MUSt3R_512.pth --input /path/to/video.mp4 --output /path/to/export --res 512 --subsamp 8 --searcher pykdtree-quadrant_x2 --viz_conf 2.5 --amp fp16 --refresh_memory

# multi-agent slam demo: several sources share the same memory / coordinate system (one webcam per agent here, videos work too)
python slam.py --chkpt /path/to/MUSt3R_512.pth --input cam:0 cam:1 --res 512 --subsamp 8 --gui --searcher pykdtree-quadrant_x2 --viz_conf 2.5 --amp fp16 --refresh_memory
```

> [!IMPORTANT]
> For faster inference, compile the RoPE cuda kernels (curope), see the [installation](#installation) section: `cd dust3r/croco/models/curope/ && pip install .` (or `pip install --no-build-isolation must3r[curope]`). Without them a much slower pytorch implementation of the positional embeddings is used (`Warning, cannot find cuda-compiled version of RoPE2D`), which noticeably lowers the framerate of the online demo.

Hit the start toggle on the top right. Keyframes (frames added to the memory) are drawn in green, the current frame in the color of its camera. Toggling start off resets the memory and rewinds the input.

#### Recent changes

The online demo was reworked for live camera usage:

- **Background memory refresh** (`--refresh_memory`): a `MemoryRefresher` thread re-renders the memory frames against the *full* current memory in the gpu idle time of the live loop (FIFO passes over the keyframes, looping again whenever new frames enter the memory). Refreshed keyframes get updated pointmaps, confidences and camera poses, which are swapped in the 3D view and in the exported trajectory. The latent memory tokens are left untouched. A two-priority gpu lock guarantees the live loop is never starved: refresh forwards only run when the live loop is not waiting for the gpu. Tune with `--refresh_bs` (memory frames per refresh forward) and `--refresh_duty` (max fraction of wall-clock time the refresher may hold the gpu, default 0.5).
- **Mixed precision** (`--amp fp16|bf16`): the decoder forward runs under autocast (the encoder and the camera pose estimation stay in float32).
- **Lower latency input**: webcams are read by a dedicated thread (`CameraStream`, V4L2/MJPG, resolution matched to `--res`) so that the model always processes the latest frame instead of a buffered backlog, and the next frame is prefetched while the gpu runs. Image preprocessing uses cv2 instead of PIL.
- **Faster overlap estimation**: `--searcher pykdtree-quadrant_x2` uses [pykdtree](https://github.com/storpipfugl/pykdtree) instead of scipy, the search trees are rebuilt lazily (once per query batch instead of once per keyframe), and points are transferred to the cpu in a single copy. Increase `--subsamp` for higher framerates at high resolution.
- **Keyframe selection**: `--min_keyframe_spacing N` forbids two keyframes closer than N frames, `--force_first_keyframes N` forces the first N frames into the memory.
- **Viewer**: frames with a mean confidence below 1.1 are not added to the 3D view (typically motion blur or a covered camera), `--viz_conf` controls the confidence threshold of the displayed points, `--hide_cameras` hides the frustrums.
- **Replay**: `--export_replay /path/to/replay.pkl` (no gui) records all predictions, `--replay /path/to/replay.pkl --gui` replays them in the gui without running the model.
- **Memory files**: `memory.pkl` now also stores the keyframe indices, `--load_memory /path/to/memory.pkl --same_kf` re-uses them as keyframes on a second pass. Older memory files still load. `--skip_every N` (process every N-th frame of a video/image folder) is now honored.

#### Options

- `--input`: one or several sources, e.g. `cam:0`, `/path/to/video.mp4`, `/path/to/image_folder` (use `--image_string` to filter image names). With several sources (multi-agent, e.g. `--input cam:0 cam:1`), the frames are read alternatively from each source and all agents share the same memory, so they are localized in a common coordinate system. Each agent gets its own frustrum color and focal length estimate; the "Next agent" button switches the video/depth/confidence panels and the follow-cam between agents.
- `--res 224|512`: inference resolution, must match the checkpoint.
- `--keyframe_overlap_thr`, `--min_conf_keyframe`, `--overlap_percentile`, `--overlap_mode`: keyframe selection hyperparameters (an incoming frame is added to the memory when its overlap with the memory pointmaps is low enough and its confidence high enough).
- `--varying_focals`: estimate the focal length per frame instead of a single one for the sequence (e.g. zooming).
- `--num_init_frames`: number of initial frames forwarded together to initialize the memory.
- `--output /path/to/export` (no gui): writes the camera trajectory `all_poses.npz` and the final memory `memory.pkl`; `--rerender` re-predicts all frames from the full memory at the end (`--rerender_bs` batch size), `--filter` applies a laplacian smoothing to the trajectory.

## Training

### Notes
The training of MUSt3R has evolved since CVPR so I'll write some notes:

- First, MUSt3R is implemented in two different classes: MUSt3R and CausalMUSt3R. MUSt3R is meant for inference and CausalMUSt3R was implemented to speed up the training. Instead of processing images 1 by 1, like at inference, CausalMUSt3R runs blocks of (5) images at once, with a causal attention mask to only attend previous images. The feeback mechanism is only applied after this.

- Two "dropout_mode" are implemented in CausalMUSt3R. "permanent" is the version used for cvpr where memory tokens are permanently removed at every step. "temporary" is the version we've been using since then where tokens are temporarily masked.

- In the dataloader, there are two parameters: min_memory_num_views and max_memory_num_views that didn't exist for the cvpr submissions. They are there to ensure that all images output by the dataloader overlap with at least one of the memory frame (== keyframe). For cvpr, it was possible to select a low number of memory frames, for example 2, and have the 10th image overlap with the 8th but not with any of the first two.

- There's no validation within train.py, you have to run it (eval.py) manually.

### Hyperparameters

Here's an example of training hyperparameters
```python
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 torchrun --nnodes 1 --nproc_per_node 8 --master_port $((RANDOM%500+29000)) train.py --decoder CausalMUSt3R(img_size=(512, 512), feedback_type='single_mlp', memory_mode="kv", mem_dropout=0.1, dropout_mode='temporary', use_xformers_mask=True, use_mem_mask=True) --epochs 100 --warmup_epochs 10 --memory_num_views 20 --memory_batch_views 5 --min_memory_num_views 2 --causal --loss_in_log --amp bf16 --use_memory_efficient_attention --batch_size 1 --accum_iter 4 --dataset "XXX @YOUR_DATASET(ROOT='/pathto/your/dataset', resolution = [(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=64, num_views=20, min_memory_num_views=2, max_memory_num_views=20, transform=ColorJitter) + ..." --chkpt "/path/to/base_checkpoint" --output_dir "/path/to/output_dir"
```
