# Table of Contents

- [Table of Contents](#table-of-contents)
- [Updated Evaluation results](#updated-evaluation-results)
  - [MUSt3R\_512](#must3r_512)
    - [3D Reconstruction](#3d-reconstruction)
    - [Multi-view Depth](#multi-view-depth)
    - [Multi-view Pose Regression](#multi-view-pose-regression)
    - [TUM RGBD SLAM](#tum-rgbd-slam)
      - [Causal](#causal)
      - [Re-Render](#re-render)
    - [ETH3D SLAM](#eth3d-slam)
      - [Causal](#causal-1)
      - [Re-Render](#re-render-1)

# Updated Evaluation results

> [!NOTE]  
> In this released, we fixed a bug in the slam where, in some rare cases, valid keyframes were incorrectly rejected. It improves the results slighly on these edge cases.

```
Here's an example:
MUSt3R_224_cvpr-C on TUM RGBD (--res 224 --subsamp 2 --keyframe_overlap_thr 0.05 --min_conf_keyframe 1.5 --overlap_percentile 85 --num_init_frames 1)

before the fix
Mean ALL RMSE= 25.9 cm
Median ALL RMSE= 7.6 cm
Avg scaling error: 31.86896%
Median scaling error: 17.90361%

after the fix
Mean ALL RMSE= 24.3 cm
Median ALL RMSE= 7.6 cm
Avg scaling error: 31.08055%
Median scaling error: 16.98365%

```

## MUSt3R_512

### 3D Reconstruction

```
Dataset: 7scenes, Accuracy: 0.02458546196051707, Completion: 0.024740870526606423, NC: 0.6199703836634916, NC1: 0.6300066052737945, NC2: 0.6099341620531887 - Acc_med: 0.0070144949541884345, Comp_med: 0.008670431775816188, NC_med: 0.6869070465276299, NC1_med: 0.7035124029461268, NC2_med: 0.6703016901091332
Dataset: DTU, Accuracy: 2.9707317256857366, Completion: 1.8963801971737977, NC: 0.6617249570146254, NC1: 0.6891046630570937, NC2: 0.6343452509721573 - Acc_med: 1.3621511875772143, Comp_med: 0.7206137830271151, NC_med: 0.7412265902861677, NC1_med: 0.778875462009568, NC2_med: 0.7035777185627672
Dataset: NRGBD, Accuracy: 0.02485924163568676, Completion: 0.01399837822097039, NC: 0.7984457233128736, NC1: 0.7996487017459476, NC2: 0.7972427448797994 - Acc_med: 0.011949956381779946, Comp_med: 0.004660009342610661, NC_med: 0.93940573326342, NC1_med: 0.9408183423118336, NC2_med: 0.9379931242150066
```

### Multi-view Depth

```
dataset,metric,0
kitti.robustmvd.mvd,absrel,4.418232788642247
kitti.robustmvd.mvd,inliers103,56.978493896863796
kitti.robustmvd.mvd,scaling_factor,1.4188890982699651
kitti.robustmvd.mvd,pred_depth_density,100.0
kitti.robustmvd.mvd,num_views,8.817204301075268
dtu.robustmvd.mvd,absrel,3.2465393443337898
dtu.robustmvd.mvd,inliers103,71.0332535004074
dtu.robustmvd.mvd,scaling_factor,0.3974136064675721
dtu.robustmvd.mvd,pred_depth_density,100.0
dtu.robustmvd.mvd,num_views,4.6
scannet.robustmvd.mvd,absrel,3.3446936160326004
scannet.robustmvd.mvd,inliers103,68.50786064006388
scannet.robustmvd.mvd,scaling_factor,0.9645559397339821
scannet.robustmvd.mvd,pred_depth_density,100.0
scannet.robustmvd.mvd,num_views,2.595
tanks_and_temples.robustmvd.mvd,absrel,2.4667293022292247
tanks_and_temples.robustmvd.mvd,inliers103,81.759253187456
tanks_and_temples.robustmvd.mvd,scaling_factor,1.8521985290707021
tanks_and_temples.robustmvd.mvd,pred_depth_density,100.0
tanks_and_temples.robustmvd.mvd,num_views,4.594202898550725
eth3d.robustmvd.mvd,absrel,2.092207147954748
eth3d.robustmvd.mvd,inliers103,83.98769838878742
eth3d.robustmvd.mvd,scaling_factor,1.5572611448856502
eth3d.robustmvd.mvd,pred_depth_density,100.0
eth3d.robustmvd.mvd,num_views,5.0576923076923075
```

### Multi-view Pose Regression


co3d_procrustes_1by1

| label         |   Racc_5 |   Tacc_5 |   Racc_15 |   Tacc_15 |   Racc_30 |   Tacc_30 |   Auc_30 |
|:--------------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| backpack      |   96.98  |   87.314 |    99.051 |    96.98  |    99.159 |    98.684 |   90.66  |
| apple         |   89.467 |   79.378 |    95.244 |    91.244 |    95.689 |    94.044 |   84.316 |
| banana        |   94.739 |   78.186 |   100     |    91.111 |   100     |    95.918 |   84.565 |
| baseballbat   |   96.032 |   76.984 |   100     |    86.984 |   100     |    90.159 |   81.529 |
| baseballglove |   93.333 |   80.593 |    93.481 |    89.037 |    93.926 |    93.481 |   83.452 |
| bench         |   94.311 |   92.044 |    98.444 |    97.956 |    99.956 |    99.244 |   92.357 |
| bicycle       |   97.022 |   88.844 |   100     |    97.822 |   100     |    99.022 |   92.207 |
| bottle        |   91.556 |   86.178 |    95.333 |    94.756 |    97.156 |    97.289 |   88.519 |
| bowl          |   91.14  |   83.752 |    93.304 |    90.88  |    94.661 |    93.737 |   86.605 |
| broccoli      |   88.333 |   73.287 |    96.944 |    92.407 |    98.889 |    97.315 |   83.437 |
| cake          |   87.407 |   72.938 |    95.654 |    89.383 |    96.889 |    94.617 |   81.004 |
| car           |   83.275 |   77.368 |    92.456 |    89.298 |    95.673 |    93.333 |   80.957 |
| carrot        |   94.541 |   77.077 |    98.527 |    92.874 |    98.961 |    96.498 |   85.015 |
| chair         |   99.183 |   90.728 |   100     |    98.212 |   100     |    99.132 |   92.926 |
| cup           |   88.178 |   77.689 |    93.067 |    89.556 |    95.2   |    93.289 |   83.081 |
| donut         |   85.6   |   71.644 |    94.044 |    91.2   |    95.111 |    95.911 |   81.813 |
| hairdryer     |   99.444 |   88.704 |   100     |    97.87  |   100     |    99.213 |   91.602 |
| handbag       |   93.74  |   81.355 |    98.157 |    94.255 |    98.509 |    97.154 |   87.017 |
| hydrant       |   95.556 |   90.756 |   100     |    98.4   |   100     |    99.556 |   92.53  |
| keyboard      |   90.717 |   72.714 |    98.453 |    89.03  |    98.706 |    93.868 |   81.541 |
| laptop        |   94.902 |   76.699 |    99.216 |    90.556 |    99.706 |    95     |   84.103 |
| microwave     |   85.333 |   77.689 |    92.711 |    88     |    94.4   |    92.089 |   82.083 |
| motorcycle    |   95.2   |   89.911 |   100     |    97.911 |   100     |    99.467 |   91.514 |
| mouse         |   95.011 |   84.263 |    99.592 |    97.052 |    99.592 |    98.957 |   89.232 |
| orange        |   89.429 |   75.937 |    94.73  |    90.095 |    95.683 |    93.81  |   82.79  |
| parkingmeter  |   88.519 |   75.556 |   100     |    91.852 |   100     |    95.556 |   83.16  |
| pizza         |   93.757 |   73.862 |    97.354 |    86.984 |    98.201 |    92.698 |   80.596 |
| plant         |   95.498 |   87.763 |    98.73  |    97.72  |    99.538 |    99.307 |   90.493 |
| stopsign      |   79.728 |   61.134 |    97.007 |    83.447 |    98.594 |    92.381 |   74.183 |
| teddybear     |   95.966 |   86.111 |    99.565 |    95.314 |    99.783 |    98.357 |   88.971 |
| toaster       |   98.311 |   92.089 |    99.733 |    99.111 |   100     |    99.689 |   93.584 |
| toilet        |   95.249 |   60.69  |   100     |    81.226 |   100     |    90.805 |   74.391 |
| toybus        |   91.282 |   85.385 |    98.205 |    95.983 |    98.205 |    97.35  |   88.53  |
| toyplane      |   84.786 |   73.048 |    92.707 |    88.775 |    94.416 |    93.561 |   80.718 |
| toytrain      |   90.903 |   82.222 |    95.417 |    91.389 |    95.417 |    94.236 |   85.718 |
| toytruck      |   85.248 |   76.454 |    91.773 |    89.645 |    93.144 |    93.901 |   81.921 |
| tv            |  100     |   91.111 |   100     |    97.778 |   100     |   100     |   93.728 |
| umbrella      |   94.8   |   88.356 |    98.933 |    97.6   |    99.644 |    99.467 |   91.215 |
| vase          |   92.222 |   85.952 |    97.619 |    95.608 |    98.598 |    98.36  |   89.042 |
| wineglass     |   87.719 |   80.078 |    93.84  |    90.37  |    95.517 |    94.035 |   83.539 |
|:--------------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| mean          |   92.11  |   80.796 |    97.232 |    92.642 |    97.973 |    96.012 |   85.866 |

realestate_procrustes_1by1

| label      |   Racc_5 |   Tacc_5 |   Racc_15 |   Tacc_15 |   Racc_30 |   Tacc_30 |   Auc_30 |
|:-----------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| realestate |    96.57 |   54.053 |    99.252 |    79.134 |    99.574 |    88.429 |   71.528 |
|:-----------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| mean       |    96.57 |   54.053 |    99.252 |    79.134 |    99.574 |    88.429 |   71.528 |

co3d_pnp_1by1

| label         |   Racc_5 |   Tacc_5 |   Racc_15 |   Tacc_15 |   Racc_30 |   Tacc_30 |   Auc_30 | 
|:--------------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| backpack      |   97.325 |   88.954 |    99.029 |    97.303 |    99.072 |    98.598 |   91.443 |
| apple         |   91.156 |   81.2   |    95.467 |    91.467 |    96.089 |    95.2   |   85.698 |
| banana        |   94.83  |   78.639 |   100     |    91.61  |   100     |    96.19  |   84.853 |
| baseballbat   |   96.19  |   75.397 |   100     |    87.46  |   100     |    90.476 |   81.873 |
| baseballglove |   93.333 |   80.741 |    93.481 |    89.185 |    93.481 |    92.148 |   83.047 |
| bench         |   98.044 |   96     |    99.6   |    99.289 |   100     |    99.867 |   95.69  |
| bicycle       |   97.378 |   89.867 |   100     |    97.689 |   100     |    99.022 |   92.844 |
| bottle        |   93.067 |   87.333 |    95.822 |    94.489 |    97.2   |    97.022 |   89.188 |
| bowl          |   91.255 |   84.589 |    93.709 |    90.678 |    94.661 |    93.449 |   86.57  |
| broccoli      |   89.352 |   75.787 |    96.667 |    93.241 |    99.028 |    97.037 |   84.623 |
| cake          |   90.025 |   77.284 |    96.444 |    90.37  |    96.642 |    94.568 |   83.105 |
| car           |   87.251 |   86.491 |    93.45  |    91.813 |    95.848 |    93.86  |   86.253 |
| carrot        |   95.338 |   77.995 |    98.913 |    93.527 |    98.961 |    96.836 |   85.78  |
| chair         |   98.34  |   93.384 |   100     |    98.672 |   100     |    99.617 |   94.198 |
| cup           |   90.578 |   81.111 |    93.156 |    90.222 |    95.289 |    93.556 |   84.615 |
| donut         |   87.733 |   74.044 |    94.578 |    90.844 |    95.111 |    95.467 |   82.006 |
| hairdryer     |   99.398 |   90.046 |   100     |    97.824 |   100     |    99.167 |   92.111 |
| handbag       |   95.014 |   82.737 |    98.401 |    94.715 |    98.645 |    97.534 |   88.173 |
| hydrant       |   99.867 |   94.178 |   100     |    98.4   |   100     |    99.156 |   93.97  |
| keyboard      |   91.055 |   74.402 |    98.397 |    89.48  |    98.678 |    94.065 |   82.325 |
| laptop        |   95.359 |   78.399 |    99.02  |    90.654 |    99.706 |    94.837 |   84.817 |
| microwave     |   87.911 |   79.111 |    93.689 |    90.044 |    95.911 |    93.6   |   84.655 |
| motorcycle    |   96.222 |   95.556 |   100     |    99.422 |   100     |    99.911 |   94.286 |
| mouse         |   95.011 |   86.485 |    99.592 |    97.687 |    99.592 |    98.957 |   90.193 |
| orange        |   90.857 |   79.587 |    94.667 |    90.794 |    95.746 |    94.222 |   84.468 |
| parkingmeter  |   88.889 |   84.444 |   100     |    96.296 |   100     |    97.407 |   88.914 |
| pizza         |   95.979 |   76.825 |    97.884 |    88.148 |    98.836 |    93.122 |   82.085 |
| plant         |   96.364 |   88.081 |    98.788 |    97.72  |    99.538 |    99.192 |   91.168 |
| stopsign      |   85.215 |   84.49  |    97.007 |    96.372 |    98.594 |    98.277 |   86.848 |
| teddybear     |   96.715 |   87.343 |    99.71  |    96.256 |    99.783 |    98.599 |   90.211 |
| toaster       |   99.956 |   94.578 |   100     |    99.422 |   100     |    99.689 |   94.979 |
| toilet        |   96.169 |   64.751 |   100     |    84.598 |   100     |    91.494 |   77.67  |
| toybus        |   94.872 |   86.581 |    98.205 |    96.581 |    98.205 |    97.778 |   90.023 |
| toyplane      |   85.755 |   74.302 |    92.365 |    88.433 |    94.245 |    93.561 |   81.286 |
| toytrain      |   92.778 |   83.403 |    95.417 |    91.875 |    95.417 |    94.236 |   86.373 |
| toytruck      |   87.234 |   78.676 |    91.489 |    89.362 |    93.286 |    93.239 |   82.827 |
| tv            |  100     |   94.815 |   100     |    97.037 |   100     |    99.259 |   94.543 |
| umbrella      |   95.422 |   90.889 |    98.978 |    97.511 |    99.644 |    99.378 |   92.249 |
| vase          |   95.873 |   89.683 |    98.651 |    97.328 |    98.915 |    98.81  |   91.761 |
| wineglass     |   89.201 |   83.041 |    93.684 |    91.111 |    95.439 |    93.45  |   85.393 |
|:--------------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| mean          |   93.558 |   83.78  |    97.406 |    93.623 |    98.039 |    96.296 |   87.578 |

realestate_pnp_1by1

| label      |   Racc_5 |   Tacc_5 |   Racc_15 |   Tacc_15 |   Racc_30 |   Tacc_30 |   Auc_30 |
|:-----------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| realestate |   97.315 |   59.338 |    99.328 |    84.409 |    99.589 |    91.627 |     76.1 |
|:-----------|---------:|---------:|----------:|----------:|----------:|----------:|---------:|
| mean       |   97.315 |   59.338 |    99.328 |    84.409 |    99.589 |    91.627 |     76.1 |

### TUM RGBD SLAM

```
--res 512 --subsamp 2 --keyframe_overlap_thr 0.05 --min_conf_keyframe 1.5 --overlap_percentile 85 --num_init_frames 1
```

#### Causal

```
Seq rgbd_dataset_freiburg1_360 scale error=31.44548%
Seq rgbd_dataset_freiburg1_desk scale error=4.04111%
Seq rgbd_dataset_freiburg1_desk2 scale error=6.83210%
Seq rgbd_dataset_freiburg1_floor scale error=0.85485%
Seq rgbd_dataset_freiburg1_plant scale error=12.97613%
Seq rgbd_dataset_freiburg1_room scale error=2.55034%
Seq rgbd_dataset_freiburg1_rpy scale error=117.09163%
Seq rgbd_dataset_freiburg1_teddy scale error=1.10095%
Seq rgbd_dataset_freiburg1_xyz scale error=9.52462%
Seq rgbd_dataset_freiburg2_360_hemisphere scale error=9.54203%
Seq rgbd_dataset_freiburg2_360_kidnap scale error=44.31427%
Seq rgbd_dataset_freiburg2_coke scale error=7.56805%
Seq rgbd_dataset_freiburg2_desk scale error=2.41235%
Seq rgbd_dataset_freiburg2_desk_with_person scale error=3.15373%
Seq rgbd_dataset_freiburg2_dishes scale error=3.59105%
Seq rgbd_dataset_freiburg2_flowerbouquet scale error=30.77483%
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground scale error=38.17346%
Seq rgbd_dataset_freiburg2_large_no_loop scale error=68.04826%
Seq rgbd_dataset_freiburg2_large_with_loop scale error=34.10837%
Seq rgbd_dataset_freiburg2_metallic_sphere scale error=6.88309%
Seq rgbd_dataset_freiburg2_metallic_sphere2 scale error=37.18356%
Seq rgbd_dataset_freiburg2_pioneer_360 scale error=10.51615%
Seq rgbd_dataset_freiburg2_pioneer_slam scale error=51.97565%
Seq rgbd_dataset_freiburg2_pioneer_slam2 scale error=1.27094%
Seq rgbd_dataset_freiburg2_pioneer_slam3 scale error=21.24183%
Seq rgbd_dataset_freiburg2_rpy scale error=30.28150%
Seq rgbd_dataset_freiburg2_xyz scale error=18.53733%
Seq rgbd_dataset_freiburg3_cabinet scale error=31.13634%
Seq rgbd_dataset_freiburg3_large_cabinet scale error=12.86953%
Seq rgbd_dataset_freiburg3_long_office_household scale error=4.88256%
Seq rgbd_dataset_freiburg3_nostructure_notexture_far scale error=37.06276%
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop scale error=122.81314%
Seq rgbd_dataset_freiburg3_nostructure_texture_far scale error=43.64944%
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop scale error=8.24306%
Seq rgbd_dataset_freiburg3_sitting_halfsphere scale error=9.50321%
Seq rgbd_dataset_freiburg3_sitting_rpy scale error=103.92274%
Seq rgbd_dataset_freiburg3_sitting_static scale error=3.91279%
Seq rgbd_dataset_freiburg3_sitting_xyz scale error=4.04688%
Seq rgbd_dataset_freiburg3_structure_notexture_far scale error=27.60070%
Seq rgbd_dataset_freiburg3_structure_notexture_near scale error=142.07437%
Seq rgbd_dataset_freiburg3_structure_texture_far scale error=9.88732%
Seq rgbd_dataset_freiburg3_structure_texture_near scale error=31.77621%
Seq rgbd_dataset_freiburg3_teddy scale error=46.00533%
Seq rgbd_dataset_freiburg3_walking_halfsphere scale error=4.70200%
Seq rgbd_dataset_freiburg3_walking_rpy scale error=43.03443%
Seq rgbd_dataset_freiburg3_walking_static scale error=360.63439%
Seq rgbd_dataset_freiburg3_walking_xyz scale error=14.54597%
####################
Seq rgbd_dataset_freiburg1_360 focal=528.0074912832971, gt=516.9
rgbd_dataset_freiburg1_360 absolute FoV error=1.1 degrees.
Seq rgbd_dataset_freiburg1_desk focal=556.1869498995463, gt=516.9
rgbd_dataset_freiburg1_desk absolute FoV error=3.7 degrees.
Seq rgbd_dataset_freiburg1_desk2 focal=551.2289891159015, gt=516.9
rgbd_dataset_freiburg1_desk2 absolute FoV error=3.2 degrees.
Seq rgbd_dataset_freiburg1_floor focal=560.0051669182675, gt=516.9
rgbd_dataset_freiburg1_floor absolute FoV error=4.0 degrees.
Seq rgbd_dataset_freiburg1_plant focal=525.0507065793664, gt=516.9
rgbd_dataset_freiburg1_plant absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg1_room focal=542.4655471495557, gt=516.9
rgbd_dataset_freiburg1_room absolute FoV error=2.4 degrees.
Seq rgbd_dataset_freiburg1_rpy focal=547.0816411460696, gt=516.9
rgbd_dataset_freiburg1_rpy absolute FoV error=2.9 degrees.
Seq rgbd_dataset_freiburg1_teddy focal=526.2970628308466, gt=516.9
rgbd_dataset_freiburg1_teddy absolute FoV error=0.9 degrees.
Seq rgbd_dataset_freiburg1_xyz focal=538.1296684777885, gt=516.9
rgbd_dataset_freiburg1_xyz absolute FoV error=2.0 degrees.
Seq rgbd_dataset_freiburg2_360_hemisphere focal=521.8636677783438, gt=520.95
rgbd_dataset_freiburg2_360_hemisphere absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_360_kidnap focal=512.0399740080279, gt=520.95
rgbd_dataset_freiburg2_360_kidnap absolute FoV error=0.9 degrees.
Seq rgbd_dataset_freiburg2_coke focal=519.046214913031, gt=520.95
rgbd_dataset_freiburg2_coke absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_desk focal=529.4729493934925, gt=520.95
rgbd_dataset_freiburg2_desk absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_desk_with_person focal=521.9285461347713, gt=520.95
rgbd_dataset_freiburg2_desk_with_person absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_dishes focal=533.8579681035243, gt=520.95
rgbd_dataset_freiburg2_dishes absolute FoV error=1.2 degrees.
Seq rgbd_dataset_freiburg2_flowerbouquet focal=512.8141995773956, gt=520.95
rgbd_dataset_freiburg2_flowerbouquet absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground focal=531.9139995207534, gt=520.95
rgbd_dataset_freiburg2_flowerbouquet_brownbackground absolute FoV error=1.1 degrees.
Seq rgbd_dataset_freiburg2_large_no_loop focal=511.3405153780792, gt=520.95
rgbd_dataset_freiburg2_large_no_loop absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg2_large_with_loop focal=529.5554414050787, gt=520.95
rgbd_dataset_freiburg2_large_with_loop absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_metallic_sphere focal=512.7171085264523, gt=520.95
rgbd_dataset_freiburg2_metallic_sphere absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_metallic_sphere2 focal=522.9916743419171, gt=520.95
rgbd_dataset_freiburg2_metallic_sphere2 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_pioneer_360 focal=523.2531758535541, gt=520.95
rgbd_dataset_freiburg2_pioneer_360 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam focal=528.9499808869756, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam2 focal=520.2467034498984, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam2 absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam3 focal=519.2994810009016, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam3 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_rpy focal=518.2055303275667, gt=520.95
rgbd_dataset_freiburg2_rpy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg2_xyz focal=570.626969580562, gt=520.95
rgbd_dataset_freiburg2_xyz absolute FoV error=4.6 degrees.
Seq rgbd_dataset_freiburg3_cabinet focal=545.5959903444857, gt=537.3
rgbd_dataset_freiburg3_cabinet absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg3_large_cabinet focal=524.2619871350039, gt=537.3
rgbd_dataset_freiburg3_large_cabinet absolute FoV error=1.2 degrees.
Seq rgbd_dataset_freiburg3_long_office_household focal=526.4510141733898, gt=537.3
rgbd_dataset_freiburg3_long_office_household absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg3_nostructure_notexture_far focal=360.53582546645737, gt=537.3
rgbd_dataset_freiburg3_nostructure_notexture_far absolute FoV error=21.6 degrees.
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop focal=574.3696174703463, gt=537.3
rgbd_dataset_freiburg3_nostructure_notexture_near_withloop absolute FoV error=3.3 degrees.
Seq rgbd_dataset_freiburg3_nostructure_texture_far focal=596.4192443642455, gt=537.3
rgbd_dataset_freiburg3_nostructure_texture_far absolute FoV error=5.1 degrees.
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop focal=547.7294207323287, gt=537.3
rgbd_dataset_freiburg3_nostructure_texture_near_withloop absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg3_sitting_halfsphere focal=540.3736144394074, gt=537.3
rgbd_dataset_freiburg3_sitting_halfsphere absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_sitting_rpy focal=555.2284584467151, gt=537.3
rgbd_dataset_freiburg3_sitting_rpy absolute FoV error=1.6 degrees.
Seq rgbd_dataset_freiburg3_sitting_static focal=512.661469974803, gt=537.3
rgbd_dataset_freiburg3_sitting_static absolute FoV error=2.4 degrees.
Seq rgbd_dataset_freiburg3_sitting_xyz focal=502.32268981960857, gt=537.3
rgbd_dataset_freiburg3_sitting_xyz absolute FoV error=3.4 degrees.
Seq rgbd_dataset_freiburg3_structure_notexture_far focal=518.320935194089, gt=537.3
rgbd_dataset_freiburg3_structure_notexture_far absolute FoV error=1.8 degrees.
Seq rgbd_dataset_freiburg3_structure_notexture_near focal=544.5279258386258, gt=537.3
rgbd_dataset_freiburg3_structure_notexture_near absolute FoV error=0.7 degrees.
Seq rgbd_dataset_freiburg3_structure_texture_far focal=553.3041280011336, gt=537.3
rgbd_dataset_freiburg3_structure_texture_far absolute FoV error=1.5 degrees.
Seq rgbd_dataset_freiburg3_structure_texture_near focal=551.5034049106755, gt=537.3
rgbd_dataset_freiburg3_structure_texture_near absolute FoV error=1.3 degrees.
Seq rgbd_dataset_freiburg3_teddy focal=534.1119403938089, gt=537.3
rgbd_dataset_freiburg3_teddy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_walking_halfsphere focal=554.0699813773972, gt=537.3
rgbd_dataset_freiburg3_walking_halfsphere absolute FoV error=1.5 degrees.
Seq rgbd_dataset_freiburg3_walking_rpy focal=540.8412115132239, gt=537.3
rgbd_dataset_freiburg3_walking_rpy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_walking_static focal=523.1747023077671, gt=537.3
rgbd_dataset_freiburg3_walking_static absolute FoV error=1.4 degrees.
Seq rgbd_dataset_freiburg3_walking_xyz focal=580.4081442559685, gt=537.3
rgbd_dataset_freiburg3_walking_xyz absolute FoV error=3.8 degrees.
####################
Seq rgbd_dataset_freiburg1_360 score RMSE=9.9 cm
Seq rgbd_dataset_freiburg1_desk score RMSE=5.1 cm
Seq rgbd_dataset_freiburg1_desk2 score RMSE=4.0 cm
Seq rgbd_dataset_freiburg1_floor score RMSE=7.3 cm
Seq rgbd_dataset_freiburg1_plant score RMSE=5.2 cm
Seq rgbd_dataset_freiburg1_room score RMSE=9.8 cm
Seq rgbd_dataset_freiburg1_rpy score RMSE=4.4 cm
Seq rgbd_dataset_freiburg1_teddy score RMSE=6.2 cm
Seq rgbd_dataset_freiburg1_xyz score RMSE=2.9 cm
Seq rgbd_dataset_freiburg2_360_hemisphere score RMSE=80.3 cm
Seq rgbd_dataset_freiburg2_360_kidnap score RMSE=131.6 cm
Seq rgbd_dataset_freiburg2_coke score RMSE=7.0 cm
Seq rgbd_dataset_freiburg2_desk score RMSE=3.3 cm
Seq rgbd_dataset_freiburg2_desk_with_person score RMSE=3.3 cm
Seq rgbd_dataset_freiburg2_dishes score RMSE=4.0 cm
Seq rgbd_dataset_freiburg2_flowerbouquet score RMSE=3.4 cm
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground score RMSE=6.0 cm
Seq rgbd_dataset_freiburg2_large_no_loop score RMSE=69.9 cm
Seq rgbd_dataset_freiburg2_large_with_loop score RMSE=64.1 cm
Seq rgbd_dataset_freiburg2_metallic_sphere score RMSE=7.1 cm
Seq rgbd_dataset_freiburg2_metallic_sphere2 score RMSE=14.1 cm
Seq rgbd_dataset_freiburg2_pioneer_360 score RMSE=13.1 cm
Seq rgbd_dataset_freiburg2_pioneer_slam score RMSE=13.5 cm
Seq rgbd_dataset_freiburg2_pioneer_slam2 score RMSE=55.5 cm
Seq rgbd_dataset_freiburg2_pioneer_slam3 score RMSE=19.5 cm
Seq rgbd_dataset_freiburg2_rpy score RMSE=2.9 cm
Seq rgbd_dataset_freiburg2_xyz score RMSE=3.1 cm
Seq rgbd_dataset_freiburg3_cabinet score RMSE=1.6 cm
Seq rgbd_dataset_freiburg3_large_cabinet score RMSE=5.9 cm
Seq rgbd_dataset_freiburg3_long_office_household score RMSE=3.4 cm
Seq rgbd_dataset_freiburg3_nostructure_notexture_far score RMSE=7.1 cm
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop score RMSE=126.9 cm
Seq rgbd_dataset_freiburg3_nostructure_texture_far score RMSE=15.6 cm
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop score RMSE=9.0 cm
Seq rgbd_dataset_freiburg3_sitting_halfsphere score RMSE=5.3 cm
Seq rgbd_dataset_freiburg3_sitting_rpy score RMSE=4.9 cm
Seq rgbd_dataset_freiburg3_sitting_static score RMSE=1.5 cm
Seq rgbd_dataset_freiburg3_sitting_xyz score RMSE=3.5 cm
Seq rgbd_dataset_freiburg3_structure_notexture_far score RMSE=2.9 cm
Seq rgbd_dataset_freiburg3_structure_notexture_near score RMSE=6.1 cm
Seq rgbd_dataset_freiburg3_structure_texture_far score RMSE=2.0 cm
Seq rgbd_dataset_freiburg3_structure_texture_near score RMSE=3.1 cm
Seq rgbd_dataset_freiburg3_teddy score RMSE=2.6 cm
Seq rgbd_dataset_freiburg3_walking_halfsphere score RMSE=5.5 cm
Seq rgbd_dataset_freiburg3_walking_rpy score RMSE=6.7 cm
Seq rgbd_dataset_freiburg3_walking_static score RMSE=2.2 cm
Seq rgbd_dataset_freiburg3_walking_xyz score RMSE=7.8 cm

Mean ALL RMSE= 16.6 cm
Median ALL RMSE= 5.9 cm
####################
Avg scaling error: 25.87991%
Median scaling error: 14.77042%
```

#### Re-Render

```
Seq rgbd_dataset_freiburg1_360 scale error=24.51150%
Seq rgbd_dataset_freiburg1_desk scale error=4.92130%
Seq rgbd_dataset_freiburg1_desk2 scale error=7.27226%
Seq rgbd_dataset_freiburg1_floor scale error=1.09376%
Seq rgbd_dataset_freiburg1_plant scale error=8.99074%
Seq rgbd_dataset_freiburg1_room scale error=1.44674%
Seq rgbd_dataset_freiburg1_rpy scale error=72.88668%
Seq rgbd_dataset_freiburg1_teddy scale error=1.40703%
Seq rgbd_dataset_freiburg1_xyz scale error=10.78309%
Seq rgbd_dataset_freiburg2_360_hemisphere scale error=22.97787%
Seq rgbd_dataset_freiburg2_360_kidnap scale error=8.62986%
Seq rgbd_dataset_freiburg2_coke scale error=7.22539%
Seq rgbd_dataset_freiburg2_desk scale error=2.75505%
Seq rgbd_dataset_freiburg2_desk_with_person scale error=2.17417%
Seq rgbd_dataset_freiburg2_dishes scale error=3.67532%
Seq rgbd_dataset_freiburg2_flowerbouquet scale error=33.69269%
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground scale error=30.47025%
Seq rgbd_dataset_freiburg2_large_no_loop scale error=67.65876%
Seq rgbd_dataset_freiburg2_large_with_loop scale error=34.73158%
Seq rgbd_dataset_freiburg2_metallic_sphere scale error=4.88980%
Seq rgbd_dataset_freiburg2_metallic_sphere2 scale error=38.92649%
Seq rgbd_dataset_freiburg2_pioneer_360 scale error=12.33180%
Seq rgbd_dataset_freiburg2_pioneer_slam scale error=47.10297%
Seq rgbd_dataset_freiburg2_pioneer_slam2 scale error=2.65148%
Seq rgbd_dataset_freiburg2_pioneer_slam3 scale error=22.02608%
Seq rgbd_dataset_freiburg2_rpy scale error=25.65367%
Seq rgbd_dataset_freiburg2_xyz scale error=17.69579%
Seq rgbd_dataset_freiburg3_cabinet scale error=31.01900%
Seq rgbd_dataset_freiburg3_large_cabinet scale error=13.92250%
Seq rgbd_dataset_freiburg3_long_office_household scale error=4.83289%
Seq rgbd_dataset_freiburg3_nostructure_notexture_far scale error=36.90347%
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop scale error=81.02254%
Seq rgbd_dataset_freiburg3_nostructure_texture_far scale error=43.18973%
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop scale error=9.38859%
Seq rgbd_dataset_freiburg3_sitting_halfsphere scale error=4.38817%
Seq rgbd_dataset_freiburg3_sitting_rpy scale error=31.27256%
Seq rgbd_dataset_freiburg3_sitting_static scale error=10.57612%
Seq rgbd_dataset_freiburg3_sitting_xyz scale error=4.79178%
Seq rgbd_dataset_freiburg3_structure_notexture_far scale error=28.32505%
Seq rgbd_dataset_freiburg3_structure_notexture_near scale error=145.94686%
Seq rgbd_dataset_freiburg3_structure_texture_far scale error=9.91230%
Seq rgbd_dataset_freiburg3_structure_texture_near scale error=30.61126%
Seq rgbd_dataset_freiburg3_teddy scale error=46.35856%
Seq rgbd_dataset_freiburg3_walking_halfsphere scale error=8.45362%
Seq rgbd_dataset_freiburg3_walking_rpy scale error=24.42151%
Seq rgbd_dataset_freiburg3_walking_static scale error=157.53075%
Seq rgbd_dataset_freiburg3_walking_xyz scale error=10.65762%
####################
Seq rgbd_dataset_freiburg1_360 focal=528.0074912832971, gt=516.9
rgbd_dataset_freiburg1_360 absolute FoV error=1.1 degrees.
Seq rgbd_dataset_freiburg1_desk focal=556.1869498995463, gt=516.9
rgbd_dataset_freiburg1_desk absolute FoV error=3.7 degrees.
Seq rgbd_dataset_freiburg1_desk2 focal=551.2289891159015, gt=516.9
rgbd_dataset_freiburg1_desk2 absolute FoV error=3.2 degrees.
Seq rgbd_dataset_freiburg1_floor focal=560.0051669182675, gt=516.9
rgbd_dataset_freiburg1_floor absolute FoV error=4.0 degrees.
Seq rgbd_dataset_freiburg1_plant focal=525.0507065793664, gt=516.9
rgbd_dataset_freiburg1_plant absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg1_room focal=542.4655471495557, gt=516.9
rgbd_dataset_freiburg1_room absolute FoV error=2.4 degrees.
Seq rgbd_dataset_freiburg1_rpy focal=547.0816411460696, gt=516.9
rgbd_dataset_freiburg1_rpy absolute FoV error=2.9 degrees.
Seq rgbd_dataset_freiburg1_teddy focal=526.2970628308466, gt=516.9
rgbd_dataset_freiburg1_teddy absolute FoV error=0.9 degrees.
Seq rgbd_dataset_freiburg1_xyz focal=538.1296684777885, gt=516.9
rgbd_dataset_freiburg1_xyz absolute FoV error=2.0 degrees.
Seq rgbd_dataset_freiburg2_360_hemisphere focal=521.8636677783438, gt=520.95
rgbd_dataset_freiburg2_360_hemisphere absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_360_kidnap focal=512.0399740080279, gt=520.95
rgbd_dataset_freiburg2_360_kidnap absolute FoV error=0.9 degrees.
Seq rgbd_dataset_freiburg2_coke focal=519.046214913031, gt=520.95
rgbd_dataset_freiburg2_coke absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_desk focal=529.4729493934925, gt=520.95
rgbd_dataset_freiburg2_desk absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_desk_with_person focal=521.9285461347713, gt=520.95
rgbd_dataset_freiburg2_desk_with_person absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_dishes focal=533.8579681035243, gt=520.95
rgbd_dataset_freiburg2_dishes absolute FoV error=1.2 degrees.
Seq rgbd_dataset_freiburg2_flowerbouquet focal=512.8141995773956, gt=520.95
rgbd_dataset_freiburg2_flowerbouquet absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground focal=531.9139995207534, gt=520.95
rgbd_dataset_freiburg2_flowerbouquet_brownbackground absolute FoV error=1.1 degrees.
Seq rgbd_dataset_freiburg2_large_no_loop focal=511.3405153780792, gt=520.95
rgbd_dataset_freiburg2_large_no_loop absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg2_large_with_loop focal=529.5554414050787, gt=520.95
rgbd_dataset_freiburg2_large_with_loop absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_metallic_sphere focal=512.7171085264523, gt=520.95
rgbd_dataset_freiburg2_metallic_sphere absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_metallic_sphere2 focal=522.9916743419171, gt=520.95
rgbd_dataset_freiburg2_metallic_sphere2 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_pioneer_360 focal=523.2531758535541, gt=520.95
rgbd_dataset_freiburg2_pioneer_360 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam focal=528.9499808869756, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam2 focal=520.2467034498984, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam2 absolute FoV error=0.1 degrees.
Seq rgbd_dataset_freiburg2_pioneer_slam3 focal=519.2994810009016, gt=520.95
rgbd_dataset_freiburg2_pioneer_slam3 absolute FoV error=0.2 degrees.
Seq rgbd_dataset_freiburg2_rpy focal=518.2055303275667, gt=520.95
rgbd_dataset_freiburg2_rpy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg2_xyz focal=570.626969580562, gt=520.95
rgbd_dataset_freiburg2_xyz absolute FoV error=4.6 degrees.
Seq rgbd_dataset_freiburg3_cabinet focal=545.5959903444857, gt=537.3
rgbd_dataset_freiburg3_cabinet absolute FoV error=0.8 degrees.
Seq rgbd_dataset_freiburg3_large_cabinet focal=524.2619871350039, gt=537.3
rgbd_dataset_freiburg3_large_cabinet absolute FoV error=1.2 degrees.
Seq rgbd_dataset_freiburg3_long_office_household focal=526.4510141733898, gt=537.3
rgbd_dataset_freiburg3_long_office_household absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg3_nostructure_notexture_far focal=360.53582546645737, gt=537.3
rgbd_dataset_freiburg3_nostructure_notexture_far absolute FoV error=21.6 degrees.
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop focal=574.3696174703463, gt=537.3
rgbd_dataset_freiburg3_nostructure_notexture_near_withloop absolute FoV error=3.3 degrees.
Seq rgbd_dataset_freiburg3_nostructure_texture_far focal=596.4192443642455, gt=537.3
rgbd_dataset_freiburg3_nostructure_texture_far absolute FoV error=5.1 degrees.
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop focal=547.7294207323287, gt=537.3
rgbd_dataset_freiburg3_nostructure_texture_near_withloop absolute FoV error=1.0 degrees.
Seq rgbd_dataset_freiburg3_sitting_halfsphere focal=540.3736144394074, gt=537.3
rgbd_dataset_freiburg3_sitting_halfsphere absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_sitting_rpy focal=555.2284584467151, gt=537.3
rgbd_dataset_freiburg3_sitting_rpy absolute FoV error=1.6 degrees.
Seq rgbd_dataset_freiburg3_sitting_static focal=512.661469974803, gt=537.3
rgbd_dataset_freiburg3_sitting_static absolute FoV error=2.4 degrees.
Seq rgbd_dataset_freiburg3_sitting_xyz focal=502.32268981960857, gt=537.3
rgbd_dataset_freiburg3_sitting_xyz absolute FoV error=3.4 degrees.
Seq rgbd_dataset_freiburg3_structure_notexture_far focal=518.320935194089, gt=537.3
rgbd_dataset_freiburg3_structure_notexture_far absolute FoV error=1.8 degrees.
Seq rgbd_dataset_freiburg3_structure_notexture_near focal=544.5279258386258, gt=537.3
rgbd_dataset_freiburg3_structure_notexture_near absolute FoV error=0.7 degrees.
Seq rgbd_dataset_freiburg3_structure_texture_far focal=553.3041280011336, gt=537.3
rgbd_dataset_freiburg3_structure_texture_far absolute FoV error=1.5 degrees.
Seq rgbd_dataset_freiburg3_structure_texture_near focal=551.5034049106755, gt=537.3
rgbd_dataset_freiburg3_structure_texture_near absolute FoV error=1.3 degrees.
Seq rgbd_dataset_freiburg3_teddy focal=534.1119403938089, gt=537.3
rgbd_dataset_freiburg3_teddy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_walking_halfsphere focal=554.0699813773972, gt=537.3
rgbd_dataset_freiburg3_walking_halfsphere absolute FoV error=1.5 degrees.
Seq rgbd_dataset_freiburg3_walking_rpy focal=540.8412115132239, gt=537.3
rgbd_dataset_freiburg3_walking_rpy absolute FoV error=0.3 degrees.
Seq rgbd_dataset_freiburg3_walking_static focal=523.1747023077671, gt=537.3
rgbd_dataset_freiburg3_walking_static absolute FoV error=1.4 degrees.
Seq rgbd_dataset_freiburg3_walking_xyz focal=580.4081442559685, gt=537.3
rgbd_dataset_freiburg3_walking_xyz absolute FoV error=3.8 degrees.
####################
Seq rgbd_dataset_freiburg1_360 score RMSE=6.6 cm
Seq rgbd_dataset_freiburg1_desk score RMSE=2.4 cm
Seq rgbd_dataset_freiburg1_desk2 score RMSE=3.7 cm
Seq rgbd_dataset_freiburg1_floor score RMSE=6.6 cm
Seq rgbd_dataset_freiburg1_plant score RMSE=3.7 cm
Seq rgbd_dataset_freiburg1_room score RMSE=7.3 cm
Seq rgbd_dataset_freiburg1_rpy score RMSE=3.5 cm
Seq rgbd_dataset_freiburg1_teddy score RMSE=4.8 cm
Seq rgbd_dataset_freiburg1_xyz score RMSE=2.0 cm
Seq rgbd_dataset_freiburg2_360_hemisphere score RMSE=39.7 cm
Seq rgbd_dataset_freiburg2_360_kidnap score RMSE=113.6 cm
Seq rgbd_dataset_freiburg2_coke score RMSE=5.1 cm
Seq rgbd_dataset_freiburg2_desk score RMSE=1.8 cm
Seq rgbd_dataset_freiburg2_desk_with_person score RMSE=2.2 cm
Seq rgbd_dataset_freiburg2_dishes score RMSE=2.1 cm
Seq rgbd_dataset_freiburg2_flowerbouquet score RMSE=2.0 cm
Seq rgbd_dataset_freiburg2_flowerbouquet_brownbackground score RMSE=1.5 cm
Seq rgbd_dataset_freiburg2_large_no_loop score RMSE=66.4 cm
Seq rgbd_dataset_freiburg2_large_with_loop score RMSE=30.3 cm
Seq rgbd_dataset_freiburg2_metallic_sphere score RMSE=6.4 cm
Seq rgbd_dataset_freiburg2_metallic_sphere2 score RMSE=9.7 cm
Seq rgbd_dataset_freiburg2_pioneer_360 score RMSE=13.5 cm
Seq rgbd_dataset_freiburg2_pioneer_slam score RMSE=11.3 cm
Seq rgbd_dataset_freiburg2_pioneer_slam2 score RMSE=28.5 cm
Seq rgbd_dataset_freiburg2_pioneer_slam3 score RMSE=16.2 cm
Seq rgbd_dataset_freiburg2_rpy score RMSE=1.6 cm
Seq rgbd_dataset_freiburg2_xyz score RMSE=2.4 cm
Seq rgbd_dataset_freiburg3_cabinet score RMSE=1.4 cm
Seq rgbd_dataset_freiburg3_large_cabinet score RMSE=5.1 cm
Seq rgbd_dataset_freiburg3_long_office_household score RMSE=3.0 cm
Seq rgbd_dataset_freiburg3_nostructure_notexture_far score RMSE=4.8 cm
Seq rgbd_dataset_freiburg3_nostructure_notexture_near_withloop score RMSE=111.6 cm
Seq rgbd_dataset_freiburg3_nostructure_texture_far score RMSE=12.7 cm
Seq rgbd_dataset_freiburg3_nostructure_texture_near_withloop score RMSE=8.8 cm
Seq rgbd_dataset_freiburg3_sitting_halfsphere score RMSE=4.6 cm
Seq rgbd_dataset_freiburg3_sitting_rpy score RMSE=4.0 cm
Seq rgbd_dataset_freiburg3_sitting_static score RMSE=1.3 cm
Seq rgbd_dataset_freiburg3_sitting_xyz score RMSE=2.4 cm
Seq rgbd_dataset_freiburg3_structure_notexture_far score RMSE=1.8 cm
Seq rgbd_dataset_freiburg3_structure_notexture_near score RMSE=4.0 cm
Seq rgbd_dataset_freiburg3_structure_texture_far score RMSE=1.7 cm
Seq rgbd_dataset_freiburg3_structure_texture_near score RMSE=2.4 cm
Seq rgbd_dataset_freiburg3_teddy score RMSE=1.8 cm
Seq rgbd_dataset_freiburg3_walking_halfsphere score RMSE=3.9 cm
Seq rgbd_dataset_freiburg3_walking_rpy score RMSE=5.0 cm
Seq rgbd_dataset_freiburg3_walking_static score RMSE=1.9 cm
Seq rgbd_dataset_freiburg3_walking_xyz score RMSE=5.4 cm
Mean ALL RMSE= 12.4 cm
Median ALL RMSE= 4.0 cm
####################
Avg scaling error: 24.05368%
Median scaling error: 15.03519%
```

### ETH3D SLAM

```
--res 512 --subsamp 2 --keyframe_overlap_thr 0.05 --min_conf_keyframe 1.5 --overlap_percentile 85 --num_init_frames 1
```

#### Causal

```
####################
ALL
####################
Seq cables_1 scale error=196.24231%
Seq cables_2 scale error=63.52993%
Seq cables_3 scale error=51.49166%
Seq camera_shake_1 scale error=144.82195%
Seq camera_shake_2 scale error=122.63395%
Seq camera_shake_3 scale error=1366.26212%
Seq ceiling_1 scale error=441.86029%
Seq ceiling_2 scale error=44.45306%
Seq desk_3 scale error=293.82908%
Seq desk_changing_1 scale error=91.90175%
Seq einstein_1 scale error=3.85232%
Seq einstein_2 scale error=3.63627%
Seq einstein_flashlight scale error=295.40100%
Seq einstein_global_light_changes_1 scale error=26.50329%
Seq einstein_global_light_changes_2 scale error=4.31855%
Seq einstein_global_light_changes_3 scale error=67.52263%
Seq kidnap_1 scale error=9.48721%
Seq large_loop_1 scale error=11.78199%
Seq mannequin_1 scale error=162.81787%
Seq mannequin_3 scale error=14.07430%
Seq mannequin_4 scale error=7.95852%
Seq mannequin_5 scale error=0.26027%
Seq mannequin_7 scale error=106.53640%
Seq mannequin_face_1 scale error=25.41532%
Seq mannequin_face_2 scale error=131.60396%
Seq mannequin_face_3 scale error=123.90409%
Seq mannequin_head scale error=336.01507%
Seq motion_1 scale error=2.11208%
Seq planar_2 scale error=8.18033%
Seq planar_3 scale error=4.89022%
Seq plant_1 scale error=62.94203%
Seq plant_2 scale error=153.51069%
Seq plant_3 scale error=25.50205%
Seq plant_4 scale error=331.59738%
Seq plant_5 scale error=187.94718%
Seq plant_scene_1 scale error=7.49816%
Seq plant_scene_2 scale error=13.86564%
Seq plant_scene_3 scale error=8.14292%
Seq reflective_1 scale error=497.71831%
Seq repetitive scale error=21.04243%
Seq sfm_bench scale error=75.86064%
Seq sfm_garden scale error=3.52679%
Seq sfm_house_loop scale error=25.59522%
Seq sfm_lab_room_1 scale error=6.77292%
Seq sfm_lab_room_2 scale error=7.09968%
Seq sofa_1 scale error=4.31742%
Seq sofa_2 scale error=9.26799%
Seq sofa_3 scale error=10.39132%
Seq sofa_4 scale error=2.40774%
Seq sofa_shake scale error=36.88788%
Seq table_3 scale error=0.32467%
Seq table_4 scale error=5.73034%
Seq table_7 scale error=76.01463%
Seq vicon_light_1 scale error=21.49352%
Seq vicon_light_2 scale error=123.82484%
####################
Seq cables_1 score RMSE=16.9 cm
Seq cables_2 score RMSE=42.6 cm
Seq cables_3 score RMSE=15.3 cm
Seq camera_shake_1 score RMSE=4.3 cm
Seq camera_shake_2 score RMSE=4.1 cm
Seq camera_shake_3 score RMSE=13.8 cm
Seq ceiling_1 score RMSE=200.7 cm
Seq ceiling_2 score RMSE=63.8 cm
Seq desk_3 score RMSE=97.0 cm
Seq desk_changing_1 score RMSE=81.0 cm
Seq einstein_1 score RMSE=2.5 cm
Seq einstein_2 score RMSE=5.0 cm
Seq einstein_flashlight score RMSE=138.8 cm
Seq einstein_global_light_changes_1 score RMSE=62.1 cm
Seq einstein_global_light_changes_2 score RMSE=3.1 cm
Seq einstein_global_light_changes_3 score RMSE=90.7 cm
Seq kidnap_1 score RMSE=15.1 cm
Seq large_loop_1 score RMSE=6.6 cm
Seq mannequin_1 score RMSE=79.4 cm
Seq mannequin_3 score RMSE=8.1 cm
Seq mannequin_4 score RMSE=12.8 cm
Seq mannequin_5 score RMSE=32.7 cm
Seq mannequin_7 score RMSE=31.7 cm
Seq mannequin_face_1 score RMSE=6.9 cm
Seq mannequin_face_2 score RMSE=2.6 cm
Seq mannequin_face_3 score RMSE=14.5 cm
Seq mannequin_head score RMSE=32.1 cm
Seq motion_1 score RMSE=8.3 cm
Seq planar_2 score RMSE=3.0 cm
Seq planar_3 score RMSE=7.5 cm
Seq plant_1 score RMSE=2.6 cm
Seq plant_2 score RMSE=2.5 cm
Seq plant_3 score RMSE=4.0 cm
Seq plant_4 score RMSE=3.3 cm
Seq plant_5 score RMSE=7.7 cm
Seq plant_scene_1 score RMSE=4.3 cm
Seq plant_scene_2 score RMSE=6.3 cm
Seq plant_scene_3 score RMSE=43.6 cm
Seq reflective_1 score RMSE=38.9 cm
Seq repetitive score RMSE=40.7 cm
Seq sfm_bench score RMSE=9.3 cm
Seq sfm_garden score RMSE=359.2 cm
Seq sfm_house_loop score RMSE=513.4 cm
Seq sfm_lab_room_1 score RMSE=25.4 cm
Seq sfm_lab_room_2 score RMSE=7.0 cm
Seq sofa_1 score RMSE=5.3 cm
Seq sofa_2 score RMSE=19.1 cm
Seq sofa_3 score RMSE=4.2 cm
Seq sofa_4 score RMSE=4.1 cm
Seq sofa_shake score RMSE=4.4 cm
Seq table_3 score RMSE=3.7 cm
Seq table_4 score RMSE=2.8 cm
Seq table_7 score RMSE=3.5 cm
Seq vicon_light_1 score RMSE=59.9 cm
Seq vicon_light_2 score RMSE=5.1 cm
Mean ALL RMSE= 41.5 cm
Median ALL RMSE= 8.3 cm
####################
Avg scaling error: 32.01520%
Median scaling error: 20.37913%
```

#### Re-Render

```
####################
ALL
####################
Seq cables_1 scale error=196.91886%
Seq cables_2 scale error=25.46701%
Seq cables_3 scale error=49.79121%
Seq camera_shake_1 scale error=137.69285%
Seq camera_shake_2 scale error=122.52162%
Seq camera_shake_3 scale error=897.85960%
Seq ceiling_1 scale error=433.88296%
Seq ceiling_2 scale error=48.78704%
Seq desk_3 scale error=317.74848%
Seq desk_changing_1 scale error=73.89780%
Seq einstein_1 scale error=1.39267%
Seq einstein_2 scale error=3.16817%
Seq einstein_flashlight scale error=280.20583%
Seq einstein_global_light_changes_1 scale error=103.68599%
Seq einstein_global_light_changes_2 scale error=2.80112%
Seq einstein_global_light_changes_3 scale error=37.50954%
Seq kidnap_1 scale error=7.21832%
Seq large_loop_1 scale error=10.67498%
Seq mannequin_1 scale error=77.84833%
Seq mannequin_3 scale error=12.53715%
Seq mannequin_4 scale error=9.59590%
Seq mannequin_5 scale error=9.14020%
Seq mannequin_7 scale error=96.94250%
Seq mannequin_face_1 scale error=15.47521%
Seq mannequin_face_2 scale error=117.29604%
Seq mannequin_face_3 scale error=130.94396%
Seq mannequin_head scale error=272.98344%
Seq motion_1 scale error=2.05682%
Seq planar_2 scale error=10.73120%
Seq planar_3 scale error=4.34915%
Seq plant_1 scale error=62.58616%
Seq plant_2 scale error=151.18493%
Seq plant_3 scale error=24.82716%
Seq plant_4 scale error=348.28657%
Seq plant_5 scale error=169.64528%
Seq plant_scene_1 scale error=6.33455%
Seq plant_scene_2 scale error=12.98017%
Seq plant_scene_3 scale error=13.70720%
Seq reflective_1 scale error=577.35584%
Seq repetitive scale error=12.49563%
Seq sfm_bench scale error=71.34305%
Seq sfm_garden scale error=8.03611%
Seq sfm_house_loop scale error=6.56751%
Seq sfm_lab_room_1 scale error=7.94395%
Seq sfm_lab_room_2 scale error=6.97130%
Seq sofa_1 scale error=4.74262%
Seq sofa_2 scale error=5.88779%
Seq sofa_3 scale error=10.74398%
Seq sofa_4 scale error=0.88204%
Seq sofa_shake scale error=26.07126%
Seq table_3 scale error=0.50033%
Seq table_4 scale error=6.26041%
Seq table_7 scale error=78.73482%
Seq vicon_light_1 scale error=6.94560%
Seq vicon_light_2 scale error=121.79519%
####################
Seq cables_1 score RMSE=16.7 cm
Seq cables_2 score RMSE=26.8 cm
Seq cables_3 score RMSE=13.2 cm
Seq camera_shake_1 score RMSE=3.9 cm
Seq camera_shake_2 score RMSE=4.2 cm
Seq camera_shake_3 score RMSE=12.6 cm
Seq ceiling_1 score RMSE=200.6 cm
Seq ceiling_2 score RMSE=65.8 cm
Seq desk_3 score RMSE=98.0 cm
Seq desk_changing_1 score RMSE=61.8 cm
Seq einstein_1 score RMSE=2.2 cm
Seq einstein_2 score RMSE=3.1 cm
Seq einstein_flashlight score RMSE=138.4 cm
Seq einstein_global_light_changes_1 score RMSE=80.1 cm
Seq einstein_global_light_changes_2 score RMSE=2.0 cm
Seq einstein_global_light_changes_3 score RMSE=74.9 cm
Seq kidnap_1 score RMSE=4.5 cm
Seq large_loop_1 score RMSE=5.9 cm
Seq mannequin_1 score RMSE=44.1 cm
Seq mannequin_3 score RMSE=6.9 cm
Seq mannequin_4 score RMSE=6.7 cm
Seq mannequin_5 score RMSE=10.5 cm
Seq mannequin_7 score RMSE=13.6 cm
Seq mannequin_face_1 score RMSE=2.9 cm
Seq mannequin_face_2 score RMSE=2.4 cm
Seq mannequin_face_3 score RMSE=10.8 cm
Seq mannequin_head score RMSE=28.6 cm
Seq motion_1 score RMSE=6.6 cm
Seq planar_2 score RMSE=3.4 cm
Seq planar_3 score RMSE=6.8 cm
Seq plant_1 score RMSE=1.1 cm
Seq plant_2 score RMSE=2.1 cm
Seq plant_3 score RMSE=2.9 cm
Seq plant_4 score RMSE=2.8 cm
Seq plant_5 score RMSE=6.6 cm
Seq plant_scene_1 score RMSE=4.1 cm
Seq plant_scene_2 score RMSE=4.8 cm
Seq plant_scene_3 score RMSE=5.7 cm
Seq reflective_1 score RMSE=39.4 cm
Seq repetitive score RMSE=28.8 cm
Seq sfm_bench score RMSE=8.5 cm
Seq sfm_garden score RMSE=424.5 cm
Seq sfm_house_loop score RMSE=433.1 cm
Seq sfm_lab_room_1 score RMSE=21.0 cm
Seq sfm_lab_room_2 score RMSE=5.7 cm
Seq sofa_1 score RMSE=2.5 cm
Seq sofa_2 score RMSE=4.7 cm
Seq sofa_3 score RMSE=3.5 cm
Seq sofa_4 score RMSE=3.5 cm
Seq sofa_shake score RMSE=3.7 cm
Seq table_3 score RMSE=2.2 cm
Seq table_4 score RMSE=2.1 cm
Seq table_7 score RMSE=2.8 cm
Seq vicon_light_1 score RMSE=13.8 cm
Seq vicon_light_2 score RMSE=4.0 cm
Mean ALL RMSE= 36.2 cm
Median ALL RMSE= 6.6 cm
####################
Avg scaling error: 30.73448%
Median scaling error: 19.88923%
```
