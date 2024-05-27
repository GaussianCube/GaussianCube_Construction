# GaussianCube Construction

This repository contains the implementation of GaussianCube construction in the paper "[GaussianCube: A Structured and Explicit Radiance Representation for 3D Generative Modeling](https://gaussiancube.github.io/)". This repository is based on the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), thanks to the authors for their great work. 

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:/GaussianCube_Construction.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/GaussianCube/GaussianCube_Construction.git --recursive
```

## Overview

The codebase has 4 main components:
- The proposed densification-constrained fitting
- Gaussian structuralization via optimal transport

## Construction

We use PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate gaussian_splatting
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.

Tip: Downloading packages and creating a new environment with Conda can require a significant amount of disk space. By default, Conda will use the main system hard drive. You can avoid this by specifying a different package download location and an environment on a different drive:

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

#### Modifications

If you can afford the disk space, we recommend using our environment files for setting up a training environment identical to ours. If you want to make modifications, please note that major version changes might affect the results of our method. However, our (limited) experiments suggest that the codebase works just fine inside a more up-to-date environment (Python 3.8, PyTorch 2.0.0, CUDA 12). Make sure to create an environment where PyTorch and its CUDA runtime version match and the installed CUDA SDK has no major version difference with PyTorch's CUDA version.

### Running

#### Densification-constrained Fitting

To run the densification-constrained fitting, we provide an example below:

```shell
python train.py -s ./example_data -m ./output_dc_fitting/ --dataset_type shapenet --white_background --test_iterations 7000 30000 --densification_interval 50 --sh_degree 0 --N_max_pts 32768
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```50``` (every 50 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.
  #### --dataset_type
  Type of dataset to use, ```shapenet``` by default. We provide loaders for ```shapenet``` (ShapeNet), ```omni``` (OmniObject3D) and ```objaverse``` (Objaverse) datasets.
  #### --N_max_pts
  Maximum number of Gaussians when fitting a single object , ```32768``` by default.
  #### --no_tqdm
  Flag to disable tqdm progress bar.
</details>
<br>

We also provide a script to perform large-scale fitting in parallel:
```shell
python run_fitting.py --source_path ./example_data --output_path ./output_dc_fitting/ --txt_file ./example_data/shapenet_car.txt --start_idx 0 --end_idx 1 --num_gpus 1 --N_max_pts 32768
``` 
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for run_fitting.py</span></summary>

  #### --source_path
  Path to the source directory of data set.
  #### --output_path
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --txt_file
  Index file of the object to fit.
  #### --start_idx
  Starting index of the object to fit.
  #### --end_idx
  Ending index of the object to fit.
  #### --num_gpus
  Number of GPUs to use for parallel fitting, ```1``` by default.
  #### --N_max_pts
  Maximum number of Gaussians when fitting a single object , ```32768``` by default.
</details>
<br>
#### Gaussian Structuralization via Optimal Transport

After obtaining the fitted Gaussians, we further structuralize them via optimal transport. The running script is as follows:

```shell
python scripts/ot_structuralization.py --source_root ./output_dc_fitting/ --save_root ./output_gaussiancube/ --txt_file ./example_data/shapenet_car.txt --start_idx 0 --end_idx 1 --num_workers 1
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/ot_structuralization.py</span></summary>

  #### --source_root
  Root path to fitted Gaussians.
  #### --save_root
  Root path to save the structuralized GaussianCubes.
  #### --txt_file
  Index file of the object to be structuralized.
  #### --start_idx
  Starting index of the object to be structuralized.
  #### --end_idx
  Ending index of the object to be structuralized.
  #### --num_workers
  Number of workers to use for parallel structuralization, ```1``` by default.
  #### --bound
  Bounding box of object, ```0.45``` by default.
  #### --visuzalize_mapping
  Flag to visualize the mapping between Gaussians and voxel grid.
</details>
<br>
The structuralized GaussianCubes after activation function are saved in ```save_root/volume_act```, which is the input for our 3D generative modeling. The non-activated structuralized GaussianCubes are saved in ```save_root/volume```. The optional visualizations of the mapping between Gaussians and voxel grid are saved in ```save_root/point_cloud```.

## Citation

If you find our work useful in your research, please consider citing:
```
@article{zhang2024gaussiancube,
  title={GaussianCube: Structuring Gaussian Splatting using Optimal Transport for 3D Generative Modeling},
  author={Zhang, Bowen and Cheng, Yiji and Yang, Jiaolong and Wang, Chunyu and Zhao, Feng and Tang, Yansong and Chen, Dong and Guo, Baining},
  journal={arXiv preprint arXiv:2403.19655},
  year={2024}
}
```
