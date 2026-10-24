<p align="center">

  <h1 align="center">📌 PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map</h1>

  <p align="center">
    <a href="#run-pings"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="#installation"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red.svg?style=flat-square" /></a>
    <a href="https://github.com/PRBonn/PINGS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>
  <h3 align="center"><a href="https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf">Paper</a> | <a href="https://www.youtube.com/watch?v=XuOLoZdcReA&t=2s">Video</a> | <a href="https://www.youtube.com/watch?v=bbAw_ZC_1fI">Demo</a> </h3>
  <div align="center"></div>
</p>

**TL;DR: PINGS is a LiDAR-visual SLAM system unifying distance fields and radiance fields within a neural point map**

> Fork notes: this README adds the local RTX 5080 / CUDA 12.8 build recipe and HKU Landmark workflow. The HKU adapter remains a local prerequisite; see [HKU Landmark dataset](#hku-landmark-dataset-local-adapter).

![teaser](https://github.com/user-attachments/assets/0ec8b71d-8902-445c-a07c-f9a7e08fb3e2)

<details>
  <summary>[Demo videos (click to expand)]</summary>


### Rendering from PINGS map of Bonn

| example 1 | example 2 |
| :-: | :-: |
| <video src='https://github.com/user-attachments/assets/bff4c53a-a5b8-4912-86af-7cb1bfa4248f'> | <video src='https://github.com/user-attachments/assets/a8e5be1c-bc64-4f39-92b1-31a6eea48b25'> |

| example 3 | example 4 |
| :-: | :-: |
| <video src='https://github.com/user-attachments/assets/f7f69d77-9b67-4cf0-a491-f57f96f6f940'> | <video src='https://github.com/user-attachments/assets/a6733384-39ff-443a-a26e-66429bf5cf4e'> |


### SLAM demo

| HKU | KITTI | IPB Car |
| :-: | :-: | :-: |
| <video src='https://github.com/user-attachments/assets/d4de3d86-589d-4f84-ad30-102fdac9fb53'> | <video src='https://github.com/user-attachments/assets/332c8633-6ff6-4dc2-bd42-19944cc8cfed'> | <video src='https://github.com/user-attachments/assets/839208c0-18ab-401a-bc67-747889f539b2'> |


</details>

----

## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Robots require high-fidelity reconstructions of their environment for effective operation. Such scene representations should be both, geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, the scalable incremental mapping of both fields consistently and at the same time with high quality remains challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to the state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by leveraging the constraints from the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction.
</details>


## Installation

### Workstation and build profile

The local HKU experiments used an Ubuntu 24.04 workstation with an NVIDIA RTX 5080 (16 GB VRAM) and 32 GB RAM. The CUDA extensions were built with **CUDA Toolkit 12.8**, selected inside the `pings` Conda environment.

| Component | Configuration |
| --- | --- |
| OS | Ubuntu 24.04, x86-64 |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| CUDA compiler | `nvcc` 12.8 |
| PyTorch | CUDA 12.8 wheel; keep the version in an existing working environment |
| NumPy / OpenCV | `numpy==1.26.4` / `opencv-python==4.11.0.86` |
| Fresh-environment example below | Python 3.10, PyTorch 2.9.0, torchvision 0.24.0, torchaudio 2.9.0 |

The pinned PyTorch command below is a reference recipe for a new environment, not a package freeze captured from the successful PINGS run. If PINGS already works, keep that environment and skip recreating it.

The CUDA version displayed by `nvidia-smi` describes driver support. The compiler used to build extensions is the one reported by `nvcc --version`. Compute capability `sm_120` on this GPU is **not** CUDA Toolkit 12.0; the old CUDA 12.0 compiler could not build the extensions for this architecture.

### 0. Clone this fork and its submodules

For a new checkout:

```bash
mkdir -p ~/my_ws
cd ~/my_ws
git clone --recursive https://github.com/ntdathp/PINGS.git
cd PINGS
git submodule update --init --recursive
```

For an existing checkout, start from its repository root and initialize any missing submodules:

```bash
cd ~/my_ws/PINGS
git submodule update --init --recursive
```

### 1. Prepare the environment

For a new environment only:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake libgl1 libglib2.0-0 libglfw3

conda create -n pings python=3.10 -y
conda activate pings
python -m pip install setuptools wheel ninja

python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

The version pairing is from [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/).

For an existing working environment, activate it instead:

```bash
conda activate pings
cd ~/my_ws/PINGS
```

### 2. Select CUDA Toolkit 12.8

If it is not already installed in this Conda environment:

```bash
conda install cuda -c nvidia/label/cuda-12.8.1
```

Select the Conda toolkit in every shell used to build the CUDA extensions:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
hash -r

command -v nvcc
nvcc --version
```

Continue only when `nvcc` reports **12.8**. If a complete system toolkit is used instead, select its actual root as `CUDA_HOME`; setting a nonexistent `/usr/local/cuda-12.8` path does not install it.

Check the selected Python/PyTorch runtime:

```bash
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is unavailable in this environment"
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Built architectures:", torch.cuda.get_arch_list())
print("CUDA tensor check:", (torch.ones(4, device="cuda") + 1).cpu())
PY
```

### 3. Apply the rasterizer header fix

The CUDA 12.8 build encountered missing `std::uintptr_t` and `uint32_t` declarations in the surfel rasterizer. Add the integer headers before building. This patch is safe to repeat:

```bash
python - <<'PY'
from pathlib import Path

path = Path("submodules/diff-gaussian-surfel-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h")
text = path.read_text()
missing = [h for h in ("#include <cstdint>", "#include <stdint.h>") if h not in text]
if missing:
    path.write_text("\n".join(missing) + "\n" + text)
print("Rasterizer integer headers are present.")
PY
```

### 4. Install dependencies and build the extensions

The fork's `requirements.txt` keeps NumPy 1.x, pins the compatible OpenCV release, and includes `rosbags` and `PyYAML` for the HKU exporter. PyTorch is installed separately.

```bash
python -m pip install -r requirements.txt --no-build-isolation
```

The two local CUDA extensions are intentionally commented out in `requirements.txt`. **For a fresh environment**, build them explicitly after installing PyTorch and selecting CUDA 12.8:

```bash
python -m pip install --no-build-isolation \
  ./submodules/diff-gaussian-surfel-rasterization-w-pose \
  ./submodules/fused-ssim
```

For an already working environment, skip that extension build. If PyTorch is changed later, rebuild the extensions against the new PyTorch installation.

Basic checks:

```bash
python -m pip check
python pings.py -h
python inspect_pings.py -h
```

### 5. Loading locally generated checkpoints with newer PyTorch

The current inspector uses `torch.load(model_path)`. If loading your own saved PINGS map fails with a weights-only / unsupported-global error on newer PyTorch, explicitly load the full local checkpoint:

```bash
python - <<'PY'
from pathlib import Path

path = Path("inspect_pings.py")
text = path.read_text()
text = text.replace(
    "torch.load(model_path)",
    "torch.load(model_path, weights_only=False)",
)
path.write_text(text)
PY
```

Use this setting only for checkpoints you created or otherwise trust. It restores full Python-object checkpoint loading.

## Docker

The Docker instructions below are retained from upstream. This Docker build has not been validated for the RTX 5080 / CUDA 12.8 workstation; the Conda instructions above describe the local build workflow.

<details>
  <summary>[Details (click to expand)]</summary>

For the prerequisite, you need to install docker and nvidia container runtime.
Firstly edit your data directory at: `- /your_data_storage_directory_here:/packages/pings/data/` in `docker-compose.yml`.
Then build the image by `make build` and run the container by `make run`. 
Afterwards you can use pings as you wish in the terminal. 

</details>

## Data Preparation

A dataset with both RGB and depth observations (LiDAR or depth camera) with extrinsic and intrinsic calibration parameters is required. Note that the input images are supposed to have been already undistorted.

If only range measurements are available, you can still run PINGS with the `--gs-off` flag while PINGS would degenerate to [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM).

To extract individual observations from a ROS bag, you may use the [ROS bag converter tool](https://github.com/YuePanEdward/rosbag_converter).

For your own dataset, you may need to implement a new dataloader class and put it in the `dataset/dataloaders` folder. 
Check [here](https://github.com/PRBonn/PINGS/blob/main/dataset/dataloaders/ipb_car.py) for an example.


## HKU Landmark dataset: local adapter

### Adapter availability

The HKU workflow below documents the local FAST-LIVO2 adapter developed for these experiments. **The adapter is not yet included in this GitHub fork.** A clean clone alone does not provide the `fastlivo2` dataloader.

Before using the HKU commands, copy these files from the working PINGS installation into the corresponding locations in the checkout:

- `pings_hku_tools/`: the complete updated tools directory, including `prepare_hku.py`, the loader source, and installer support.
- `config/run_fastlivo2_stable.yaml`: the stable configuration used in the successful tracking run.

Run the adapter installer to register the loader and apply its dataset hook:

```bash
conda activate pings
cd ~/my_ws/PINGS

python pings_hku_tools/prepare_hku.py install \
  --pings-root "$PWD" --update

test -f dataset/dataloaders/fastlivo2.py
test -f config/run_fastlivo2_stable.yaml
python pings.py -h
```

These instructions do not install an upstream HKU loader: they require the updated local adapter and stable configuration described above.

### Input bag and sensor topics

The source is the original **ROS1** FAST-LIVO2 bag:

```text
/home/ntd/data/fast_livo2/HKU_Landmark.bag
```

| Topic | Message | Use in this adapter |
| --- | --- | --- |
| `/livox/lidar` | Livox CustomMsg | LiDAR points and per-point timing |
| `/left_camera/image` | sensor_msgs/Image | Camera observations |
| `/livox/imu` | sensor_msgs/Imu | Present in the bag; not used by this PINGS adaptation |

No ROS2 conversion or prior FAST-LIVO2 execution is required. The exporter prepares files for PINGS, which estimates its own trajectory.

### Landmark calibration

Use the user-verified `hku-landmark` preset. Its spatial calibration is:

```yaml
Rcl: [-0.00200, -0.99975, -0.02211,
      -0.00366,  0.02212, -0.99975,
       0.99999, -0.00192, -0.00371]
Pcl: [0.00260, 0.05057, -0.00587]

cam_width: 1280
cam_height: 1024
scale: 0.5
cam_fx: 1176.2874292149932
cam_fy: 1176.21585445307
cam_cx: 592.1187382755453
cam_cy: 509.0864309628322
cam_d0: -0.13218037625958456
cam_d1: 0.15360732717073536
cam_d2: 0.00036918417348059815
cam_d3: -0.00031715324469463964
```

The extrinsic direction is **LiDAR to camera**:

```text
p_camera = Rcl @ p_lidar + Pcl
```

Do not invert this transform when projecting points that are still in the LiDAR frame. The image scale applies to the images and camera intrinsics, not to the extrinsics.

The exporter undistorts the images. The selected camera timestamp adjustment is **+0.1 s**, added to the image header timestamp; images are matched to the nearest LiDAR scan-end time. This offset was accepted after inspecting projection overlays for this bag. It is not a universal offset for other HKU recordings.

### Short projection check

Start with a new output directory:

```bash
python pings_hku_tools/prepare_hku.py export \
  --bag /home/ntd/data/fast_livo2/HKU_Landmark.bag \
  --output /home/ntd/data/fast_livo2/pings_landmark_check_v2 \
  --preset hku-landmark \
  --image-offset 0.1 \
  --max-scans 100

xdg-open /home/ntd/data/fast_livo2/pings_landmark_check_v2/qa
```

Check projected edges before approving the dataset. After accepting the overlays:

```bash
python pings_hku_tools/prepare_hku.py approve \
  /home/ntd/data/fast_livo2/pings_landmark_check_v2
```

### Full export and corrupt-scan handling

The bag contains 911 LiDAR messages and 911 camera images. One LiDAR scan has widespread invalid point offsets reaching approximately **3.05 s**, instead of the normal approximately 0.1 s scan interval. The updated exporter skips that entire scan, leaving **910 usable LiDAR scans**.

Use the updated exporter with its explicit skip option:

```bash
python pings_hku_tools/prepare_hku.py export \
  --bag /home/ntd/data/fast_livo2/HKU_Landmark.bag \
  --output /home/ntd/data/fast_livo2/pings_landmark_full_v3 \
  --preset hku-landmark \
  --image-offset 0.1 \
  --skip-corrupt-scans
```

Omitting `--max-scans` exports the complete sequence. Use a fresh output directory if exporting again. The updated loader also handles the gap created by the skipped scan; replacing only the exporter with an old loader is insufficient.

Inspect the full export's QA overlays, then approve:

```bash
xdg-open /home/ntd/data/fast_livo2/pings_landmark_full_v3/qa

python pings_hku_tools/prepare_hku.py approve \
  /home/ntd/data/fast_livo2/pings_landmark_full_v3
```

### Stable configuration and deskew note

The local `config/run_fastlivo2_stable.yaml` uses these settings:

```yaml
setting:
  name: hku_landmark_stable
neuralpoints:
  color_feature_dim: 16
tracker:
  source_vox_down_m: 0.2
gs:
  gs_down_rate: 0
  gs_vis_down_rate: 0
  gs_iters: 100
  gs_keyframe_accu_dist: 0.1
  img_pool_size: 60
  long_term_img_pool_size: 120
  learn_color_residual: false
```

This is an excerpt, **not a complete runnable configuration**. Retain the other entries from the working stable file. The images have already been resized during export; `gs_down_rate: 0` avoids another training downsampling step.

In the local comparison, the LiDAR-only run with PINGS deskew enabled lost tracking near frame 320. With `--no-deskew`, it completed all 910 frames. Keep deskew disabled for this baseline. The precise deskew fault has not been established, and IMU-assisted compensation has not been implemented here.

This is a local configuration, not a verified reproduction of the paper's rendering quality. SDF registration residuals are not ground-truth trajectory errors.

## Run PINGS

The following HKU commands require the local adapter, the complete stable configuration, and the approved full export described above. Run them from the PINGS repository root.

### 1. Verify LiDAR tracking first

```bash
conda activate pings
cd ~/my_ws/PINGS

python pings.py config/run_fastlivo2_stable.yaml fastlivo2 \
  -i /home/ntd/data/fast_livo2/pings_landmark_full_v3 \
  -o /home/ntd/data/fast_livo2/pings_results/tracking_no_deskew_full \
  --gs-off --no-deskew -s -l
```

This was the successful **910-frame** tracking baseline. It saves the neural-point map and checkpoint without training the GS branch.

### 2. Run joint Gaussian/SDF mapping

```bash
python pings.py config/run_fastlivo2_stable.yaml fastlivo2 \
  -i /home/ntd/data/fast_livo2/pings_landmark_full_v3 \
  -o /home/ntd/data/fast_livo2/pings_results/gs_no_deskew \
  --no-deskew -v -s -l
```

| Option | Meaning |
| --- | --- |
| `--no-deskew` | Disable PINGS motion compensation for this baseline |
| `--gs-off` | Disable GS training for the tracking check |
| `-v` | Enable visualization |
| `-s` | Save the neural map and checkpoint |
| `-m` | Also reconstruct a mesh at the end; optional |
| `-l` | Print detailed logs |

To reproduce the earlier command that also reconstructs a mesh, append `-m` to the joint-mapping command. Meshing can add substantial time after SLAM finishes.

Each run creates a timestamped subdirectory under the chosen output directory.

### 3. Inspect the saved run

Set `RUN` to the actual timestamped run directory printed by PINGS. For the recorded HKU GS run:

```bash
RUN=/home/ntd/data/fast_livo2/pings_results/gs_no_deskew/hku_landmark_stable_fastlivo2__2026-10-22_13-07-25

python inspect_pings.py "$RUN" -g
```

If checkpoint loading fails on newer PyTorch, see the trusted-checkpoint compatibility note in Installation.

A coarse offline mesh can be reconstructed with:

```bash
python inspect_pings.py "$RUN" -g -m --mesh-mc-m 0.12
```

`--mesh-mc-m` is in **meters**: `0.12` is 12 cm, `0.012` is 12 mm, and `0.0001` is 0.1 mm. Start coarse. Very small marching-cubes spacing can create an impractically large query grid; the local submillimeter attempt hit query limits and required extensive subdivision.

### Other datasets and CLI help

```bash
python pings.py -h
python inspect_pings.py -h
```

## Citation

<details>
  <summary>[Details (click to expand)]</summary>


If you use PINGS for any academic work, please cite our original [paper](https://www.roboticsproceedings.org/rss21/p040.pdf).

```
@inproceedings{pan2025rss,
title = {{PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map}},
author = {Y. Pan and X. Zhong and L. Jin and L. Wiesmann and M. Popovi\'c and J. Behley and C. Stachniss},
booktitle= {Robotics: Science and Systems (RSS)},
year = {2025},
codeurl = {https://github.com/PRBonn/PINGS},
url = {https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/pan2025rss.pdf}
}
```
</details>

## Contact
If you have any questions, please contact:

- [Yue Pan](https://www.ipb.uni-bonn.de/people/yue-pan/index.html) {[yue.pan@igg.uni-bonn.de]()}


## Acknowledgement

<details>
  <summary>[Details (click to expand)]</summary>

PINGS is built on top of our previous work [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM) and we thank the authors for the following works:

* [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
* [Gaussian Surfels](https://github.com/turandai/gaussian_surfels)
* [2DGS](https://github.com/hbb1/2d-gaussian-splatting)
* [Scaffold-GS](https://github.com/city-super/Scaffold-GS)
* [MonoGS](https://github.com/muskie82/MonoGS)
* [Oxford Spires Dataset](https://github.com/ori-drs/oxford_spires_dataset)

</details>
