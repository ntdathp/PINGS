# Run PINGS on HKU_Landmark.bag

This adapter reads the original ROS1 Livox CustomMsg + camera bag without ROS,
exports undistorted RGB and per-scan local LiDAR points, and installs a small
PINGS dataset loader. PINGS estimates its own trajectory; FAST-LIVO2 and its IMU
estimator are not run. The input bag is never modified. The installer adds a
small, backed-up change to PINGS's `dataset/slam_dataset.py` so the adapter can
disable constant-period motion compensation on individual frames near a gap.
Other loaders keep their original behavior.

## 1. Install

Extract this ZIP in `~/Downloads`, then:

```bash
conda activate pings
cd ~/Downloads/pings_hku_tools
python -m pip install rosbags opencv-python pyyaml
python prepare_hku.py install --pings-root /home/ntd/my_ws/PINGS
```

This adds `dataset/dataloaders/fastlivo2.py` and
`config/run_fastlivo2_gs.yaml`. It refuses to overwrite either file unless you
pass `--update`; that option keeps one backup of the previous files. The config
is copied from your installed `run_r3live_gs.yaml`, with an end-of-scan deskew
reference matching the exporter. Gaussian splatting remains enabled.
The installer uses image pyramid level 0 for training and viewing: the exporter
has already resized images to 640x512. Earlier bundle versions inherited level
1 from R3LIVE, which reduced effective training images again to 320x256.

## 2. Export a short calibration/timing check

The preferred calibration is the pair of YAMLs that already worked with your
FAST-LIVO2 Landmark run:

```bash
python prepare_hku.py export \
  --bag /home/ntd/data/fast_livo2/HKU_Landmark.bag \
  --output /home/ntd/data/fast_livo2/pings_landmark_check \
  --mapping-config /actual/path/to/your/avia.yaml \
  --camera-config /actual/path/to/your/camera_pinhole.yaml \
  --max-scans 100
```

The `hku-landmark` preset contains the exact sequence-specific Rcl, Pcl,
intrinsics, and distortion supplied by the user on 22 October 2026:

```bash
python prepare_hku.py export \
  --bag /home/ntd/data/fast_livo2/HKU_Landmark.bag \
  --output /home/ntd/data/fast_livo2/pings_landmark_check \
  --preset hku-landmark --image-offset 0.1 \
  --max-scans 100
```

**Spatial calibration is now specific to HKU_Landmark.** It uses the original
1280x1024 intrinsics, Rcl = [-0.00200, -0.99975, -0.02211; -0.00366, 0.02212,
-0.99975; 0.99999, -0.00192, -0.00371], and Pcl = [0.00260, 0.05057,
-0.00587] metres. The older `upstream-avia` preset is retained only for other
datasets; do not use it for this Landmark sequence.

**The supplied spatial calibration does not specify a camera time offset.**
The command above explicitly tries +0.1 seconds as a timing candidate (as in
upstream Avia time-offset settings), not as a verified Landmark timing value.
Check the overlay/report and use the value from your working FAST-LIVO2
configuration when available. `--image-offset SECONDS` is required for this
preset and means seconds ADDED to the camera header. No automatic clock shift
or extrinsic inversion is performed.

The exporter undistorts at original resolution and resizes to half resolution
for the RTX 5080's memory budget. Intrinsics are scaled accordingly. Use
`--image-scale 1` for full-resolution images. `scale` from the original camera
YAML is deliberately not applied again. `Rcl/Pcl` are read directly as
`p_camera = Rcl @ p_lidar + Pcl`.

Inspect `pings_landmark_check/qa/overlay_*.jpg` and `qa/report.json`. Colored
points should follow walls, poles, window frames, and other edges. These are
raw-scan overlays without IMU deskew: residual distortion during motion is
possible. Good overlays are a sanity check, not a quantitative calibration
validation. If alignment is wrong, fix the calibration or time offset before
running mapping. A large fraction of unmatched scans also needs investigation.

## 3. Run a short PINGS check

After you inspect the overlays:

```bash
cd ~/Downloads/pings_hku_tools
python prepare_hku.py approve /home/ntd/data/fast_livo2/pings_landmark_check

cd /home/ntd/my_ws/PINGS
python pings.py config/run_fastlivo2_gs.yaml fastlivo2 \
  -i /home/ntd/data/fast_livo2/pings_landmark_check \
  -o /home/ntd/data/fast_livo2/pings_results/check \
  -v -s -l
```

The loader only accepts exports you have marked as visually checked. This is
an adapter workflow choice; upstream PINGS does not impose this approval step.

## 4. Export and run the entire bag

Repeat the export with a **new output folder** and omit `--max-scans`. Reuse
the same verified YAMLs and offset. For example, if you verified the preset:

```bash
cd ~/Downloads/pings_hku_tools
python prepare_hku.py export \
  --bag /home/ntd/data/fast_livo2/HKU_Landmark.bag \
  --output /home/ntd/data/fast_livo2/pings_landmark_full \
  --preset hku-landmark --image-offset 0.1 \
  --skip-corrupt-scans

# Inspect the full run's QA overlays before approving.
python prepare_hku.py approve /home/ntd/data/fast_livo2/pings_landmark_full

cd /home/ntd/my_ws/PINGS
python pings.py config/run_fastlivo2_gs.yaml fastlivo2 \
  -i /home/ntd/data/fast_livo2/pings_landmark_full \
  -o /home/ntd/data/fast_livo2/pings_results/full \
  -v -s -m -l
```

`-v` opens the viewer, `-s` saves the map, `-m` saves a mesh, and `-l` prints
logs. Omit `-v` if the viewer fails or you want to reduce overhead. Artifacts
are saved beneath the output directory in PINGS's experiment subfolder.

## Assumptions and limits

### Quality profile

`run_fastlivo2_quality.yaml` is a separate experimental quality profile based
on the R3LIVE baseline. Copy it into PINGS's config directory, and rerun against
the existing export; no new bag conversion is required. It uses level 0 images,
100 joint GS/SDF iterations, 16-dimensional appearance features, 60/120 short/
long-term image pools, and a 0.1m image keyframe threshold instead of the 1m
default derived from max_range=50m. It keeps the baseline 0.2m neural-point
spacing to isolate supervision changes before increasing map density.
These are proposed settings, not a verified reproduction of the paper's HKU
demo. Training can take substantially longer; GPU memory needs must be checked
on the actual scene. The paper's main quantitative datasets and camera views
differ from this Landmark sequence.

```bash
cp ~/my_ws/PINGS/pings_hku_tools/run_fastlivo2_quality.yaml ~/my_ws/PINGS/config/
cd ~/my_ws/PINGS
python pings.py config/run_fastlivo2_quality.yaml fastlivo2 \
  -i /home/ntd/data/fast_livo2/pings_landmark_full_v3 \
  -o /home/ntd/data/fast_livo2/pings_results/quality \
  -v -s -m -l
```

- Expected topics: `/livox/lidar` (CustomMsg) and `/left_camera/image` (Image).
  Override with `--lidar-topic` and `--image-topic` if necessary.
- ROS1 custom message definitions are read from the bag. Supported image types
  include rgb8/bgr8, mono8, common 8-bit Bayer patterns, and CompressedImage.
- The adapter expects approximately 10Hz contiguous scans, as assumed by
  PINGS's camera timestamp compensation. It rejects unmatched interior scans
  and unexpected header-time gaps rather than silently accepting them.
  Isolated points with offset_time outside [0, 0.125] seconds are removed and
  counted in the terminal and QA report, without dropping the scan or changing
  retained timestamps. Filtering stops if more than 1% of a scan's points are
  affected. For the known Landmark scan with approximately 3-second offsets,
  `--skip-corrupt-scans` explicitly discards the entire scan and records its
  source index/header time/reason in dataset.json and qa/report.json. At most
  one scan may be discarded by default (`--max-skipped-scans`).
- An isolated discarded scan creates approximately a 0.2-second step. The first
  valid frame after it, and the following frame, disable source/map deskew and
  residual camera-motion interpolation. Pose registration still runs. All
  other frames retain normal motion compensation. This avoids applying a
  0.1-second motion assumption across the gap, but those two scans may show
  extra motion distortion; real-bag localization quality remains to be checked.
  Larger/irregular gaps are rejected. The installer must update both the loader
  and PINGS's frame-compensation hook; copying only the exporter is insufficient.
- Real Livox `offset_time` values are retained and normalized within each scan.
  PINGS deskews from its estimated frame motion, not from the bag's IMU.
- LiDAR points stay in the LiDAR frame; no global map or prior trajectory is
  needed. Blind range defaults to 0.8m; this is Avia data, not Horizon data.
- Output directories must not already exist. For interrupted exports, choose a
  fresh directory or manually inspect and remove the incomplete one.
- Tested locally on a synthetic ROS1 bag with embedded CustomMsg definitions:
  export, RGB ordering/row padding, timestamps, intrinsic scaling, projection,
  QA gate, loader interface, and config installation passed. The actual user
  bag and GPU PINGS execution were unavailable here and remain untested.

## Sources

- PINGS: https://github.com/PRBonn/PINGS
- Loader baseline: https://github.com/PRBonn/PINGS/blob/main/dataset/dataloaders/r3live.py
- Config baseline: https://github.com/PRBonn/PINGS/blob/main/config/run_r3live_gs.yaml
- Calibration candidate: https://github.com/hku-mars/FAST-LIVO2/blob/main/config/avia.yaml
- Camera candidate: https://github.com/hku-mars/FAST-LIVO2/blob/main/config/camera_pinhole.yaml

The two Python adapter scripts are supplied under the MIT license.

## Tracking failure near frame 652

The observed 326 valid source points out of 3892 is approximately 8.4%, below
PINGS tracker.py's 10% validity requirement. The final stop occurs after more
than ten consecutive invalid registrations, so inspect earlier frames too.
The log alone does not establish the underlying cause.

`run_fastlivo2_stable.yaml` is a proposed diagnostic profile: it disables
learn_color_residual, which in PINGS also disables the camera-FOV-only filtering
of geometry used for SDF mapping, and reduces source voxel size from 0.4m to
0.2m to retain more tracking samples. It keeps the quality-profile GS settings
and original acceptance thresholds. These are unverified on the actual bag.

First run this profile with `--gs-off`, without GUI, against the entire export
to test LiDAR/SDF tracking quickly. If it completes, rerun the same profile with
GS enabled in a new output directory. No bag re-export is needed.
