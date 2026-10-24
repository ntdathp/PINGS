#!/usr/bin/env python3
"""Export FAST-LIVO2 Livox CustomMsg ROS1 bags for PINGS, without ROS. MIT license.

No poses or IMU are used: PINGS estimates its own LiDAR-visual trajectory.
Calibration can come from the user's working FAST-LIVO2 YAMLs, or an explicitly
selected upstream Avia candidate. Projection QA is required in either case.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


class ScanTimestampError(ValueError):
    """A scan's per-point timestamps cannot be used safely."""


def stamp_ns(msg):
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


def calibration(args):
    if args.mapping_config or args.camera_config:
        if not (args.mapping_config and args.camera_config):
            raise ValueError("Supply both --mapping-config and --camera-config")
        mapping = yaml.safe_load(args.mapping_config.read_text())
        cam = yaml.safe_load(args.camera_config.read_text())
        source = f"User YAMLs: {args.mapping_config}, {args.camera_config}"
    elif args.preset == "hku-landmark":
        # Exact sequence-specific calibration supplied by the user.
        # A camera time offset is separate from spatial calibration.
        if args.image_offset is None:
            raise ValueError("HKU calibration supplied no timing offset. Pass --image-offset explicitly (seconds added to camera stamp).")
        mapping = {"extrin_calib": {
            "Rcl": [-0.00200, -0.99975, -0.02211,
                    -0.00366, 0.02212, -0.99975,
                    0.99999, -0.00192, -0.00371],
            "Pcl": [0.00260, 0.05057, -0.00587]}}
        cam = dict(cam_model="Pinhole", cam_width=1280, cam_height=1024,
                   cam_fx=1176.2874292149932, cam_fy=1176.21585445307,
                   cam_cx=592.1187382755453, cam_cy=509.0864309628322,
                   cam_d0=-0.13218037625958456, cam_d1=0.15360732717073536,
                   cam_d2=0.00036918417348059815, cam_d3=-0.00031715324469463964)
        source = "User-supplied HKU_Landmark spatial calibration; time offset selected explicitly"
    elif args.preset == "upstream-avia":
        # Upstream FAST-LIVO2 config/avia.yaml and config/camera_pinhole.yaml.
        # These are a candidate, NOT a claim that every HKU Landmark bag uses this rig.
        mapping = {"extrin_calib": {
            "Rcl": [0.00610193, -0.999863, -0.0154172,
                    -0.00615449, 0.0153796, -0.999863,
                    0.999962, 0.00619598, -0.0060598],
            "Pcl": [0.0194384, 0.104689, -0.0251952]},
            "time_offset": {"img_time_offset": 0.1}}
        cam = dict(cam_model="Pinhole", cam_width=1280, cam_height=1024,
                   cam_fx=1293.56944, cam_fy=1293.3155,
                   cam_cx=626.91359, cam_cy=522.799224,
                   cam_d0=-0.076160, cam_d1=0.123001,
                   cam_d2=-0.00113, cam_d3=0.000251)
        source = "Upstream FAST-LIVO2 Avia candidate; Landmark calibration not verified"
    else:
        raise ValueError("Use your working --mapping-config and --camera-config, or --preset hku-landmark --image-offset SECONDS")
    if str(cam.get("cam_model", "Pinhole")).lower() != "pinhole":
        raise ValueError("This exporter supports the FAST-LIVO2 pinhole camera model only")
    K = np.array([[cam["cam_fx"], 0, cam["cam_cx"]],
                  [0, cam["cam_fy"], cam["cam_cy"]], [0, 0, 1]], dtype=float)
    dist = np.array([cam.get(f"cam_d{i}", 0.0) for i in range(5)], dtype=float)
    T = np.eye(4)
    # FAST-LIVO2 defines p_camera = Rcl @ p_lidar + Pcl. Do not invert this.
    T[:3, :3] = np.array(mapping["extrin_calib"]["Rcl"]).reshape(3, 3)
    T[:3, 3] = mapping["extrin_calib"]["Pcl"]
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=0.005):
        raise ValueError("Rcl is not a rotation matrix")
    if not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=0.005):
        raise ValueError("Rcl has invalid determinant")
    offset = mapping.get("time_offset", {}).get("img_time_offset", 0.0)
    if args.image_offset is not None:
        offset = args.image_offset
    return K, dist, T, int(cam["cam_width"]), int(cam["cam_height"]), float(offset), source


def decode_image(msg, msgtype):
    if msgtype.endswith("/CompressedImage"):
        img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode compressed image")
        return img
    enc = msg.encoding.lower()
    channels = {"mono8": 1, "8uc1": 1, "rgb8": 3, "bgr8": 3,
                "rgba8": 4, "bgra8": 4}
    bayer = {"bayer_rggb8": cv2.COLOR_BayerRGGB2BGR,
             "bayer_bggr8": cv2.COLOR_BayerBGGR2BGR,
             "bayer_grbg8": cv2.COLOR_BayerGRBG2BGR,
             "bayer_gbrg8": cv2.COLOR_BayerGBRG2BGR}
    if enc not in channels and enc not in bayer:
        raise ValueError(f"Unsupported image encoding {enc!r}; add an explicit conversion")
    n = channels.get(enc, 1)
    rows = np.asarray(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    img = rows[:, :int(msg.width) * n].reshape(int(msg.height), int(msg.width), n)
    if enc in bayer:
        return cv2.cvtColor(img[:, :, 0], bayer[enc])
    code = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR,
            "8uc1": cv2.COLOR_GRAY2BGR}.get(enc)
    return cv2.cvtColor(img, code) if code is not None else img.copy()


def custom_points(msg, blind):
    if not len(msg.points):
        raise ValueError("Empty Livox scan")
    xyz = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)
    offsets = np.array([p.offset_time for p in msg.points], dtype=np.int64)
    # This adapter expects 10Hz Livox scans. Allow 25ms tolerance, but never
    # stretch a scan to seconds because of a few corrupt offset_time values.
    time_valid = (offsets >= 0) & (offsets <= 125_000_000)
    bad_time_count = int((~time_valid).sum())
    if bad_time_count:
        fraction = bad_time_count / len(offsets)
        quantiles = np.percentile(offsets / 1e9, [0, 50, 95, 99, 100]).tolist()
        detail = f"{bad_time_count}/{len(offsets)} points outside [0,0.125]s; offsets p0/p50/p95/p99/max={quantiles}"
        if fraction > 0.01 or not time_valid.any():
            raise ScanTimestampError("Widespread timestamp anomaly; inspect this scan before filtering: " + detail)
        print("[timing QA] Rejecting isolated timestamp outliers: " + detail, flush=True)
    # Retained acquisition times are unchanged; normalize only for PINGS.
    duration = int(offsets[time_valid].max())
    if not 10_000_000 <= duration <= 125_000_000:
        raise ScanTimestampError(f"Unexpected Livox scan span {duration / 1e9:.6f}s; inspect timing before proceeding")
    tags = np.array([getattr(p, "tag", 0) for p in msg.points], dtype=np.uint8)
    # Match FAST-LIVO2's accepted Livox return categories (0x00 and 0x10).
    valid = time_valid & np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) >= blind)
    valid &= np.isin(tags & 0x30, [0x00, 0x10])
    if not valid.any():
        raise ValueError("No valid points in scan")
    return xyz[valid], (offsets[valid] / duration).astype(np.float32), duration, bad_time_count


def overlay(img, xyz, K, T):
    pc = xyz @ T[:3, :3].T + T[:3, 3]
    mask = pc[:, 2] > 0.5
    pc = pc[mask]
    uvw = pc @ K.T
    uv = np.rint(uvw[:, :2] / uvw[:, 2:]).astype(int)
    h, w = img.shape[:2]
    good = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, depth = uv[good], pc[good, 2]
    out = img.copy()
    colors = cv2.applyColorMap((np.minimum(depth, 30) / 30 * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    for (u, v), color in zip(uv[::3], colors[::3]):
        cv2.circle(out, (int(u), int(v)), 1, tuple(map(int, color)), -1)
    return out, len(uv)


def export(args):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    if not args.bag.is_file():
        raise FileNotFoundError(f"Bag not found on this machine: {args.bag}")
    K, dist, T, width, height, offset, source = calibration(args)
    # Do not overwrite a completed or interrupted export.
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("scans", "images", "qa"):
        (args.output / name).mkdir()
    images, scans, skipped_scans = [], [], []
    source_scan_index = -1
    print(f"Calibration: {source}\nImage offset added to header stamp: {offset:+.6f}s", flush=True)
    limit_end = None
    with AnyReader([args.bag], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
        connections = [c for c in reader.connections if c.topic in (args.lidar_topic, args.image_topic)]
        for topic in (args.lidar_topic, args.image_topic):
            found = [c for c in connections if c.topic == topic]
            if not found:
                raise ValueError(f"Missing topic {topic}. Available: {list(reader.topics)}")
            print(topic, [(c.msgtype, c.msgcount) for c in found], flush=True)
        for connection, bag_ns, raw in reader.messages(connections=connections):
            if limit_end is not None and bag_ns > limit_end + 1_000_000_000:
                break
            msg = reader.deserialize(raw, connection.msgtype)
            header_ns = stamp_ns(msg)
            if header_ns <= 0:
                raise ValueError("Zero/negative header timestamp; refusing to replace with bag time")
            if connection.topic == args.image_topic:
                img = decode_image(msg, connection.msgtype)
                if img.shape[:2] != (height, width):
                    raise ValueError(f"Image is {img.shape[1]}x{img.shape[0]}, calibration expects {width}x{height}")
                img = cv2.undistort(img, K, dist)
                out_w, out_h = round(width * args.image_scale), round(height * args.image_scale)
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                rel = f"images/{len(images):06d}.png"
                if not cv2.imwrite(str(args.output / rel), img):
                    raise OSError(f"Could not save {rel}")
                images.append(dict(image=rel, header_ns=header_ns,
                                   corrected_ns=header_ns + round(offset * 1e9)))
            else:
                source_scan_index += 1
                if not connection.msgtype.endswith("/CustomMsg"):
                    raise ValueError(f"Expected Livox CustomMsg, got {connection.msgtype}")
                if args.max_scans and len(scans) >= args.max_scans:
                    continue
                try:
                    xyz, ts, duration, bad_time_count = custom_points(msg, args.blind)
                except ScanTimestampError as exc:
                    if not args.skip_corrupt_scans:
                        raise
                    if len(skipped_scans) >= args.max_skipped_scans:
                        raise ValueError("Exceeded --max-skipped-scans; inspect the bag") from exc
                    skipped_scans.append(dict(source_scan_index=source_scan_index,
                                              header_ns=header_ns, reason=str(exc)))
                    print(f"[timing QA] SKIPPING entire corrupt scan {source_scan_index}: {exc}", flush=True)
                    continue
                rel = f"scans/{len(scans):06d}.npz"
                np.savez(args.output / rel, xyz=xyz, point_ts=ts)
                timebase = int(getattr(msg, "timebase", header_ns))
                scans.append(dict(scan=rel, source_scan_index=source_scan_index, scan_start_ns=header_ns,
                                  scan_end_ns=header_ns + duration,
                                  duration_ns=duration, rejected_timestamp_points=bad_time_count,
                                  timebase_header_diff_ns=timebase - header_ns))
                if len(scans) % 50 == 0:
                    print(f"Exported {len(scans)} scans / {len(images)} images", flush=True)
                if args.max_scans and len(scans) == args.max_scans:
                    limit_end = bag_ns
    if not scans or not images:
        raise ValueError("No scans/images exported")
    images.sort(key=lambda x: x["corrected_ns"])
    times = np.array([x["corrected_ns"] for x in images], dtype=np.int64)
    frames, gaps = [], []
    for scan in scans:
        end = scan["scan_end_ns"]
        j = int(np.searchsorted(times, end))
        choices = [k for k in (j - 1, j) if 0 <= k < len(images)]
        k = min(choices, key=lambda k: abs(int(times[k]) - end))
        gap = abs(int(times[k]) - end) / 1e9
        if gap > args.max_image_gap:
            continue
        frames.append(dict(**scan, image=images[k]["image"],
                           image_corrected_ns=int(times[k]), image_gap_s=gap))
        gaps.append(gap)
    if len(frames) < 2:
        raise ValueError("Fewer than two synchronized frames; check image time offset")
    # Image association must not silently discard interior valid scans.
    used = [int(Path(f["scan"]).stem) for f in frames]
    if np.any(np.diff(used) != 1):
        raise ValueError("Unmatched interior scans: inspect timestamps/offset instead of running discontinuous scans")
    K_scaled = K.copy()
    K_scaled[0] *= out_w / width
    K_scaled[1] *= out_h / height
    durations = np.array([f["duration_ns"] for f in frames]) / 1e9
    steps = np.diff([f["scan_end_ns"] for f in frames]) / 1e9
    source_steps = np.diff([f["source_scan_index"] for f in frames])
    if np.any(np.abs(steps - source_steps * 0.1) > 0.025) or np.any(steps > 0.25):
        raise ValueError("Unexpected inter-scan gap; only isolated skipped scans in approximately 10Hz data are supported")
    # Do not apply a constant-period motion estimate across a missing scan.
    # The first retained frame after the gap and the next frame both skip
    # source/map deskew and residual camera-motion interpolation. Registration
    # still estimates their poses; all other frames keep normal compensation.
    affected = set()
    for j in np.flatnonzero(source_steps > 1):
        affected.update([int(j) + 1, int(j) + 2])
    for j, frame in enumerate(frames):
        frame["skip_motion_compensation"] = j in affected
    meta = dict(format="pings-fastlivo2-v1", source_bag=str(args.bag.resolve()),
                qa_approved=False, calibration_source=source,
                image_offset_added_s=offset,
                calibration=dict(K=K_scaled.tolist(), T_camera_lidar=T.tolist(),
                                 width=out_w, height=out_h,
                                 original_K=K.tolist(), original_distortion=dist.tolist()),
                skipped_scans=skipped_scans, frames=frames)
    (args.output / "dataset.json").write_text(json.dumps(meta, indent=2))
    checks = []
    qa_indices = set(np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int))
    qa_indices.update(j for j in affected if j < len(frames))
    for idx in sorted(qa_indices):
        frame = frames[idx]
        with np.load(args.output / frame["scan"]) as pts:
            img = cv2.imread(str(args.output / frame["image"]))
            view, n = overlay(img, pts["xyz"], K_scaled, T)
        cv2.putText(view, f"frame {idx}, dt={frame['image_gap_s']*1000:.1f}ms, projected={n}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.imwrite(str(args.output / "qa" / f"overlay_{idx:06d}.jpg"), view)
        checks.append(dict(frame=int(idx), projected_points=n))
    report = dict(scans_exported=len(scans), images_exported=len(images), matched_frames=len(frames),
                  dropped_edge_scans=len(scans)-len(frames),
                  rejected_timestamp_points=sum(f["rejected_timestamp_points"] for f in scans),
                  skipped_corrupt_scans=skipped_scans,
                  motion_compensation_disabled_frames=[j for j in sorted(affected) if j < len(frames)],
                  image_gap_median_ms=float(np.median(gaps)*1000),
                  image_gap_max_ms=float(np.max(gaps)*1000),
                  scan_span_median_s=float(np.median(durations)),
                  scan_period_median_s=float(np.median(steps)),
                  max_abs_timebase_header_diff_s=max(abs(f["timebase_header_diff_ns"]) for f in frames)/1e9,
                  overlays=checks,
                  limitation="Overlays use raw scans, without IMU deskew; moving scenes can have residual distortion")
    (args.output / "qa" / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Inspect {args.output}/qa/*.jpg for alignment of walls, poles, and edges.\n"
          "This is a calibration/timing sanity check, not a validation of motion compensation.\n"
          f"After inspection: python prepare_hku.py approve {args.output}")


def patch_gap_support(source):
    marker = "    supports_frame_motion_skip = True  # HKU adapter"
    if marker in source:
        return source
    replacements = [
        ("class SLAMDataset():", "class SLAMDataset():\n" + marker, 1),
        ("        frame_data = self.loader[frame_id_in_folder]",
         "        frame_data = self.loader[frame_id_in_folder]\n"
         "        self.skip_frame_motion_compensation = bool(frame_data.get('skip_motion_compensation', False)) if isinstance(frame_data, dict) else False",
         1),
        ("if self.config.deskew and not self.lose_track:",
         "if self.config.deskew and not self.lose_track and not getattr(self, 'skip_frame_motion_compensation', False):", 2),
        ("if tran_in_frame is not None and self.cur_sensor_ts is not None:",
         "if tran_in_frame is not None and self.cur_sensor_ts is not None and not getattr(self, 'skip_frame_motion_compensation', False):", 1),
    ]
    for before, after, count in replacements:
        if source.count(before) < count:
            raise ValueError("PINGS source differs from the tested revision; cannot safely patch frame compensation")
        source = source.replace(before, after, count)
    compile(source, "slam_dataset.py", "exec")
    return source


def install(root, update=False):
    base = Path(__file__).resolve().parent
    loader = root / "dataset/dataloaders/fastlivo2.py"
    config = root / "config/run_fastlivo2_gs.yaml"
    for p in (loader, config):
        if p.exists() and not update:
            raise FileExistsError(f"Refusing to overwrite {p}")
    if not (root / "pings.py").is_file():
        raise ValueError("--pings-root must point to the PINGS repository")
    cfg = yaml.safe_load((root / "config/run_r3live_gs.yaml").read_text())
    cfg["setting"].update(name="hku_landmark", deskew_ref_ratio=1.0)
    # The export is already half resolution. Level 0 avoids a second halving.
    cfg["gs"]["gs_down_rate"] = 0
    cfg["gs"]["gs_vis_down_rate"] = 0
    slam = root / "dataset/slam_dataset.py"
    source = slam.read_text()
    patched = patch_gap_support(source)
    # Keep a single original backup, and make repeated installs idempotent.
    for p in (loader, config):
        backup = p.with_name(p.name + ".before_hku_update")
        if p.exists() and not backup.exists():
            shutil.copy2(p, backup)
    if patched != source:
        backup = slam.with_name(slam.name + ".before_hku_update")
        if not backup.exists():
            shutil.copy2(slam, backup)
        slam.write_text(patched)
    shutil.copy2(base / "fastlivo2.py", loader)
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Installed {loader}\nInstalled {config}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--bag", type=Path, default=Path("/home/ntd/data/fast_livo2/HKU_Landmark.bag"))
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--mapping-config", type=Path)
    e.add_argument("--camera-config", type=Path)
    e.add_argument("--preset", choices=["hku-landmark", "upstream-avia"])
    e.add_argument("--image-offset", type=float, help="Seconds ADDED to image header stamp; overrides YAML")
    e.add_argument("--image-scale", type=float, default=0.5)
    e.add_argument("--lidar-topic", default="/livox/lidar")
    e.add_argument("--image-topic", default="/left_camera/image")
    e.add_argument("--blind", type=float, default=0.8)
    e.add_argument("--max-image-gap", type=float, default=0.03)
    e.add_argument("--max-scans", type=int, default=0, help="0 means entire bag")
    e.add_argument("--skip-corrupt-scans", action="store_true", help="Discard entire scans with unusable point timestamps; logged in QA")
    e.add_argument("--max-skipped-scans", type=int, default=1, help="Stop if more corrupt scans would be skipped")
    i = sub.add_parser("install")
    i.add_argument("--pings-root", type=Path, required=True)
    i.add_argument("--update", action="store_true", help="Update this adapter's loader/config and back up previous versions")
    a = sub.add_parser("approve", help="Record your visual approval after inspecting overlays")
    a.add_argument("dataset", type=Path)
    args = p.parse_args()
    if args.command == "export":
        if not 0 < args.image_scale <= 1 or args.max_scans < 0 or args.max_image_gap <= 0 or args.blind < 0 or args.max_skipped_scans < 0:
            p.error("Invalid scale, max-scans, gap, or blind value")
        export(args)
    elif args.command == "install":
        install(args.pings_root, args.update)
    else:
        path = args.dataset / "dataset.json"
        meta = json.loads(path.read_text())
        meta["qa_approved"] = True
        path.write_text(json.dumps(meta, indent=2))
        print("Visual QA approval recorded. No numerical calibration accuracy is implied.")


if __name__ == "__main__":
    main()
