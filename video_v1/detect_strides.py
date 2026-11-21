from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def smooth(values: pd.Series, window: int = 9) -> np.ndarray:
    array = values.interpolate(limit_direction="both").to_numpy(dtype=float)
    return savgol_filter(array, window, 2) if len(array) >= window else array


def robust_standardize(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    return (values - median) / max(scale, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect same-foot strides from side-view COCO-17 landmarks")
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-stride-s", type=float, default=0.65)
    parser.add_argument("--max-stride-s", type=float, default=1.15)
    parser.add_argument("--min-pelvis-speed", type=float, default=0.004)
    parser.add_argument("--max-side-score", type=float, default=0.28)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--mode", choices=["auto", "overground", "treadmill"], default="auto")
    parser.add_argument("--facing", choices=["auto", "left", "right"], default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.landmarks_csv)
    stem = args.landmarks_csv.stem.replace("_yolo_landmarks", "")
    coordinate_columns = [
        column for column in data.columns if column.endswith("_x") or column.endswith("_y")
    ]
    values = {column: smooth(data[column]) for column in coordinate_columns}

    mid_hip_x = (values["left_hip_x"] + values["right_hip_x"]) / 2.0
    mid_hip_y = (values["left_hip_y"] + values["right_hip_y"]) / 2.0
    mid_shoulder_x = (values["left_shoulder_x"] + values["right_shoulder_x"]) / 2.0
    mid_shoulder_y = (values["left_shoulder_y"] + values["right_shoulder_y"]) / 2.0
    torso_length = np.hypot(mid_shoulder_x - mid_hip_x, mid_shoulder_y - mid_hip_y)
    side_score = (
        np.abs(values["left_shoulder_x"] - values["right_shoulder_x"])
        + np.abs(values["left_hip_x"] - values["right_hip_x"])
    ) / np.clip(2.0 * torso_length, 1e-6, None)

    pelvis_velocity = savgol_filter(mid_hip_x, 21, 2, deriv=1)
    pelvis_excursion = float(np.percentile(mid_hip_x, 90) - np.percentile(mid_hip_x, 10))
    resolved_mode = (
        "overground" if pelvis_excursion >= 0.15 else "treadmill"
    ) if args.mode == "auto" else args.mode
    if resolved_mode == "overground":
        direction = np.sign(pelvis_velocity)
        direction[np.abs(pelvis_velocity) < args.min_pelvis_speed] = 0.0
        facing_source = "pelvis_travel"
    else:
        if args.facing == "auto":
            nose_offset = smooth(data["nose_x"]) - mid_hip_x
            facing_sign = float(np.sign(np.median(nose_offset)) or 1.0)
            facing_source = "nose_relative_to_pelvis"
        else:
            facing_sign = 1.0 if args.facing == "right" else -1.0
            facing_source = "user_argument"
        direction = np.full(len(data), facing_sign, dtype=float)

    event_rows: list[dict] = []
    signals: dict[str, np.ndarray] = {}
    minimum_peak_distance = max(3, int(round(args.min_stride_s * args.fps * 0.85)))
    for tracked_side in ("left", "right"):
        relative_ankle_x = values[f"{tracked_side}_ankle_x"] - mid_hip_x
        if resolved_mode == "overground":
            # This proxy was validated against synchronized V3D heel minima for
            # the moving BML-MoVi clip.
            contact_signal = savgol_filter(-direction * relative_ankle_x, 9, 2)
            prominence_threshold = 0.025
        else:
            # On a treadmill the pelvis is nearly stationary. Heel-strike proxies
            # are therefore based on the ankle being forward and low relative to
            # the ipsilateral hip, rather than on pelvis travel velocity.
            forward_position = direction * relative_ankle_x
            vertical_extension = values[f"{tracked_side}_ankle_y"] - values[f"{tracked_side}_hip_y"]
            contact_signal = savgol_filter(
                0.70 * robust_standardize(forward_position)
                + 0.30 * robust_standardize(vertical_extension),
                9, 2,
            )
            prominence_threshold = 0.35
        signals[tracked_side] = contact_signal
        peak_indices, properties = find_peaks(
            contact_signal,
            distance=minimum_peak_distance,
            prominence=prominence_threshold,
        )
        for peak, prominence in zip(peak_indices, properties["prominences"]):
            confidence = float(np.mean([
                data.loc[peak, f"{tracked_side}_hip_confidence"],
                data.loc[peak, f"{tracked_side}_knee_confidence"],
                data.loc[peak, f"{tracked_side}_ankle_confidence"],
            ]))
            event_rows.append({
                "tracked_side": tracked_side,
                "frame_index": int(peak),
                "time_s": float(peak / args.fps),
                "travel_direction": (
                    ("treadmill_facing_right" if direction[peak] > 0 else "treadmill_facing_left")
                    if resolved_mode == "treadmill"
                    else "right" if direction[peak] > 0 else "left" if direction[peak] < 0 else "turning"
                ),
                "pelvis_speed_normalized_per_frame": float(abs(pelvis_velocity[peak])),
                "side_view_score": float(side_score[peak]),
                "keypoint_confidence": confidence,
                "peak_prominence": float(prominence),
                "accepted_event": bool(
                    (resolved_mode == "treadmill" or direction[peak] != 0)
                    and side_score[peak] <= args.max_side_score
                    and confidence >= args.min_confidence
                ),
            })

    events = pd.DataFrame(event_rows).sort_values(["tracked_side", "frame_index"]).reset_index(drop=True)
    stride_rows: list[dict] = []
    stride_number = 0
    for tracked_side in ("left", "right"):
        side_events = events[
            (events["tracked_side"] == tracked_side) & events["accepted_event"]
        ].sort_values("frame_index")
        records = side_events.to_dict("records")
        for first, second in zip(records[:-1], records[1:]):
            start = int(first["frame_index"])
            end = int(second["frame_index"])
            duration_s = (end - start) / args.fps
            same_direction = first["travel_direction"] == second["travel_direction"]
            confidence_columns = [
                f"{tracked_side}_hip_confidence",
                f"{tracked_side}_knee_confidence",
                f"{tracked_side}_ankle_confidence",
            ]
            interval_confidence = float(
                data.loc[start:end, confidence_columns].min(axis=1).mean()
            )
            interval_side_score = float(np.median(side_score[start:end]))
            accepted_stride = bool(
                args.min_stride_s <= duration_s <= args.max_stride_s
                and same_direction
                and interval_confidence >= args.min_confidence
                and interval_side_score <= args.max_side_score
            )
            if accepted_stride:
                stride_number += 1
            stride_rows.append({
                "stride_id": f"stride_{stride_number:03d}" if accepted_stride else "",
                "tracked_side": tracked_side,
                "start_frame_inclusive": start,
                "end_frame_exclusive": end,
                "duration_s": duration_s,
                "travel_direction": first["travel_direction"] if same_direction else "turnaround",
                "mean_keypoint_confidence": interval_confidence,
                "median_side_view_score": interval_side_score,
                "accepted_stride": accepted_stride,
                "rejection_reason": "" if accepted_stride else (
                    "duration" if not args.min_stride_s <= duration_s <= args.max_stride_s
                    else "turnaround" if not same_direction
                    else "confidence" if interval_confidence < args.min_confidence
                    else "not_side_view"
                ),
            })
    strides = pd.DataFrame(stride_rows).sort_values(
        ["start_frame_inclusive", "tracked_side"]
    ).reset_index(drop=True)

    events_path = args.output_dir / f"{stem}_auto_heel_strikes.csv"
    strides_path = args.output_dir / f"{stem}_auto_strides.csv"
    report_path = args.output_dir / f"{stem}_auto_stride_report.json"
    plot_path = args.output_dir / f"{stem}_auto_stride_detection.png"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    strides.to_csv(strides_path, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    time = np.arange(len(data)) / args.fps
    axes[0].plot(time, mid_hip_x, color="#202020", linewidth=2)
    axes[0].set_ylabel("Pelvis x\n(normalized)")
    axes[1].plot(time, signals["left"], color="#377EB8", label="YOLO left")
    axes[1].plot(time, signals["right"], color="#E41A1C", label="YOLO right")
    accepted = events[events["accepted_event"]]
    for _, row in accepted.iterrows():
        color = "#377EB8" if row["tracked_side"] == "left" else "#E41A1C"
        axes[1].axvline(row["time_s"], color=color, alpha=0.45, linewidth=1)
    axes[1].set_ylabel("Contact proxy")
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(time, side_score, color="#4D4D4D")
    axes[2].axhline(args.max_side_score, color="#E17C05", linestyle="--", linewidth=1)
    for _, row in strides[strides["accepted_stride"]].iterrows():
        axes[2].axvspan(
            row["start_frame_inclusive"] / args.fps,
            row["end_frame_exclusive"] / args.fps,
            color="#59A14F", alpha=0.14, linewidth=0,
        )
    axes[2].set_ylabel("Side-view score")
    axes[2].set_xlabel("Clip time (s)")
    axes[0].set_title("Automatic stride detection from video keypoints", loc="left")
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(plot_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    accepted_strides = strides[strides["accepted_stride"]]
    report = {
        "landmarks_csv": str(args.landmarks_csv),
        "frames": int(len(data)),
        "fps": args.fps,
        "accepted_heel_strikes": int(events["accepted_event"].sum()),
        "accepted_strides": int(len(accepted_strides)),
        "accepted_strides_by_tracked_side": {
            side: int((accepted_strides["tracked_side"] == side).sum())
            for side in ("left", "right")
        },
        "requested_mode": args.mode,
        "resolved_mode": resolved_mode,
        "pelvis_excursion_normalized": pelvis_excursion,
        "facing_source": facing_source,
        "method": (
            "direction-aware ankle position relative to pelvis"
            if resolved_mode == "overground"
            else "forward-and-low ankle position relative to ipsilateral hip and pelvis"
        ),
        "outputs": {
            "heel_strikes_csv": str(events_path),
            "strides_csv": str(strides_path),
            "diagnostic_plot": str(plot_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(accepted_strides.to_string(index=False))


if __name__ == "__main__":
    main()
