from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
PHASE_NAMES = {1: "LANDING", 2: "BRAKING", 3: "PROPULSION", 4: "RECOVERY"}
PHASE_COLORS_BGR = {
    1: (30, 205, 245),
    2: (70, 70, 225),
    3: (75, 185, 85),
    4: (215, 130, 55),
}
PHASE_COLORS_HEX = {1: "#F5CD1E", 2: "#E14646", 3: "#55B94B", 4: "#3782D7"}


def constrained_decode(probabilities: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(probabilities, 1e-12, 1.0))
    n, states = logp.shape
    score = np.full((n, states), -np.inf)
    back = np.full((n, states), -1, dtype=np.int16)
    score[0, 0] = logp[0, 0]
    for time_index in range(1, n):
        for state in range(states):
            candidates = [(score[time_index - 1, state], state)]
            if state > 0:
                candidates.append((score[time_index - 1, state - 1], state - 1))
            best, previous = max(candidates, key=lambda item: item[0])
            score[time_index, state] = best + logp[time_index, state]
            back[time_index, state] = previous
    state = states - 1
    decoded = np.empty(n, dtype=np.int8)
    decoded[-1] = state + 1
    for time_index in range(n - 1, 0, -1):
        state = int(back[time_index, state])
        decoded[time_index - 1] = state + 1
    return decoded


def temporal_prior_decode(
    cycle_fraction: np.ndarray,
    target_speed: float,
    dataset_path: Path,
) -> tuple[np.ndarray, dict]:
    """Decode with speed-specific medians learned from train+validation only."""
    dataset = np.load(dataset_path)
    speeds = np.asarray(dataset["event_speeds"], dtype=float)
    splits = dataset["event_splits"].astype(str)
    development = np.isin(splits, ["train", "validation"])
    available = np.unique(speeds[development])
    selected_speed = float(available[np.argmin(np.abs(available - target_speed))])
    event_indices = np.flatnonzero(development & np.isclose(speeds, selected_speed))
    boundaries = []
    for event_index in event_indices:
        indices = np.flatnonzero(dataset["sample_event_index"] == event_index)
        truth = dataset["y"][indices]
        length = len(truth)
        boundaries.append([
            int(np.flatnonzero(truth == 2)[0]) / length,
            int(np.flatnonzero(truth == 3)[0]) / length,
            int(np.flatnonzero(truth == 4)[0]) / length,
        ])
    landing_end, propulsion_start, recovery_start = np.median(boundaries, axis=0)
    labels = np.full(len(cycle_fraction), 4, dtype=np.int8)
    labels[cycle_fraction < recovery_start] = 3
    labels[cycle_fraction < propulsion_start] = 2
    labels[cycle_fraction < landing_end] = 1
    metadata = {
        "nearest_training_speed_m_s": selected_speed,
        "training_strides": int(len(event_indices)),
        "threshold_cycle_fractions": {
            "landing_end": float(landing_end),
            "propulsion_start": float(propulsion_start),
            "recovery_start": float(recovery_start),
        },
    }
    return labels, metadata


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.sum(first * second, axis=1) / np.clip(denominator, 1e-9, None)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def smooth(values: pd.Series, window: int = 9) -> np.ndarray:
    array = values.interpolate(limit_direction="both").to_numpy(dtype=float)
    if len(array) >= window:
        return savgol_filter(array, window, 2)
    return array


def phase_ranges(labels: np.ndarray, sample_rate: float) -> list[dict]:
    rows = []
    for phase in range(1, 5):
        indices = np.flatnonzero(labels == phase)
        rows.append({
            "phase_label": phase,
            "phase_name": PHASE_NAMES[phase].lower(),
            "start_ms": float(indices[0] / sample_rate * 1000.0),
            "end_ms": float((indices[-1] + 1) / sample_rate * 1000.0),
            "frames_100hz": int(len(indices)),
        })
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="RunScope video-to-four-stage prototype")
    parser.add_argument("source_video", type=Path)
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("--model", type=Path, default=root / "model_v1/artifacts/runscope_phase_model_v1.joblib")
    parser.add_argument("--output-dir", type=Path, default=root / "video_v1/outputs")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--stride-start", type=int, default=104, help="Inclusive source frame")
    parser.add_argument("--stride-end", type=int, default=130, help="Exclusive source frame")
    parser.add_argument("--speed", type=float, default=1.0314, help="V3D pelvis-derived planar speed in m/s")
    parser.add_argument(
        "--decoder",
        choices=["temporal-prior-v2", "kinematic-ml-v1"],
        default="temporal-prior-v2",
        help="v2 uses speed-specific boundaries learned from Reznick train+validation.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    landmarks = pd.read_csv(args.landmarks_csv)
    if args.stride_end > len(landmarks) or args.stride_end <= args.stride_start:
        raise SystemExit("Invalid stride range")

    coordinate_columns = [
        column for column in landmarks.columns if column.endswith("_x") or column.endswith("_y")
    ]
    smoothed = {column: smooth(landmarks[column]) for column in coordinate_columns}

    def points(name: str, frame_slice: slice) -> np.ndarray:
        return np.column_stack([smoothed[f"{name}_x"][frame_slice], smoothed[f"{name}_y"][frame_slice]])

    source_slice = slice(args.stride_start, args.stride_end)
    source_frames = np.arange(args.stride_start, args.stride_end)
    source_time = (source_frames - args.stride_start) / fps
    duration_s = (args.stride_end - args.stride_start) / fps
    sample_count = int(round(duration_s * 100.0))
    target_time = np.arange(sample_count) / 100.0

    left_hip = points("left_hip", source_slice)
    right_hip = points("right_hip", source_slice)
    left_shoulder = points("left_shoulder", source_slice)
    right_shoulder = points("right_shoulder", source_slice)
    hip = points(f"{args.side}_hip", source_slice)
    knee = points(f"{args.side}_knee", source_slice)
    ankle = points(f"{args.side}_ankle", source_slice)
    mid_hip = (left_hip + right_hip) / 2.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    forward_sign = float(np.sign(np.polyfit(source_frames, mid_hip[:, 0], 1)[0]) or 1.0)

    hip_flexion = forward_sign * np.degrees(
        np.arctan2(knee[:, 0] - hip[:, 0], knee[:, 1] - hip[:, 1])
    )
    knee_flexion = 180.0 - joint_angle(hip, knee, ankle)
    shank_orientation = forward_sign * np.degrees(
        np.arctan2(ankle[:, 0] - knee[:, 0], ankle[:, 1] - knee[:, 1])
    )
    ankle_flexion_proxy = shank_orientation + 20.0
    pelvis_tilt_proxy = forward_sign * np.degrees(
        np.arctan2(mid_shoulder[:, 0] - mid_hip[:, 0], mid_hip[:, 1] - mid_shoulder[:, 1])
    )

    source_angles = np.column_stack([
        hip_flexion, knee_flexion, ankle_flexion_proxy, pelvis_tilt_proxy
    ])
    target_angles = np.column_stack([
        np.interp(target_time, source_time, source_angles[:, column]) for column in range(4)
    ])
    target_angles = savgol_filter(target_angles, 11, 3, axis=0, mode="interp")
    velocities = np.gradient(target_angles, 0.01, axis=0)
    accelerations = np.gradient(velocities, 0.01, axis=0)
    cycle_fraction = np.clip(target_time / duration_s, 0.0, 0.999999)
    features = np.column_stack([
        np.full(sample_count, args.speed),
        cycle_fraction,
        target_angles,
        velocities,
        accelerations,
    ]).astype(np.float32)

    bundle = joblib.load(args.model)
    expected_features = list(bundle["feature_names"])
    if features.shape[1] != len(expected_features):
        raise RuntimeError("Video feature count does not match the model schema")
    probabilities = bundle["model"].predict_proba(features)
    decoder_metadata = None
    if args.decoder == "temporal-prior-v2":
        labels, decoder_metadata = temporal_prior_decode(
            cycle_fraction,
            args.speed,
            root / "model_v1/data/runscope_phase_frames_v1.npz",
        )
    else:
        labels = constrained_decode(probabilities)
    predicted_probability = probabilities[np.arange(sample_count), labels - 1]

    output_stem = f"{args.source_video.stem}_{args.side}_stride_{args.stride_start}_{args.stride_end}"
    feature_table = pd.DataFrame(features, columns=expected_features)
    feature_table.insert(0, "time_s", target_time)
    feature_table.insert(0, "source_frame_nearest", args.stride_start + np.rint(target_time * fps).astype(int))
    feature_table["phase_label"] = labels
    feature_table["phase_name"] = [PHASE_NAMES[int(label)].lower() for label in labels]
    feature_table["predicted_probability"] = predicted_probability
    feature_path = args.output_dir / f"{output_stem}_features_and_prediction.csv"
    feature_table.to_csv(feature_path, index=False, encoding="utf-8-sig")

    ranges = phase_ranges(labels, 100.0)
    range_path = args.output_dir / f"{output_stem}_phase_ranges.csv"
    pd.DataFrame(ranges).to_csv(range_path, index=False, encoding="utf-8-sig")

    overlay_path = args.output_dir / f"{output_stem}_phase_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    cap = cv2.VideoCapture(str(args.source_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.stride_start)
    for source_frame in range(args.stride_start, args.stride_end):
        ok, frame = cap.read()
        if not ok:
            break
        local_time = (source_frame - args.stride_start) / fps
        prediction_index = int(np.clip(round(local_time * 100.0), 0, sample_count - 1))
        phase = int(labels[prediction_index])
        color = PHASE_COLORS_BGR[phase]
        for first, second in CONNECTIONS:
            confidence = min(
                float(landmarks.loc[source_frame, f"{first}_confidence"]),
                float(landmarks.loc[source_frame, f"{second}_confidence"]),
            )
            if confidence < 0.20:
                continue
            first_point = (
                int(landmarks.loc[source_frame, f"{first}_x"] * width),
                int(landmarks.loc[source_frame, f"{first}_y"] * height),
            )
            second_point = (
                int(landmarks.loc[source_frame, f"{second}_x"] * width),
                int(landmarks.loc[source_frame, f"{second}_y"] * height),
            )
            cv2.line(frame, first_point, second_point, color, 5, cv2.LINE_AA)
            cv2.circle(frame, first_point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, second_point, 6, color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (width, 86), (18, 18, 18), -1)
        cv2.putText(
            frame, f"PHASE {phase}: {PHASE_NAMES[phase]}", (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "PROTOTYPE - 2D proxy / speed outside trained range", (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (230, 230, 230), 1, cv2.LINE_AA,
        )
        writer.write(frame)
    cap.release()
    writer.release()

    plot_path = args.output_dir / f"{output_stem}_joint_phase.png"
    figure, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    angle_names = ["Hip flexion", "Knee flexion", "Ankle proxy", "Trunk/pelvis proxy"]
    for axis, angle_name, angle_values in zip(axes, angle_names, target_angles.T):
        for phase in range(1, 5):
            indices = np.flatnonzero(labels == phase)
            axis.axvspan(
                target_time[indices[0]] * 1000.0,
                (target_time[indices[-1]] + 0.01) * 1000.0,
                color=PHASE_COLORS_HEX[phase], alpha=0.16, linewidth=0,
            )
        axis.plot(target_time * 1000.0, angle_values, color="#202020", linewidth=2)
        axis.set_ylabel(f"{angle_name}\n(deg)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("RunScope video prototype: joint-angle proxies and predicted phases", loc="left")
    axes[-1].set_xlabel("Time from proxy heel strike (ms)")
    figure.savefig(plot_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    report = {
        "status": "experimental_out_of_domain_prototype",
        "source_video": str(args.source_video),
        "pose_landmarks": str(args.landmarks_csv),
        "tracked_side": args.side,
        "stride_source_frames": [args.stride_start, args.stride_end],
        "stride_duration_s": duration_s,
        "nominal_speed_m_s_from_v3d": args.speed,
        "trained_speed_range_m_s": [1.8, 2.4],
        "speed_in_training_range": 1.8 <= args.speed <= 2.4,
        "mean_predicted_probability": float(np.mean(predicted_probability)),
        "decoder": args.decoder,
        "decoder_metadata": decoder_metadata,
        "phase_ranges": ranges,
        "warnings": [
            "YOLO COCO-17 has no heel or foot-index landmark; ankle flexion is a shank-orientation proxy.",
            "Pelvis tilt is approximated by trunk lean in a monocular side view.",
            "The source speed is below the model training range.",
            "The model was trained on laboratory 3D joint angles, not 2D video angles.",
            "BML-MoVi has no force-derived four-stage labels, so this clip cannot measure phase accuracy.",
        ],
        "outputs": {
            "overlay_video": str(overlay_path),
            "features_and_prediction_csv": str(feature_path),
            "phase_ranges_csv": str(range_path),
            "joint_phase_plot": str(plot_path),
        },
    }
    report_path = args.output_dir / f"{output_stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
