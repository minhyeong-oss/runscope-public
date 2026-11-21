from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/twente_full_duration_sync_audit_v1"
PROTOCOL_PATH = EXPERIMENT / "config/protocol.json"
VIDEO_ROOT = ROOT / "data/twente_rgb/side_running/Videos_anonymized"
MODEL_PATH = ROOT / "experiments/pose_foot_keypoints_v1/models/pose_landmarker_full.task"
POSE_ROOT = EXPERIMENT / "artifacts/raw_pose"
REPORT_ROOT = EXPERIMENT / "reports"

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mat_path(subject: str, run: int) -> Path:
    if subject == "Subj05":
        return ROOT / f"data/twente_sensor/{subject}/extracted_run{run}/{subject}/{subject}_run_{run}.mat"
    return ROOT / f"data/twente_sensor/{subject}/development_extracted/{subject}/{subject}_run_{run}.mat"


def interpolate_column(table: pd.DataFrame, name: str) -> np.ndarray:
    return table[name].interpolate(limit_direction="both").to_numpy(dtype=float)


def make_landmarker_options() -> vision.PoseLandmarkerOptions:
    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )


def extract_window(video_path: Path, start_s: float, end_s: float) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps
    if end_s > duration_s + 1e-6:
        capture.release()
        raise ValueError(f"Window {start_s}-{end_s}s exceeds video duration {duration_s:.3f}s")
    first_frame = int(round(start_s * fps))
    final_frame = int(round(end_s * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    rows: list[dict] = []
    source_frame = first_frame
    options = make_landmarker_options()
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while source_frame < final_frame:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round((source_frame - first_frame) / fps * 1000.0))
            result = landmarker.detect_for_video(image, timestamp_ms)
            row: dict[str, float | int | bool] = {
                "frame_index": source_frame - first_frame,
                "source_frame_index": source_frame,
                "time_s": (source_frame - first_frame) / fps,
                "source_time_s": source_frame / fps,
                "fps": fps,
                "pose_detected": bool(result.pose_landmarks),
            }
            if result.pose_landmarks:
                for index, name in enumerate(LANDMARK_NAMES):
                    point = result.pose_landmarks[0][index]
                    row[f"{name}_x"] = point.x
                    row[f"{name}_y"] = point.y
                    row[f"{name}_z"] = point.z
                    row[f"{name}_visibility"] = point.visibility
            rows.append(row)
            source_frame += 1
    capture.release()
    return pd.DataFrame(rows)


def synchronization_scan(marker: dict, pose: pd.DataFrame, protocol: dict) -> dict:
    labels = list(marker["DataLabel"])
    marker_data = np.asarray(marker["MarkerData"], dtype=float)
    rate = float(marker["FrameRate"])
    sensor_time = np.arange(marker_data.shape[2], dtype=float) / rate
    video_time = pose["source_time_s"].to_numpy(dtype=float)
    midhip = (
        interpolate_column(pose, "left_hip_x")
        + interpolate_column(pose, "right_hip_x")
    ) / 2.0
    marker_pairs = protocol["synchronization"]["marker_pairs"]
    lo_ms, hi_ms = protocol["synchronization"]["offset_search_ms"]
    step_ms = protocol["synchronization"]["offset_step_ms"]
    candidates: list[dict] = []
    for offset_ms in np.arange(lo_ms, hi_ms + step_ms / 2.0, step_ms):
        offset_s = float(offset_ms / 1000.0)
        correlations: dict[str, float] = {}
        for pose_name, marker_names in marker_pairs.items():
            available = [name for name in marker_names if name in labels]
            if not available:
                continue
            video_signal = interpolate_column(pose, f"{pose_name}_x") - midhip
            marker_signal = np.nanmean(
                np.vstack([marker_data[labels.index(name), 0] for name in available]), axis=0
            ) - marker_data[labels.index("SACR"), 0]
            finite = np.isfinite(marker_signal)
            if int(finite.sum()) < 2:
                continue
            filled = np.interp(sensor_time, sensor_time[finite], marker_signal[finite])
            aligned = np.interp(video_time + offset_s, sensor_time, filled)
            corr = abs(float(np.corrcoef(video_signal, aligned)[0, 1]))
            if np.isfinite(corr):
                correlations[pose_name] = corr
        if correlations:
            candidates.append(
                {
                    "offset_ms": float(offset_ms),
                    "mean_abs_correlation": float(np.mean(list(correlations.values()))),
                    "correlations": correlations,
                }
            )
    if not candidates:
        raise RuntimeError("No finite synchronization candidates")
    best = max(candidates, key=lambda item: item["mean_abs_correlation"])
    return best


def visibility_summary(pose: pd.DataFrame) -> tuple[float, float]:
    columns = [
        "left_heel_visibility", "right_heel_visibility",
        "left_foot_index_visibility", "right_foot_index_visibility",
    ]
    values = np.concatenate(
        [pose[column].to_numpy(dtype=float) for column in columns if column in pose]
    )
    return float(np.nanmean(values)), float(np.nanmedian(values))


def classify_trial(group: pd.DataFrame, protocol: dict) -> dict:
    thresholds = protocol["fixed_audit_thresholds"]
    expected_windows = len(protocol["windows_s"])
    offsets = group["offset_ms"].to_numpy(dtype=float)
    centers = group["window_center_s"].to_numpy(dtype=float)
    correlations = group["mean_abs_correlation"].to_numpy(dtype=float)
    coefficients = np.polyfit(centers, offsets, 1)
    fitted = np.polyval(coefficients, centers)
    residuals = offsets - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((offsets - offsets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rmse = float(np.sqrt(np.mean(residuals**2)))
    offset_range = float(offsets.max() - offsets.min())
    boundary = float(thresholds["offset_scan_boundary_margin_ms"])
    lo_ms, hi_ms = protocol["synchronization"]["offset_search_ms"]
    scan_boundary_hit = bool(np.any(offsets <= lo_ms + boundary) or np.any(offsets >= hi_ms - boundary))
    low_corr = bool(np.any(correlations < float(thresholds["low_correlation"])))
    incomplete = len(group) != expected_windows
    if incomplete:
        classification = "INCOMPLETE_WINDOWS"
    elif scan_boundary_hit:
        classification = "OFFSET_SCAN_BOUNDARY"
    elif low_corr:
        classification = "LOW_CORRELATION"
    elif offset_range <= float(thresholds["constant_offset_range_ms_max"]):
        classification = "STABLE_CONSTANT"
    elif (
        r2 >= float(thresholds["linear_drift_r2_min"])
        and rmse <= float(thresholds["linear_drift_rmse_ms_max"])
    ):
        classification = "POSSIBLE_LINEAR_DRIFT"
    else:
        classification = "NONLINEAR_OR_NOISY"
    return {
        "windows": int(len(group)),
        "offset_mean_ms": float(offsets.mean()),
        "offset_median_ms": float(np.median(offsets)),
        "offset_min_ms": float(offsets.min()),
        "offset_max_ms": float(offsets.max()),
        "offset_range_ms": offset_range,
        "start_to_end_change_ms": float(offsets[-1] - offsets[0]),
        "linear_slope_ms_per_s": float(coefficients[0]),
        "linear_slope_ms_per_min": float(coefficients[0] * 60.0),
        "linear_fit_r2": float(r2),
        "linear_fit_rmse_ms": rmse,
        "correlation_min": float(correlations.min()),
        "correlation_mean": float(correlations.mean()),
        "scan_boundary_hit": scan_boundary_hit,
        "low_correlation_window": low_corr,
        "classification": classification,
    }


def write_reports(window_table: pd.DataFrame, protocol: dict, scope: str) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    window_path = REPORT_ROOT / f"window_sync_audit_{scope}.csv"
    trial_path = REPORT_ROOT / f"trial_drift_summary_{scope}.csv"
    json_path = REPORT_ROOT / f"AUDIT_SUMMARY_{scope}.json"
    markdown_path = REPORT_ROOT / f"AUDIT_REPORT_{scope}_KO.md"
    window_table.sort_values(["subject", "run", "window_start_s"]).to_csv(
        window_path, index=False, encoding="utf-8-sig"
    )
    trial_rows = []
    for (subject, run), group in window_table.groupby(["subject", "run"], sort=True):
        ordered = group.sort_values("window_center_s")
        trial_rows.append(
            {"subject": subject, "run": int(run), **classify_trial(ordered, protocol)}
        )
    trial_table = pd.DataFrame(trial_rows)
    trial_table.to_csv(trial_path, index=False, encoding="utf-8-sig")
    counts = trial_table["classification"].value_counts().to_dict()
    summary = {
        "status": "SYNC_DRIFT_AUDIT_COMPLETE_NO_FORCE_EVENTS_USED_NO_MODEL_TRAINING",
        "scope": scope,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "development_subjects_only": sorted(window_table["subject"].unique().tolist()),
        "forbidden_subjects_accessed": False,
        "force_events_used_for_synchronization": False,
        "windows": int(len(window_table)),
        "trials": int(len(trial_table)),
        "classification_counts": {str(k): int(v) for k, v in counts.items()},
        "stable_constant_trials": int((trial_table["classification"] == "STABLE_CONSTANT").sum()),
        "outputs": {
            "window_csv": str(window_path),
            "trial_csv": str(trial_path),
            "markdown": str(markdown_path),
        },
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Twente full-duration synchronization drift audit — {scope}",
        "",
        "- 힘판 접지 이벤트를 정렬 offset 선택에 사용하지 않았다.",
        f"- 분석 구간: {len(window_table)} windows / {len(trial_table)} trials",
        f"- 프로토콜 SHA-256: `{summary['protocol_sha256']}`",
        "",
        "## Trial classification",
        "",
    ]
    for name, count in counts.items():
        lines.append(f"- {name}: {int(count)}")
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "이 결과는 동기화 안정성 감사이며 모델 성능 향상 결과가 아니다. "
            "STABLE_CONSTANT가 아닌 trial은 전체 길이 확장에 자동 포함하지 않는다.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Audit Twente video-marker offset drift")
    parser.add_argument("--subjects", nargs="+", default=protocol["development_subjects"])
    parser.add_argument("--runs", nargs="+", type=int, default=protocol["runs"])
    parser.add_argument("--scope", default="full")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    forbidden = set(protocol["forbidden_subjects"])
    requested = set(args.subjects)
    if requested & forbidden:
        raise ValueError(f"Forbidden subjects requested: {sorted(requested & forbidden)}")
    unknown = requested - set(protocol["development_subjects"])
    if unknown:
        raise ValueError(f"Unknown development subjects: {sorted(unknown)}")
    POSE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for subject in args.subjects:
        for run in args.runs:
            video_path = VIDEO_ROOT / subject / f"run_{run}_side_anonymized.avi"
            datastr = loadmat(
                mat_path(subject, run), simplify_cells=True, variable_names=["Datastr"]
            )["Datastr"]
            for start_s, end_s in protocol["windows_s"]:
                stem = f"{subject}_run{run}_side_{start_s}s_to_{end_s}s_mediapipe"
                pose_path = POSE_ROOT / f"{stem}.csv"
                if args.skip_existing and pose_path.exists():
                    pose = pd.read_csv(pose_path)
                else:
                    pose = extract_window(video_path, float(start_s), float(end_s))
                    pose.to_csv(pose_path, index=False, encoding="utf-8-sig")
                best = synchronization_scan(datastr["Marker"], pose, protocol)
                mean_vis, median_vis = visibility_summary(pose)
                row = {
                    "subject": subject,
                    "run": int(run),
                    "window_start_s": float(start_s),
                    "window_end_s": float(end_s),
                    "window_center_s": float((start_s + end_s) / 2.0),
                    "frames": int(len(pose)),
                    "fps": float(pose["fps"].iloc[0]),
                    "pose_detection_rate": float(pose["pose_detected"].mean()),
                    "foot_visibility_mean": mean_vis,
                    "foot_visibility_median": median_vis,
                    "offset_ms": float(best["offset_ms"]),
                    "mean_abs_correlation": float(best["mean_abs_correlation"]),
                    "left_heel_correlation": best["correlations"].get("left_heel", np.nan),
                    "right_heel_correlation": best["correlations"].get("right_heel", np.nan),
                    "left_foot_index_correlation": best["correlations"].get("left_foot_index", np.nan),
                    "right_foot_index_correlation": best["correlations"].get("right_foot_index", np.nan),
                    "pose_csv": str(pose_path),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    write_reports(pd.DataFrame(rows), protocol, args.scope)


if __name__ == "__main__":
    main()
