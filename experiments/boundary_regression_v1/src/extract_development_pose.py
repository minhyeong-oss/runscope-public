from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
RUNS = [63, 81, 99]


def extract_clip(video_path: Path, landmarker: vision.PoseLandmarker, start_s: float, end_s: float) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first_frame = int(round(start_s * fps))
    final_frame = int(round(end_s * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    rows: list[dict] = []
    source_frame = first_frame
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
            points = result.pose_landmarks[0]
            for index, name in enumerate(LANDMARK_NAMES):
                point = points[index]
                row[f"{name}_x"] = point.x
                row[f"{name}_y"] = point.y
                row[f"{name}_z"] = point.z
                row[f"{name}_visibility"] = point.visibility
        rows.append(row)
        source_frame += 1
    capture.release()
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Extract development-only MediaPipe pose clips")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--start-s", type=float, default=10.0)
    parser.add_argument("--end-s", type=float, default=22.0)
    args = parser.parse_args()
    output_root = args.root / "experiments/boundary_regression_v1/artifacts/raw_pose"
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = args.root / "experiments/pose_foot_keypoints_v1/models/pose_landmarker_full.task"
    video_root = args.root / "data/twente_rgb/side_running/Videos_anonymized"
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    reports = []
    for subject in SUBJECTS:
        for run in RUNS:
            # VIDEO mode requires monotonically increasing timestamps. A fresh
            # tracker per independent trial also prevents state leaking across
            # different people and speeds.
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                video_path = video_root / subject / f"run_{run}_side_anonymized.avi"
                table = extract_clip(video_path, landmarker, args.start_s, args.end_s)
                path = output_root / f"{subject}_run{run}_side_{int(args.start_s)}s_to_{int(args.end_s)}s_mediapipe.csv"
                table.to_csv(path, index=False, encoding="utf-8-sig")
                detected = int(table["pose_detected"].sum())
                report = {
                    "subject": subject,
                    "run": run,
                    "frames": int(len(table)),
                    "detected_frames": detected,
                    "detection_rate": detected / max(1, len(table)),
                    "output": str(path),
                }
                reports.append(report)
                print(json.dumps(report), flush=True)
    summary = {
        "experiment": "boundary_regression_v1",
        "scope": "development subjects only",
        "forbidden_subjects_accessed": False,
        "clips": reports,
        "mean_detection_rate": float(np.mean([item["detection_rate"] for item in reports])),
    }
    summary_path = args.root / "experiments/boundary_regression_v1/artifacts/pose_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "mean_detection_rate": summary["mean_detection_rate"]}, indent=2))


if __name__ == "__main__":
    main()
