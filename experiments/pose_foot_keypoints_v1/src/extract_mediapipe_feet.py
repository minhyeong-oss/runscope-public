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
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]

FOOT_CONNECTIONS = [
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe heel and foot-index landmarks")
    parser.add_argument("video", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    overlay_path = args.output_dir / f"{args.video.stem}_mediapipe_feet_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    rows = []
    detected = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(frame_index / fps * 1000.0))
            result = landmarker.detect_for_video(image, timestamp_ms)
            row = {
                "frame_index": frame_index,
                "time_s": frame_index / fps,
                "pose_detected": bool(result.pose_landmarks),
            }
            if result.pose_landmarks:
                detected += 1
                landmarks = result.pose_landmarks[0]
                world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                for index, name in enumerate(LANDMARK_NAMES):
                    point = landmarks[index]
                    row[f"{name}_x"] = point.x
                    row[f"{name}_y"] = point.y
                    row[f"{name}_z"] = point.z
                    row[f"{name}_visibility"] = point.visibility
                    row[f"{name}_presence"] = point.presence
                    if world:
                        row[f"{name}_world_x"] = world[index].x
                        row[f"{name}_world_y"] = world[index].y
                        row[f"{name}_world_z"] = world[index].z
                for first, second in FOOT_CONNECTIONS:
                    first_point, second_point = landmarks[first], landmarks[second]
                    if min(first_point.visibility, second_point.visibility) < args.min_visibility:
                        continue
                    p1 = (int(first_point.x * width), int(first_point.y * height))
                    p2 = (int(second_point.x * width), int(second_point.y * height))
                    color = (55, 225, 95) if first in (23, 25, 27, 29, 31) else (255, 155, 55)
                    cv2.line(frame, p1, p2, color, 5, cv2.LINE_AA)
                    cv2.circle(frame, p1, 6, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, p2, 6, color, -1, cv2.LINE_AA)
            rows.append(row)
            cv2.rectangle(frame, (0, 0), (width, 58), (18, 18, 18), -1)
            cv2.putText(
                frame, "MediaPipe feet experiment: heel + foot index", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 235, 235), 2, cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    cap.release()
    writer.release()

    table = pd.DataFrame(rows)
    csv_path = args.output_dir / f"{args.video.stem}_mediapipe_33_landmarks.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    foot_names = ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]
    visibility = {
        name: float(table[f"{name}_visibility"].mean()) for name in foot_names
    }
    report = {
        "experiment": "pose_foot_keypoints_v1",
        "source_video": str(args.video),
        "model": str(args.model),
        "frames_expected": frames_expected,
        "frames_read": int(len(table)),
        "detected_frames": detected,
        "detection_rate": detected / max(1, len(table)),
        "mean_foot_visibility": visibility,
        "outputs": {"landmarks_csv": str(csv_path), "overlay_video": str(overlay_path)},
    }
    report_path = args.output_dir / f"{args.video.stem}_quality.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
