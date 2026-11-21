from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
DEFAULT_MODEL = Path(__file__).parent / "models" / "yolo11n-pose.pt"


def select_tracked_person(
    boxes_xyxy: np.ndarray,
    box_scores: np.ndarray,
    keypoint_confidence: np.ndarray,
    width: int,
    height: int,
    previous_center: np.ndarray | None,
    previous_area: float | None,
) -> tuple[int | None, np.ndarray | None, float | None, float | None]:
    """Select one runner consistently instead of taking the top score per frame.

    The first frame uses person size and lower-body visibility. Later frames use
    the previous bounding-box center and area as a simple single-person tracker.
    This prevents a confident background gym user from replacing the runner.
    """
    if len(boxes_xyxy) == 0:
        return None, previous_center, previous_area, None

    frame_diagonal = max(float(np.hypot(width, height)), 1.0)
    frame_area = max(float(width * height), 1.0)
    centers = np.column_stack([
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0,
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0,
    ])
    areas = (
        np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0.0, None)
        * np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0.0, None)
    ) / frame_area
    lower_body = keypoint_confidence[:, [11, 12, 13, 14, 15, 16]].mean(axis=1)

    if previous_center is None or previous_area is None:
        # A treadmill runner close to the camera normally occupies the largest
        # coherent person box. Lower-body visibility breaks close size ties.
        scores = 2.5 * areas + 0.35 * lower_body + 0.15 * box_scores
        selected = int(np.argmax(scores))
        tracking_distance = 0.0
    else:
        distances = np.linalg.norm(centers - previous_center, axis=1) / frame_diagonal
        area_change = np.abs(np.log(np.clip(areas, 1e-6, None) / max(previous_area, 1e-6)))
        costs = (
            5.0 * distances
            + 0.55 * area_change
            - 0.20 * lower_body
            - 0.05 * box_scores
        )
        selected = int(np.argmin(costs))
        tracking_distance = float(distances[selected])

        # If every detection jumped far away, keep the track missing for this
        # frame instead of silently switching to another person.
        if tracking_distance > 0.22:
            return None, previous_center, previous_area, tracking_distance

    return selected, centers[selected], float(areas[selected]), tracking_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract COCO-17 pose landmarks from a running video")
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    stem = args.input_video.stem
    overlay_path = args.output_dir / f"{stem}_yolo_pose_overlay.mp4"
    landmarks_path = args.output_dir / f"{stem}_yolo_landmarks.csv"
    quality_path = args.output_dir / f"{stem}_yolo_pose_quality.json"
    writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    model = YOLO(str(args.model))
    predictions = model.predict(
        source=str(args.input_video),
        imgsz=args.imgsz,
        conf=0.20,
        iou=0.50,
        device="cpu",
        stream=True,
        verbose=False,
    )
    rows: list[dict[str, float | int | bool]] = []
    detected = 0
    previous_center: np.ndarray | None = None
    previous_area: float | None = None
    multi_person_frames = 0
    rejected_track_jumps = 0
    for frame_index, result in enumerate(predictions):
        frame = result.orig_img.copy()
        row: dict[str, float | int | bool] = {
            "frame_index": frame_index,
            "time_s": frame_index / fps,
            "pose_detected": False,
            "person_candidates": 0,
            "selected_person_index": -1,
            "tracking_distance_normalized": np.nan,
        }
        person_index = None
        if (
            result.boxes is not None
            and len(result.boxes)
            and result.keypoints is not None
        ):
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            if result.keypoints.conf is None:
                all_keypoint_confidence = np.ones((len(boxes_xyxy), len(KEYPOINT_NAMES)))
            else:
                all_keypoint_confidence = result.keypoints.conf.detach().cpu().numpy()
            row["person_candidates"] = int(len(boxes_xyxy))
            multi_person_frames += int(len(boxes_xyxy) > 1)
            person_index, new_center, new_area, tracking_distance = select_tracked_person(
                boxes_xyxy,
                scores,
                all_keypoint_confidence,
                width,
                height,
                previous_center,
                previous_area,
            )
            row["tracking_distance_normalized"] = (
                np.nan if tracking_distance is None else tracking_distance
            )
            if person_index is not None:
                previous_center = new_center
                previous_area = new_area
                row["selected_person_index"] = person_index
            else:
                rejected_track_jumps += 1
        if person_index is not None and result.keypoints is not None:
            xy = result.keypoints.xy[person_index].detach().cpu().numpy()
            if result.keypoints.conf is None:
                confidence = np.ones(len(xy), dtype=float)
            else:
                confidence = result.keypoints.conf[person_index].detach().cpu().numpy()
            row["pose_detected"] = True
            detected += 1
            for index, name in enumerate(KEYPOINT_NAMES):
                row[f"{name}_x"] = float(xy[index, 0] / width)
                row[f"{name}_y"] = float(xy[index, 1] / height)
                row[f"{name}_confidence"] = float(confidence[index])
            for start, end in CONNECTIONS:
                if min(confidence[start], confidence[end]) < 0.20:
                    continue
                pa = tuple(np.rint(xy[start]).astype(int))
                pb = tuple(np.rint(xy[end]).astype(int))
                cv2.line(frame, pa, pb, (64, 220, 255), 3, cv2.LINE_AA)
            for index in range(len(xy)):
                if confidence[index] < 0.20:
                    continue
                point = tuple(np.rint(xy[index]).astype(int))
                cv2.circle(frame, point, 4, (30, 60, 255), -1, cv2.LINE_AA)
        else:
            for name in KEYPOINT_NAMES:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_confidence"] = np.nan
        cv2.putText(
            frame,
            f"YOLO pose {'OK' if row['pose_detected'] else 'missing'}  frame {frame_index}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(row)
        writer.write(frame)
    writer.release()

    table = pd.DataFrame(rows)
    table.to_csv(landmarks_path, index=False, encoding="utf-8-sig")
    lower_body = [
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    ]
    confidence_summary = {
        name: float(table[f"{name}_confidence"].mean(skipna=True)) for name in lower_body
    }
    quality = {
        "pose_backend": "YOLO11n-pose COCO-17",
        "input_video": str(args.input_video),
        "frames_expected": expected_frames,
        "frames_read": len(table),
        "fps": fps,
        "width": width,
        "height": height,
        "detected_frames": detected,
        "detection_rate": detected / len(table) if len(table) else 0.0,
        "tracking_method": "single-person temporal center-and-area association",
        "multi_person_frames": multi_person_frames,
        "rejected_track_jumps": rejected_track_jumps,
        "mean_lower_body_confidence": confidence_summary,
        "limitation": "COCO-17 has ankles but no heel or foot-index landmarks.",
        "landmarks_csv": str(landmarks_path),
        "overlay_video": str(overlay_path),
    }
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
