from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_adaptation_v1"
DATA_ROOT = ROOT / "data/landing_video_kth_v1"
VIDEO_ROOT = DATA_ROOT / "extracted"
SEQUENCE_FILE = DATA_ROOT / "00sequences.txt"
V1_SOURCE = ROOT / "experiments/landing_visual_contact_v1/src/train_and_test.py"
PREP_SOURCE = ROOT / "experiments/landing_visual_contact_v1/src/prepare_dataset.py"
V2_SOURCE = ROOT / "experiments/landing_visual_contact_v2/src/train_loso.py"
V2_MODEL = ROOT / "experiments/landing_visual_contact_v2/artifacts/landing_visual_contact_v2.pt"
DEFAULT_POSE_MODEL = ROOT / "video_v1/models/yolo11n-pose.pt"

POSE_RATE = 30.0
CONSENSUS_TOLERANCE = 3
MIN_CONSENSUS_EVENTS = 3
MIN_DETECTION_RATE = 0.55
MIN_LOWER_BODY_CONFIDENCE = 0.35

SPLITS = {
    "train": ["11", "12", "13", "14", "15", "16", "17", "18"],
    "validation": ["19", "20", "21", "23", "24", "25", "01", "04"],
    "untouched_test": ["22", "02", "03", "05", "06", "07", "08", "09", "10"],
}
PRIOR_LOCKED_TEST = ["zenodo_19720803:07", "zenodo_19720803:14"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("landing_kth_base", V1_SOURCE)
prep = load_module("landing_kth_prep", PREP_SOURCE)
v2 = load_module("landing_kth_v2", V2_SOURCE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_annotations() -> dict[str, list[tuple[int, int]]]:
    annotations: dict[str, list[tuple[int, int]]] = {}
    pattern = re.compile(r"^(person\d+_running_d\d)\s+frames\s+(.+)$")
    for raw in SEQUENCE_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(raw.strip())
        if not match:
            continue
        ranges = []
        for item in match.group(2).split(","):
            start, end = (int(value) for value in item.strip().split("-"))
            # KTH's text file uses one-based, inclusive frame indices.
            ranges.append((start - 1, end - 1))
        annotations[match.group(1)] = ranges
    if len(annotations) != 100:
        raise RuntimeError(f"Expected 100 KTH running annotations, found {len(annotations)}")
    return annotations


def locate_video(stem: str) -> Path:
    candidates = sorted(VIDEO_ROOT.rglob(f"{stem}*.avi"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one AVI for {stem}, found {candidates}")
    return candidates[0]


def annotated_frames(path: Path, ranges: list[tuple[int, int]]) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    wanted = set()
    for start, end in ranges:
        wanted.update(range(start, end + 1))
    frames = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in wanted:
            frames.append(frame)
        frame_index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No annotated frames decoded from {path}")
    return frames, fps


def primary_pose(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    if result.boxes is None or not len(result.boxes) or result.keypoints is None:
        return None, None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    box_confidence = result.boxes.conf.detach().cpu().numpy()
    xy = result.keypoints.xy.detach().cpu().numpy()
    if result.keypoints.conf is None:
        confidence = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        confidence = result.keypoints.conf.detach().cpu().numpy()
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    lower_body = confidence[:, [11, 12, 13, 14, 15, 16]].mean(axis=1)
    score = areas / max(float(areas.max()), 1.0) + 0.35 * lower_body + 0.10 * box_confidence
    selected = int(np.argmax(score))
    return xy[selected].astype(np.float32), confidence[selected].astype(np.float32)


def interpolate_pose(coordinates: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for keypoint in range(17):
        valid = confidence[:, keypoint] > 0.05
        for axis in range(2):
            coordinates[:, keypoint, axis] = prep.interpolate(
                coordinates[:, keypoint, axis], valid
            )
    return coordinates, confidence


def resample_pose(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    source_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    if abs(source_rate - POSE_RATE) < 1e-6:
        return coordinates, confidence
    old_time = np.arange(len(coordinates), dtype=np.float64) / source_rate
    new_length = max(2, int(round(len(coordinates) * POSE_RATE / source_rate)))
    new_time = np.arange(new_length, dtype=np.float64) / POSE_RATE
    new_time = np.clip(new_time, old_time[0], old_time[-1])
    new_coordinates = np.empty((new_length, 17, 2), dtype=np.float32)
    new_confidence = np.empty((new_length, 17), dtype=np.float32)
    for keypoint in range(17):
        for axis in range(2):
            new_coordinates[:, keypoint, axis] = np.interp(
                new_time, old_time, coordinates[:, keypoint, axis]
            )
        new_confidence[:, keypoint] = np.interp(
            new_time, old_time, confidence[:, keypoint]
        )
    return new_coordinates, new_confidence


def ankle_extrema(features: np.ndarray) -> np.ndarray:
    candidates = []
    # Feature blocks are [x, y, vx, vy, ax, ay, confidence]. Ankle y is the
    # second feature of blocks 6 and 7. Image-positive y points downward.
    for column in (6 * 7 + 1, 7 * 7 + 1):
        signal = features[:, column]
        prominence = max(0.04, float(np.std(signal)) * 0.18)
        peaks, _ = find_peaks(signal, distance=8, prominence=prominence)
        candidates.extend(peaks.tolist())
    return np.asarray(sorted(set(candidates)), dtype=np.int64)


def consensus_events(
    teacher_probability: np.ndarray,
    threshold: float,
    distance: int,
    kinematic_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    teacher_events = base.predicted_events(teacher_probability, threshold, distance)
    accepted = []
    for event in teacher_events:
        if len(kinematic_events) and np.min(np.abs(kinematic_events - event)) <= CONSENSUS_TOLERANCE:
            accepted.append(int(event))
    return teacher_events, np.asarray(accepted, dtype=np.int64)


def load_teacher():
    checkpoint = torch.load(V2_MODEL, map_location="cpu", weights_only=False)
    model = base.LandingEventTCN(int(checkpoint["input_features"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["threshold"]), int(checkpoint["minimum_peak_distance_frames"])


def process_video(
    stem: str,
    ranges: list[tuple[int, int]],
    split: str,
    pose_model: YOLO,
    teacher,
    threshold: float,
    distance: int,
    output_dir: Path,
    imgsz: int,
) -> dict:
    path = locate_video(stem)
    frames, fps = annotated_frames(path, ranges)
    predictions = pose_model.predict(
        source=frames,
        imgsz=imgsz,
        conf=0.10,
        iou=0.50,
        device="cpu",
        batch=32,
        stream=False,
        verbose=False,
    )
    coordinates = np.zeros((len(frames), 17, 2), dtype=np.float32)
    confidence = np.zeros((len(frames), 17), dtype=np.float32)
    detected = np.zeros(len(frames), dtype=bool)
    for index, result in enumerate(predictions):
        xy, conf = primary_pose(result)
        if xy is None or conf is None:
            continue
        coordinates[index] = xy
        confidence[index] = conf
        detected[index] = True
    detection_rate = float(detected.mean())
    raw_lower_body_confidence = float(
        confidence[:, [11, 12, 13, 14, 15, 16]].mean()
    )
    coordinates, confidence = interpolate_pose(coordinates, confidence)
    coordinates, confidence = resample_pose(coordinates, confidence, fps)
    features, feature_names = prep.extract_features(coordinates, confidence)
    normalized_features = v2.subject_normalize(features)
    probability = base.probabilities(
        teacher,
        normalized_features,
        np.zeros((1, 1, normalized_features.shape[1]), dtype=np.float32),
        np.ones((1, 1, normalized_features.shape[1]), dtype=np.float32),
    )
    kinematic = ankle_extrema(normalized_features)
    teacher_events, pseudo_events = consensus_events(
        probability, threshold, distance, kinematic
    )
    labels = np.zeros(len(features), dtype=np.uint8)
    labels[pseudo_events] = 1
    accepted = (
        detection_rate >= MIN_DETECTION_RATE
        and raw_lower_body_confidence >= MIN_LOWER_BODY_CONFIDENCE
        and len(pseudo_events) >= MIN_CONSENSUS_EVENTS
    )
    subject = stem[6:8]
    artifact = output_dir / f"{stem}.npz"
    np.savez_compressed(
        artifact,
        features=normalized_features,
        labels=labels,
        events=pseudo_events,
        teacher_probability=probability.astype(np.float32),
        teacher_events=teacher_events,
        kinematic_events=kinematic,
        feature_names=np.asarray(feature_names),
        subject=np.asarray(subject),
        split=np.asarray(split),
        source_video=np.asarray(str(path)),
        pose_rate_hz=np.asarray(POSE_RATE),
        accepted=np.asarray(accepted),
    )
    return {
        "stem": stem,
        "subject": subject,
        "split": split,
        "condition": stem[-2:],
        "source_video": str(path),
        "source_fps": fps,
        "annotated_source_frames": len(frames),
        "resampled_frames": len(features),
        "detection_rate": detection_rate,
        "mean_lower_body_confidence": raw_lower_body_confidence,
        "teacher_events": int(len(teacher_events)),
        "ankle_extrema": int(len(kinematic)),
        "consensus_pseudo_events": int(len(pseudo_events)),
        "accepted": bool(accepted),
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conditions", nargs="+", default=["d1", "d2", "d3", "d4"])
    parser.add_argument("--max-videos", type=int)
    args = parser.parse_args()

    if set(SPLITS["untouched_test"]) & set(SPLITS["train"] + SPLITS["validation"]):
        raise RuntimeError("KTH official test leakage")
    annotations = parse_annotations()
    jobs = []
    for split in ("train", "validation"):
        for subject in SPLITS[split]:
            for condition in args.conditions:
                stem = f"person{subject}_running_{condition}"
                jobs.append((stem, annotations[stem], split))
    if args.max_videos is not None:
        jobs = jobs[: args.max_videos]

    artifact_dir = EXPERIMENT / "artifacts/sequences"
    report_dir = EXPERIMENT / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    pose_model = YOLO(str(args.pose_model))
    teacher, threshold, distance = load_teacher()
    rows = []
    for index, (stem, ranges, split) in enumerate(jobs, start=1):
        row = process_video(
            stem, ranges, split, pose_model, teacher, threshold, distance,
            artifact_dir, args.imgsz,
        )
        rows.append(row)
        print(json.dumps({"progress": f"{index}/{len(jobs)}", **row}), flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "kth_pose_and_pseudolabel_quality.csv", index=False, encoding="utf-8-sig")
    accepted = table[table["accepted"]]
    manifest = {
        "status": "KTH_VIDEO_POSE_AND_PSEUDO_LABELS_PREPARED",
        "source": "KTH Human Actions running videos",
        "source_url": "https://www.csc.kth.se/cvap/actions/",
        "pose_model": str(args.pose_model),
        "pose_model_sha256": sha256(args.pose_model),
        "teacher_model": str(V2_MODEL),
        "teacher_model_sha256": sha256(V2_MODEL),
        "teacher_threshold": threshold,
        "teacher_minimum_peak_distance_frames": distance,
        "pseudo_label_rule": (
            f"frozen v2 peak within {CONSENSUS_TOLERANCE} frames of an ankle-y extremum"
        ),
        "quality_gates": {
            "minimum_pose_detection_rate": MIN_DETECTION_RATE,
            "minimum_mean_lower_body_confidence_including_missing_frames": MIN_LOWER_BODY_CONFIDENCE,
            "minimum_consensus_events_per_video": MIN_CONSENSUS_EVENTS,
        },
        "split": SPLITS,
        "prior_locked_test_subjects": PRIOR_LOCKED_TEST,
        "prior_locked_test_data_accessed": False,
        "kth_official_test_data_accessed": False,
        "conditions": args.conditions,
        "processed_videos": int(len(table)),
        "accepted_videos": int(len(accepted)),
        "accepted_by_split": {
            name: int((accepted["split"] == name).sum()) for name in ("train", "validation")
        },
        "accepted_pseudo_events": int(accepted["consensus_pseudo_events"].sum()),
        "limitations": [
            "KTH provides no force-plate or manual foot-contact timing.",
            "Pseudo-label agreement must not be reported as force-ground-truth F1.",
            "The source videos are low resolution (160x120) and recorded at 25 fps.",
        ],
    }
    (EXPERIMENT / "artifacts/DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
