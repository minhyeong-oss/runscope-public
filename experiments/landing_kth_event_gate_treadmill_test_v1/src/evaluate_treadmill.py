from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_treadmill_test_v1"
VIDEO = ROOT / "test_videos/twente/Subj05_run81_side_10s_to_22s.mp4"
POSE_CSV = ROOT / "video_v1/test_outputs/twente_subj05_run81_pilot/Subj05_run81_side_10s_to_22s_yolo_landmarks.csv"
POSE_QUALITY = ROOT / "video_v1/test_outputs/twente_subj05_run81_pilot/Subj05_run81_side_10s_to_22s_yolo_pose_quality.json"
FORCE_MATCHED = ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"

V1_MODEL_SOURCE = ROOT / "experiments/landing_visual_contact_v1/src/train_and_test.py"
V1_PREP_SOURCE = ROOT / "experiments/landing_visual_contact_v1/src/prepare_dataset.py"
V2_SOURCE = ROOT / "experiments/landing_visual_contact_v2/src/train_loso.py"
V2_MODEL = ROOT / "experiments/landing_visual_contact_v2/artifacts/landing_visual_contact_v2.pt"
GATE_SOURCE = ROOT / "experiments/landing_kth_event_gate_v1/src/train_event_gate.py"
GATE_MODEL = ROOT / "experiments/landing_kth_event_gate_v1/artifacts/landing_kth_event_gate_v1.joblib"

SUBJECT = "Subj05"
RUN = 81
CLIP_SOURCE_OFFSET_S = 10.0
POSE_RATE = 30.0
TOLERANCES = [1, 2, 3]
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("treadmill_test_base", V1_MODEL_SOURCE)
prep = load_module("treadmill_test_prep", V1_PREP_SOURCE)
v2 = load_module("treadmill_test_v2", V2_SOURCE)
gate_source = load_module("treadmill_test_gate", GATE_SOURCE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_sequence() -> tuple[np.ndarray, np.ndarray, dict]:
    table = pd.read_csv(POSE_CSV)
    quality = json.loads(POSE_QUALITY.read_text(encoding="utf-8"))
    frames = len(table)
    coordinates = np.zeros((frames, 17, 2), dtype=np.float32)
    confidence = np.zeros((frames, 17), dtype=np.float32)
    width = float(quality["width"])
    height = float(quality["height"])
    for index, name in enumerate(KEYPOINTS):
        coordinates[:, index, 0] = table[f"{name}_x"].to_numpy(dtype=np.float32) * width
        coordinates[:, index, 1] = table[f"{name}_y"].to_numpy(dtype=np.float32) * height
        confidence[:, index] = table[f"{name}_confidence"].fillna(0).to_numpy(dtype=np.float32)
        valid = confidence[:, index] > 0.05
        for axis in range(2):
            coordinates[:, index, axis] = prep.interpolate(
                coordinates[:, index, axis], valid
            )
    source_rate = float(quality["fps"])
    old_time = np.arange(frames, dtype=np.float64) / source_rate
    new_length = int(round(frames * POSE_RATE / source_rate))
    new_time = np.arange(new_length, dtype=np.float64) / POSE_RATE
    new_time = np.clip(new_time, old_time[0], old_time[-1])
    resampled_coordinates = np.empty((new_length, 17, 2), dtype=np.float32)
    resampled_confidence = np.empty((new_length, 17), dtype=np.float32)
    for keypoint in range(17):
        for axis in range(2):
            resampled_coordinates[:, keypoint, axis] = np.interp(
                new_time, old_time, coordinates[:, keypoint, axis]
            )
        resampled_confidence[:, keypoint] = np.interp(
            new_time, old_time, confidence[:, keypoint]
        )
    features, _ = prep.extract_features(resampled_coordinates, resampled_confidence)
    return v2.subject_normalize(features), resampled_confidence, quality


def teacher_outputs(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(V2_MODEL, map_location="cpu", weights_only=False)
    model = base.LandingEventTCN(int(checkpoint["input_features"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probability = base.probabilities(
        model,
        features,
        np.zeros((1, 1, features.shape[1]), dtype=np.float32),
        np.ones((1, 1, features.shape[1]), dtype=np.float32),
    )
    events = base.predicted_events(
        probability,
        float(checkpoint["threshold"]),
        int(checkpoint["minimum_peak_distance_frames"]),
    )
    return probability, events, checkpoint


def gated_outputs(
    features: np.ndarray,
    teacher_probability: np.ndarray,
    teacher_events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    artifact = joblib.load(GATE_MODEL)
    event_features = np.stack([
        gate_source.context_features(features, teacher_probability, int(event))
        for event in teacher_events
    ])
    probability = artifact["model"].predict_proba(event_features)[:, 1]
    accepted = teacher_events[probability >= float(artifact["threshold"])]
    return probability, accepted.astype(np.int64), artifact


def force_truth(duration_frames: int) -> tuple[np.ndarray, pd.DataFrame]:
    table = pd.read_csv(FORCE_MATCHED)
    table = table[(table["subject"] == SUBJECT) & (table["run"] == RUN)].copy()
    table["force_initial_contact_source_time_s"] = (
        table["detected_start_source_time_s"] + table["target_ic_offset_s"]
    )
    table["force_initial_contact_clip_time_s"] = (
        table["force_initial_contact_source_time_s"] - CLIP_SOURCE_OFFSET_S
    )
    table = table[
        (table["force_initial_contact_clip_time_s"] >= 0)
        & (table["force_initial_contact_clip_time_s"] < duration_frames / POSE_RATE)
    ].copy()
    table["force_initial_contact_frame_30hz"] = np.rint(
        table["force_initial_contact_clip_time_s"] * POSE_RATE
    ).astype(int)
    events = np.asarray(
        sorted(set(table["force_initial_contact_frame_30hz"].astype(int))),
        dtype=np.int64,
    )
    return events, table


def event_metrics(truth: np.ndarray, prediction: np.ndarray, tolerance: int) -> dict:
    tp, fp, fn, errors = base.match_events(truth, prediction, tolerance)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "matched_onset_mae_ms": (
            float(np.mean(np.abs(errors))) * 1000.0 / POSE_RATE if errors else None
        ),
    }


def event_rows(
    truth: np.ndarray,
    teacher_events: np.ndarray,
    teacher_probability: np.ndarray,
    gate_probability: np.ndarray,
    accepted: np.ndarray,
) -> pd.DataFrame:
    accepted_set = set(accepted.tolist())
    rows = []
    for event, gate_score in zip(teacher_events, gate_probability):
        nearest_index = int(np.argmin(np.abs(truth - event)))
        nearest = int(truth[nearest_index])
        rows.append({
            "teacher_event_frame_30hz": int(event),
            "clip_time_s": float(event / POSE_RATE),
            "source_time_s": float(CLIP_SOURCE_OFFSET_S + event / POSE_RATE),
            "teacher_probability": float(teacher_probability[event]),
            "gate_probability": float(gate_score),
            "gate_accepted": int(event) in accepted_set,
            "nearest_force_event_frame_30hz": nearest,
            "error_frames": int(event - nearest),
            "absolute_error_ms": abs(int(event - nearest)) * 1000.0 / POSE_RATE,
        })
    return pd.DataFrame(rows)


def main() -> None:
    model_hash_before = sha256(GATE_MODEL)
    teacher_hash_before = sha256(V2_MODEL)
    features, confidence, quality = pose_sequence()
    teacher_probability, teacher_events, teacher_checkpoint = teacher_outputs(features)
    gate_probability, accepted_events, gate_artifact = gated_outputs(
        features, teacher_probability, teacher_events
    )
    truth, force_table = force_truth(len(features))

    teacher_metrics = {
        f"within_{tolerance}_frames": event_metrics(truth, teacher_events, tolerance)
        for tolerance in TOLERANCES
    }
    gate_metrics = {
        f"within_{tolerance}_frames": event_metrics(truth, accepted_events, tolerance)
        for tolerance in TOLERANCES
    }
    if sha256(GATE_MODEL) != model_hash_before or sha256(V2_MODEL) != teacher_hash_before:
        raise RuntimeError("A frozen model changed during test")

    report_dir = EXPERIMENT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    force_table.to_csv(report_dir / "force_initial_contact_truth.csv", index=False, encoding="utf-8-sig")
    event_rows(
        truth, teacher_events, teacher_probability, gate_probability, accepted_events
    ).to_csv(report_dir / "predicted_events_vs_force_truth.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        report_dir / "frozen_test_outputs.npz",
        features=features,
        mean_lower_body_confidence=confidence[:, [11, 12, 13, 14, 15, 16]].mean(axis=1),
        teacher_probability=teacher_probability,
        teacher_events=teacher_events,
        gate_probability=gate_probability,
        gate_accepted_events=accepted_events,
        force_truth_events=truth,
    )
    report = {
        "status": "FROZEN_EXTERNAL_TREADMILL_TEST_COMPLETED_NO_RETRAINING",
        "experiment": EXPERIMENT.name,
        "video": str(VIDEO),
        "video_sha256": sha256(VIDEO),
        "pose_csv": str(POSE_CSV),
        "pose_detection_rate": quality["detection_rate"],
        "tested_stack": {
            "teacher": str(V2_MODEL),
            "teacher_sha256": teacher_hash_before,
            "teacher_threshold": float(teacher_checkpoint["threshold"]),
            "teacher_minimum_peak_distance_frames": int(
                teacher_checkpoint["minimum_peak_distance_frames"]
            ),
            "gate": str(GATE_MODEL),
            "gate_sha256": model_hash_before,
            "gate_threshold": float(gate_artifact["threshold"]),
        },
        "force_truth": {
            "source": str(FORCE_MATCHED),
            "events": int(len(truth)),
            "definition": "force-platform synchronized initial contact",
        },
        "predicted_events": {
            "teacher_before_gate": int(len(teacher_events)),
            "accepted_after_gate": int(len(accepted_events)),
        },
        "teacher_metrics": teacher_metrics,
        "event_gate_metrics": gate_metrics,
        "training_or_refitting_performed": False,
        "threshold_or_parameter_tuning_performed": False,
        "post_test_retraining_performed": False,
        "test_scope_note": (
            f"{SUBJECT} run{RUN} was not used to train the tested v2+KTH-gate stack, "
            "but it exists in older separate RunScope development experiments."
        ),
        "limitations": [
            "The video is 33 fps and predictions are evaluated on a 30 Hz resampled timeline.",
            "This report tests landing onset only; it does not test braking, propulsion, or recovery labels.",
            "One treadmill trial is insufficient for a population-level performance claim.",
        ],
    }
    report_path = report_dir / "TEST_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
