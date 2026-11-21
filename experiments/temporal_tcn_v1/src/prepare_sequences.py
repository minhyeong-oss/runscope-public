from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/temporal_tcn_v1"
BOUNDARY_EXPERIMENT = ROOT / "experiments/boundary_regression_v1"
SEQUENCE_LENGTH = 64
PRE_WINDOW_S = 0.12
FORBIDDEN = {"Subj04", "Subj11"}


def column(table: pd.DataFrame, name: str) -> np.ndarray:
    return table[name].interpolate(limit_direction="both").to_numpy(dtype=float)


def resample(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, np.arange(len(values), dtype=float), values)


def trial_features(pose: pd.DataFrame, side: str) -> tuple[dict[str, np.ndarray], float]:
    opposite = "right" if side == "left" else "left"
    midhip_x = (column(pose, "left_hip_x") + column(pose, "right_hip_x")) / 2.0
    midhip_y = (column(pose, "left_hip_y") + column(pose, "right_hip_y")) / 2.0
    shoulder_x = (column(pose, "left_shoulder_x") + column(pose, "right_shoulder_x")) / 2.0
    shoulder_y = (column(pose, "left_shoulder_y") + column(pose, "right_shoulder_y")) / 2.0
    torso = np.clip(np.hypot(shoulder_x - midhip_x, shoulder_y - midhip_y), 1e-4, None)
    direction = float(np.sign(np.nanmedian(column(pose, "nose_x") - midhip_x)) or 1.0)
    signals: dict[str, np.ndarray] = {}
    for role, anatomical_side in (("tracked", side), ("opposite", opposite)):
        for joint in ("heel", "foot_index", "ankle", "knee", "hip"):
            signals[f"{role}_{joint}_x"] = direction * (column(pose, f"{anatomical_side}_{joint}_x") - midhip_x) / torso
            signals[f"{role}_{joint}_y"] = (column(pose, f"{anatomical_side}_{joint}_y") - midhip_y) / torso
        signals[f"{role}_heel_visibility"] = column(pose, f"{anatomical_side}_heel_visibility")
        signals[f"{role}_toe_visibility"] = column(pose, f"{anatomical_side}_foot_index_visibility")
    signals["trunk_x"] = direction * (shoulder_x - midhip_x) / torso
    signals["trunk_y"] = (shoulder_y - midhip_y) / torso
    return signals, direction


def main() -> None:
    matched = pd.read_csv(BOUNDARY_EXPERIMENT / "artifacts/development_matched_strides.csv")
    if set(matched["subject"]) & FORBIDDEN:
        raise RuntimeError("Forbidden subject found in sequence source")
    pose_root = BOUNDARY_EXPERIMENT / "artifacts/raw_pose"
    feature_names: list[str] | None = None
    sequences, labels, subjects, runs, sides, source_rows = [], [], [], [], [], []
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for source_row, row in matched.iterrows():
        subject, run, side = str(row["subject"]), int(row["run"]), str(row["side"])
        key = (subject, run)
        if key not in cache:
            cache[key] = pd.read_csv(pose_root / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv")
        pose = cache[key]
        fps = float(pose["fps"].iloc[0])
        source_time = pose["source_time_s"].to_numpy(dtype=float)
        detected_start = float(row["detected_start_source_time_s"])
        detected_end = detected_start + float(row["detected_stride_duration_s"])
        start_position = float(np.interp(detected_start - PRE_WINDOW_S, source_time, np.arange(len(pose))))
        end_position = float(np.interp(detected_end, source_time, np.arange(len(pose))))
        positions = np.linspace(start_position, end_position, SEQUENCE_LENGTH)
        sample_times = np.interp(positions, np.arange(len(pose)), source_time) - detected_start
        signals, _ = trial_features(pose, side)
        if feature_names is None:
            feature_names = list(signals)
        sequence = np.column_stack([resample(signals[name], positions) for name in feature_names])
        ic = float(row["target_ic_offset_s"])
        propulsion = float(row["target_propulsion_offset_s"])
        recovery = float(row["target_recovery_offset_s"])
        target = np.where(
            sample_times < ic, 4,
            np.where(
                sample_times < ic + 0.030, 1,
                np.where(sample_times < propulsion, 2, np.where(sample_times < recovery, 3, 4)),
            ),
        ).astype(np.int64)
        sequences.append(sequence.astype(np.float32))
        labels.append(target)
        subjects.append(subject)
        runs.append(run)
        sides.append(side)
        source_rows.append(source_row)
    output_dir = EXPERIMENT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "development_sequences_64.npz"
    np.savez_compressed(
        output_path,
        x=np.stack(sequences), y=np.stack(labels),
        subjects=np.asarray(subjects), runs=np.asarray(runs), sides=np.asarray(sides),
        source_rows=np.asarray(source_rows), feature_names=np.asarray(feature_names),
    )
    y = np.stack(labels)
    summary = {
        "experiment": "temporal_tcn_v1", "output": str(output_path),
        "sequences": len(sequences), "sequence_length": SEQUENCE_LENGTH,
        "features": len(feature_names or []), "development_subjects": sorted(set(subjects)),
        "forbidden_subjects_accessed": False,
        "class_samples": {str(stage): int((y == stage).sum()) for stage in (1, 2, 3, 4)},
    }
    (output_dir / "sequence_preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
