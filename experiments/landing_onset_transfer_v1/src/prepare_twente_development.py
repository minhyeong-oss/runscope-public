from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_onset_transfer_v1"
MATCHED = ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"
POSE_ROOT = ROOT / "experiments/boundary_regression_v1/artifacts/raw_pose"
FEATURE_SOURCE = ROOT / "experiments/temporal_tcn_v1/src/prepare_sequences.py"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}
SEQUENCE_LENGTH = 64
SEQUENCE_DURATION_S = 0.48
PRECONTACT_CONTEXT_S = 0.19
LANDING_WINDOW_S = 0.030


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    feature_module = load_module("twente_sequence_features", FEATURE_SOURCE)
    matched = pd.read_csv(MATCHED)
    if set(matched["subject"]) != set(SUBJECTS) or set(matched["subject"]) & FORBIDDEN:
        raise RuntimeError("Development table violates the frozen subject split")

    sequences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    onset_indices: list[int] = []
    subjects: list[str] = []
    runs: list[int] = []
    sides: list[str] = []
    source_rows: list[int] = []
    target_offsets: list[float] = []
    feature_names: list[str] | None = None
    cache: dict[tuple[str, int], pd.DataFrame] = {}

    for source_row, row in matched.iterrows():
        subject, run, side = str(row["subject"]), int(row["run"]), str(row["side"])
        key = (subject, run)
        if key not in cache:
            path = POSE_ROOT / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
            cache[key] = pd.read_csv(path)
        pose = cache[key]
        source_time = pose["source_time_s"].to_numpy(dtype=float)
        detected_start = float(row["detected_start_source_time_s"])
        true_onset = detected_start + float(row["target_ic_offset_s"])
        window_start = detected_start - PRECONTACT_CONTEXT_S
        sample_times = window_start + np.linspace(0.0, SEQUENCE_DURATION_S, SEQUENCE_LENGTH)
        if true_onset <= sample_times[0] or true_onset + LANDING_WINDOW_S >= sample_times[-1]:
            raise RuntimeError("Force onset fell outside the frozen inference window")
        positions = np.interp(sample_times, source_time, np.arange(len(pose), dtype=float))
        signals, _ = feature_module.trial_features(pose, side)
        if feature_names is None:
            feature_names = list(signals)
        sequence = np.column_stack([
            feature_module.resample(signals[name], positions) for name in feature_names
        ]).astype(np.float32)
        relative = sample_times - true_onset
        target = ((relative >= 0.0) & (relative < LANDING_WINDOW_S)).astype(np.int64)
        onset = int(np.argmin(np.abs(sample_times - true_onset)))
        sequences.append(sequence)
        labels.append(target)
        onset_indices.append(onset)
        subjects.append(subject)
        runs.append(run)
        sides.append(side)
        source_rows.append(int(source_row))
        target_offsets.append(float(row["target_ic_offset_s"]))

    output_dir = EXPERIMENT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "twente_development_onset_sequences_64.npz"
    x = np.stack(sequences)
    y = np.stack(labels)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        onset_indices=np.asarray(onset_indices),
        subjects=np.asarray(subjects),
        runs=np.asarray(runs),
        sides=np.asarray(sides),
        source_rows=np.asarray(source_rows),
        target_ic_offset_s=np.asarray(target_offsets, dtype=np.float32),
        feature_names=np.asarray(feature_names),
    )
    summary = {
        "experiment": "landing_onset_transfer_v1",
        "source": "Twente development RGB pose synchronized to force-platform initial contact",
        "sequences": int(len(x)),
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_duration_s": SEQUENCE_DURATION_S,
        "precontact_context_s": PRECONTACT_CONTEXT_S,
        "landing_window_s": LANDING_WINDOW_S,
        "subjects": {subject: int(np.sum(np.asarray(subjects) == subject)) for subject in SUBJECTS},
        "force_onset_offset_s": {
            "minimum": float(np.min(target_offsets)),
            "median": float(np.median(target_offsets)),
            "maximum": float(np.max(target_offsets)),
        },
        "locked_test_subjects": sorted(FORBIDDEN),
        "locked_test_subjects_accessed": False,
        "output": str(output_path),
    }
    (output_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
