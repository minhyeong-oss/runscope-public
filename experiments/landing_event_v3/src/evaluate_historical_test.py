from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_event_v3"
FROZEN_SELECTION = EXPERIMENT / "artifacts/FROZEN_SELECTION.json"
POSE_ROOT = ROOT / "experiments/landing_gate_hybrid_v1/artifacts/raw_pose"
BOUNDARY_SOURCE = ROOT / "experiments/boundary_regression_v1/src/train_boundary_regression.py"
FEATURE_SOURCE = ROOT / "experiments/temporal_tcn_v1/src/prepare_sequences.py"
TRIALS = [
    ("Subj04", 81),
    ("Subj04", 99),
    ("Subj11", 63),
    ("Subj11", 81),
    ("Subj11", 99),
]
SPEEDS = {63: 1.75, 81: 2.25, 99: 2.75}
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mat_path(subject: str, run: int) -> Path:
    return (
        ROOT
        / f"data/twente_sensor/{subject}/locked_test_extracted/{subject}/{subject}_run_{run}.mat"
    )


def build_matched_strides(boundary) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    trial_rows: list[dict] = []
    for subject, run in TRIALS:
        pose_path = POSE_ROOT / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
        pose = pd.read_csv(pose_path)
        datastr = loadmat(
            mat_path(subject, run), simplify_cells=True, variable_names=["Datastr"]
        )["Datastr"]
        sync_offset, sync_corr = boundary.synchronization_offset(datastr["Marker"], pose)
        force_rate = float(datastr["Force"]["FrameRate"])
        mass = float(datastr["Info"]["subjMass"])
        trial_matches = 0
        trial_peaks = 0
        for side in ("left", "right"):
            events = boundary.force_strides(
                datastr["Force"][side.capitalize() + "ForceData"], force_rate, mass
            )
            peaks, _, direction = boundary.detect_contacts(pose, side)
            trial_peaks += len(peaks)
            used: set[int] = set()
            for first, second in zip(peaks[:-1], peaks[1:]):
                detected_duration = (second - first) / float(pose["fps"].iloc[0])
                if not 0.45 <= detected_duration <= 1.20:
                    continue
                detected_sensor_time = float(pose["source_time_s"].iloc[first]) + sync_offset
                match = boundary.match_force_event(detected_sensor_time, events, used)
                if match is None:
                    continue
                event_index, event = match
                if not 0.45 <= event["next_contact_s"] - event["initial_contact_s"] <= 1.20:
                    continue
                used.add(event_index)
                trial_matches += 1
                features = boundary.sample_features(
                    pose, int(first), int(second), side, direction, SPEEDS[run]
                )
                rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "side": side,
                        "detected_start_source_time_s": float(
                            pose["source_time_s"].iloc[first]
                        ),
                        "sync_offset_s": sync_offset,
                        "sync_correlation": sync_corr,
                        "target_ic_offset_s": event["initial_contact_s"]
                        - detected_sensor_time,
                        **features,
                    }
                )
        trial_rows.append(
            {
                "subject": subject,
                "run": run,
                "sync_offset_s": sync_offset,
                "sync_correlation": sync_corr,
                "detected_contact_peaks": trial_peaks,
                "matched_video_force_strides": trial_matches,
                "pose_detection_rate": float(pose["pose_detected"].mean()),
            }
        )
        print(json.dumps(trial_rows[-1]), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trial_rows)


def build_sequences(matched: pd.DataFrame, feature_module, expected_features: list[str]):
    sequences: list[np.ndarray] = []
    onsets: list[int] = []
    accepted_rows: list[int] = []
    cache: dict[tuple[str, int], pd.DataFrame] = {}
    for source_row, row in matched.iterrows():
        subject, run, side = str(row["subject"]), int(row["run"]), str(row["side"])
        key = (subject, run)
        if key not in cache:
            cache[key] = pd.read_csv(
                POSE_ROOT / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
            )
        pose = cache[key]
        source_time = pose["source_time_s"].to_numpy(dtype=float)
        detected_start = float(row["detected_start_source_time_s"])
        true_onset = detected_start + float(row["target_ic_offset_s"])
        window_start = detected_start - PRECONTACT_CONTEXT_S
        sample_times = window_start + np.linspace(
            0.0, SEQUENCE_DURATION_S, SEQUENCE_LENGTH
        )
        if true_onset <= sample_times[0] or true_onset + LANDING_WINDOW_S >= sample_times[-1]:
            continue
        positions = np.interp(sample_times, source_time, np.arange(len(pose), dtype=float))
        signals, _ = feature_module.trial_features(pose, side)
        if list(signals) != expected_features:
            raise RuntimeError("Historical test feature order differs from frozen model")
        sequence = np.column_stack(
            [feature_module.resample(signals[name], positions) for name in expected_features]
        ).astype(np.float32)
        if not np.isfinite(sequence).all():
            raise RuntimeError("Non-finite test sequence")
        sequences.append(sequence)
        onsets.append(int(np.argmin(np.abs(sample_times - true_onset))))
        accepted_rows.append(int(source_row))
    if not sequences:
        raise RuntimeError("No historical test sequences were constructed")
    return (
        np.stack(sequences),
        np.asarray(onsets, dtype=int),
        matched.iloc[accepted_rows].reset_index(drop=True),
    )


def decode_window(onset: np.ndarray, length: int = 64, window_samples: int = 4) -> np.ndarray:
    result = np.zeros((len(onset), length), dtype=np.uint8)
    for row, index in enumerate(onset):
        result[row, index : min(length, index + window_samples + 1)] = 1
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray, sample_interval_s: float) -> dict:
    error = np.abs(prediction - truth)
    return {
        "sequences": int(len(truth)),
        "event_f1_15ms": float(np.mean(error * sample_interval_s <= 0.015)),
        "event_f1_30ms": float(np.mean(error * sample_interval_s <= 0.030)),
        "event_f1_60ms": float(np.mean(error * sample_interval_s <= 0.060)),
        "onset_mae_ms": float(np.mean(error) * sample_interval_s * 1000.0),
        "landing_window_f1": float(
            f1_score(
                decode_window(truth).reshape(-1),
                decode_window(prediction).reshape(-1),
                zero_division=0,
            )
        ),
    }


def main() -> None:
    # Verify the development-only selection and artifact before reading any test force labels.
    selection = json.loads(FROZEN_SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "LANDING_EVENT_V3_FROZEN_BEFORE_HISTORICAL_TEST":
        raise RuntimeError("Model is not in the required frozen state")
    if selection["selected_candidate"] != "ridge_alpha1000":
        raise RuntimeError("Unexpected selected candidate")
    artifact_path = Path(selection["selected_artifact"])
    artifact_hash = sha256(artifact_path)
    if artifact_hash != selection["selected_artifact_sha256"]:
        raise RuntimeError("Frozen landing artifact hash mismatch")
    artifact = joblib.load(artifact_path)
    if set(artifact["forbidden_subjects"]) != {"Subj04", "Subj11"}:
        raise RuntimeError("Historical test split mismatch")

    boundary = load_module("landing_v3_boundary_features", BOUNDARY_SOURCE)
    feature_module = load_module("landing_v3_sequence_features", FEATURE_SOURCE)
    matched, trials = build_matched_strides(boundary)
    expected_features = list(artifact["feature_names"])
    x, truth, accepted = build_sequences(matched, feature_module, expected_features)
    if set(accepted["subject"].astype(str)) != {"Subj04", "Subj11"}:
        raise RuntimeError("Historical test subject coverage is incomplete")

    x_flat = x[:, ::4].reshape(len(x), -1)
    prediction = np.clip(
        np.rint(artifact["model"].predict(x_flat)), 0, SEQUENCE_LENGTH - 1
    ).astype(int)
    sample_interval_s = float(artifact["sample_interval_s"])
    overall = metrics(truth, prediction, sample_interval_s)
    per_subject = {
        subject: metrics(
            truth[accepted["subject"].astype(str).to_numpy() == subject],
            prediction[accepted["subject"].astype(str).to_numpy() == subject],
            sample_interval_s,
        )
        for subject in ("Subj04", "Subj11")
    }
    per_trial = {}
    for (subject, run), group in accepted.groupby(["subject", "run"]):
        indices = group.index.to_numpy(dtype=int)
        per_trial[f"{subject}_run{int(run)}"] = metrics(
            truth[indices], prediction[indices], sample_interval_s
        )
    requested_trials = [f"{subject}_run{run}" for subject, run in TRIALS]
    evaluated_trials = list(per_trial)
    unevaluated_trials = [name for name in requested_trials if name not in per_trial]

    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    accepted = accepted.copy()
    accepted["true_onset_index"] = truth
    accepted["predicted_onset_index"] = prediction
    accepted["absolute_error_ms"] = (
        np.abs(prediction - truth) * sample_interval_s * 1000.0
    )
    matched_path = artifacts / "historical_test_matched_strides.csv"
    trial_path = reports / "historical_test_trial_quality.csv"
    prediction_path = reports / "historical_test_predictions.csv"
    report_path = reports / "HISTORICAL_TEST_REPORT.json"
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
    trials.to_csv(trial_path, index=False, encoding="utf-8-sig")
    accepted.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "LANDING_EVENT_V3_HISTORICAL_TEST_COMPLETED_NO_REFIT_NO_RETUNING",
        "selected_candidate": selection["selected_candidate"],
        "verified_artifact": str(artifact_path),
        "verified_artifact_sha256": artifact_hash,
        "development_subjects": artifact["development_subjects"],
        "historical_test_subjects": ["Subj04", "Subj11"],
        "requested_historical_test_trials": requested_trials,
        "evaluated_historical_test_trials": evaluated_trials,
        "unevaluated_historical_test_trials": {
            name: "zero video contact candidates matched to force events"
            for name in unevaluated_trials
        },
        "test_training_performed": False,
        "test_threshold_or_parameter_tuning_performed": False,
        "matched_strides_before_window_filter": int(len(matched)),
        "evaluated_sequences": int(len(accepted)),
        "overall_metrics": overall,
        "per_subject_metrics": per_subject,
        "per_trial_metrics": per_trial,
        "development_selected_metrics": selection["selected_development_metrics"],
        "warning": (
            "Subj04/Subj11 are a reused historical reference, not a new untouched benchmark. "
            "Do not tune or retrain v3 from this report. Overall metrics cover only trials "
            "listed under evaluated_historical_test_trials."
        ),
        "outputs": {
            "matched_strides_csv": str(matched_path),
            "trial_quality_csv": str(trial_path),
            "predictions_csv": str(prediction_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
