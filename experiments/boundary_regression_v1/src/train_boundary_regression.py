from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "video_v1"))
from evaluate_twente_force_f1 import force_strides  # noqa: E402


EXPERIMENT = ROOT / "experiments/boundary_regression_v1"
POSE_ROOT = EXPERIMENT / "artifacts/raw_pose"
SPLIT_PATH = ROOT / "data/twente_sensor/SPLIT_LOCK_v1.json"
SPEEDS = {63: 1.75, 81: 2.25, 99: 2.75}
STAGES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}
RANDOM_SEED = 20261107


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


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    iqr = np.nanpercentile(values, 75) - np.nanpercentile(values, 25)
    return (values - median) / max(float(iqr), 1e-6)


def synchronization_offset(marker: dict, pose: pd.DataFrame) -> tuple[float, float]:
    labels = list(marker["DataLabel"])
    marker_data = np.asarray(marker["MarkerData"], dtype=float)
    rate = float(marker["FrameRate"])
    sensor_time = np.arange(marker_data.shape[2], dtype=float) / rate
    video_time = pose["source_time_s"].to_numpy(dtype=float)
    midhip = (interpolate_column(pose, "left_hip_x") + interpolate_column(pose, "right_hip_x")) / 2.0
    pairs = {
        "left_heel": ("LCAL",), "right_heel": ("RCAL",),
        "left_foot_index": ("L1MT", "L5MT"), "right_foot_index": ("R1MT", "R5MT"),
    }
    candidates = []
    for offset in np.arange(-0.120, 0.1201, 0.0005):
        correlations = []
        for pose_name, marker_names in pairs.items():
            available_marker_names = [name for name in marker_names if name in labels]
            if not available_marker_names:
                continue
            video_signal = interpolate_column(pose, f"{pose_name}_x") - midhip
            marker_signal = np.nanmean(
                np.vstack([marker_data[labels.index(name), 0] for name in available_marker_names]), axis=0
            ) - marker_data[labels.index("SACR"), 0]
            finite = np.isfinite(marker_signal)
            if int(finite.sum()) < 2:
                continue
            filled = np.interp(sensor_time, sensor_time[finite], marker_signal[finite])
            aligned = np.interp(video_time + offset, sensor_time, filled)
            value = abs(float(np.corrcoef(video_signal, aligned)[0, 1]))
            if np.isfinite(value):
                correlations.append(value)
        if correlations:
            candidates.append((float(np.mean(correlations)), float(offset)))
    if not candidates:
        raise RuntimeError("No finite synchronization candidates")
    return max(candidates, key=lambda item: item[0])[1], max(candidates, key=lambda item: item[0])[0]


def detect_contacts(pose: pd.DataFrame, side: str) -> tuple[np.ndarray, np.ndarray, float]:
    fps = float(pose["fps"].iloc[0])
    midhip_x = (interpolate_column(pose, "left_hip_x") + interpolate_column(pose, "right_hip_x")) / 2.0
    nose_offset = interpolate_column(pose, "nose_x") - midhip_x
    direction = float(np.sign(np.nanmedian(nose_offset)) or 1.0)
    heel_x = interpolate_column(pose, f"{side}_heel_x")
    heel_y = interpolate_column(pose, f"{side}_heel_y")
    hip_y = interpolate_column(pose, f"{side}_hip_y")
    toe_y = interpolate_column(pose, f"{side}_foot_index_y")
    signal = (
        0.62 * robust_z(direction * (heel_x - midhip_x))
        + 0.25 * robust_z(heel_y - hip_y)
        + 0.13 * robust_z(toe_y - hip_y)
    )
    signal = savgol_filter(signal, 9, 2)
    peaks, _ = find_peaks(signal, distance=int(round(0.55 * fps)), prominence=0.35)
    return peaks, signal, direction


def sample_features(pose: pd.DataFrame, start: int, end: int, side: str, direction: float, speed: float) -> dict[str, float]:
    opposite = "right" if side == "left" else "left"
    midhip_x = (interpolate_column(pose, "left_hip_x") + interpolate_column(pose, "right_hip_x")) / 2.0
    midhip_y = (interpolate_column(pose, "left_hip_y") + interpolate_column(pose, "right_hip_y")) / 2.0
    midshoulder_x = (interpolate_column(pose, "left_shoulder_x") + interpolate_column(pose, "right_shoulder_x")) / 2.0
    midshoulder_y = (interpolate_column(pose, "left_shoulder_y") + interpolate_column(pose, "right_shoulder_y")) / 2.0
    torso = np.hypot(midshoulder_x - midhip_x, midshoulder_y - midhip_y)
    torso = np.clip(torso, 1e-4, None)
    signals = {
        "heel_x": direction * (interpolate_column(pose, f"{side}_heel_x") - midhip_x) / torso,
        "heel_y": (interpolate_column(pose, f"{side}_heel_y") - midhip_y) / torso,
        "toe_x": direction * (interpolate_column(pose, f"{side}_foot_index_x") - midhip_x) / torso,
        "toe_y": (interpolate_column(pose, f"{side}_foot_index_y") - midhip_y) / torso,
        "ankle_x": direction * (interpolate_column(pose, f"{side}_ankle_x") - midhip_x) / torso,
        "knee_y": (interpolate_column(pose, f"{side}_knee_y") - midhip_y) / torso,
        "opposite_heel_x": direction * (interpolate_column(pose, f"{opposite}_heel_x") - midhip_x) / torso,
    }
    fractions = np.linspace(0.0, 0.875, 8)
    positions = start + fractions * max(1, end - start)
    indices = np.arange(len(pose), dtype=float)
    result = {
        "speed_m_s": speed,
        "detected_stride_duration_s": (end - start) / float(pose["fps"].iloc[0]),
        "mean_tracked_foot_visibility": float(np.nanmean([
            pose.loc[start:end, f"{side}_heel_visibility"].mean(),
            pose.loc[start:end, f"{side}_foot_index_visibility"].mean(),
        ])),
    }
    for name, values in signals.items():
        for index, position in enumerate(positions):
            result[f"{name}_q{index}"] = float(np.interp(position, indices, values))
    return result


def match_force_event(peak_sensor_time: float, events: list[dict], used: set[int]) -> tuple[int, dict] | None:
    candidates = [
        (abs(event["initial_contact_s"] - peak_sensor_time), index, event)
        for index, event in enumerate(events) if index not in used
    ]
    if not candidates:
        return None
    distance, index, event = min(candidates, key=lambda item: item[0])
    return (index, event) if distance <= 0.14 else None


def build_dataset(subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    trial_rows: list[dict] = []
    for subject in subjects:
        for run, speed in SPEEDS.items():
            pose_path = POSE_ROOT / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
            pose = pd.read_csv(pose_path)
            datastr = loadmat(mat_path(subject, run), simplify_cells=True, variable_names=["Datastr"])["Datastr"]
            sync_offset, sync_corr = synchronization_offset(datastr["Marker"], pose)
            force_rate = float(datastr["Force"]["FrameRate"])
            mass = float(datastr["Info"]["subjMass"])
            trial_matches = 0
            trial_peaks = 0
            for side in ("left", "right"):
                events = force_strides(datastr["Force"][side.capitalize() + "ForceData"], force_rate, mass)
                peaks, _, direction = detect_contacts(pose, side)
                trial_peaks += len(peaks)
                used: set[int] = set()
                for first, second in zip(peaks[:-1], peaks[1:]):
                    detected_duration = (second - first) / float(pose["fps"].iloc[0])
                    if not 0.45 <= detected_duration <= 1.20:
                        continue
                    detected_sensor_time = float(pose["source_time_s"].iloc[first]) + sync_offset
                    match = match_force_event(detected_sensor_time, events, used)
                    if match is None:
                        continue
                    event_index, event = match
                    if not 0.45 <= event["next_contact_s"] - event["initial_contact_s"] <= 1.20:
                        continue
                    used.add(event_index)
                    trial_matches += 1
                    features = sample_features(pose, int(first), int(second), side, direction, speed)
                    row = {
                        "subject": subject, "run": run, "side": side,
                        "detected_start_source_time_s": float(pose["source_time_s"].iloc[first]),
                        "sync_offset_s": sync_offset, "sync_correlation": sync_corr,
                        "target_ic_offset_s": event["initial_contact_s"] - detected_sensor_time,
                        "target_propulsion_offset_s": event["propulsion_start_s"] - detected_sensor_time,
                        "target_recovery_offset_s": event["toe_off_s"] - detected_sensor_time,
                        "target_next_ic_offset_s": event["next_contact_s"] - detected_sensor_time,
                        **features,
                    }
                    rows.append(row)
            trial_rows.append({
                "subject": subject, "run": run, "sync_offset_s": sync_offset,
                "sync_correlation": sync_corr, "detected_contact_peaks": trial_peaks,
                "matched_video_force_strides": trial_matches,
                "pose_detection_rate": float(pose["pose_detected"].mean()),
            })
            print(json.dumps(trial_rows[-1]), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trial_rows)


TARGETS = [
    "target_ic_offset_s", "target_propulsion_offset_s",
    "target_recovery_offset_s", "target_next_ic_offset_s",
]


def make_model(name: str):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=350, max_depth=8, min_samples_leaf=5,
            max_features=0.65, n_jobs=-1, random_state=RANDOM_SEED,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=350, max_depth=10, min_samples_leaf=4,
            max_features=0.75, n_jobs=-1, random_state=RANDOM_SEED,
        )
    raise ValueError(name)


def baseline_predict(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    medians = train.groupby("run")[["target_propulsion_offset_s", "target_recovery_offset_s"]].median()
    predictions = []
    for row in validation.itertuples(index=False):
        duration = float(row.detected_stride_duration_s)
        run_medians = medians.loc[int(row.run)]
        predictions.append([0.0, run_medians.iloc[0], run_medians.iloc[1], duration])
    return np.asarray(predictions, dtype=float)


def constrain(predictions: np.ndarray) -> np.ndarray:
    result = predictions.copy()
    result[:, 0] = np.clip(result[:, 0], -0.12, 0.12)
    result[:, 1] = np.clip(result[:, 1], result[:, 0] + 0.055, 0.65)
    result[:, 2] = np.clip(result[:, 2], result[:, 1] + 0.07, 0.85)
    result[:, 3] = np.clip(result[:, 3], result[:, 2] + 0.12, 1.20)
    return result


def phase_arrays(truth_boundaries: np.ndarray, predicted_boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truths, predictions = [], []
    for truth, prediction in zip(truth_boundaries, predicted_boundaries):
        times = np.arange(int(round((truth[3] - truth[0]) * 100.0))) / 100.0 + truth[0]
        truth_labels = np.where(
            times < truth[0] + 0.030, 1,
            np.where(times < truth[1], 2, np.where(times < truth[2], 3, 4)),
        )
        predicted_labels = np.where(
            times < prediction[0], 4,
            np.where(
                times < prediction[0] + 0.030, 1,
                np.where(times < prediction[1], 2, np.where(times < prediction[2], 3, 4)),
            ),
        )
        truths.append(truth_labels.astype(np.int8))
        predictions.append(predicted_labels.astype(np.int8))
    return np.concatenate(truths), np.concatenate(predictions)


def score_predictions(table: pd.DataFrame, predictions: np.ndarray) -> dict:
    truth_boundaries = table[TARGETS].to_numpy(dtype=float)
    predictions = constrain(predictions)
    truth, prediction = phase_arrays(truth_boundaries, predictions)
    per_stage = {
        STAGES[label]: float(f1_score(truth, prediction, labels=[label], average="macro"))
        for label in STAGES
    }
    return {
        "macro_f1": float(f1_score(truth, prediction, labels=[1, 2, 3, 4], average="macro")),
        "per_stage_f1": per_stage,
        "evaluated_frames_100hz": int(len(truth)),
    }


def main() -> None:
    (EXPERIMENT / "artifacts").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "reports").mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    development = list(split["development_subjects"])
    forbidden = set(split["test_subjects"])
    if set(development) & forbidden or forbidden != {"Subj04", "Subj11"}:
        raise RuntimeError("Invalid subject split")
    dataset, trials = build_dataset(development)
    feature_names = [
        column for column in dataset.columns
        if column not in {"subject", "run", "side", "detected_start_source_time_s", "sync_offset_s", "sync_correlation", *TARGETS}
    ]
    dataset_path = EXPERIMENT / "artifacts/development_matched_strides.csv"
    trials_path = EXPERIMENT / "artifacts/development_trial_quality.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    candidates = ["v3_timing_baseline", "ridge", "random_forest", "extra_trees"]
    cv_rows = []
    for held_out in development:
        train = dataset[dataset["subject"] != held_out]
        validation = dataset[dataset["subject"] == held_out]
        for candidate in candidates:
            if candidate == "v3_timing_baseline":
                predictions = baseline_predict(train, validation)
            else:
                model = make_model(candidate)
                model.fit(train[feature_names].to_numpy(), train[TARGETS].to_numpy())
                predictions = model.predict(validation[feature_names].to_numpy())
            scores = score_predictions(validation, predictions)
            cv_rows.append({
                "held_out_subject": held_out, "candidate": candidate,
                "validation_strides": int(len(validation)), **scores["per_stage_f1"],
                "macro_f1": scores["macro_f1"], "evaluated_frames_100hz": scores["evaluated_frames_100hz"],
            })
    cv = pd.DataFrame(cv_rows)
    cv_path = EXPERIMENT / "reports/development_loso_cv.csv"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")
    summary = cv.groupby("candidate").agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"), landing_f1_mean=("landing", "mean"),
        braking_f1_mean=("braking", "mean"), propulsion_f1_mean=("propulsion", "mean"),
        recovery_f1_mean=("recovery", "mean"),
    ).reset_index().sort_values(["macro_f1_mean", "macro_f1_min"], ascending=False)
    summary_path = EXPERIMENT / "reports/development_loso_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    eligible = summary[summary["candidate"] != "v3_timing_baseline"]
    selected = str(eligible.iloc[0]["candidate"])
    model = make_model(selected)
    model.fit(dataset[feature_names].to_numpy(), dataset[TARGETS].to_numpy())
    artifact = {
        "experiment": "boundary_regression_v1", "selected_candidate": selected,
        "feature_names": feature_names, "target_names": TARGETS,
        "development_subjects": development, "forbidden_subjects": sorted(forbidden),
        "clip_window_s": [10.0, 22.0], "model": model,
    }
    artifact_path = EXPERIMENT / "artifacts/boundary_regression_v1.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    selected_row = summary[summary["candidate"] == selected].iloc[0]
    baseline_row = summary[summary["candidate"] == "v3_timing_baseline"].iloc[0]
    manifest = {
        "status": "DEVELOPMENT_ONLY_MODEL_FROZEN",
        "experiment": "boundary_regression_v1",
        "selected_candidate": selected,
        "selection_rule": "highest mean development-only subject-disjoint LOSO macro F1 among learned candidates",
        "artifact": str(artifact_path), "artifact_sha256": sha256(artifact_path),
        "development_subjects": development, "forbidden_subjects": sorted(forbidden),
        "forbidden_subjects_accessed": False,
        "matched_development_strides": int(len(dataset)),
        "mean_sync_absolute_correlation": float(trials["sync_correlation"].mean()),
        "selected_loso_macro_f1_mean": float(selected_row["macro_f1_mean"]),
        "baseline_loso_macro_f1_mean": float(baseline_row["macro_f1_mean"]),
        "macro_f1_improvement": float(selected_row["macro_f1_mean"] - baseline_row["macro_f1_mean"]),
        "landing_f1_selected": float(selected_row["landing_f1_mean"]),
        "landing_f1_baseline": float(baseline_row["landing_f1_mean"]),
        "limitations": [
            "This is development-only LOSO evidence, not a new untouched-test result.",
            "Subj04 and Subj11 were already used by the historical v3 final test and are deliberately excluded here.",
            "A future genuinely untouched subject/video set is required for final generalization claims."
        ],
    }
    manifest_path = EXPERIMENT / "reports/MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nLOSO summary\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
