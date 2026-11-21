from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat

from evaluate_twente_force_f1 import force_strides, ground_truth_label
from evaluate_twente_temporal_prior_v2 import metrics


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model_v1/twente_subject_split_v3"
ARTIFACT_PATH = MODEL_DIR / "runscope_twente_v3_frozen.joblib"
FREEZE_PATH = MODEL_DIR / "MODEL_FREEZE.json"
OUTPUT_DIR = ROOT / "video_v1/test_outputs/twente_locked_test/final_v3"
SPEEDS = {63: 1.75, 81: 2.25, 99: 2.75}
TRIALS = [("Subj04", 81), ("Subj04", 99), ("Subj11", 63), ("Subj11", 81), ("Subj11", 99)]
STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trial_paths(subject: str, run: int) -> tuple[Path, Path, Path]:
    mat = ROOT / f"data/twente_sensor/{subject}/locked_test_extracted/{subject}/{subject}_run_{run}.mat"
    out = ROOT / f"video_v1/test_outputs/twente_locked_test/{subject}_run{run}"
    stem = f"{subject}_run{run}_side_10s_to_22s"
    return mat, out / f"{stem}_yolo_landmarks.csv", out / f"{stem}_auto_strides.csv"


def synchronization_offset(marker: dict, landmarks: pd.DataFrame) -> tuple[float, float, list[float]]:
    labels = list(marker["DataLabel"])
    data = np.asarray(marker["MarkerData"], dtype=float)
    rate = float(marker["FrameRate"])
    sensor_time = np.arange(data.shape[2], dtype=float) / rate
    video_time = 10.0 + landmarks["time_s"].to_numpy(dtype=float)
    midhip_x = (
        landmarks["left_hip_x"].to_numpy(dtype=float)
        + landmarks["right_hip_x"].to_numpy(dtype=float)
    ) / 2.0
    candidates = []
    for offset in np.arange(-0.100, 0.1001, 0.0005):
        correlations = []
        for side, marker_name in (("left", "LLM"), ("right", "RLM")):
            video_signal = landmarks[f"{side}_ankle_x"].to_numpy(dtype=float) - midhip_x
            sensor_signal_raw = (
                data[labels.index(marker_name), 0]
                - data[labels.index("SACR"), 0]
            )
            finite = np.isfinite(sensor_signal_raw)
            if int(finite.sum()) < 2:
                raise RuntimeError(f"Insufficient finite marker samples for {marker_name}")
            sensor_signal = np.interp(
                sensor_time,
                sensor_time[finite],
                sensor_signal_raw[finite],
            )
            interpolated = np.interp(video_time + offset, sensor_time, sensor_signal)
            correlations.append(abs(float(np.corrcoef(video_signal, interpolated)[0, 1])))
        candidates.append((float(np.mean(correlations)), float(offset), correlations))
    mean_correlation, offset, correlations = max(candidates, key=lambda item: item[0])
    return offset, mean_correlation, correlations


def frozen_boundaries(artifact: dict, run: int, speed: float, duration: float) -> tuple[float, float]:
    bundle = artifact["model"]
    if bundle["kind"] == "speed_median":
        values = bundle["medians"][run]
        return float(values["propulsion_start_fraction"]), float(values["recovery_start_fraction"])
    x = np.asarray([[speed, duration]], dtype=float)
    p3 = float(bundle["models"][0].predict(x)[0])
    p4 = float(bundle["models"][1].predict(x)[0])
    p3 = float(np.clip(p3, 0.12, 0.55))
    p4 = float(np.clip(p4, p3 + 0.08, 0.75))
    return p3, p4


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256(ARTIFACT_PATH)
    if actual_hash != freeze["artifact_sha256"]:
        raise RuntimeError("Frozen artifact hash mismatch")
    if freeze["status"] != "FROZEN_BEFORE_TEST_ACCESS":
        raise RuntimeError("Invalid freeze status")
    artifact = joblib.load(ARTIFACT_PATH)
    if set(artifact["locked_test_subjects"]) != {"Subj04", "Subj11"}:
        raise RuntimeError("Artifact test split does not match the locked test")

    rows = []
    trial_metadata = []
    for subject, run in TRIALS:
        mat_path, landmarks_path, strides_path = trial_paths(subject, run)
        datastr = loadmat(mat_path, simplify_cells=True, variable_names=["Datastr"])["Datastr"]
        landmarks = pd.read_csv(landmarks_path)
        strides = pd.read_csv(strides_path)
        strides = strides[
            strides["accepted_stride"].astype(str).str.lower().isin(["true", "1"])
        ].copy()
        sync_offset, sync_correlation, side_correlations = synchronization_offset(
            datastr["Marker"], landmarks
        )
        force_rate = float(datastr["Force"]["FrameRate"])
        body_mass = float(datastr["Info"]["subjMass"])
        force_events = {
            side: force_strides(
                datastr["Force"][side.capitalize() + "ForceData"],
                force_rate,
                body_mass,
            )
            for side in ("left", "right")
        }
        speed = SPEEDS[run]
        for stride in strides.itertuples(index=False):
            side = str(stride.tracked_side)
            start = int(stride.start_frame_inclusive)
            end = int(stride.end_frame_exclusive)
            duration = (end - start) / 33.0
            p3, p4 = frozen_boundaries(artifact, run, speed, duration)
            for sample_index in range(int(round(duration * 100.0))):
                local_time = sample_index / 100.0
                fraction = min(0.999999, local_time / duration)
                prediction = 1 if local_time < 0.030 else 2 if fraction < p3 else 3 if fraction < p4 else 4
                video_time = 10.0 + start / 33.0 + local_time
                sensor_time = video_time + sync_offset
                truth = ground_truth_label(sensor_time, force_events[side])
                if truth is None:
                    continue
                rows.append({
                    "subject": subject,
                    "run": run,
                    "speed_m_s": speed,
                    "stride_id": str(stride.stride_id),
                    "side": side,
                    "video_time_s": video_time,
                    "sensor_time_s": sensor_time,
                    "cycle_fraction": fraction,
                    "predicted_label": prediction,
                    "predicted_stage": STAGE_NAMES[prediction],
                    "ground_truth_label": int(truth),
                    "ground_truth_stage": STAGE_NAMES[int(truth)],
                    "correct": prediction == truth,
                })
        trial_metadata.append({
            "subject": subject,
            "run": run,
            "speed_m_s": speed,
            "accepted_video_strides": int(len(strides)),
            "sync_offset_s": sync_offset,
            "sync_mean_absolute_correlation": sync_correlation,
            "sync_side_absolute_correlations": side_correlations,
            "force_strides_left": len(force_events["left"]),
            "force_strides_right": len(force_events["right"]),
        })

    comparison = pd.DataFrame(rows)
    overall = metrics(comparison, "predicted_label")
    per_subject = {
        subject: metrics(group, "predicted_label")
        for subject, group in comparison.groupby("subject")
    }
    per_trial = {
        f"{subject}_run_{run}": metrics(group, "predicted_label")
        for (subject, run), group in comparison.groupby(["subject", "run"])
    }
    comparison_path = OUTPUT_DIR / "locked_test_predictions_and_force_truth.csv"
    report_path = OUTPUT_DIR / "LOCKED_TEST_REPORT.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    report = {
        "status": "FINAL_LOCKED_TEST_COMPLETED_NO_REFIT",
        "model_artifact": str(ARTIFACT_PATH),
        "verified_artifact_sha256": actual_hash,
        "development_subjects": artifact["development_subjects"],
        "test_subjects": artifact["locked_test_subjects"],
        "selected_candidate": artifact["selected_candidate"],
        "test_trials": trial_metadata,
        "overall_metrics": overall,
        "per_subject_metrics": per_subject,
        "per_trial_metrics": per_trial,
        "target_macro_f1": 0.85,
        "meets_target": overall["macro_f1"] >= 0.85,
        "post_test_training_performed": False,
        "rule": "This result is final for v3. The model must not be retrained or tuned using these test subjects.",
        "outputs": {"comparison_csv": str(comparison_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "model_sha256": actual_hash,
        "test_subjects": artifact["locked_test_subjects"],
        "test_trials": len(trial_metadata),
        "test_strides": sum(item["accepted_video_strides"] for item in trial_metadata),
        "overall_macro_f1": overall["macro_f1"],
        "overall_weighted_f1": overall["weighted_f1"],
        "overall_accuracy": overall["accuracy"],
        "per_subject_macro_f1": {key: value["macro_f1"] for key, value in per_subject.items()},
        "per_trial_macro_f1": {key: value["macro_f1"] for key, value in per_trial.items()},
        "meets_target": report["meets_target"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
