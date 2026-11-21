from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_gate_hybrid_v1"
LANDING_MODEL_PATH = ROOT / "experiments/landing_aux_v2/artifacts/landing_onset_laas_v2.pt"
LANDING_MANIFEST_PATH = ROOT / "experiments/landing_aux_v2/reports/MODEL_MANIFEST.json"
BASE_COMPARISON_PATH = ROOT / "video_v1/test_outputs/twente_locked_test/final_v3/locked_test_predictions_and_force_truth.csv"
BASE_REPORT_PATH = ROOT / "video_v1/test_outputs/twente_locked_test/final_v3/LOCKED_TEST_REPORT.json"
TCN_SOURCE = ROOT / "experiments/landing_aux_v2/src/train_landing_onset.py"
METRICS_SOURCE = ROOT / "video_v1/evaluate_twente_temporal_prior_v2.py"
POSE_ROOT = EXPERIMENT / "artifacts/raw_pose"
STRIDE_ROOT = ROOT / "video_v1/test_outputs/twente_locked_test"
TRIALS = [("Subj04", 81), ("Subj04", 99), ("Subj11", 63), ("Subj11", 81), ("Subj11", 99)]
STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}

# Frozen before this script reads any force truth. It is the midpoint of the
# auxiliary training pre-contact range [0.10, 0.28] seconds.
PRECONTACT_CONTEXT_S = 0.19
SEQUENCE_DURATION_S = 0.48
SEQUENCE_LENGTH = 64
LANDING_DURATION_S = 0.030


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


def column(table: pd.DataFrame, name: str) -> np.ndarray:
    return table[name].interpolate(limit_direction="both").to_numpy(dtype=float)


def resample(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, np.arange(len(values), dtype=float), values)


def pose_features(pose: pd.DataFrame, side: str, positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
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
            signals[f"{role}_{joint}_x"] = direction * (
                column(pose, f"{anatomical_side}_{joint}_x") - midhip_x
            ) / torso
            signals[f"{role}_{joint}_y"] = (
                column(pose, f"{anatomical_side}_{joint}_y") - midhip_y
            ) / torso
        signals[f"{role}_heel_visibility"] = column(
            pose, f"{anatomical_side}_heel_visibility"
        )
        signals[f"{role}_toe_visibility"] = column(
            pose, f"{anatomical_side}_foot_index_visibility"
        )
    signals["trunk_x"] = direction * (shoulder_x - midhip_x) / torso
    signals["trunk_y"] = (shoulder_y - midhip_y) / torso
    names = list(signals)
    values = np.column_stack([resample(signals[name], positions) for name in names])
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite pose feature sequence")
    return values.astype(np.float32), names


def paths(subject: str, run: int) -> tuple[Path, Path]:
    pose = POSE_ROOT / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
    stem = f"{subject}_run{run}_side_10s_to_22s_auto_strides.csv"
    strides = STRIDE_ROOT / f"{subject}_run{run}" / stem
    return pose, strides


def main() -> None:
    # The frozen artifact and threshold must exist before force truth is loaded.
    manifest = json.loads(LANDING_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["locked_twente_test_subjects_accessed"]:
        raise RuntimeError("Auxiliary model manifest reports test access")
    artifact_hash = sha256(LANDING_MODEL_PATH)
    if artifact_hash != manifest["artifact_sha256"]:
        raise RuntimeError("Landing artifact hash mismatch")
    artifact = torch.load(LANDING_MODEL_PATH, map_location="cpu", weights_only=False)
    threshold = float(artifact["presence_threshold"])
    if abs(threshold - float(manifest["development_selected_threshold"])) > 1e-12:
        raise RuntimeError("Threshold mismatch")

    tcn_module = load_module("landing_aux_v2_frozen", TCN_SOURCE)
    model = tcn_module.LandingOnsetTCN(int(artifact["input_features"]))
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    mean = artifact["normalization_mean"].numpy()
    std = artifact["normalization_std"].numpy()
    expected_features = list(artifact["feature_names"])

    stride_predictions: list[dict] = []
    for subject, run in TRIALS:
        pose_path, strides_path = paths(subject, run)
        pose = pd.read_csv(pose_path)
        strides = pd.read_csv(strides_path)
        strides = strides[
            strides["accepted_stride"].astype(str).str.lower().isin(["true", "1"])
        ].copy()
        source_time = pose["source_time_s"].to_numpy(dtype=float)
        row_positions = np.arange(len(pose), dtype=float)
        for stride in strides.itertuples(index=False):
            detected_start_video_s = 10.0 + int(stride.start_frame_inclusive) / 33.0
            window_start_s = detected_start_video_s - PRECONTACT_CONTEXT_S
            sample_times = window_start_s + np.linspace(
                0.0, SEQUENCE_DURATION_S, SEQUENCE_LENGTH
            )
            positions = np.interp(sample_times, source_time, row_positions)
            features, names = pose_features(pose, str(stride.tracked_side), positions)
            if names != expected_features:
                raise RuntimeError("Feature order does not match the frozen auxiliary model")
            normalized = (features[None] - mean) / std
            with torch.no_grad():
                logits_tensor = model(torch.tensor(normalized, dtype=torch.float32))
                global_logit = torch.logsumexp(logits_tensor, dim=1) - math.log(SEQUENCE_LENGTH)
                presence = float(torch.sigmoid(global_logit)[0])
                peak_index = int(torch.argmax(logits_tensor[0]).item())
            onset_video_s = float(sample_times[peak_index])
            stride_predictions.append({
                "subject": subject,
                "run": run,
                "stride_id": str(stride.stride_id),
                "side": str(stride.tracked_side),
                "detected_start_video_s": detected_start_video_s,
                "aux_presence_probability": presence,
                "aux_gate_activated": presence >= threshold,
                "aux_onset_video_s": onset_video_s,
                "aux_onset_shift_from_detected_ms": (
                    onset_video_s - detected_start_video_s
                ) * 1000.0,
            })

    # Force truth is read only after every gate output has been frozen above.
    comparison = pd.read_csv(BASE_COMPARISON_PATH)
    gates = pd.DataFrame(stride_predictions)
    merged = comparison.merge(
        gates,
        on=["subject", "run", "stride_id", "side"],
        how="left",
        validate="many_to_one",
    )
    if merged["aux_presence_probability"].isna().any():
        raise RuntimeError("Missing gate result for a test stride")
    merged["baseline_predicted_label"] = merged["predicted_label"].astype(int)
    hybrid = merged["baseline_predicted_label"].to_numpy(dtype=int).copy()
    video_time = merged["video_time_s"].to_numpy(dtype=float)
    onset = merged["aux_onset_video_s"].to_numpy(dtype=float)
    active = merged["aux_gate_activated"].to_numpy(dtype=bool)
    base_landing = hybrid == 1
    gated_landing = active & (video_time >= onset) & (video_time < onset + LANDING_DURATION_S)
    # If the new gate is confident, it replaces only the landing window. All
    # braking/propulsion/recovery predictions outside that tiny window stay frozen.
    before_onset = video_time < onset
    hybrid[active & base_landing & before_onset] = 4
    hybrid[active & base_landing & ~before_onset] = 2
    hybrid[gated_landing] = 1
    merged["hybrid_predicted_label"] = hybrid
    merged["hybrid_predicted_stage"] = [STAGE_NAMES[value] for value in hybrid]
    merged["hybrid_correct"] = hybrid == merged["ground_truth_label"].to_numpy(dtype=int)

    metrics_module = load_module("locked_metrics", METRICS_SOURCE)
    baseline_metrics = metrics_module.metrics(merged, "baseline_predicted_label")
    hybrid_metrics = metrics_module.metrics(merged, "hybrid_predicted_label")
    historical = json.loads(BASE_REPORT_PATH.read_text(encoding="utf-8"))["overall_metrics"]
    if abs(baseline_metrics["macro_f1"] - historical["macro_f1"]) > 1e-12:
        raise RuntimeError("Baseline reproduction mismatch")

    outputs = EXPERIMENT / "reports"
    outputs.mkdir(parents=True, exist_ok=True)
    comparison_path = outputs / "locked_test_baseline_vs_landing_gate.csv"
    stride_path = outputs / "locked_test_stride_gate_outputs.csv"
    merged.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    gates.to_csv(stride_path, index=False, encoding="utf-8-sig")
    delta = {
        "macro_f1": hybrid_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "weighted_f1": hybrid_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
        "accuracy": hybrid_metrics["accuracy"] - baseline_metrics["accuracy"],
        "landing_f1": (
            hybrid_metrics["per_stage"]["landing"]["f1-score"]
            - baseline_metrics["per_stage"]["landing"]["f1-score"]
        ),
    }
    adopt = delta["landing_f1"] > 0.0 and delta["macro_f1"] >= 0.0
    report = {
        "status": "HISTORICAL_LOCKED_TEST_REUSE_COMPLETED_NO_REFIT_NO_RETUNING",
        "experiment": "landing_gate_hybrid_v1",
        "landing_artifact": str(LANDING_MODEL_PATH),
        "landing_artifact_sha256": artifact_hash,
        "frozen_presence_threshold": threshold,
        "frozen_precontact_context_s": PRECONTACT_CONTEXT_S,
        "integration_rule": (
            "If auxiliary presence >= frozen threshold, replace only the 30 ms landing window; "
            "otherwise retain the frozen v3 baseline prediction."
        ),
        "test_subjects": ["Subj04", "Subj11"],
        "test_training_performed": False,
        "test_threshold_or_parameter_tuning_performed": False,
        "baseline_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "delta_hybrid_minus_baseline": delta,
        "gate_summary": {
            "strides": int(len(gates)),
            "activated_strides": int(gates["aux_gate_activated"].sum()),
            "activation_rate": float(gates["aux_gate_activated"].mean()),
            "mean_presence_probability": float(gates["aux_presence_probability"].mean()),
            "median_onset_shift_ms": float(gates["aux_onset_shift_from_detected_ms"].median()),
        },
        "deployment_decision": "ADOPT" if adopt else "REJECT",
        "target_macro_f1": 0.85,
        "meets_target": hybrid_metrics["macro_f1"] >= 0.85,
        "warning": (
            "Subj04/Subj11 were already used by the historical v3 evaluation. "
            "This is a fixed-parameter comparison for the new gate, but not a newly untouched benchmark."
        ),
        "outputs": {
            "comparison_csv": str(comparison_path),
            "stride_gate_csv": str(stride_path),
        },
    }
    report_path = outputs / "LOCKED_TEST_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "hybrid_macro_f1": hybrid_metrics["macro_f1"],
        "baseline_landing_f1": baseline_metrics["per_stage"]["landing"]["f1-score"],
        "hybrid_landing_f1": hybrid_metrics["per_stage"]["landing"]["f1-score"],
        "delta": delta,
        "gate_summary": report["gate_summary"],
        "deployment_decision": report["deployment_decision"],
        "meets_target": report["meets_target"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
