from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_visual_contact_v2"
V1_EXPERIMENT = ROOT / "experiments/landing_visual_contact_v1"
V1_SOURCE = V1_EXPERIMENT / "src/train_and_test.py"
V1_MANIFEST = V1_EXPERIMENT / "artifacts/DATASET_MANIFEST.json"

SEED = 20261110
EPOCHS = 25
TOLERANCES = [1, 2, 3]


def load_v1_module():
    spec = importlib.util.spec_from_file_location("landing_visual_contact_v1_base", V1_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load v1 model source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEED = SEED
    module.EPOCHS = EPOCHS
    return module


base = load_v1_module()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_normalize(features: np.ndarray) -> np.ndarray:
    center = np.median(features, axis=0, keepdims=True)
    scale = np.percentile(features, 75, axis=0, keepdims=True) - np.percentile(
        features, 25, axis=0, keepdims=True
    )
    return ((features - center) / np.clip(scale, 0.05, None)).astype(np.float32)


def subject_data(subject: str) -> dict[str, np.ndarray]:
    bundle = base.load_subject(subject)
    bundle["features"] = subject_normalize(bundle["features"].astype(np.float32))
    return bundle


def make_windows(subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_windows = []
    label_windows = []
    for subject in subjects:
        bundle = subject_data(subject)
        features = bundle["features"]
        labels = bundle["labels"].astype(np.float32)
        starts = list(range(0, len(features) - base.WINDOW + 1, base.STRIDE))
        if starts[-1] != len(features) - base.WINDOW:
            starts.append(len(features) - base.WINDOW)
        for start in starts:
            feature_windows.append(features[start:start + base.WINDOW])
            label_windows.append(labels[start:start + base.WINDOW])
    return np.stack(feature_windows), np.stack(label_windows)


def decode(probability: np.ndarray, threshold: float, distance: int) -> np.ndarray:
    return base.predicted_events(probability, threshold, distance)


def aggregate_metrics(
    subjects: list[str],
    truth: dict[str, np.ndarray],
    probability: dict[str, np.ndarray],
    threshold: float,
    distance: int,
    tolerance: int,
) -> tuple[dict[str, float], list[dict]]:
    totals = np.zeros(3, dtype=int)
    errors = []
    rows = []
    for subject in subjects:
        prediction = decode(probability[subject], threshold, distance)
        tp, fp, fn, subject_errors = base.match_events(
            truth[subject], prediction, tolerance
        )
        totals += [tp, fp, fn]
        errors.extend(subject_errors)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "subject": subject,
            "tolerance_frames": tolerance,
            "truth_events": int(len(truth[subject])),
            "predicted_events": int(len(prediction)),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        })
    tp, fp, fn = totals.tolist()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "event_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "event_precision": precision,
        "event_recall": recall,
        "matched_onset_mae_ms": (
            float(np.mean(np.abs(errors))) * 1000.0 / base.POSE_RATE
            if errors else float("nan")
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }, rows


def main() -> None:
    torch.set_num_threads(4)
    manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    development = manifest["split"]["train"] + manifest["split"]["validation"]
    forbidden = manifest["split"]["locked_test"]
    if set(development) & set(forbidden):
        raise RuntimeError("Forbidden participant leakage")

    truth: dict[str, np.ndarray] = {}
    oof_probability: dict[str, np.ndarray] = {}
    folds = []
    zero_mean = zero_std = None
    for fold, held_out in enumerate(development):
        training = [subject for subject in development if subject != held_out]
        train_x, train_y = make_windows(training)
        zero_mean = np.zeros((1, 1, train_x.shape[2]), dtype=np.float32)
        zero_std = np.ones_like(zero_mean)
        base.SEED = SEED + fold
        model = base.train_model(train_x, train_y, zero_mean, zero_std)
        held_bundle = subject_data(held_out)
        probability = base.probabilities(
            model, held_bundle["features"], zero_mean, zero_std
        )
        truth[held_out] = held_bundle["events"].astype(np.int64)
        oof_probability[held_out] = probability.astype(np.float32)
        folds.append({
            "held_out_subject": held_out,
            "training_subjects": training,
            "training_windows": int(len(train_x)),
            "truth_events": int(len(truth[held_out])),
            "probability_max": float(probability.max()),
            "probability_p95": float(np.percentile(probability, 95)),
        })
        print(json.dumps(folds[-1]), flush=True)

    grid_rows = []
    for distance in range(7, 15):
        for threshold in np.linspace(0.05, 0.95, 91):
            metrics, _ = aggregate_metrics(
                development, truth, oof_probability,
                float(threshold), distance, tolerance=1,
            )
            grid_rows.append({"threshold": float(threshold), "distance": distance, **metrics})
    grid = pd.DataFrame(grid_rows).sort_values(
        ["event_f1", "matched_onset_mae_ms", "threshold"],
        ascending=[False, True, False],
    )
    selected = grid.iloc[0]
    threshold = float(selected["threshold"])
    distance = int(selected["distance"])

    metrics_by_tolerance = {}
    subject_rows = []
    for tolerance in TOLERANCES:
        metrics, rows = aggregate_metrics(
            development, truth, oof_probability,
            threshold, distance, tolerance=tolerance,
        )
        metrics_by_tolerance[f"within_{tolerance}_frames"] = metrics
        subject_rows.extend(rows)

    report_dir = EXPERIMENT / "reports"
    artifact_dir = EXPERIMENT / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(report_dir / "development_oof_decoder_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(subject_rows).to_csv(
        report_dir / "development_oof_subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        report_dir / "development_oof_predictions.npz",
        subjects=np.asarray(development),
        probabilities=np.asarray([oof_probability[s] for s in development], dtype=object),
        truths=np.asarray([truth[s] for s in development], dtype=object),
    )

    # Final deployment artifact uses development participants only. The v1 test
    # participants are deliberately never loaded in this experiment.
    final_x, final_y = make_windows(development)
    zero_mean = np.zeros((1, 1, final_x.shape[2]), dtype=np.float32)
    zero_std = np.ones_like(zero_mean)
    base.SEED = SEED + 100
    final_model = base.train_model(final_x, final_y, zero_mean, zero_std)
    model_path = artifact_dir / "landing_visual_contact_v2.pt"
    torch.save({
        "experiment": "landing_visual_contact_v2",
        "architecture": "v1 residual TCN with robust per-video normalization",
        "model_state_dict": final_model.state_dict(),
        "input_features": int(final_x.shape[2]),
        "normalizer": "per-video median and IQR, scale floor 0.05",
        "threshold": threshold,
        "minimum_peak_distance_frames": distance,
        "pose_rate_hz": base.POSE_RATE,
        "development_subjects": development,
        "forbidden_v1_test_subjects": forbidden,
    }, model_path)

    report = {
        "status": "DEVELOPMENT_OOF_MODEL_NO_NEW_TEST_CLAIM",
        "experiment": "landing_visual_contact_v2",
        "model_sha256": sha256(model_path),
        "development_subjects": development,
        "forbidden_v1_test_subjects": forbidden,
        "forbidden_subjects_accessed": False,
        "folds": folds,
        "selected_threshold": threshold,
        "selected_minimum_peak_distance_frames": distance,
        "development_oof_metrics": metrics_by_tolerance,
        "limitations": [
            "Decoder selection and OOF reporting use the same development OOF predictions.",
            "The v1 locked test was already observed and is not reused for v2 evaluation.",
            "A new independent synchronized running source is required for another test F1.",
            "Targets are kinematic visual-contact onsets, not force-plate contacts.",
        ],
    }
    (report_dir / "MODEL_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
