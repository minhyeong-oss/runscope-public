from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_adaptation_v1"
SEQUENCE_ROOT = EXPERIMENT / "artifacts/sequences"
DATASET_MANIFEST = EXPERIMENT / "artifacts/DATASET_MANIFEST.json"
V1_SOURCE = ROOT / "experiments/landing_visual_contact_v1/src/train_and_test.py"
V2_MODEL = ROOT / "experiments/landing_visual_contact_v2/artifacts/landing_visual_contact_v2.pt"

SEED = 20261110
WINDOW = 96
STRIDE = 32
EPOCHS = 18
TOLERANCES = [1, 2, 3]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("landing_kth_train_base", V1_SOURCE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def accepted_sequences(split: str) -> list[dict]:
    bundles = []
    for path in sorted(SEQUENCE_ROOT.glob("*.npz")):
        bundle = np.load(path)
        if str(bundle["split"].item()) != split or not bool(bundle["accepted"].item()):
            continue
        bundles.append({key: bundle[key] for key in bundle.files} | {"path": path})
    return bundles


def make_windows(bundles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    feature_windows = []
    label_windows = []
    for bundle in bundles:
        features = bundle["features"].astype(np.float32)
        labels = bundle["labels"].astype(np.float32)
        if len(features) < WINDOW:
            padding = WINDOW - len(features)
            features = np.pad(features, ((0, padding), (0, 0)), mode="edge")
            labels = np.pad(labels, (0, padding))
        starts = list(range(0, len(features) - WINDOW + 1, STRIDE))
        if starts[-1] != len(features) - WINDOW:
            starts.append(len(features) - WINDOW)
        for start in starts:
            feature_windows.append(features[start:start + WINDOW])
            label_windows.append(labels[start:start + WINDOW])
    if not feature_windows:
        raise RuntimeError("No accepted KTH sequences for training")
    return np.stack(feature_windows), np.stack(label_windows)


def augment(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    output += torch.randn_like(output) * 0.012
    reshaped = output.reshape(output.shape[0], output.shape[1], 8, 7)
    flip_mask = torch.rand(output.shape[0]) < 0.5
    if flip_mask.any():
        flipped = reshaped[flip_mask][:, :, [1, 0, 3, 2, 5, 4, 7, 6], :].clone()
        flipped[:, :, :, [0, 2, 4]] *= -1.0
        reshaped[flip_mask] = flipped
    # Simulate intermittent low-confidence landmarks in deployment video.
    dropout = torch.rand(output.shape[0], output.shape[1], 8) < 0.025
    confidence = reshaped[:, :, :, 6]
    confidence[dropout] *= 0.25
    return reshaped.reshape_as(output)


def soft_targets(labels: torch.Tensor) -> torch.Tensor:
    target = labels.clone()
    target[:, 1:] = torch.maximum(target[:, 1:], labels[:, :-1] * 0.35)
    target[:, :-1] = torch.maximum(target[:, :-1], labels[:, 1:] * 0.35)
    return target


def train(train_x: np.ndarray, train_y: np.ndarray):
    seed_everything(SEED)
    checkpoint = torch.load(V2_MODEL, map_location="cpu", weights_only=False)
    model = base.LandingEventTCN(int(checkpoint["input_features"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=4.0e-3)
    positive = float(train_y.sum())
    negative = float(train_y.size - positive)
    positive_weight = torch.tensor(np.sqrt(negative / max(positive, 1.0)), dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    history = []
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for features, labels in loader:
            logits = model(augment(features))
            loss = loss_function(logits, soft_targets(labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
        row = {"epoch": epoch + 1, "loss": total / len(dataset)}
        history.append(row)
        print(json.dumps(row), flush=True)
    return model.eval(), history


def predict(model, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(features, dtype=torch.float32)[None]
        return torch.sigmoid(model(tensor))[0].numpy()


def aggregate(
    bundles: list[dict],
    probabilities: dict[str, np.ndarray],
    threshold: float,
    distance: int,
    tolerance: int,
) -> tuple[dict, list[dict]]:
    totals = np.zeros(3, dtype=int)
    errors = []
    rows = []
    for bundle in bundles:
        name = Path(str(bundle["source_video"].item())).stem.replace("_uncomp", "")
        truth = bundle["events"].astype(np.int64)
        prediction = base.predicted_events(probabilities[name], threshold, distance)
        tp, fp, fn, event_errors = base.match_events(truth, prediction, tolerance)
        totals += [tp, fp, fn]
        errors.extend(event_errors)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "video": name,
            "subject": str(bundle["subject"].item()),
            "tolerance_frames": tolerance,
            "pseudo_truth_events": len(truth),
            "predicted_events": len(prediction),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "pseudo_label_agreement_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        })
    tp, fp, fn = totals.tolist()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "pseudo_label_agreement_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "precision": precision,
        "recall": recall,
        "matched_onset_mae_ms": float(np.mean(np.abs(errors))) * 1000.0 / 30.0 if errors else None,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }, rows


def main() -> None:
    torch.set_num_threads(4)
    dataset_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if dataset_manifest["prior_locked_test_data_accessed"]:
        raise RuntimeError("Prior locked test was accessed")
    if dataset_manifest["kth_official_test_data_accessed"]:
        raise RuntimeError("KTH official test was accessed")

    training = accepted_sequences("train")
    validation = accepted_sequences("validation")
    if not training or not validation:
        raise RuntimeError(
            f"Need accepted train and validation videos, got {len(training)} and {len(validation)}"
        )
    train_x, train_y = make_windows(training)
    model, history = train(train_x, train_y)
    validation_probability = {}
    for bundle in validation:
        name = Path(str(bundle["source_video"].item())).stem.replace("_uncomp", "")
        validation_probability[name] = predict(model, bundle["features"].astype(np.float32))

    baseline_checkpoint = torch.load(V2_MODEL, map_location="cpu", weights_only=False)
    baseline_model = base.LandingEventTCN(int(baseline_checkpoint["input_features"]))
    baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline_model.eval()
    baseline_probability = {}
    for bundle in validation:
        name = Path(str(bundle["source_video"].item())).stem.replace("_uncomp", "")
        baseline_probability[name] = predict(
            baseline_model, bundle["features"].astype(np.float32)
        )

    grid_rows = []
    for distance in range(7, 14):
        for threshold in np.linspace(0.10, 0.90, 81):
            metrics, _ = aggregate(
                validation, validation_probability, float(threshold), distance, tolerance=2
            )
            grid_rows.append({"threshold": float(threshold), "distance": distance, **metrics})
    grid = pd.DataFrame(grid_rows).sort_values(
        ["pseudo_label_agreement_f1", "matched_onset_mae_ms", "threshold"],
        ascending=[False, True, False],
    )
    selected = grid.iloc[0]
    threshold = float(selected["threshold"])
    distance = int(selected["distance"])
    metrics_by_tolerance = {}
    subject_rows = []
    for tolerance in TOLERANCES:
        metrics, rows = aggregate(
            validation, validation_probability, threshold, distance, tolerance
        )
        metrics_by_tolerance[f"within_{tolerance}_frames"] = metrics
        subject_rows.extend(rows)

    baseline_original_metrics = {}
    for tolerance in TOLERANCES:
        metrics, _ = aggregate(
            validation,
            baseline_probability,
            float(baseline_checkpoint["threshold"]),
            int(baseline_checkpoint["minimum_peak_distance_frames"]),
            tolerance,
        )
        baseline_original_metrics[f"within_{tolerance}_frames"] = metrics

    baseline_grid_rows = []
    for baseline_distance in range(7, 14):
        for baseline_threshold in np.linspace(0.10, 0.90, 81):
            metrics, _ = aggregate(
                validation,
                baseline_probability,
                float(baseline_threshold),
                baseline_distance,
                tolerance=2,
            )
            baseline_grid_rows.append({
                "threshold": float(baseline_threshold),
                "distance": baseline_distance,
                **metrics,
            })
    baseline_grid = pd.DataFrame(baseline_grid_rows).sort_values(
        ["pseudo_label_agreement_f1", "matched_onset_mae_ms", "threshold"],
        ascending=[False, True, False],
    )
    baseline_selected = baseline_grid.iloc[0]
    baseline_tuned_threshold = float(baseline_selected["threshold"])
    baseline_tuned_distance = int(baseline_selected["distance"])
    baseline_tuned_metrics = {}
    for tolerance in TOLERANCES:
        metrics, _ = aggregate(
            validation,
            baseline_probability,
            baseline_tuned_threshold,
            baseline_tuned_distance,
            tolerance,
        )
        baseline_tuned_metrics[f"within_{tolerance}_frames"] = metrics

    artifact_dir = EXPERIMENT / "artifacts"
    report_dir = EXPERIMENT / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "landing_kth_adaptation_v1.pt"
    torch.save({
        "experiment": "landing_kth_adaptation_v1",
        "architecture": "v2 residual TCN fine-tuned on KTH consensus pseudo-labels",
        "model_state_dict": model.state_dict(),
        "input_features": int(train_x.shape[2]),
        "normalizer": "per-video median and IQR, scale floor 0.05",
        "threshold": threshold,
        "minimum_peak_distance_frames": distance,
        "pose_rate_hz": 30.0,
        "training_subjects": dataset_manifest["split"]["train"],
        "validation_subjects": dataset_manifest["split"]["validation"],
        "untouched_kth_test_subjects": dataset_manifest["split"]["untouched_test"],
        "forbidden_prior_test_subjects": dataset_manifest["prior_locked_test_subjects"],
    }, model_path)

    pd.DataFrame(history).to_csv(report_dir / "training_history.csv", index=False)
    grid.to_csv(report_dir / "validation_decoder_grid.csv", index=False, encoding="utf-8-sig")
    baseline_grid.to_csv(
        report_dir / "baseline_validation_decoder_grid.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(subject_rows).to_csv(
        report_dir / "validation_video_metrics.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "status": "KTH_DOMAIN_ADAPTATION_VALIDATED_ON_PSEUDO_LABELS_NO_FORCE_F1_CLAIM",
        "model_sha256": sha256(model_path),
        "initialized_from_frozen_v2_sha256": sha256(V2_MODEL),
        "training_videos": len(training),
        "validation_videos": len(validation),
        "training_windows": int(len(train_x)),
        "selected_threshold": threshold,
        "selected_minimum_peak_distance_frames": distance,
        "validation_pseudo_label_agreement": metrics_by_tolerance,
        "frozen_v2_original_decoder_pseudo_label_agreement": baseline_original_metrics,
        "frozen_v2_validation_tuned_decoder": {
            "threshold": baseline_tuned_threshold,
            "minimum_peak_distance_frames": baseline_tuned_distance,
            "metrics": baseline_tuned_metrics,
        },
        "adapted_minus_tuned_baseline_f1": {
            key: (
                metrics_by_tolerance[key]["pseudo_label_agreement_f1"]
                - baseline_tuned_metrics[key]["pseudo_label_agreement_f1"]
            )
            for key in metrics_by_tolerance
        },
        "prior_locked_test_data_accessed": False,
        "kth_official_test_data_accessed": False,
        "post_validation_retraining_performed": False,
        "limitations": [
            "Validation targets are teacher/ankle consensus pseudo-labels, not manual or force-plate contacts.",
            "Pseudo-label agreement is not an independent biomechanical accuracy score.",
            "An untouched manually annotated or synchronized test video is still required for a true F1 claim.",
        ],
    }
    (report_dir / "MODEL_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
