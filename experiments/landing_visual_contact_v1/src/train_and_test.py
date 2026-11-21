from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_visual_contact_v1"
SEQUENCE_ROOT = EXPERIMENT / "artifacts/sequences"
MANIFEST_PATH = EXPERIMENT / "artifacts/DATASET_MANIFEST.json"

SEED = 20261109
WINDOW = 128
STRIDE = 64
EPOCHS = 35
TOLERANCE_FRAMES = 1
POSE_RATE = 30.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class LandingEventTCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 48):
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, channels, 1)
        self.temporal = nn.Sequential(
            ResidualBlock(channels, 1),
            ResidualBlock(channels, 2),
            ResidualBlock(channels, 4),
            ResidualBlock(channels, 8),
        )
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        hidden = self.temporal(self.input_projection(values))
        return self.head(hidden).squeeze(1)


def load_subject(subject: str) -> dict[str, np.ndarray]:
    bundle = np.load(SEQUENCE_ROOT / f"subject_{subject}.npz")
    return {key: bundle[key] for key in bundle.files}


def windows(subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    feature_windows = []
    label_windows = []
    for subject in subjects:
        bundle = load_subject(subject)
        features = bundle["features"].astype(np.float32)
        labels = bundle["labels"].astype(np.float32)
        starts = list(range(0, len(features) - WINDOW + 1, STRIDE))
        if starts[-1] != len(features) - WINDOW:
            starts.append(len(features) - WINDOW)
        for start in starts:
            feature_windows.append(features[start:start + WINDOW])
            label_windows.append(labels[start:start + WINDOW])
    return np.stack(feature_windows), np.stack(label_windows)


def augment(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    output += torch.randn_like(output) * 0.006
    reshaped = output.reshape(output.shape[0], output.shape[1], 8, 7)
    flip_mask = torch.rand(output.shape[0]) < 0.5
    if flip_mask.any():
        flipped = reshaped[flip_mask][:, :, [1, 0, 3, 2, 5, 4, 7, 6], :].clone()
        flipped[:, :, :, [0, 2, 4]] *= -1.0
        reshaped[flip_mask] = flipped
    return reshaped.reshape_as(output)


def soft_targets(labels: torch.Tensor) -> torch.Tensor:
    target = labels.clone()
    target[:, 1:] = torch.maximum(target[:, 1:], labels[:, :-1] * 0.35)
    target[:, :-1] = torch.maximum(target[:, :-1], labels[:, 1:] * 0.35)
    return target


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> LandingEventTCN:
    seed_everything(SEED)
    model = LandingEventTCN(x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-3)
    positive = float(y.sum())
    negative = float(y.size - positive)
    positive_weight = torch.tensor(np.sqrt(negative / max(positive, 1.0)), dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for raw, labels in loader:
            features = (augment(raw) - mean_tensor) / std_tensor
            logits = model(features)
            loss = loss_function(logits, soft_targets(labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(raw)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(json.dumps({"epoch": epoch + 1, "loss": total / len(dataset)}), flush=True)
    return model.eval()


def probabilities(
    model: LandingEventTCN,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor((features - mean[0, 0]) / std[0, 0], dtype=torch.float32)[None]
        return torch.sigmoid(model(tensor))[0].numpy()


def predicted_events(probability: np.ndarray, threshold: float, distance: int) -> np.ndarray:
    peaks, _ = find_peaks(probability, height=threshold, distance=distance)
    return peaks.astype(np.int64)


def match_events(
    truth: np.ndarray,
    prediction: np.ndarray,
    tolerance: int,
) -> tuple[int, int, int, list[int]]:
    unmatched = set(range(len(truth)))
    errors = []
    true_positive = 0
    for predicted in prediction:
        candidates = [index for index in unmatched if abs(int(truth[index]) - int(predicted)) <= tolerance]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: abs(int(truth[index]) - int(predicted)))
        unmatched.remove(chosen)
        true_positive += 1
        errors.append(int(predicted) - int(truth[chosen]))
    false_positive = len(prediction) - true_positive
    false_negative = len(truth) - true_positive
    return true_positive, false_positive, false_negative, errors


def evaluate(
    model: LandingEventTCN,
    subjects: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    distance: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    totals = np.zeros(3, dtype=int)
    errors: list[int] = []
    rows = []
    for subject in subjects:
        bundle = load_subject(subject)
        truth = bundle["events"].astype(np.int64)
        probability = probabilities(model, bundle["features"].astype(np.float32), mean, std)
        prediction = predicted_events(probability, threshold, distance)
        tp, fp, fn, subject_errors = match_events(truth, prediction, TOLERANCE_FRAMES)
        totals += [tp, fp, fn]
        errors.extend(subject_errors)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "subject": subject,
            "truth_events": len(truth),
            "predicted_events": len(prediction),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "matched_onset_mae_ms": (
                float(np.mean(np.abs(subject_errors))) * 1000.0 / POSE_RATE
                if subject_errors else float("nan")
            ),
        })
    tp, fp, fn = totals.tolist()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "event_f1_within_one_frame": 2 * precision * recall / max(precision + recall, 1e-12),
        "event_precision_within_one_frame": precision,
        "event_recall_within_one_frame": recall,
        "matched_onset_mae_ms": (
            float(np.mean(np.abs(errors))) * 1000.0 / POSE_RATE if errors else None
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "truth_events": int(tp + fn),
        "predicted_events": int(tp + fp),
    }
    return metrics, pd.DataFrame(rows)


def main() -> None:
    torch.set_num_threads(4)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_subjects = manifest["split"]["train"]
    validation_subjects = manifest["split"]["validation"]
    test_subjects = manifest["split"]["locked_test"]
    if set(test_subjects) & (set(train_subjects) | set(validation_subjects)):
        raise RuntimeError("Locked test leakage")

    train_x, train_y = windows(train_subjects)
    mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.clip(train_x.std(axis=(0, 1), keepdims=True), 1e-4, None).astype(np.float32)
    model = train_model(train_x, train_y, mean, std)

    report_dir = EXPERIMENT / "reports"
    artifact_dir = EXPERIMENT / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    grid_rows = []
    for distance in range(7, 15):
        for threshold in np.linspace(0.05, 0.95, 91):
            metrics, _ = evaluate(
                model, validation_subjects, mean, std, float(threshold), distance
            )
            grid_rows.append({"threshold": float(threshold), "distance": distance, **metrics})
    grid = pd.DataFrame(grid_rows)
    grid = grid.sort_values(
        ["event_f1_within_one_frame", "matched_onset_mae_ms", "threshold"],
        ascending=[False, True, False],
    )
    selected = grid.iloc[0]
    threshold = float(selected["threshold"])
    distance = int(selected["distance"])
    validation_metrics, validation_subject_table = evaluate(
        model, validation_subjects, mean, std, threshold, distance
    )

    model_path = artifact_dir / "landing_visual_contact_v1.pt"
    torch.save({
        "experiment": "landing_visual_contact_v1",
        "model_state_dict": model.state_dict(),
        "input_features": int(train_x.shape[2]),
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "threshold": threshold,
        "minimum_peak_distance_frames": distance,
        "event_tolerance_frames": TOLERANCE_FRAMES,
        "pose_rate_hz": POSE_RATE,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "locked_test_subjects": test_subjects,
    }, model_path)
    frozen_hash = sha256(model_path)

    development_report = {
        "status": "MODEL_AND_DECODER_FROZEN_BEFORE_TEST",
        "model_sha256": frozen_hash,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "locked_test_subjects_not_accessed_during_selection": test_subjects,
        "selected_threshold": threshold,
        "selected_minimum_peak_distance_frames": distance,
        "validation_metrics": validation_metrics,
    }
    grid.to_csv(report_dir / "validation_decoder_grid.csv", index=False, encoding="utf-8-sig")
    validation_subject_table.to_csv(
        report_dir / "validation_subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (report_dir / "DEVELOPMENT_REPORT.json").write_text(
        json.dumps(development_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(development_report, ensure_ascii=False, indent=2), flush=True)

    # Locked test access begins only after the artifact and decoding parameters freeze.
    test_metrics, test_subject_table = evaluate(
        model, test_subjects, mean, std, threshold, distance
    )
    if sha256(model_path) != frozen_hash:
        raise RuntimeError("Model changed during locked test evaluation")
    test_subject_table.to_csv(
        report_dir / "locked_test_subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    test_report = {
        "status": "LOCKED_TEST_EVALUATED_ONCE_NO_RETRAINING",
        "model_sha256": frozen_hash,
        "locked_test_subjects": test_subjects,
        "decoder": {
            "threshold": threshold,
            "minimum_peak_distance_frames": distance,
            "event_tolerance_frames": TOLERANCE_FRAMES,
        },
        "metrics": test_metrics,
        "target_scope": "visual/kinematic contact onset; not force-plate contact",
        "raw_movie_limitation": "Public archive provides video-derived pose CSV, not raw movie files.",
        "post_test_retraining_performed": False,
    }
    (report_dir / "LOCKED_TEST_REPORT.json").write_text(
        json.dumps(test_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
