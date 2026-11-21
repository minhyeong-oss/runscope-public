from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_onset_transfer_v1"
DATA_PATH = EXPERIMENT / "artifacts/twente_development_onset_sequences_64.npz"
PRETRAINED_PATH = ROOT / "experiments/landing_aux_v2/artifacts/landing_onset_laas_v2.pt"
MODEL_SOURCE = ROOT / "experiments/landing_aux_v2/src/train_landing_onset.py"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}
SEED = 20261109
EPOCHS = 55
SAMPLE_INTERVAL_S = 0.48 / 63.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_module("landing_aux_v2_architecture", MODEL_SOURCE)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=(0, 1), keepdims=True)
    std = np.clip(values.std(axis=(0, 1), keepdims=True), 1e-4, None)
    return mean.astype(np.float32), std.astype(np.float32)


def corrupt(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    coordinate_indices = torch.tensor([
        index for index in range(output.shape[2]) if index not in (10, 11, 22, 23)
    ])
    output[:, :, coordinate_indices] += torch.randn_like(
        output[:, :, coordinate_indices]
    ) * 0.012
    visibility_indices = torch.tensor([10, 11, 22, 23])
    visibility_scale = torch.empty((output.shape[0], 1, 1)).uniform_(0.75, 1.0)
    output[:, :, visibility_indices] = torch.clamp(
        output[:, :, visibility_indices] * visibility_scale, 0.0, 1.0
    )
    return output


def new_model(input_features: int) -> nn.Module:
    pretrained = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
    model = model_module.LandingOnsetTCN(input_features)
    model.load_state_dict(pretrained["model_state_dict"])
    return model


def train_model(
    x: np.ndarray,
    onset: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> nn.Module:
    seed_everything(seed)
    model = new_model(x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=4e-3)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(onset, dtype=torch.long),
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for raw, targets in loader:
            logits = model((corrupt(raw) - mean_tensor) / std_tensor)
            loss = nn.functional.cross_entropy(logits, targets, label_smoothing=0.015)
            # Encourage a stable local peak without broadening the evaluation label.
            soft = torch.zeros_like(logits)
            soft[torch.arange(len(targets)), targets] = 1.0
            for delta, weight in ((-1, 0.30), (1, 0.30)):
                indices = targets + delta
                valid = (indices >= 0) & (indices < logits.shape[1])
                soft[torch.arange(len(targets))[valid], indices[valid]] = weight
            loss = loss + 0.20 * nn.functional.binary_cross_entropy_with_logits(logits, soft)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def predict(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor((x - mean) / std, dtype=torch.float32))
        return torch.argmax(logits, dim=1).numpy()


def decode(onsets: np.ndarray, length: int = 64, landing_frames: int = 4) -> np.ndarray:
    output = np.zeros((len(onsets), length), dtype=np.int64)
    for row, onset in enumerate(onsets):
        output[row, onset:min(length, onset + landing_frames)] = 1
    return output


def metrics(truth: np.ndarray, prediction: np.ndarray, true_onset: np.ndarray, predicted_onset: np.ndarray) -> dict:
    return {
        "landing_f1": float(f1_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
        "landing_precision": float(precision_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
        "landing_recall": float(recall_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
        "onset_mae_ms": float(np.mean(np.abs(predicted_onset - true_onset)) * SAMPLE_INTERVAL_S * 1000.0),
        "within_one_frame": float((np.abs(predicted_onset - true_onset) <= 1).mean()),
        "within_two_frames": float((np.abs(predicted_onset - true_onset) <= 2).mean()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    bundle = np.load(DATA_PATH)
    x = bundle["x"].astype(np.float32)
    y = bundle["y"].astype(np.int64)
    onset = bundle["onset_indices"].astype(np.int64)
    subjects = bundle["subjects"].astype(str)
    if set(subjects) != set(SUBJECTS) or set(subjects) & FORBIDDEN:
        raise RuntimeError("Invalid development split")

    baseline_onset = np.full(len(onset), int(round(0.19 / SAMPLE_INTERVAL_S)), dtype=np.int64)
    baseline_prediction = decode(baseline_onset)
    baseline_metrics = metrics(y, baseline_prediction, onset, baseline_onset)
    oof_onset = np.zeros_like(onset)
    rows = []
    for fold, held_out in enumerate(SUBJECTS):
        training, validation = subjects != held_out, subjects == held_out
        mean, std = normalization(x[training])
        model = train_model(x[training], onset[training], mean, std, SEED + fold)
        predicted = predict(model, x[validation], mean, std)
        oof_onset[validation] = predicted
        fold_metrics = metrics(y[validation], decode(predicted), onset[validation], predicted)
        row = {
            "held_out_subject": held_out,
            "training_sequences": int(training.sum()),
            "validation_sequences": int(validation.sum()),
            **fold_metrics,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    oof_metrics = metrics(y, decode(oof_onset), onset, oof_onset)
    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(reports / "development_subject_loso.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        reports / "development_oof_predictions.npz",
        truth=y, true_onset=onset, predicted_onset=oof_onset, subjects=subjects,
    )

    full_mean, full_std = normalization(x)
    final_model = train_model(x, onset, full_mean, full_std, SEED + 100)
    artifact_path = artifacts / "landing_onset_transfer_v1.pt"
    torch.save({
        "experiment": "landing_onset_transfer_v1",
        "architecture": "LAAS-pretrained 4-block residual onset TCN fine-tuned on Twente development video",
        "model_state_dict": final_model.state_dict(),
        "input_features": int(x.shape[2]),
        "feature_names": bundle["feature_names"].astype(str).tolist(),
        "normalization_mean": torch.from_numpy(full_mean),
        "normalization_std": torch.from_numpy(full_std),
        "sequence_length": int(x.shape[1]),
        "sequence_duration_s": 0.48,
        "precontact_context_s": 0.19,
        "landing_window_s": 0.030,
        "landing_frames": 4,
        "development_subjects": SUBJECTS,
        "forbidden_test_subjects": sorted(FORBIDDEN),
        "inference_contract": "One accepted running stride candidate in; one visual contact onset out.",
    }, artifact_path)
    manifest = {
        "status": "VIDEO_DOMAIN_TRANSFER_MODEL_FROZEN_DEVELOPMENT_ONLY",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "initialization": str(PRETRAINED_PATH),
        "training_sequences": int(len(x)),
        "development_subjects": SUBJECTS,
        "forbidden_test_subjects": sorted(FORBIDDEN),
        "forbidden_test_subjects_accessed": False,
        "baseline_detected_start_metrics": baseline_metrics,
        "development_subject_disjoint_oof_metrics": oof_metrics,
        "mean_fold_metrics": {
            column: float(fold_table[column].mean())
            for column in (
                "landing_f1", "landing_precision", "landing_recall",
                "onset_mae_ms", "within_one_frame", "within_two_frames",
            )
        },
        "user_video_policy": {
            "force_plate_required_for_inference": False,
            "true_f1_available_without_ground_truth": False,
            "output": "predicted visual contact onset, 30 ms landing window, and model confidence",
        },
        "limitations": [
            "Development F1 uses public force-synchronized data and is not a user-video F1.",
            "A video-only user cannot obtain true F1 without manual or sensor ground truth.",
            "High-frame-rate side video is recommended for a 30 ms landing definition.",
        ],
    }
    (reports / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
