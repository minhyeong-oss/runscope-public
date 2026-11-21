from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/temporal_tcn_v1"
DATA_PATH = EXPERIMENT / "artifacts/development_sequences_64.npz"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}
SEED = 20261107
EPOCHS = 45


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.10):
        super().__init__()
        padding = dilation
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class RunScopeTCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 48):
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, channels, 1)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
            ResidualTemporalBlock(channels, 8),
        )
        self.classifier = nn.Conv1d(channels, 4, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        return self.classifier(self.temporal(self.input_projection(values))).transpose(1, 2)


def normalize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.clip(std, 1e-4, None)
    return (train - mean) / std, (other - mean) / std, mean, std


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.asarray([(labels == stage).sum() for stage in (1, 2, 3, 4)], dtype=float)
    weights = np.sqrt(counts.sum() / np.clip(counts, 1, None))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_model(x: np.ndarray, y: np.ndarray, seed: int) -> RunScopeTCN:
    seed_everything(seed)
    model = RunScopeTCN(x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    loss_function = nn.CrossEntropyLoss(weight=class_weights(y), label_smoothing=0.02)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y - 1, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    model.train()
    for _ in range(EPOCHS):
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits.reshape(-1, 4), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def monotonic_decode(logits: np.ndarray) -> np.ndarray:
    # Hidden states are previous recovery, landing, braking, propulsion, recovery.
    log_prob = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    emissions = np.column_stack([log_prob[:, 3], log_prob[:, 0], log_prob[:, 1], log_prob[:, 2], log_prob[:, 3]])
    length, states = emissions.shape
    score = np.full((length, states), -np.inf)
    back = np.zeros((length, states), dtype=np.int8)
    score[0, 0] = emissions[0, 0]
    for time in range(1, length):
        for state in range(states):
            stay = score[time - 1, state]
            advance = score[time - 1, state - 1] if state > 0 else -np.inf
            if advance > stay:
                score[time, state] = advance + emissions[time, state]
                back[time, state] = state - 1
            else:
                score[time, state] = stay + emissions[time, state]
                back[time, state] = state
    final_state = 4
    path = np.zeros(length, dtype=np.int8)
    path[-1] = final_state
    for time in range(length - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    return np.asarray([4, 1, 2, 3, 4], dtype=np.int8)[path]


def predict(model: RunScopeTCN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return np.stack([monotonic_decode(sequence) for sequence in logits])


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth_flat, prediction_flat = truth.reshape(-1), prediction.reshape(-1)
    return {
        "macro_f1": float(f1_score(truth_flat, prediction_flat, labels=[1, 2, 3, 4], average="macro")),
        "landing": float(f1_score(truth_flat, prediction_flat, labels=[1], average="macro")),
        "braking": float(f1_score(truth_flat, prediction_flat, labels=[2], average="macro")),
        "propulsion": float(f1_score(truth_flat, prediction_flat, labels=[3], average="macro")),
        "recovery": float(f1_score(truth_flat, prediction_flat, labels=[4], average="macro")),
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
    x, y, subjects = bundle["x"], bundle["y"], bundle["subjects"].astype(str)
    if set(subjects) & FORBIDDEN or set(subjects) != set(SUBJECTS):
        raise RuntimeError("Invalid development subject set")
    rows = []
    for fold, held_out in enumerate(SUBJECTS):
        train_mask, validation_mask = subjects != held_out, subjects == held_out
        train_x, validation_x, mean, std = normalize(x[train_mask], x[validation_mask])
        model = train_model(train_x, y[train_mask], SEED + fold)
        prediction = predict(model, validation_x)
        result = metrics(y[validation_mask], prediction)
        rows.append({
            "held_out_subject": held_out, "training_sequences": int(train_mask.sum()),
            "validation_sequences": int(validation_mask.sum()), **result,
        })
        print(json.dumps(rows[-1]), flush=True)
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(rows)
    cv_path = reports / "development_loso_cv.csv"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")
    summary = {
        "macro_f1_mean": float(cv["macro_f1"].mean()),
        "macro_f1_std": float(cv["macro_f1"].std()),
        "macro_f1_min": float(cv["macro_f1"].min()),
        "per_stage_f1_mean": {stage: float(cv[stage].mean()) for stage in ("landing", "braking", "propulsion", "recovery")},
    }
    full_x, _, mean, std = normalize(x, x)
    final_model = train_model(full_x, y, SEED + 100)
    artifact_path = EXPERIMENT / "artifacts/temporal_tcn_v1.pt"
    torch.save({
        "experiment": "temporal_tcn_v1", "model_state_dict": final_model.state_dict(),
        "input_features": int(x.shape[2]), "feature_names": bundle["feature_names"].astype(str).tolist(),
        "normalization_mean": torch.from_numpy(mean.astype(np.float32)),
        "normalization_std": torch.from_numpy(std.astype(np.float32)),
        "development_subjects": SUBJECTS, "forbidden_subjects": sorted(FORBIDDEN),
        "sequence_length": int(x.shape[1]), "pre_contact_window_s": 0.12,
    }, artifact_path)
    manifest = {
        "status": "DEVELOPMENT_ONLY_MODEL_FROZEN", "experiment": "temporal_tcn_v1",
        "architecture": "4-block dilated residual TCN, 48 channels",
        "training_epochs_fixed": EPOCHS, "decoder": "monotonic 4-1-2-3-4 Viterbi",
        "artifact": str(artifact_path), "artifact_sha256": sha256(artifact_path),
        "development_subjects": SUBJECTS, "forbidden_subjects": sorted(FORBIDDEN),
        "forbidden_subjects_accessed": False, "development_sequences": int(len(x)),
        "development_loso": summary,
        "limitations": [
            "Development-only LOSO result; it is not a new untouched-test result.",
            "Labels inherit video-force synchronization and contact matching quality.",
            "Subj04 and Subj11 are deliberately excluded."
        ],
    }
    (reports / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
