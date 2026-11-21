from __future__ import annotations

import hashlib
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
EXPERIMENT = ROOT / "experiments/landing_aux_v1"
DATA_PATH = EXPERIMENT / "artifacts/laas_landing_sequences_64.npz"
SUBJECTS = ["PKFC", "PKLP", "PKLT", "PKMT", "PKRY"]
FORBIDDEN_TWENTE_SUBJECTS = {"Subj04", "Subj11"}
SEED = 20261109
EPOCHS = 80


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.12),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class LandingAuxTCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 24):
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, channels, 1)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
        )
        self.classifier = nn.Conv1d(channels, 2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        values = self.temporal(self.input_projection(values))
        return self.classifier(values).transpose(1, 2)


def corrupt(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    coordinates = torch.tensor([i for i in range(output.shape[2]) if i not in (10, 11, 22, 23)])
    output[:, :, coordinates] += torch.randn_like(output[:, :, coordinates]) * 0.018
    for sample in range(output.shape[0]):
        if float(torch.rand(())) < 0.50:
            block = int(torch.randint(2, 7, ()).item())
            start = int(torch.randint(1, output.shape[1] - block - 1, ()).item())
            end = start + block
            output[sample, start:end, :12] = output[sample, start - 1, :12]
            output[sample, start:end, 10:12] *= 0.55
    return output


def normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=(0, 1), keepdims=True)
    std = np.clip(values.std(axis=(0, 1), keepdims=True), 1e-4, None)
    return mean.astype(np.float32), std.astype(np.float32)


def train_model(x: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray, seed: int) -> LandingAuxTCN:
    seed_everything(seed)
    model = LandingAuxTCN(x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-3)
    counts = np.asarray([(y == label).sum() for label in (0, 1)], dtype=float)
    weights = np.sqrt(counts.sum() / np.clip(counts, 1.0, None))
    weights /= weights.mean()
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32), label_smoothing=0.01)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for raw, targets in loader:
            augmented = corrupt(raw)
            features = torch.cat((raw, augmented), dim=0)
            combined_targets = torch.cat((targets, targets), dim=0)
            features = (features - mean_tensor) / std_tensor
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits.reshape(-1, 2), combined_targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def probabilities(model: LandingAuxTCN, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = (x - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(normalized, dtype=torch.float32))
        return torch.softmax(logits, dim=-1)[..., 1].numpy()


def frame_metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "landing_f1": float(f1_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
        "landing_precision": float(precision_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
        "landing_recall": float(recall_score(truth.reshape(-1), prediction.reshape(-1), zero_division=0)),
    }


def onset_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    kinds: np.ndarray,
    relative_times: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    landing_mask = kinds == "landing"
    negative_mask = kinds == "negative"
    hits = np.max(probability[landing_mask], axis=1) >= threshold
    false_alarms = np.max(probability[negative_mask], axis=1) >= threshold if negative_mask.any() else np.asarray([])
    peak_indices = np.argmax(probability[landing_mask], axis=1)
    peak_times = relative_times[landing_mask][np.arange(landing_mask.sum()), peak_indices]
    return {
        "event_recall": float(hits.mean()) if hits.size else 0.0,
        "negative_sequence_false_alarm_rate": float(false_alarms.mean()) if false_alarms.size else 0.0,
        "peak_onset_mae_ms": float(np.mean(np.abs(peak_times)) * 1000.0) if peak_times.size else float("nan"),
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
    subjects = bundle["subjects"].astype(str)
    kinds = bundle["kinds"].astype(str)
    relative_times = bundle["relative_times"].astype(np.float32)
    if set(subjects) != set(SUBJECTS):
        raise RuntimeError(f"Unexpected subject set: {sorted(set(subjects))}")

    rows: list[dict] = []
    oof_probability = np.zeros_like(y, dtype=np.float32)
    for fold, held_out in enumerate(SUBJECTS):
        training = subjects != held_out
        validation = subjects == held_out
        mean, std = normalization(x[training])
        model = train_model(x[training], y[training], mean, std, SEED + fold)
        probability = probabilities(model, x[validation], mean, std)
        oof_probability[validation] = probability
        result = {
            "held_out_subject": held_out,
            "training_sequences": int(training.sum()),
            "validation_sequences": int(validation.sum()),
            **frame_metrics(y[validation], probability, 0.5),
            **onset_metrics(
                y[validation], probability, kinds[validation], relative_times[validation], 0.5
            ),
        }
        rows.append(result)
        print(json.dumps(result), flush=True)

    thresholds = np.linspace(0.20, 0.80, 121)
    threshold_scores = np.asarray([
        frame_metrics(y, oof_probability, float(threshold))["landing_f1"]
        for threshold in thresholds
    ])
    selected_threshold = float(thresholds[int(np.argmax(threshold_scores))])
    fixed_metrics = frame_metrics(y, oof_probability, 0.5)
    selected_metrics = frame_metrics(y, oof_probability, selected_threshold)
    selected_event_metrics = onset_metrics(
        y, oof_probability, kinds, relative_times, selected_threshold
    )

    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(reports / "laas_subject_loso.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        reports / "laas_oof_predictions.npz",
        truth=y,
        probability=oof_probability,
        subjects=subjects,
        kinds=kinds,
        relative_times=relative_times,
    )

    full_mean, full_std = normalization(x)
    final_model = train_model(x, y, full_mean, full_std, SEED + 100)
    artifact_path = artifacts / "landing_aux_laas_v1.pt"
    torch.save(
        {
            "experiment": "landing_aux_v1",
            "architecture": "3-block residual TCN, 24 channels, binary frame classifier",
            "model_state_dict": final_model.state_dict(),
            "input_features": int(x.shape[2]),
            "feature_names": bundle["feature_names"].astype(str).tolist(),
            "normalization_mean": torch.from_numpy(full_mean),
            "normalization_std": torch.from_numpy(full_std),
            "decision_threshold": selected_threshold,
            "development_subjects": SUBJECTS,
            "forbidden_twente_test_subjects": sorted(FORBIDDEN_TWENTE_SUBJECTS),
            "sequence_length": int(x.shape[1]),
            "source": "LAAS Parkour Dataset sv+kv force-labeled contacts",
            "intended_use": "auxiliary landing gate; requires running-domain integration and validation",
        },
        artifact_path,
    )
    manifest = {
        "status": "AUXILIARY_MODEL_TRAINED_NOT_FINAL_RUNNING_MODEL",
        "experiment": "landing_aux_v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "epochs_fixed": EPOCHS,
        "development_subjects": SUBJECTS,
        "sequences": int(len(x)),
        "landing_sequences": int((kinds == "landing").sum()),
        "negative_sequences": int((kinds == "negative").sum()),
        "locked_twente_test_subjects": sorted(FORBIDDEN_TWENTE_SUBJECTS),
        "locked_twente_test_subjects_accessed": False,
        "fixed_threshold_0_5_oof": fixed_metrics,
        "development_selected_threshold": selected_threshold,
        "selected_threshold_oof": {**selected_metrics, **selected_event_metrics},
        "mean_fold_metrics_at_0_5": {
            name: float(fold_table[name].mean())
            for name in (
                "landing_f1", "landing_precision", "landing_recall",
                "event_recall", "negative_sequence_false_alarm_rate", "peak_onset_mae_ms",
            )
        },
        "limitations": [
            "LAAS contains parkour landings, not treadmill running strides.",
            "The five-subject LOSO metrics measure auxiliary-domain generalization only.",
            "The OOF-selected threshold is a development setting and is not an untouched-test result.",
            "The model must be integrated as a landing gate and tested once on the locked running set.",
        ],
    }
    (reports / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
