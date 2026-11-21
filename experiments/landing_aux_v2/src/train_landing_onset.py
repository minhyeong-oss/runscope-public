from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_aux_v2"
DATA_PATH = ROOT / "experiments/landing_aux_v1/artifacts/laas_landing_sequences_64.npz"
SUBJECTS = ["PKFC", "PKLP", "PKLT", "PKMT", "PKRY"]
FORBIDDEN_TWENTE_SUBJECTS = {"Subj04", "Subj11"}
SEED = 20261109
EPOCHS = 100
LANDING_FRAMES = 4


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
            nn.GroupNorm(4, channels), nn.GELU(), nn.Dropout(0.10),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels), nn.GELU(), nn.Dropout(0.10),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class LandingOnsetTCN(nn.Module):
    def __init__(self, input_features: int, channels: int = 32):
        super().__init__()
        self.input_projection = nn.Conv1d(input_features, channels, 1)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(channels, 1),
            ResidualTemporalBlock(channels, 2),
            ResidualTemporalBlock(channels, 4),
            ResidualTemporalBlock(channels, 8),
        )
        self.onset_head = nn.Conv1d(channels, 1, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        hidden = self.temporal(self.input_projection(values))
        return self.onset_head(hidden).squeeze(1)


def normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=(0, 1), keepdims=True)
    std = np.clip(values.std(axis=(0, 1), keepdims=True), 1e-4, None)
    return mean.astype(np.float32), std.astype(np.float32)


def corrupt(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    coordinates = torch.tensor([i for i in range(output.shape[2]) if i not in (10, 11, 22, 23)])
    output[:, :, coordinates] += torch.randn_like(output[:, :, coordinates]) * 0.012
    if output.shape[0] > 1:
        scales = torch.empty((output.shape[0], 1, 1)).uniform_(0.96, 1.04)
        output[:, :, coordinates] *= scales
    return output


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    kinds: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seed: int,
) -> LandingOnsetTCN:
    seed_everything(seed)
    model = LandingOnsetTCN(x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=3e-3)
    presence = (kinds == "landing").astype(np.float32)
    onset_indices = np.asarray([
        int(np.flatnonzero(row)[0]) if row.any() else -1 for row in y
    ], dtype=np.int64)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(presence, dtype=torch.float32),
        torch.tensor(onset_indices, dtype=torch.long),
    )
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    presence_loss = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(EPOCHS):
        for raw, present, onset in loader:
            features = corrupt(raw)
            logits = model((features - mean_tensor) / std_tensor)
            global_logit = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
            loss = presence_loss(global_logit, present)
            positive = onset >= 0
            if positive.any():
                loss = loss + 1.5 * nn.functional.cross_entropy(logits[positive], onset[positive])
                # A small local smoothness target makes adjacent video frames robust
                # without redefining the 30 ms evaluation label.
                target = torch.zeros_like(logits[positive])
                positive_onsets = onset[positive]
                target[torch.arange(len(positive_onsets)), positive_onsets] = 1.0
                for delta, weight in ((-1, 0.35), (1, 0.35)):
                    indices = positive_onsets + delta
                    valid = (indices >= 0) & (indices < logits.shape[1])
                    target[torch.arange(len(indices))[valid], indices[valid]] = weight
                loss = loss + 0.25 * nn.functional.binary_cross_entropy_with_logits(
                    logits[positive], target
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def raw_outputs(
    model: LandingOnsetTCN,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(torch.tensor((x - mean) / std, dtype=torch.float32))
        global_logit = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
        return torch.sigmoid(global_logit).numpy(), logits.numpy()


def decode(presence: np.ndarray, logits: np.ndarray, threshold: float) -> np.ndarray:
    output = np.zeros_like(logits, dtype=np.int64)
    for index in range(len(output)):
        if presence[index] < threshold:
            continue
        onset = int(np.argmax(logits[index]))
        output[index, onset:min(output.shape[1], onset + LANDING_FRAMES)] = 1
    return output


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    presence: np.ndarray,
    logits: np.ndarray,
    kinds: np.ndarray,
    relative_times: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    truth_flat, prediction_flat = truth.reshape(-1), prediction.reshape(-1)
    landing = kinds == "landing"
    negative = kinds == "negative"
    peak_indices = np.argmax(logits[landing], axis=1)
    peak_times = relative_times[landing][np.arange(landing.sum()), peak_indices]
    return {
        "landing_f1": float(f1_score(truth_flat, prediction_flat, zero_division=0)),
        "landing_precision": float(precision_score(truth_flat, prediction_flat, zero_division=0)),
        "landing_recall": float(recall_score(truth_flat, prediction_flat, zero_division=0)),
        "event_recall": float((presence[landing] >= threshold).mean()) if landing.any() else 0.0,
        "negative_sequence_false_alarm_rate": float((presence[negative] >= threshold).mean()) if negative.any() else 0.0,
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
        raise RuntimeError("Unexpected auxiliary subject set")

    oof_presence = np.zeros(len(x), dtype=np.float32)
    oof_logits = np.zeros_like(y, dtype=np.float32)
    fold_cache: list[dict] = []
    for fold, held_out in enumerate(SUBJECTS):
        training, validation = subjects != held_out, subjects == held_out
        mean, std = normalization(x[training])
        model = train_model(x[training], y[training], kinds[training], mean, std, SEED + fold)
        presence, logits = raw_outputs(model, x[validation], mean, std)
        oof_presence[validation], oof_logits[validation] = presence, logits
        fold_cache.append({
            "subject": held_out,
            "training": int(training.sum()),
            "validation": int(validation.sum()),
        })
        print(json.dumps({
            "held_out_subject": held_out,
            "training_sequences": int(training.sum()),
            "validation_sequences": int(validation.sum()),
            "presence_mean_landing": float(presence[kinds[validation] == "landing"].mean()),
            "presence_mean_negative": float(presence[kinds[validation] == "negative"].mean()),
        }), flush=True)

    thresholds = np.linspace(0.05, 0.95, 181)
    scored = []
    for threshold in thresholds:
        prediction = decode(oof_presence, oof_logits, float(threshold))
        scored.append(metrics(
            y, prediction, oof_presence, oof_logits, kinds, relative_times, float(threshold)
        ))
    selected_index = int(np.argmax([row["landing_f1"] for row in scored]))
    selected_threshold = float(thresholds[selected_index])
    selected_metrics = scored[selected_index]

    rows = []
    for item in fold_cache:
        mask = subjects == item["subject"]
        prediction = decode(oof_presence[mask], oof_logits[mask], selected_threshold)
        rows.append({
            "held_out_subject": item["subject"],
            "training_sequences": item["training"],
            "validation_sequences": item["validation"],
            **metrics(
                y[mask], prediction, oof_presence[mask], oof_logits[mask],
                kinds[mask], relative_times[mask], selected_threshold,
            ),
        })

    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(reports / "laas_subject_loso.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        reports / "laas_oof_predictions.npz",
        truth=y, presence=oof_presence, logits=oof_logits,
        subjects=subjects, kinds=kinds, relative_times=relative_times,
    )

    full_mean, full_std = normalization(x)
    final_model = train_model(x, y, kinds, full_mean, full_std, SEED + 100)
    artifact_path = artifacts / "landing_onset_laas_v2.pt"
    torch.save({
        "experiment": "landing_aux_v2",
        "architecture": "4-block residual TCN onset localizer, 32 channels",
        "model_state_dict": final_model.state_dict(),
        "input_features": int(x.shape[2]),
        "feature_names": bundle["feature_names"].astype(str).tolist(),
        "normalization_mean": torch.from_numpy(full_mean),
        "normalization_std": torch.from_numpy(full_std),
        "presence_threshold": selected_threshold,
        "landing_frames": LANDING_FRAMES,
        "development_subjects": SUBJECTS,
        "forbidden_twente_test_subjects": sorted(FORBIDDEN_TWENTE_SUBJECTS),
        "source": "LAAS Parkour Dataset sv+kv force-labeled contacts",
        "intended_use": "auxiliary landing-onset gate; running-domain validation required",
    }, artifact_path)
    manifest = {
        "status": "AUXILIARY_ONSET_MODEL_TRAINED_NOT_FINAL_RUNNING_MODEL",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "epochs_fixed": EPOCHS,
        "development_subjects": SUBJECTS,
        "sequences": int(len(x)),
        "landing_sequences": int((kinds == "landing").sum()),
        "negative_sequences": int((kinds == "negative").sum()),
        "development_selected_threshold": selected_threshold,
        "oof_selected_threshold_metrics": selected_metrics,
        "mean_subject_loso_metrics": {
            column: float(fold_table[column].mean())
            for column in (
                "landing_f1", "landing_precision", "landing_recall", "event_recall",
                "negative_sequence_false_alarm_rate", "peak_onset_mae_ms",
            )
        },
        "locked_twente_test_subjects": sorted(FORBIDDEN_TWENTE_SUBJECTS),
        "locked_twente_test_subjects_accessed": False,
        "limitations": [
            "LAAS contains parkour landings rather than treadmill running.",
            "Threshold selection uses development OOF predictions and is not a final test result.",
            "A separate integration experiment must test transfer to video-derived running pose.",
        ],
    }
    (reports / "MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
