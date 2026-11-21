from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_event_v3"
DATA = EXPERIMENT / "artifacts/landing_event_development_v3.npz"
VISUAL_MODEL = ROOT / "experiments/landing_visual_contact_v2/artifacts/landing_visual_contact_v2.pt"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}
SEED = 20261115
EPOCHS = 35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


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
        hidden = self.temporal(self.input_projection(values.transpose(1, 2)))
        return self.head(hidden).squeeze(1)


def new_tcn(input_features: int, variant: str, seed: int) -> LandingEventTCN:
    seed_everything(seed)
    model = LandingEventTCN(input_features)
    if variant == "visual_temporal_init_weighted":
        visual = torch.load(VISUAL_MODEL, map_location="cpu", weights_only=False)
        compatible = {
            name: value
            for name, value in visual["model_state_dict"].items()
            if (name.startswith("temporal.") or name.startswith("head."))
            and name in model.state_dict()
            and model.state_dict()[name].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if unexpected or not compatible or not any(name.startswith("temporal.") for name in compatible):
            raise RuntimeError("Visual temporal initialization failed")
        if not all(name.startswith("input_projection.") for name in missing):
            raise RuntimeError(f"Unexpected missing pretrained layers: {missing}")
    return model


def normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.clip(x.std(axis=(0, 1), keepdims=True), 1e-4, None).astype(np.float32)
    return mean, std


def augment(values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    coordinates = [index for index in range(output.shape[2]) if index not in (10, 11, 22, 23)]
    output[:, :, coordinates] += torch.randn_like(output[:, :, coordinates]) * 0.008
    visibility = [10, 11, 22, 23]
    scale = torch.empty((output.shape[0], 1, 1)).uniform_(0.82, 1.0)
    output[:, :, visibility] = torch.clamp(output[:, :, visibility] * scale, 0.0, 1.0)
    return output


def train_tcn(
    x: np.ndarray,
    hard_onset: np.ndarray,
    soft_target: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    variant: str,
    seed: int,
) -> LandingEventTCN:
    model = new_tcn(x.shape[2], variant, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=4e-3)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(hard_onset, dtype=torch.long),
        torch.tensor(soft_target, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for raw, onset, soft, weight in loader:
            logits = model((augment(raw) - mean_tensor) / std_tensor)
            log_probability = nn.functional.log_softmax(logits, dim=1)
            soft_loss = -(soft * log_probability).sum(dim=1)
            hard_loss = nn.functional.cross_entropy(logits, onset, reduction="none")
            loss = ((soft_loss + 0.35 * hard_loss) * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def predict_tcn(model: nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor((x - mean) / std, dtype=torch.float32))
        return torch.argmax(logits, dim=1).numpy().astype(int)


def decode_window(onset: np.ndarray, length: int = 64, window_samples: int = 4) -> np.ndarray:
    result = np.zeros((len(onset), length), dtype=np.uint8)
    for row, index in enumerate(onset):
        result[row, index:min(length, index + window_samples + 1)] = 1
    return result


def metrics(truth: np.ndarray, prediction: np.ndarray, sample_interval_s: float) -> dict:
    error = np.abs(prediction - truth)
    return {
        "event_f1_15ms": float(np.mean(error * sample_interval_s <= 0.015)),
        "event_f1_30ms": float(np.mean(error * sample_interval_s <= 0.030)),
        "event_f1_60ms": float(np.mean(error * sample_interval_s <= 0.060)),
        "onset_mae_ms": float(np.mean(error) * sample_interval_s * 1000.0),
        "landing_window_f1": float(
            f1_score(
                decode_window(truth).reshape(-1),
                decode_window(prediction).reshape(-1),
                zero_division=0,
            )
        ),
    }


def main() -> None:
    torch.set_num_threads(4)
    bundle = np.load(DATA)
    x = bundle["x"].astype(np.float32)
    onset = bundle["onset_indices"].astype(int)
    soft = bundle["soft_event_target"].astype(np.float32)
    subjects = bundle["subjects"].astype(str)
    tiers = bundle["quality_tier"].astype(str)
    sample_interval_s = float(bundle["sample_interval_s"])
    if set(subjects) != set(SUBJECTS) or set(subjects) & FORBIDDEN:
        raise RuntimeError("Development split violation")
    quality_weights = np.asarray(
        [{"high": 1.0, "medium": 0.55, "low": 0.25}[tier] for tier in tiers],
        dtype=np.float32,
    )

    candidates = [
        "ridge_alpha100",
        "ridge_alpha1000",
        "scratch_weighted",
        "visual_temporal_init_weighted",
    ]
    oof = {name: np.zeros(len(x), dtype=int) for name in candidates}
    fold_rows = []
    for fold, held_out in enumerate(SUBJECTS):
        training = subjects != held_out
        validation = subjects == held_out
        x_flat_train = x[training, ::4].reshape(int(training.sum()), -1)
        x_flat_validation = x[validation, ::4].reshape(int(validation.sum()), -1)
        for alpha in (100.0, 1000.0):
            name = f"ridge_alpha{int(alpha)}"
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x_flat_train, onset[training])
            oof[name][validation] = np.clip(
                np.rint(model.predict(x_flat_validation)), 0, x.shape[1] - 1
            ).astype(int)
        mean, std = normalization(x[training])
        for candidate_index, name in enumerate(("scratch_weighted", "visual_temporal_init_weighted")):
            model = train_tcn(
                x[training], onset[training], soft[training], quality_weights[training],
                mean, std, name, SEED + fold * 10 + candidate_index,
            )
            oof[name][validation] = predict_tcn(model, x[validation], mean, std)
        for name in candidates:
            fold_rows.append(
                {
                    "held_out_subject": held_out,
                    "candidate": name,
                    "training_sequences": int(training.sum()),
                    "validation_sequences": int(validation.sum()),
                    **metrics(onset[validation], oof[name][validation], sample_interval_s),
                }
            )
        print(json.dumps({"completed_fold": held_out}, ensure_ascii=False), flush=True)

    summary_rows = []
    for name in candidates:
        summary_rows.append(
            {"candidate": name, **metrics(onset, oof[name], sample_interval_s)}
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["event_f1_60ms", "event_f1_30ms", "onset_mae_ms"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    selected = str(summary.iloc[0]["candidate"])

    artifacts = EXPERIMENT / "artifacts"
    reports = EXPERIMENT / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(
        reports / "development_candidate_loso.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(reports / "development_candidate_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        reports / "development_oof_predictions_v3.npz",
        truth_onset=onset,
        subjects=subjects,
        **{name: prediction for name, prediction in oof.items()},
    )

    artifact_paths = {}
    x_flat = x[:, ::4].reshape(len(x), -1)
    for alpha in (100.0, 1000.0):
        name = f"ridge_alpha{int(alpha)}"
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_flat, onset)
        path = artifacts / f"{name}.joblib"
        joblib.dump(
            {
                "model_type": "ridge_flattened_every_4_samples",
                "model": model,
                "sequence_length": int(x.shape[1]),
                "input_features": int(x.shape[2]),
                "feature_names": bundle["feature_names"].astype(str).tolist(),
                "sample_interval_s": sample_interval_s,
                "development_subjects": SUBJECTS,
                "forbidden_subjects": sorted(FORBIDDEN),
            },
            path,
        )
        artifact_paths[name] = path
    full_mean, full_std = normalization(x)
    for candidate_index, name in enumerate(("scratch_weighted", "visual_temporal_init_weighted")):
        model = train_tcn(
            x, onset, soft, quality_weights, full_mean, full_std,
            name, SEED + 100 + candidate_index,
        )
        path = artifacts / f"{name}.pt"
        torch.save(
            {
                "model_type": "landing_event_tcn",
                "variant": name,
                "model_state_dict": model.state_dict(),
                "normalization_mean": torch.from_numpy(full_mean),
                "normalization_std": torch.from_numpy(full_std),
                "sequence_length": int(x.shape[1]),
                "input_features": int(x.shape[2]),
                "feature_names": bundle["feature_names"].astype(str).tolist(),
                "sample_interval_s": sample_interval_s,
                "development_subjects": SUBJECTS,
                "forbidden_subjects": sorted(FORBIDDEN),
            },
            path,
        )
        artifact_paths[name] = path

    selected_path = artifact_paths[selected]
    selection = {
        "status": "LANDING_EVENT_V3_FROZEN_BEFORE_HISTORICAL_TEST",
        "selection_rule": "highest development subject-LOSO F1 at 60 ms, then 30 ms, then lowest MAE",
        "selected_candidate": selected,
        "selected_artifact": str(selected_path),
        "selected_artifact_sha256": sha256(selected_path),
        "selected_development_metrics": summary.iloc[0].to_dict(),
        "all_candidate_metrics": summary.to_dict(orient="records"),
        "development_subjects": SUBJECTS,
        "forbidden_historical_test_subjects": sorted(FORBIDDEN),
        "historical_test_accessed_during_training": False,
        "quality_weights": {"high": 1.0, "medium": 0.55, "low": 0.25},
        "warning": "Subj04/Subj11 can be used only as a historical reference, not a new untouched benchmark.",
    }
    (artifacts / "FROZEN_SELECTION.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
