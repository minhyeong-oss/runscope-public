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
EXPERIMENT = ROOT / "experiments/temporal_tcn_robust_v2"
SOURCE_EXPERIMENT = ROOT / "experiments/temporal_tcn_v1"
DATA_PATH = SOURCE_EXPERIMENT / "artifacts/development_sequences_64.npz"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}
SEED = 20261108
EPOCHS = 45


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("robust_v2_base_tcn", SOURCE_EXPERIMENT / "src/train_tcn.py")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corruption(
    values: torch.Tensor,
    coordinate_indices: list[int],
    visibility_indices: list[int],
    *,
    strong: bool,
) -> torch.Tensor:
    """Simulate low-light/occlusion landmark degradation without changing labels."""
    output = values.clone()
    batch, length, _ = output.shape
    coord = torch.tensor(coordinate_indices, dtype=torch.long)
    vis = torch.tensor(visibility_indices, dtype=torch.long)
    noise_max = 0.045 if strong else 0.030
    noise_min = 0.020 if strong else 0.004
    noise_scale = torch.empty((batch, 1, 1)).uniform_(noise_min, noise_max)
    output[:, :, coord] += torch.randn_like(output[:, :, coord]) * noise_scale

    visibility_low, visibility_high = ((0.45, 0.72) if strong else (0.60, 1.00))
    visibility_scale = torch.empty((batch, 1, 1)).uniform_(visibility_low, visibility_high)
    output[:, :, vis] = torch.clamp(output[:, :, vis] * visibility_scale, 0.0, 1.0)

    foot_indices = torch.tensor([0, 1, 2, 3, 4, 5, 10, 11], dtype=torch.long)
    blocks = 2 if strong else 1
    probability = 1.0 if strong else 0.55
    for sample in range(batch):
        if float(torch.rand(())) > probability:
            continue
        for _ in range(blocks):
            block_length = int(torch.randint(5 if strong else 3, 13 if strong else 10, ()).item())
            start = int(torch.randint(1, max(2, length - block_length - 1), ()).item())
            end = min(length - 1, start + block_length)
            before = output[sample, start - 1, foot_indices]
            after = output[sample, end, foot_indices]
            fractions = torch.linspace(0.0, 1.0, end - start + 1).unsqueeze(1)
            output[sample, start : end + 1, foot_indices] = (
                before.unsqueeze(0) * (1.0 - fractions) + after.unsqueeze(0) * fractions
            )
            output[sample, start : end + 1, vis[:2]] *= 0.45

    # Random frame loss represents lower capture rates; neighboring samples fill gaps.
    drop_probability = 0.12 if strong else 0.05
    mask = torch.rand((batch, length - 2)) < drop_probability
    for sample in range(batch):
        indices = torch.nonzero(mask[sample], as_tuple=False).flatten() + 1
        if len(indices):
            output[sample, indices] = (output[sample, indices - 1] + output[sample, indices + 1]) / 2.0
    return output


def train_robust(
    raw_x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    coordinate_indices: list[int],
    visibility_indices: list[int],
    seed: int,
) -> base.RunScopeTCN:
    seed_everything(seed)
    model = base.RunScopeTCN(raw_x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
    loss_function = nn.CrossEntropyLoss(weight=base.class_weights(y), label_smoothing=0.02)
    dataset = TensorDataset(torch.tensor(raw_x, dtype=torch.float32), torch.tensor(y - 1, dtype=torch.long))
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        for raw_features, targets in loader:
            augmented = corruption(
                raw_features, coordinate_indices, visibility_indices, strong=False
            )
            combined = torch.cat((raw_features, augmented), dim=0)
            combined_targets = torch.cat((targets, targets), dim=0)
            normalized = (combined - mean_tensor) / std_tensor
            optimizer.zero_grad(set_to_none=True)
            logits = model(normalized)
            loss = loss_function(logits.reshape(-1, 4), combined_targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def predict(model: base.RunScopeTCN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).numpy()
    return np.stack([base.monotonic_decode(sequence) for sequence in logits])


def corrupted_evaluation(
    raw_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    coordinate_indices: list[int],
    visibility_indices: list[int],
    seed: int,
) -> np.ndarray:
    seed_everything(seed)
    values = corruption(
        torch.tensor(raw_x, dtype=torch.float32),
        coordinate_indices,
        visibility_indices,
        strong=True,
    ).numpy()
    return (values - mean) / std


def main() -> None:
    torch.set_num_threads(4)
    bundle = np.load(DATA_PATH)
    x = bundle["x"].astype(np.float32)
    y = bundle["y"].astype(np.int64)
    subjects = bundle["subjects"].astype(str)
    feature_names = bundle["feature_names"].astype(str).tolist()
    if set(subjects) != set(SUBJECTS) or set(subjects) & FORBIDDEN:
        raise RuntimeError("Invalid development split or forbidden subject access")
    visibility_indices = [index for index, name in enumerate(feature_names) if "visibility" in name]
    coordinate_indices = [index for index in range(len(feature_names)) if index not in visibility_indices]

    rows = []
    for fold, held_out in enumerate(SUBJECTS):
        train_mask, validation_mask = subjects != held_out, subjects == held_out
        train_raw, validation_raw = x[train_mask], x[validation_mask]
        mean = train_raw.mean(axis=(0, 1), keepdims=True)
        std = np.clip(train_raw.std(axis=(0, 1), keepdims=True), 1e-4, None)
        train_clean = (train_raw - mean) / std
        validation_clean = (validation_raw - mean) / std
        validation_corrupted = corrupted_evaluation(
            validation_raw, mean, std, coordinate_indices, visibility_indices, SEED + 1000 + fold
        )

        baseline = base.train_model(train_clean, y[train_mask], base.SEED + fold)
        robust = train_robust(
            train_raw, y[train_mask], mean, std,
            coordinate_indices, visibility_indices, SEED + fold,
        )
        results = {
            "baseline_clean": base.metrics(y[validation_mask], predict(baseline, validation_clean)),
            "baseline_corrupted": base.metrics(y[validation_mask], predict(baseline, validation_corrupted)),
            "robust_clean": base.metrics(y[validation_mask], predict(robust, validation_clean)),
            "robust_corrupted": base.metrics(y[validation_mask], predict(robust, validation_corrupted)),
        }
        row = {"held_out_subject": held_out, "validation_sequences": int(validation_mask.sum())}
        for condition, metrics in results.items():
            for name, value in metrics.items():
                row[f"{condition}_{name}"] = value
        rows.append(row)
        print(json.dumps(row), flush=True)

    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(reports / "development_loso_robustness_comparison.csv", index=False, encoding="utf-8-sig")

    full_mean = x.mean(axis=(0, 1), keepdims=True)
    full_std = np.clip(x.std(axis=(0, 1), keepdims=True), 1e-4, None)
    final_model = train_robust(
        x, y, full_mean, full_std,
        coordinate_indices, visibility_indices, SEED + 100,
    )
    artifact_path = artifacts / "temporal_tcn_robust_v2.pt"
    torch.save(
        {
            "experiment": "temporal_tcn_robust_v2",
            "model_state_dict": final_model.state_dict(),
            "input_features": int(x.shape[2]),
            "feature_names": feature_names,
            "normalization_mean": torch.from_numpy(full_mean.astype(np.float32)),
            "normalization_std": torch.from_numpy(full_std.astype(np.float32)),
            "development_subjects": SUBJECTS,
            "forbidden_subjects": sorted(FORBIDDEN),
            "sequence_length": int(x.shape[1]),
            "pre_contact_window_s": 0.12,
            "augmentation": "landmark jitter + foot occlusion interpolation + visibility decay + frame loss",
        },
        artifact_path,
    )
    means = {
        column: float(comparison[column].mean())
        for column in comparison.columns
        if column.endswith("macro_f1")
    }
    summary = {
        "status": "DEVELOPMENT_ONLY_ROBUST_MODEL_FROZEN",
        "experiment": "temporal_tcn_robust_v2",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "epochs": EPOCHS,
        "development_subjects": SUBJECTS,
        "forbidden_subjects": sorted(FORBIDDEN),
        "forbidden_subjects_accessed": False,
        "development_loso_macro_f1_means": means,
        "robust_corrupted_gain_over_baseline": (
            means["robust_corrupted_macro_f1"] - means["baseline_corrupted_macro_f1"]
        ),
        "clean_change_from_baseline": means["robust_clean_macro_f1"] - means["baseline_clean_macro_f1"],
        "limitations": [
            "Corruption validation is synthetic and does not replace force-labeled external validation.",
            "The failed Vimeo video was not used as a label source or training example.",
            "Subj04 and Subj11 remain untouched for one-time final testing."
        ],
    }
    (reports / "MODEL_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
