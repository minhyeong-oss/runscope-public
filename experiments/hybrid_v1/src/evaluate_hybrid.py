from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/hybrid_v1"
BOUNDARY_DIR = ROOT / "experiments/boundary_regression_v1"
TCN_DIR = ROOT / "experiments/temporal_tcn_v1"
SUBJECTS = ["Subj05", "Subj06", "Subj07", "Subj08", "Subj09", "Subj10", "Subj12"]
FORBIDDEN = {"Subj04", "Subj11"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


br = load_module("runscope_boundary_train", BOUNDARY_DIR / "src/train_boundary_regression.py")
tcn = load_module("runscope_tcn_train", TCN_DIR / "src/train_tcn.py")


CONFIGS = {
    "ridge_only": np.asarray([0.0, 0.0, 0.0, 0.0]),
    "tcn_only": np.asarray([1.0, 1.0, 1.0, 1.0]),
    "equal": np.asarray([0.5, 0.5, 0.5, 0.5]),
    "tcn_landing_ridge_others": np.asarray([0.75, 0.25, 0.25, 0.25]),
    "tcn_landing_equal_others": np.asarray([0.75, 0.50, 0.50, 0.50]),
    "ridge_landing_tcn_others": np.asarray([0.25, 0.75, 0.75, 0.75]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def boundary_probabilities(table: pd.DataFrame, predictions: np.ndarray, sequence_length: int) -> np.ndarray:
    probabilities = []
    for row, boundary in zip(table.itertuples(index=False), br.constrain(predictions)):
        times = np.linspace(-0.12, float(row.detected_stride_duration_s), sequence_length)
        labels = np.where(
            times < boundary[0], 4,
            np.where(
                times < boundary[0] + 0.030, 1,
                np.where(times < boundary[1], 2, np.where(times < boundary[2], 3, 4)),
            ),
        ).astype(int)
        probability = np.full((sequence_length, 4), 0.025, dtype=np.float32)
        probability[np.arange(sequence_length), labels - 1] = 0.925
        probabilities.append(probability)
    return np.stack(probabilities)


def decoded_hybrid(tcn_probability: np.ndarray, boundary_probability: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mixed = tcn_probability * weights + boundary_probability * (1.0 - weights)
    mixed = mixed / np.clip(mixed.sum(axis=-1, keepdims=True), 1e-8, None)
    return np.stack([tcn.monotonic_decode(np.log(np.clip(item, 1e-8, 1.0))) for item in mixed])


def score(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth_flat, prediction_flat = truth.reshape(-1), prediction.reshape(-1)
    result = {
        "macro_f1": float(f1_score(truth_flat, prediction_flat, labels=[1, 2, 3, 4], average="macro"))
    }
    for label, stage in ((1, "landing"), (2, "braking"), (3, "propulsion"), (4, "recovery")):
        result[stage] = float(f1_score(truth_flat, prediction_flat, labels=[label], average="macro"))
    return result


def main() -> None:
    torch.set_num_threads(4)
    sequence_bundle = np.load(TCN_DIR / "artifacts/development_sequences_64.npz")
    x, y = sequence_bundle["x"], sequence_bundle["y"]
    subjects = sequence_bundle["subjects"].astype(str)
    source_rows = sequence_bundle["source_rows"].astype(int)
    matched = pd.read_csv(BOUNDARY_DIR / "artifacts/development_matched_strides.csv").iloc[source_rows].reset_index(drop=True)
    if set(subjects) != set(SUBJECTS) or set(subjects) & FORBIDDEN:
        raise RuntimeError("Invalid subject split")
    boundary_artifact = joblib.load(BOUNDARY_DIR / "artifacts/boundary_regression_v1.joblib")
    feature_names = list(boundary_artifact["feature_names"])
    tcn_oof = np.zeros((len(x), x.shape[1], 4), dtype=np.float32)
    boundary_oof = np.zeros_like(tcn_oof)
    for fold, held_out in enumerate(SUBJECTS):
        train_mask, validation_mask = subjects != held_out, subjects == held_out
        train_table, validation_table = matched[train_mask], matched[validation_mask]
        boundary_model = br.make_model("ridge")
        boundary_model.fit(train_table[feature_names].to_numpy(), train_table[br.TARGETS].to_numpy())
        boundary_prediction = boundary_model.predict(validation_table[feature_names].to_numpy())
        boundary_oof[validation_mask] = boundary_probabilities(validation_table, boundary_prediction, x.shape[1])
        train_x, validation_x, _, _ = tcn.normalize(x[train_mask], x[validation_mask])
        temporal_model = tcn.train_model(train_x, y[train_mask], tcn.SEED + fold)
        with torch.no_grad():
            logits = temporal_model(torch.tensor(validation_x, dtype=torch.float32)).numpy()
        tcn_oof[validation_mask] = softmax(logits)
        print(json.dumps({"base_oof_complete": held_out, "validation_sequences": int(validation_mask.sum())}), flush=True)

    rows = []
    for held_out in SUBJECTS:
        meta_train = subjects != held_out
        meta_validation = subjects == held_out
        selection_scores = {}
        for name, weights in CONFIGS.items():
            prediction = decoded_hybrid(tcn_oof[meta_train], boundary_oof[meta_train], weights)
            selection_scores[name] = score(y[meta_train], prediction)["macro_f1"]
        selected = max(selection_scores, key=selection_scores.get)
        validation_prediction = decoded_hybrid(
            tcn_oof[meta_validation], boundary_oof[meta_validation], CONFIGS[selected]
        )
        rows.append({
            "held_out_subject": held_out, "selected_meta_config": selected,
            "meta_training_macro_f1": selection_scores[selected],
            "validation_sequences": int(meta_validation.sum()),
            **score(y[meta_validation], validation_prediction),
        })
    reports = EXPERIMENT / "reports"
    artifacts = EXPERIMENT / "artifacts"
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame(rows)
    cv_path = reports / "development_meta_loso_cv.csv"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")

    component_scores = {}
    for name, weights in {"boundary_regression": CONFIGS["ridge_only"], "temporal_tcn": CONFIGS["tcn_only"]}.items():
        prediction = decoded_hybrid(tcn_oof, boundary_oof, weights)
        by_subject = [score(y[subjects == subject], prediction[subjects == subject]) for subject in SUBJECTS]
        component_scores[name] = {
            key + "_mean": float(np.mean([item[key] for item in by_subject]))
            for key in ("macro_f1", "landing", "braking", "propulsion", "recovery")
        }
    all_selection_scores = {}
    for name, weights in CONFIGS.items():
        all_selection_scores[name] = score(y, decoded_hybrid(tcn_oof, boundary_oof, weights))["macro_f1"]
    deployed_config = max(all_selection_scores, key=all_selection_scores.get)
    np.savez_compressed(
        artifacts / "development_oof_component_probabilities.npz",
        tcn=tcn_oof, boundary=boundary_oof, truth=y, subjects=subjects,
    )
    boundary_model_path = BOUNDARY_DIR / "artifacts/boundary_regression_v1.joblib"
    tcn_model_path = TCN_DIR / "artifacts/temporal_tcn_v1.pt"
    hybrid_artifact = {
        "experiment": "hybrid_v1", "selected_deployment_config": deployed_config,
        "classwise_tcn_weights": CONFIGS[deployed_config].tolist(),
        "class_order": ["landing", "braking", "propulsion", "recovery"],
        "boundary_component": {"path": str(boundary_model_path), "sha256": sha256(boundary_model_path)},
        "tcn_component": {"path": str(tcn_model_path), "sha256": sha256(tcn_model_path)},
        "decoder": "monotonic 4-1-2-3-4 Viterbi",
        "development_subjects": SUBJECTS, "forbidden_subjects": sorted(FORBIDDEN),
    }
    artifact_path = artifacts / "hybrid_v1.json"
    artifact_path.write_text(json.dumps(hybrid_artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "DEVELOPMENT_ONLY_HYBRID_FROZEN",
        "experiment": "hybrid_v1", "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "meta_loso_macro_f1_mean": float(cv["macro_f1"].mean()),
        "meta_loso_macro_f1_std": float(cv["macro_f1"].std()),
        "meta_loso_macro_f1_min": float(cv["macro_f1"].min()),
        "meta_loso_per_stage_f1_mean": {
            stage: float(cv[stage].mean()) for stage in ("landing", "braking", "propulsion", "recovery")
        },
        "component_scores_on_common_64_step_grid": component_scores,
        "selected_deployment_config": deployed_config,
        "all_oof_config_scores_for_deployment": all_selection_scores,
        "development_subjects": SUBJECTS, "forbidden_subjects": sorted(FORBIDDEN),
        "forbidden_subjects_accessed": False,
        "limitations": [
            "Development-only nested meta-LOSO evidence, not a new untouched-test result.",
            "Both components inherit the same pose and force-label preparation limitations.",
            "Final generalization requires new subjects not used by any prior experiment."
        ],
    }
    manifest_path = reports / "MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
