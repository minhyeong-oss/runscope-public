from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/expected_f1_calibrator_v1"
ALPHA = 1.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_trial_table() -> pd.DataFrame:
    hybrid = load_module(
        "expected_f1_hybrid",
        ROOT / "experiments/hybrid_v1/src/evaluate_hybrid.py",
    )
    sequence = np.load(
        ROOT / "experiments/temporal_tcn_v1/artifacts/development_sequences_64.npz"
    )
    oof = np.load(
        ROOT / "experiments/hybrid_v1/artifacts/development_oof_component_probabilities.npz"
    )
    matched_all = pd.read_csv(
        ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"
    )
    matched = matched_all.iloc[sequence["source_rows"].astype(int)].reset_index(drop=True)
    weights = hybrid.CONFIGS["tcn_landing_equal_others"]
    hybrid_probability = oof["tcn"] * weights + oof["boundary"] * (1.0 - weights)
    hybrid_probability /= np.clip(
        hybrid_probability.sum(axis=-1, keepdims=True), 1e-8, None
    )
    prediction = hybrid.decoded_hybrid(oof["tcn"], oof["boundary"], weights)

    rows = []
    for (subject, run), group in matched.groupby(["subject", "run"]):
        indices = group.index.to_numpy()
        truth = oof["truth"][indices].reshape(-1)
        predicted = prediction[indices].reshape(-1)
        rows.append(
            {
                "subject": subject,
                "run": int(run),
                "sequences": int(len(indices)),
                "hybrid_mean_max_probability": float(
                    hybrid_probability[indices].max(axis=2).mean()
                ),
                "observed_macro_f1": float(
                    f1_score(
                        truth,
                        predicted,
                        labels=[1, 2, 3, 4],
                        average="macro",
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)


def main() -> None:
    artifacts = EXPERIMENT / "artifacts"
    reports = EXPERIMENT / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    table = build_trial_table()
    features = ["hybrid_mean_max_probability"]
    x = table[features].to_numpy(dtype=float)
    y = table["observed_macro_f1"].to_numpy(dtype=float)
    groups = table["subject"].to_numpy(dtype=str)

    oof_prediction = np.zeros(len(table), dtype=float)
    for train_index, validation_index in LeaveOneGroupOut().split(x, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
        model.fit(x[train_index], y[train_index])
        oof_prediction[validation_index] = model.predict(x[validation_index])

    residual = np.abs(y - oof_prediction)
    error_90 = float(np.quantile(residual, 0.90, method="higher"))
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    final_model.fit(x, y)
    artifact = {
        "experiment": "expected_f1_calibrator_v1",
        "target": "hybrid four-stage frame Macro F1",
        "feature_names": features,
        "model": final_model,
        "empirical_absolute_error_90": error_90,
        "training_feature_min": float(x.min()),
        "training_feature_max": float(x.max()),
        "training_target_min": float(y.min()),
        "training_target_max": float(y.max()),
        "development_subjects": sorted(set(groups)),
        "development_trials": int(len(table)),
    }
    artifact_path = artifacts / "expected_f1_calibrator_v1.joblib"
    joblib.dump(artifact, artifact_path)

    trial_path = reports / "development_trial_features_and_f1.csv"
    table.to_csv(trial_path, index=False, encoding="utf-8-sig")
    oof_table = table.copy()
    oof_table["predicted_expected_f1"] = oof_prediction
    oof_table["absolute_error"] = residual
    oof_path = reports / "subject_loso_predictions.csv"
    oof_table.to_csv(oof_path, index=False, encoding="utf-8-sig")

    metrics = {
        "mae": float(mean_absolute_error(y, oof_prediction)),
        "rmse": float(mean_squared_error(y, oof_prediction) ** 0.5),
        "r2": float(r2_score(y, oof_prediction)),
        "correlation": float(np.corrcoef(y, oof_prediction)[0, 1]),
        "empirical_absolute_error_90": error_90,
    }
    manifest = {
        "status": "EXPECTED_F1_CALIBRATOR_DEVELOPMENT_ONLY",
        "experiment": "expected_f1_calibrator_v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "model": "StandardScaler + Ridge(alpha=1.0)",
        "input": features,
        "target": "observed hybrid Macro F1 on force-labeled development trials",
        "validation": "leave-one-subject-out across 7 subjects and 21 trials",
        "metrics": metrics,
        "development_trials": int(len(table)),
        "development_subjects": sorted(set(groups)),
        "feature_range": [float(x.min()), float(x.max())],
        "target_range": [float(y.min()), float(y.max())],
        "limitations": [
            "The output is an expected F1 estimate, not an observed F1 on the new video.",
            "Only 21 laboratory treadmill trials from 7 subjects were available.",
            "The calibration may not transfer to a curved non-motorized treadmill.",
            "The 90 percent band is an empirical LOSO residual band, not a guarantee.",
        ],
    }
    manifest_path = reports / "MODEL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
