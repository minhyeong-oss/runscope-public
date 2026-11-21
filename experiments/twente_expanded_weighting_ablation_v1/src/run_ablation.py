from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/twente_expanded_weighting_ablation_v1"
DATASET = ROOT / "experiments/twente_expanded_alignment_v1/artifacts/expanded_development_matched_strides.csv"
BOUNDARY_SOURCE = ROOT / "experiments/boundary_regression_v1/src/train_boundary_regression.py"


def load_boundary():
    spec = importlib.util.spec_from_file_location("runscope_boundary_ablation", BOUNDARY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load boundary implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights(subjects: pd.Series) -> np.ndarray:
    counts = subjects.value_counts()
    values = subjects.map(lambda subject: 1.0 / counts[subject]).to_numpy(dtype=float)
    return values * len(values) / values.sum()


def main() -> None:
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary()
    data = pd.read_csv(DATASET)
    forbidden = {"Subj04", "Subj11"}
    if set(data["subject"]) & forbidden:
        raise RuntimeError("Forbidden subject found")
    metadata = {
        "subject", "run", "side", "window_start_s", "window_end_s",
        "force_event_index", "force_initial_contact_s",
        "detected_start_source_time_s", "sync_offset_s", "sync_correlation",
        "candidate_force_timing_error_s", "abs_candidate_force_timing_error_s",
        "duplicate_force_event", *boundary.TARGETS,
    }
    features = [column for column in data.columns if column not in metadata]
    rows = []
    for held_out in sorted(data["subject"].unique()):
        train = data[data["subject"] != held_out]
        valid = data[data["subject"] == held_out]
        for name, balanced in [
            ("ridge_alpha10_unweighted", False),
            ("ridge_alpha10_subject_balanced", True),
        ]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            fit_kwargs = {"ridge__sample_weight": weights(train["subject"])} if balanced else {}
            model.fit(
                train[features].to_numpy(dtype=float),
                train[boundary.TARGETS].to_numpy(dtype=float),
                **fit_kwargs,
            )
            prediction = model.predict(valid[features].to_numpy(dtype=float))
            score = boundary.score_predictions(valid, prediction)
            rows.append({
                "held_out_subject": held_out,
                "variant": name,
                "validation_strides": int(len(valid)),
                **score["per_stage_f1"],
                "macro_f1": score["macro_f1"],
            })
    table = pd.DataFrame(rows)
    summary = table.groupby("variant").agg(
        participants=("held_out_subject", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_min=("macro_f1", "min"),
        landing_f1_mean=("landing", "mean"),
        braking_f1_mean=("braking", "mean"),
        propulsion_f1_mean=("propulsion", "mean"),
        recovery_f1_mean=("recovery", "mean"),
    ).reset_index()
    table.to_csv(reports / "weighting_ablation_loso.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(reports / "weighting_ablation_summary.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "POST_HOC_DIAGNOSTIC_COMPLETE_NOT_MODEL_SELECTION",
        "forbidden_subjects_accessed": False,
        "dataset_strides": int(len(data)),
        "subjects": sorted(data["subject"].unique().tolist()),
        "summary": summary.to_dict(orient="records"),
    }
    (reports / "ABLATION_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
