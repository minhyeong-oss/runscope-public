from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from evaluate_twente_force_f1 import force_strides


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/twente_sensor"
SPLIT_LOCK = DATA_ROOT / "SPLIT_LOCK_v1.json"
MODEL_DIR = ROOT / "model_v1/twente_subject_split_v3"
SPEEDS = {63: 1.75, 81: 2.25, 99: 2.75}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mat_path(subject: str, run: int) -> Path:
    if subject == "Subj05":
        return DATA_ROOT / subject / f"extracted_run{run}" / subject / f"{subject}_run_{run}.mat"
    return DATA_ROOT / subject / "development_extracted" / subject / f"{subject}_run_{run}.mat"


def build_events(subjects: list[str]) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        for run, speed in SPEEDS.items():
            path = mat_path(subject, run)
            datastr = loadmat(path, simplify_cells=True, variable_names=["Datastr"])["Datastr"]
            frame_rate = float(datastr["Force"]["FrameRate"])
            mass = float(datastr["Info"]["subjMass"])
            for side in ("left", "right"):
                events = force_strides(
                    datastr["Force"][side.capitalize() + "ForceData"],
                    frame_rate,
                    mass,
                )
                for index, event in enumerate(events):
                    duration = event["next_contact_s"] - event["initial_contact_s"]
                    p3 = (event["propulsion_start_s"] - event["initial_contact_s"]) / duration
                    p4 = (event["toe_off_s"] - event["initial_contact_s"]) / duration
                    if 0.10 < p3 < p4 < 0.80 and 0.45 < duration < 1.20:
                        rows.append({
                            "subject": subject,
                            "run": run,
                            "speed_m_s": speed,
                            "side": side,
                            "stride_index": index,
                            "duration_s": duration,
                            "propulsion_start_fraction": p3,
                            "recovery_start_fraction": p4,
                            "ap_sign_consistency": event["ap_sign_consistency"],
                            "source_mat": str(path),
                        })
    return pd.DataFrame(rows)


def make_models(kind: str):
    if kind == "ridge_poly2":
        return [
            make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=5.0)),
            make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=5.0)),
        ]
    if kind == "hist_gbr":
        return [
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=9, l2_regularization=2.0, random_state=20261107),
            HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=9, l2_regularization=2.0, random_state=20261107),
        ]
    if kind == "random_forest":
        return [
            RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=20261107),
            RandomForestRegressor(n_estimators=250, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=20261107),
        ]
    raise ValueError(kind)


def fit_candidate(kind: str, train: pd.DataFrame) -> dict:
    if kind == "speed_median":
        medians = train.groupby("run")[["propulsion_start_fraction", "recovery_start_fraction"]].median()
        return {"kind": kind, "medians": medians.to_dict(orient="index")}
    x = train[["speed_m_s", "duration_s"]].to_numpy()
    models = make_models(kind)
    models[0].fit(x, train["propulsion_start_fraction"].to_numpy())
    models[1].fit(x, train["recovery_start_fraction"].to_numpy())
    return {"kind": kind, "models": models}


def predict_boundaries(bundle: dict, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if bundle["kind"] == "speed_median":
        p3 = np.asarray([bundle["medians"][int(run)]["propulsion_start_fraction"] for run in table["run"]])
        p4 = np.asarray([bundle["medians"][int(run)]["recovery_start_fraction"] for run in table["run"]])
    else:
        x = table[["speed_m_s", "duration_s"]].to_numpy()
        p3 = bundle["models"][0].predict(x)
        p4 = bundle["models"][1].predict(x)
    p3 = np.clip(p3, 0.12, 0.55)
    p4 = np.clip(p4, p3 + 0.08, 0.75)
    return p3, p4


def frame_metric(table: pd.DataFrame, p3: np.ndarray, p4: np.ndarray) -> tuple[float, int]:
    truth_parts, prediction_parts = [], []
    for (_, row), pred3, pred4 in zip(table.iterrows(), p3, p4):
        duration = float(row["duration_s"])
        time = np.arange(int(round(duration * 100.0))) / 100.0
        fraction = time / duration
        truth = np.full(len(time), 4, dtype=np.int8)
        truth[fraction < row["recovery_start_fraction"]] = 3
        truth[fraction < row["propulsion_start_fraction"]] = 2
        truth[time < 0.030] = 1
        prediction = np.full(len(time), 4, dtype=np.int8)
        prediction[fraction < pred4] = 3
        prediction[fraction < pred3] = 2
        prediction[time < 0.030] = 1
        truth_parts.append(truth)
        prediction_parts.append(prediction)
    truth = np.concatenate(truth_parts)
    prediction = np.concatenate(prediction_parts)
    return float(f1_score(truth, prediction, labels=[1, 2, 3, 4], average="macro")), len(truth)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_LOCK.read_text(encoding="utf-8"))
    development = list(split["development_subjects"])
    test_subjects = set(split["test_subjects"])
    if set(development) & test_subjects:
        raise RuntimeError("Development/test subject overlap")
    if any((DATA_ROOT / subject / "development_extracted").exists() for subject in test_subjects):
        raise RuntimeError("A locked test archive was extracted before model freeze")

    events = build_events(development)
    events_path = MODEL_DIR / "development_force_events.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    candidate_names = ["speed_median", "ridge_poly2", "hist_gbr", "random_forest"]
    cv_rows = []
    for held_out in development:
        train = events[events["subject"] != held_out]
        validation = events[events["subject"] == held_out]
        for candidate in candidate_names:
            bundle = fit_candidate(candidate, train)
            p3, p4 = predict_boundaries(bundle, validation)
            macro_f1, frames = frame_metric(validation, p3, p4)
            cv_rows.append({
                "held_out_subject": held_out,
                "candidate": candidate,
                "validation_strides": len(validation),
                "validation_frames": frames,
                "macro_f1": macro_f1,
            })
    cv = pd.DataFrame(cv_rows)
    cv_path = MODEL_DIR / "development_loso_cv.csv"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")
    summary = cv.groupby("candidate")["macro_f1"].agg(["mean", "std", "min", "max"]).reset_index()
    summary = summary.sort_values(["mean", "min"], ascending=False).reset_index(drop=True)
    summary_path = MODEL_DIR / "development_loso_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    selected = str(summary.iloc[0]["candidate"])
    final_bundle = fit_candidate(selected, events)
    artifact = {
        "version": "twente_subject_split_v3_frozen",
        "selected_candidate": selected,
        "feature_names": ["speed_m_s", "stride_duration_s"],
        "landing_window_s": 0.030,
        "development_subjects": development,
        "locked_test_subjects": sorted(test_subjects),
        "speed_mapping_m_s": SPEEDS,
        "model": final_bundle,
    }
    artifact_path = MODEL_DIR / "runscope_twente_v3_frozen.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    artifact_hash = sha256(artifact_path)
    frozen = {
        "status": "FROZEN_BEFORE_TEST_ACCESS",
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "selected_candidate": selected,
        "selection_basis": "highest mean subject-disjoint LOSO macro F1 on development subjects only",
        "development_subjects": development,
        "locked_test_subjects": sorted(test_subjects),
        "development_strides": int(len(events)),
        "development_loso_macro_f1": {
            row["candidate"]: {
                key: float(row[key]) for key in ("mean", "std", "min", "max")
            }
            for _, row in summary.iterrows()
        },
        "test_access_before_freeze": False,
        "prohibition": "Never refit, recalibrate, select, or tune this artifact using Subj04 or Subj11.",
        "inputs_sha256": {str(path): sha256(path) for path in sorted({Path(value) for value in events["source_mat"]})},
        "split_lock_sha256": sha256(SPLIT_LOCK),
    }
    freeze_path = MODEL_DIR / "MODEL_FREEZE.json"
    freeze_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "development_subjects": development,
        "locked_test_subjects": sorted(test_subjects),
        "development_strides": len(events),
        "selected_candidate": selected,
        "selected_loso_macro_f1_mean": float(summary.iloc[0]["mean"]),
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_hash,
        "freeze_manifest": str(freeze_path),
    }, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
