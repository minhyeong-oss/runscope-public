from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_v1"
SOURCE_EXPERIMENT = ROOT / "experiments/landing_kth_adaptation_v1"
SEQUENCE_ROOT = SOURCE_EXPERIMENT / "artifacts/sequences"
DATASET_MANIFEST = SOURCE_EXPERIMENT / "artifacts/DATASET_MANIFEST.json"
SEED = 20261110


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def context_features(features: np.ndarray, probability: np.ndarray, event: int) -> np.ndarray:
    # Blocks 2..7 are left/right hip, knee, and ankle. The gate never receives
    # the kinematic-event array or the label-generating peak distance.
    lower_body = features[:, 2 * 7:]
    indices = np.clip(np.arange(event - 4, event + 5), 0, len(features) - 1)
    window = lower_body[indices]
    summary = np.concatenate([
        lower_body[event],
        window.mean(axis=0),
        window.std(axis=0),
        window.min(axis=0),
        window.max(axis=0),
        window[-1] - window[0],
    ])
    probability_indices = np.clip(np.arange(event - 5, event + 6), 0, len(probability) - 1)
    probability_window = probability[probability_indices]
    teacher_summary = np.asarray([
        probability[event],
        probability_window.mean(),
        probability_window.std(),
        probability_window.min(),
        probability_window.max(),
        probability[event] - probability_window.mean(),
    ], dtype=np.float32)
    return np.concatenate([summary, teacher_summary]).astype(np.float32)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    x = []
    y = []
    for path in sorted(SEQUENCE_ROOT.glob("*.npz")):
        bundle = np.load(path)
        if str(bundle["split"].item()) != split or not bool(bundle["accepted"].item()):
            continue
        features = bundle["features"].astype(np.float32)
        probability = bundle["teacher_probability"].astype(np.float32)
        positives = set(bundle["events"].astype(int).tolist())
        for event in bundle["teacher_events"].astype(int):
            label = int(event in positives)
            x.append(context_features(features, probability, int(event)))
            y.append(label)
            rows.append({
                "split": split,
                "video": path.stem,
                "subject": str(bundle["subject"].item()),
                "event_frame_30hz": int(event),
                "consensus_pseudo_label": label,
                "teacher_probability": float(probability[event]),
            })
    if not x:
        raise RuntimeError(f"No events in {split}")
    return np.stack(x), np.asarray(y, dtype=np.uint8), pd.DataFrame(rows)


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = probability >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, prediction, average="binary", zero_division=0
    )
    true_positive = int(np.sum(prediction & (y_true == 1)))
    false_positive = int(np.sum(prediction & (y_true == 0)))
    false_negative = int(np.sum((~prediction) & (y_true == 1)))
    return {
        "pseudo_label_agreement_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def candidates() -> dict[str, object]:
    models: dict[str, object] = {}
    for c in (0.03, 0.10, 0.30, 1.0, 3.0):
        models[f"logistic_c_{c:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c, class_weight="balanced", max_iter=5000, random_state=SEED
            ),
        )
    for depth in (2, 3):
        for l2 in (0.1, 1.0):
            models[f"histgb_depth_{depth}_l2_{l2:g}"] = HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.05,
                max_depth=depth,
                l2_regularization=l2,
                class_weight="balanced",
                random_state=SEED,
            )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=4,
    )
    return models


def main() -> None:
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if manifest["prior_locked_test_data_accessed"] or manifest["kth_official_test_data_accessed"]:
        raise RuntimeError("A forbidden test split was accessed")
    train_x, train_y, train_rows = load_split("train")
    validation_x, validation_y, validation_rows = load_split("validation")

    search_rows = []
    fitted = {}
    probabilities = {}
    for name, model in candidates().items():
        model.fit(train_x, train_y)
        fitted[name] = model
        probability = model.predict_proba(validation_x)[:, 1]
        probabilities[name] = probability
        for threshold in np.linspace(0.05, 0.95, 91):
            search_rows.append({
                "model": name,
                "threshold": float(threshold),
                **metrics(validation_y, probability, float(threshold)),
            })
    search = pd.DataFrame(search_rows).sort_values(
        ["pseudo_label_agreement_f1", "precision", "threshold"],
        ascending=[False, False, False],
    )
    selected = search.iloc[0]
    model_name = str(selected["model"])
    threshold = float(selected["threshold"])
    model = fitted[model_name]
    validation_probability = probabilities[model_name]
    selected_metrics = metrics(validation_y, validation_probability, threshold)

    artifact_dir = EXPERIMENT / "artifacts"
    report_dir = EXPERIMENT / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "landing_kth_event_gate_v1.joblib"
    joblib.dump({
        "experiment": "landing_kth_event_gate_v1",
        "model_name": model_name,
        "model": model,
        "threshold": threshold,
        "input_features": int(train_x.shape[1]),
        "source_teacher": "landing_visual_contact_v2",
        "training_subjects": manifest["split"]["train"],
        "validation_subjects": manifest["split"]["validation"],
        "untouched_kth_test_subjects": manifest["split"]["untouched_test"],
        "forbidden_prior_test_subjects": manifest["prior_locked_test_subjects"],
    }, model_path)

    validation_output = validation_rows.copy()
    validation_output["gate_probability"] = validation_probability
    validation_output["gate_prediction"] = validation_probability >= threshold
    subject_metrics = []
    for subject, group in validation_output.groupby("subject"):
        subject_metrics.append({
            "subject": str(subject),
            "teacher_events": int(len(group)),
            "consensus_pseudo_events": int(group["consensus_pseudo_label"].sum()),
            **metrics(
                group["consensus_pseudo_label"].to_numpy(dtype=np.uint8),
                group["gate_probability"].to_numpy(dtype=np.float32),
                threshold,
            ),
        })
    subject_table = pd.DataFrame(subject_metrics)
    train_rows.to_csv(report_dir / "training_teacher_events.csv", index=False, encoding="utf-8-sig")
    validation_output.to_csv(
        report_dir / "validation_teacher_events.csv", index=False, encoding="utf-8-sig"
    )
    subject_table.to_csv(
        report_dir / "validation_subject_metrics.csv", index=False, encoding="utf-8-sig"
    )
    search.to_csv(report_dir / "model_and_threshold_search.csv", index=False, encoding="utf-8-sig")

    baseline = metrics(
        validation_y, np.ones(len(validation_y), dtype=np.float32), threshold=0.5
    )
    report = {
        "status": "KTH_EVENT_GATE_VALIDATED_ON_PSEUDO_LABELS_NO_FORCE_F1_CLAIM",
        "model_sha256": sha256(model_path),
        "selected_model": model_name,
        "selected_threshold": threshold,
        "training_events": int(len(train_y)),
        "training_positive_consensus_events": int(train_y.sum()),
        "validation_events": int(len(validation_y)),
        "validation_positive_consensus_events": int(validation_y.sum()),
        "validation_pseudo_label_agreement": selected_metrics,
        "accept_all_frozen_teacher_baseline": baseline,
        "f1_improvement_over_accept_all_teacher": (
            selected_metrics["pseudo_label_agreement_f1"]
            - baseline["pseudo_label_agreement_f1"]
        ),
        "validation_subject_f1_range": {
            "minimum": float(subject_table["pseudo_label_agreement_f1"].min()),
            "maximum": float(subject_table["pseudo_label_agreement_f1"].max()),
        },
        "prior_locked_test_data_accessed": False,
        "kth_official_test_data_accessed": False,
        "post_validation_retraining_performed": False,
        "limitations": [
            "Targets are frozen-teacher/ankle consensus pseudo-labels.",
            "The score is not manual-contact or force-plate F1.",
            "Model family and threshold were selected on the KTH validation split.",
            "The KTH official test split remains untouched for a later frozen evaluation plan.",
        ],
    }
    (report_dir / "MODEL_MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
