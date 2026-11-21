from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/twente_sample_participant_curve_v1"
PROTOCOL_PATH = EXPERIMENT / "config/protocol.json"
DATASET_PATH = ROOT / "experiments/twente_expanded_alignment_v1/artifacts/expanded_development_matched_strides.csv"
WINDOW_DECISIONS = ROOT / "experiments/twente_expanded_alignment_v1/reports/window_inclusion_decisions.csv"
WINDOW_QUALITY = ROOT / "experiments/twente_expanded_alignment_v1/reports/included_window_quality.csv"
OLD_DATASET_PATH = ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"
BOUNDARY_SOURCE = ROOT / "experiments/boundary_regression_v1/src/train_boundary_regression.py"
REPORTS = EXPERIMENT / "reports"


def load_boundary():
    spec = importlib.util.spec_from_file_location("runscope_boundary_curve", BOUNDARY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load boundary implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_weights(subjects: pd.Series) -> np.ndarray:
    counts = subjects.value_counts()
    values = subjects.map(lambda x: 1.0 / float(counts[x])).to_numpy(dtype=float)
    return values * len(values) / values.sum()


def sample_old_scale(
    table: pd.DataFrame, subjects: tuple[str, ...], old_counts: dict[str, int], seed: int
) -> pd.DataFrame:
    pieces = []
    for index, subject in enumerate(subjects):
        subset = table[table["subject"] == subject]
        n = min(int(old_counts[subject]), len(subset))
        pieces.append(subset.sample(n=n, replace=False, random_state=seed + index * 1009))
    return pd.concat(pieces, ignore_index=True)


def fit_predict(
    boundary,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    alpha: float,
) -> tuple[np.ndarray, dict]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(
        train[feature_names].to_numpy(dtype=float),
        train[boundary.TARGETS].to_numpy(dtype=float),
        ridge__sample_weight=subject_weights(train["subject"]),
    )
    prediction = model.predict(validation[feature_names].to_numpy(dtype=float))
    score = boundary.score_predictions(validation, prediction)
    return prediction, score


def choose_alpha(boundary, train: pd.DataFrame, feature_names: list[str], alphas: list[float]) -> float:
    subjects = sorted(train["subject"].unique())
    if len(subjects) < 2:
        raise RuntimeError("Nested LOSO requires at least two training subjects")
    rows = []
    for alpha in alphas:
        fold_scores = []
        for held_out in subjects:
            inner_train = train[train["subject"] != held_out]
            inner_valid = train[train["subject"] == held_out]
            _, score = fit_predict(boundary, inner_train, inner_valid, feature_names, alpha)
            fold_scores.append(float(score["macro_f1"]))
        rows.append((float(np.mean(fold_scores)), float(alpha)))
    return max(rows, key=lambda item: (item[0], item[1]))[1]


def event_metrics(boundary, validation: pd.DataFrame, prediction: np.ndarray) -> dict:
    constrained = boundary.constrain(prediction)
    error_ms = (constrained[:, 0] - validation["target_ic_offset_s"].to_numpy(dtype=float)) * 1000.0
    frame_ms = 1000.0 / float(validation["fps"].median()) if "fps" in validation else 1000.0 / 33.0
    return {
        "landing_mae_ms": float(np.mean(np.abs(error_ms))),
        "landing_signed_error_ms": float(np.mean(error_ms)),
        "landing_within_1_frame": float(np.mean(np.abs(error_ms) <= frame_ms)),
        "landing_within_2_frames": float(np.mean(np.abs(error_ms) <= 2.0 * frame_ms)),
        "native_frame_ms": frame_ms,
    }


def bootstrap_ci(values: np.ndarray, seed: int = 20261121, draws: int = 100000) -> dict:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "participants": int(len(values)),
    }


def candidate_coverage(boundary, dataset: pd.DataFrame) -> pd.DataFrame:
    windows = pd.read_csv(WINDOW_DECISIONS)
    windows = windows[windows["included"]].copy()
    quality = pd.read_csv(WINDOW_QUALITY)
    rows = []
    for (subject, run), group in windows.groupby(["subject", "run"], sort=True):
        datastr = loadmat(
            boundary.mat_path(str(subject), int(run)),
            simplify_cells=True,
            variable_names=["Datastr"],
        )["Datastr"]
        force_rate = float(datastr["Force"]["FrameRate"])
        mass = float(datastr["Info"]["subjMass"])
        intervals = [
            (
                float(item.window_start_s) + float(item.offset_ms) / 1000.0,
                float(item.window_end_s) + float(item.offset_ms) / 1000.0,
            )
            for item in group.itertuples(index=False)
        ]
        eligible: set[tuple[str, int]] = set()
        for side in ("left", "right"):
            events = boundary.force_strides(
                datastr["Force"][side.capitalize() + "ForceData"], force_rate, mass
            )
            for event_index, event in enumerate(events):
                duration = float(event["next_contact_s"] - event["initial_contact_s"])
                if not 0.45 <= duration <= 1.20:
                    continue
                if any(
                    start <= float(event["initial_contact_s"])
                    and float(event["next_contact_s"]) <= end
                    for start, end in intervals
                ):
                    eligible.add((side, int(event_index)))
        matched_subset = dataset[
            (dataset["subject"] == subject) & (dataset["run"] == int(run))
        ]
        matched = set(
            zip(matched_subset["side"], matched_subset["force_event_index"].astype(int))
        )
        q = quality[(quality["subject"] == subject) & (quality["run"] == int(run))]
        candidates = int(q["candidate_strides"].sum())
        rows.append(
            {
                "subject": subject,
                "run": int(run),
                "approved_windows": int(len(group)),
                "eligible_force_contacts": int(len(eligible)),
                "matched_contacts": int(len(matched & eligible)),
                "candidate_strides": candidates,
                "candidate_recall": float(len(matched & eligible) / len(eligible)) if eligible else np.nan,
                "candidates_per_eligible_contact": float(candidates / len(eligible)) if eligible else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    boundary = load_boundary()
    data = pd.read_csv(DATASET_PATH)
    old = pd.read_csv(OLD_DATASET_PATH)
    forbidden = {"Subj04", "Subj11"}
    if set(data["subject"]) & forbidden:
        raise RuntimeError("Forbidden subject present")
    old_counts = old.groupby("subject").size().astype(int).to_dict()
    metadata = {
        "subject", "run", "side", "window_start_s", "window_end_s",
        "force_event_index", "force_initial_contact_s", "detected_start_source_time_s",
        "sync_offset_s", "sync_correlation", "candidate_force_timing_error_s",
        "abs_candidate_force_timing_error_s", "duplicate_force_event", *boundary.TARGETS,
    }
    feature_names = [column for column in data.columns if column not in metadata]
    subjects = sorted(data["subject"].unique())
    alphas = [float(value) for value in protocol["alpha_candidates"]]
    seeds = [int(value) for value in protocol["random_seeds"]]
    rows = []
    for held_out in subjects:
        validation = data[data["subject"] == held_out]
        available = tuple(subject for subject in subjects if subject != held_out)
        for participant_count in protocol["training_participant_counts"]:
            for combo in itertools.combinations(available, int(participant_count)):
                expanded_train = data[data["subject"].isin(combo)].copy()
                alpha = choose_alpha(boundary, expanded_train, feature_names, alphas)
                prediction, score = fit_predict(
                    boundary, expanded_train, validation, feature_names, alpha
                )
                rows.append(
                    {
                        "held_out_subject": held_out,
                        "training_subjects": "+".join(combo),
                        "training_participants": int(participant_count),
                        "stride_budget": "EXPANDED_ALL",
                        "seed": -1,
                        "alpha": alpha,
                        "training_strides": int(len(expanded_train)),
                        "validation_strides": int(len(validation)),
                        **score["per_stage_f1"],
                        "macro_f1": score["macro_f1"],
                        **event_metrics(boundary, validation, prediction),
                    }
                )
                for seed in seeds:
                    sampled = sample_old_scale(data, combo, old_counts, seed)
                    sampled_alpha = choose_alpha(boundary, sampled, feature_names, alphas)
                    sampled_prediction, sampled_score = fit_predict(
                        boundary, sampled, validation, feature_names, sampled_alpha
                    )
                    rows.append(
                        {
                            "held_out_subject": held_out,
                            "training_subjects": "+".join(combo),
                            "training_participants": int(participant_count),
                            "stride_budget": "OLD_SCALE",
                            "seed": seed,
                            "alpha": sampled_alpha,
                            "training_strides": int(len(sampled)),
                            "validation_strides": int(len(validation)),
                            **sampled_score["per_stage_f1"],
                            "macro_f1": sampled_score["macro_f1"],
                            **event_metrics(boundary, validation, sampled_prediction),
                        }
                    )
        print(json.dumps({"completed_held_out": held_out}), flush=True)

    curve = pd.DataFrame(rows)
    curve_path = REPORTS / "learning_curve_all_runs.csv"
    curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
    seed_averaged = (
        curve.groupby(
            ["held_out_subject", "training_subjects", "training_participants", "stride_budget"],
            as_index=False,
        )
        .agg(
            macro_f1=("macro_f1", "mean"),
            landing_f1=("landing", "mean"),
            landing_mae_ms=("landing_mae_ms", "mean"),
            landing_within_1_frame=("landing_within_1_frame", "mean"),
            landing_within_2_frames=("landing_within_2_frames", "mean"),
            training_strides=("training_strides", "mean"),
            alpha_median=("alpha", "median"),
        )
    )
    summary = (
        seed_averaged.groupby(["training_participants", "stride_budget"], as_index=False)
        .agg(
            evaluations=("macro_f1", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            landing_f1_mean=("landing_f1", "mean"),
            landing_mae_ms_mean=("landing_mae_ms", "mean"),
            landing_within_1_frame_mean=("landing_within_1_frame", "mean"),
            landing_within_2_frames_mean=("landing_within_2_frames", "mean"),
            training_strides_mean=("training_strides", "mean"),
        )
    )
    summary.to_csv(REPORTS / "learning_curve_summary.csv", index=False, encoding="utf-8-sig")

    k5 = seed_averaged[seed_averaged["training_participants"] == 5]
    wide = k5.pivot(index="held_out_subject", columns="stride_budget", values=[
        "macro_f1", "landing_f1", "landing_mae_ms", "landing_within_1_frame"
    ])
    paired = pd.DataFrame({"held_out_subject": wide.index})
    paired["macro_delta_expanded_minus_old"] = (
        wide[("macro_f1", "EXPANDED_ALL")].to_numpy()
        - wide[("macro_f1", "OLD_SCALE")].to_numpy()
    )
    paired["landing_f1_delta_expanded_minus_old"] = (
        wide[("landing_f1", "EXPANDED_ALL")].to_numpy()
        - wide[("landing_f1", "OLD_SCALE")].to_numpy()
    )
    paired["landing_mae_delta_ms_expanded_minus_old"] = (
        wide[("landing_mae_ms", "EXPANDED_ALL")].to_numpy()
        - wide[("landing_mae_ms", "OLD_SCALE")].to_numpy()
    )
    paired["landing_within1_delta_expanded_minus_old"] = (
        wide[("landing_within_1_frame", "EXPANDED_ALL")].to_numpy()
        - wide[("landing_within_1_frame", "OLD_SCALE")].to_numpy()
    )
    paired.to_csv(REPORTS / "paired_k5_participant_deltas.csv", index=False, encoding="utf-8-sig")
    bootstrap = {
        column: bootstrap_ci(paired[column].to_numpy(dtype=float))
        for column in paired.columns
        if column != "held_out_subject"
    }

    coverage = candidate_coverage(boundary, data)
    coverage.to_csv(REPORTS / "candidate_coverage_by_trial.csv", index=False, encoding="utf-8-sig")
    total_eligible = int(coverage["eligible_force_contacts"].sum())
    total_matched = int(coverage["matched_contacts"].sum())
    total_candidates = int(coverage["candidate_strides"].sum())
    result = {
        "status": "POST_HOC_SAMPLE_PARTICIPANT_CURVE_COMPLETE",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "forbidden_subjects_accessed": False,
        "participants": subjects,
        "maximum_training_participants_with_independent_evaluation": 5,
        "paired_k5_bootstrap": bootstrap,
        "candidate_coverage": {
            "eligible_force_contacts": total_eligible,
            "matched_contacts": total_matched,
            "candidate_strides": total_candidates,
            "candidate_recall": float(total_matched / total_eligible),
            "candidates_per_eligible_contact": float(total_candidates / total_eligible),
            "selection_precision_note": "Not reported: force matching is oracle evaluation, not a deployable force-free selector."
        },
        "interpretation_limit": "Post-hoc development-only diagnostic; not model-selection or untouched-test evidence."
    }
    (REPORTS / "LEARNING_CURVE_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
