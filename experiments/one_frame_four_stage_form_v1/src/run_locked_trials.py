from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/one_frame_four_stage_form_v1"
REPORTS = EXPERIMENT / "reports"
LOCKED_COMPARISON = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/final_v3/locked_test_predictions_and_force_truth.csv"
)
FORM_ANALYZER = ROOT / "experiments/running_form_feedback_v1/src/analyze_form.py"
FORM_RENDERER = ROOT / "experiments/running_form_feedback_v1/src/render_feedback.py"
LANDING_OOF = ROOT / "experiments/landing_event_v3/reports/development_oof_predictions_v3.npz"
LANDING_HISTORICAL = ROOT / "experiments/landing_event_v3/reports/historical_test_predictions.csv"
LANDING_SELECTION = ROOT / "experiments/landing_event_v3/artifacts/FROZEN_SELECTION.json"
TRIALS = [
    ("Subj04", 81),
    ("Subj04", 99),
    ("Subj11", 63),
    ("Subj11", 81),
    ("Subj11", 99),
]
STAGES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def paths(subject: str, run: int) -> tuple[Path, Path]:
    stem = f"{subject}_run{run}_side_10s_to_22s"
    video = ROOT / f"test_videos/twente_locked_test/{stem}.mp4"
    landmarks = (
        ROOT
        / f"video_v1/test_outputs/twente_locked_test/{subject}_run{run}/{stem}_yolo_landmarks.csv"
    )
    return video, landmarks


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid fps: {path}")
    return fps


def phase_metrics(table: pd.DataFrame) -> dict:
    truth = table["ground_truth_label"].to_numpy(dtype=int)
    prediction = table["predicted_label"].to_numpy(dtype=int)
    labels = list(STAGES)
    classification = classification_report(
        truth,
        prediction,
        labels=labels,
        target_names=[STAGES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": int(len(table)),
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro")),
        "weighted_f1": float(
            f1_score(truth, prediction, labels=labels, average="weighted")
        ),
        "per_stage": {
            STAGES[label]: {
                key: float(classification[STAGES[label]][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def first_stage_time_after(
    group: pd.DataFrame, column: str, stage: int, start_time: float
) -> float | None:
    candidates = group[
        (group["video_time_s"] >= start_time) & (group[column].astype(int) == stage)
    ]
    if candidates.empty:
        return None
    return float(candidates["video_time_s"].iloc[0])


def boundary_rows(table: pd.DataFrame, fps_lookup: dict[tuple[str, int], float]) -> pd.DataFrame:
    records: list[dict] = []
    keys = ["subject", "run", "stride_id", "side"]
    for (subject, run, stride_id, side), group in table.groupby(keys, sort=True):
        group = group.sort_values("video_time_s").reset_index(drop=True)
        true_landing_rows = group[group["ground_truth_label"].astype(int) == 1]
        predicted_landing_rows = group[group["predicted_label"].astype(int) == 1]
        if true_landing_rows.empty or predicted_landing_rows.empty:
            continue
        true_start = float(true_landing_rows["video_time_s"].iloc[0])
        predicted_start = float(predicted_landing_rows["video_time_s"].iloc[0])
        fps = fps_lookup[(str(subject), int(run))]
        tolerance_s = 1.0 / fps
        for stage in STAGES:
            truth_time = first_stage_time_after(
                group, "ground_truth_label", stage, true_start
            )
            prediction_time = first_stage_time_after(
                group, "predicted_label", stage, predicted_start
            )
            error_ms = (
                abs(prediction_time - truth_time) * 1000.0
                if truth_time is not None and prediction_time is not None
                else None
            )
            records.append(
                {
                    "subject": subject,
                    "run": int(run),
                    "stride_id": stride_id,
                    "side": side,
                    "boundary_into_stage": stage,
                    "boundary_name": STAGES[stage],
                    "truth_time_s": truth_time,
                    "prediction_time_s": prediction_time,
                    "absolute_error_ms": error_ms,
                    "fps": fps,
                    "one_frame_tolerance_ms": tolerance_s * 1000.0,
                    "within_one_frame": bool(
                        error_ms is not None and error_ms <= tolerance_s * 1000.0 + 1e-9
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_boundaries(rows: pd.DataFrame) -> dict:
    per_boundary = {}
    for stage, group in rows.groupby("boundary_into_stage"):
        valid = group["absolute_error_ms"].notna()
        hits = int(group.loc[valid, "within_one_frame"].sum())
        expected = int(len(group))
        predicted = int(valid.sum())
        precision = hits / predicted if predicted else 0.0
        recall = hits / expected if expected else 0.0
        event_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        per_boundary[STAGES[int(stage)]] = {
            "expected_events": expected,
            "predicted_events": predicted,
            "one_frame_hits": hits,
            "precision": precision,
            "recall": recall,
            "event_f1": event_f1,
            "mae_ms": float(group.loc[valid, "absolute_error_ms"].mean()),
        }
    return {
        "definition": (
            "A boundary is correct when its predicted transition is within ±1 native video "
            "frame of force-derived truth. One transition is expected per stage per stride."
        ),
        "macro_event_f1": float(
            np.mean([value["event_f1"] for value in per_boundary.values()])
        ),
        "per_boundary": per_boundary,
    }


def landing_one_frame_metrics() -> dict:
    selection = json.loads(LANDING_SELECTION.read_text(encoding="utf-8"))
    selected = str(selection["selected_candidate"])
    sample_interval_s = 0.48 / 63.0
    tolerance_s = 1.0 / 33.0
    oof = np.load(LANDING_OOF)
    truth = oof["truth_onset"].astype(int)
    candidate_metrics = {}
    for candidate in (
        "ridge_alpha100",
        "ridge_alpha1000",
        "scratch_weighted",
        "visual_temporal_init_weighted",
    ):
        prediction = oof[candidate].astype(int)
        error_s = np.abs(prediction - truth) * sample_interval_s
        candidate_metrics[candidate] = {
            "development_subject_loso_one_frame_event_f1": float(
                np.mean(error_s <= tolerance_s)
            ),
            "onset_mae_ms": float(error_s.mean() * 1000.0),
        }
    historical = pd.read_csv(LANDING_HISTORICAL)
    historical_error_s = (
        np.abs(
            historical["predicted_onset_index"].to_numpy(dtype=int)
            - historical["true_onset_index"].to_numpy(dtype=int)
        )
        * sample_interval_s
    )
    ranked = sorted(
        candidate_metrics,
        key=lambda name: (
            -candidate_metrics[name]["development_subject_loso_one_frame_event_f1"],
            candidate_metrics[name]["onset_mae_ms"],
        ),
    )
    return {
        "policy": "±1 native video frame",
        "reference_fps": 33.0,
        "tolerance_ms": tolerance_s * 1000.0,
        "selected_candidate": selected,
        "selected_candidate_is_also_best_under_one_frame_policy": ranked[0] == selected,
        "development_candidates": candidate_metrics,
        "historical_reference": {
            "sequences": int(len(historical)),
            "one_frame_event_f1": float(np.mean(historical_error_s <= tolerance_s)),
            "onset_mae_ms": float(historical_error_s.mean() * 1000.0),
            "scope": "4 evaluated historical trials; Subj11_run99 unavailable in landing_event_v3",
        },
    }


def run_form_feedback(
    subject: str,
    run: int,
    table: pd.DataFrame,
    metrics: dict,
    video: Path,
    landmarks: Path,
) -> dict:
    trial_name = f"{subject}_run{run}"
    output_dir = REPORTS / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = table.copy().rename(
        columns={"side": "tracked_side", "predicted_label": "v2_predicted_label"}
    )
    adapter_path = output_dir / "four_stage_predictions_adapter.csv"
    metrics_path = output_dir / "four_stage_metrics.json"
    adapter.to_csv(adapter_path, index=False, encoding="utf-8-sig")
    metrics_path.write_text(
        json.dumps({"temporal_prior_v2": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    analyze_command = [
        sys.executable,
        str(FORM_ANALYZER),
        "--video",
        str(video),
        "--landmarks",
        str(landmarks),
        "--comparison",
        str(adapter_path),
        "--phase-metrics",
        str(metrics_path),
        "--output-dir",
        str(output_dir),
        "--clip-offset-s",
        "10.0",
    ]
    subprocess.run(analyze_command, cwd=ROOT, check=True)
    subprocess.run(
        [
            sys.executable,
            str(FORM_RENDERER),
            "--report-dir",
            str(output_dir),
            "--landmarks",
            str(landmarks),
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads((output_dir / "FORM_FEEDBACK.json").read_text(encoding="utf-8"))
    return {
        "trial": trial_name,
        "usable_strides": int(report["usable_strides"]),
        "four_stage_macro_f1": float(metrics["macro_f1"]),
        "posture_ground_truth_available": False,
        "posture_f1": None,
        "assessments": {
            item["stage_name"]: {
                "status": item["status"],
                "status_ko": item["status_ko"],
                "evidence_ko": item["evidence_ko"],
                "action_ko": item["action_ko"],
                "remediation_links": item["remediation_links"],
            }
            for item in report["assessments"]
        },
        "board": str(output_dir / "running_form_feedback_board.png"),
        "html": str(output_dir / "running_form_feedback.html"),
        "report": str(output_dir / "FORM_FEEDBACK.json"),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    locked = pd.read_csv(LOCKED_COMPARISON)
    fps_lookup: dict[tuple[str, int], float] = {}
    trial_outputs = []
    for subject, run in TRIALS:
        video, landmarks = paths(subject, run)
        fps_lookup[(subject, run)] = video_fps(video)
        trial = locked[(locked["subject"] == subject) & (locked["run"] == run)].copy()
        if trial.empty:
            raise RuntimeError(f"No frozen four-stage rows for {subject} run {run}")
        metrics = phase_metrics(trial)
        trial_outputs.append(
            run_form_feedback(subject, run, trial, metrics, video, landmarks)
        )

    boundaries = boundary_rows(locked, fps_lookup)
    boundary_path = REPORTS / "one_frame_boundary_events.csv"
    boundaries.to_csv(boundary_path, index=False, encoding="utf-8-sig")
    overall_phase = phase_metrics(locked)
    report = {
        "status": "ONE_FRAME_FOUR_STAGE_FORM_EVALUATION_COMPLETE",
        "policy": {
            "event_tolerance": "±1 native video frame",
            "twente_fps": sorted(set(fps_lookup.values())),
            "twente_tolerance_ms": sorted(
                set(round(1000.0 / value, 6) for value in fps_lookup.values())
            ),
            "posture_colors": {
                "GOOD": "green",
                "IMPROVE": "red",
                "UNKNOWN": "gray",
            },
        },
        "landing_event": landing_one_frame_metrics(),
        "four_stage_framewise": overall_phase,
        "four_stage_boundary_one_frame": summarize_boundaries(boundaries),
        "trial_posture_feedback": trial_outputs,
        "posture_evaluation": {
            "ground_truth_available": False,
            "f1": None,
            "note": (
                "GOOD/IMPROVE is an evidence-informed screening rule, not a learned "
                "diagnosis. UNKNOWN is retained when no validated side-view cutoff exists."
            ),
        },
        "leakage_note": (
            "Frozen predictions are evaluated only. No test refit, threshold tuning, or "
            "post-test model selection is performed. Subj04/Subj11 remain historical references."
        ),
        "outputs": {"boundary_events_csv": str(boundary_path)},
    }
    report_path = REPORTS / "ONE_FRAME_FOUR_STAGE_FORM_REPORT.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "landing_one_frame_development_f1": report["landing_event"][
                    "development_candidates"
                ][report["landing_event"]["selected_candidate"]][
                    "development_subject_loso_one_frame_event_f1"
                ],
                "landing_one_frame_historical_f1": report["landing_event"][
                    "historical_reference"
                ]["one_frame_event_f1"],
                "four_stage_macro_f1": overall_phase["macro_f1"],
                "four_stage_boundary_one_frame_macro_f1": report[
                    "four_stage_boundary_one_frame"
                ]["macro_event_f1"],
                "trial_statuses": {
                    item["trial"]: {
                        stage: values["status"]
                        for stage, values in item["assessments"].items()
                    }
                    for item in trial_outputs
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
