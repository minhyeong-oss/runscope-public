from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
REPORT_DIR = EXPERIMENT / "reports"
TRAINING_DATASET = ROOT / "model_v1/data/runscope_phase_frames_v1.npz"
OLD_FOUR_STAGE = (
    ROOT
    / "video_v1/test_outputs/twente_subj05_run81_pilot/temporal_prior_v2"
    / "twente_temporal_prior_v2_comparison.csv"
)
LOCKED_FOUR_STAGE = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/final_v3"
    / "locked_test_predictions_and_force_truth.csv"
)
LANDING_REPORTS = {
    "Subj05_run81": (
        ROOT
        / "experiments/landing_kth_event_gate_treadmill_test_v1/reports/TEST_REPORT.json"
    ),
    "Subj11_run99": (
        ROOT
        / "experiments/landing_kth_event_gate_treadmill_test_v2/reports/TEST_REPORT.json"
    ),
}

sys.path.insert(0, str(ROOT / "video_v1"))
from evaluate_twente_temporal_prior_v2 import metrics, training_thresholds  # noqa: E402


STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_temporal_prior(table: pd.DataFrame, target_speed: float) -> tuple[pd.DataFrame, dict]:
    selected_speed, thresholds, training_stride_count = training_thresholds(
        TRAINING_DATASET, target_speed
    )
    landing_end, propulsion_start, recovery_start = thresholds

    def predict(fraction: float) -> int:
        if fraction < landing_end:
            return 1
        if fraction < propulsion_start:
            return 2
        if fraction < recovery_start:
            return 3
        return 4

    result = table.copy()
    result["same_model_predicted_label"] = result["cycle_fraction"].map(predict).astype(int)
    result["same_model_predicted_stage"] = result["same_model_predicted_label"].map(STAGE_NAMES)
    result["same_model_correct"] = result["same_model_predicted_label"].eq(
        result["ground_truth_label"]
    )
    metadata = {
        "target_speed_m_s": target_speed,
        "nearest_training_speed_m_s": selected_speed,
        "training_strides_used_for_thresholds": training_stride_count,
        "threshold_cycle_fractions": {
            "landing_end": landing_end,
            "propulsion_start": propulsion_start,
            "recovery_start": recovery_start,
        },
        "target_speed_outside_available_training_range": bool(target_speed > selected_speed),
    }
    return result, metadata


def landing_comparison() -> tuple[dict, pd.DataFrame]:
    result: dict[str, dict] = {}
    rows = []
    for trial, path in LANDING_REPORTS.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        result[trial] = {
            "force_truth_events": report["force_truth"]["events"],
            "teacher_candidates": report["predicted_events"]["teacher_before_gate"],
            "gate_accepted": report["predicted_events"]["accepted_after_gate"],
            "within_1_frame": report["event_gate_metrics"]["within_1_frames"],
            "within_2_frames": report["event_gate_metrics"]["within_2_frames"],
            "teacher_sha256": report["tested_stack"]["teacher_sha256"],
            "gate_sha256": report["tested_stack"]["gate_sha256"],
        }
        for tolerance, key in ((1, "within_1_frames"), (2, "within_2_frames")):
            values = report["event_gate_metrics"][key]
            rows.append(
                {
                    "trial": trial,
                    "tolerance_frames": tolerance,
                    "event_f1": values["event_f1"],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "true_positive": values["true_positive"],
                    "false_positive": values["false_positive"],
                    "false_negative": values["false_negative"],
                }
            )
    return result, pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(OLD_FOUR_STAGE)
    old, old_prior = apply_temporal_prior(old, target_speed=2.25)

    locked = pd.read_csv(LOCKED_FOUR_STAGE)
    new = locked[(locked["subject"] == "Subj11") & (locked["run"] == 99)].copy()
    new, new_prior = apply_temporal_prior(new, target_speed=2.75)

    old_metrics = metrics(old, "same_model_predicted_label")
    new_metrics = metrics(new, "same_model_predicted_label")
    landing, landing_table = landing_comparison()

    four_stage_rows = []
    for trial, values in (("Subj05_run81", old_metrics), ("Subj11_run99", new_metrics)):
        four_stage_rows.append(
            {
                "trial": trial,
                "accuracy": values["accuracy"],
                "macro_f1": values["macro_f1"],
                **{
                    f"{stage}_f1": values["per_stage"][stage]["f1-score"]
                    for stage in ("landing", "braking", "propulsion", "recovery")
                },
            }
        )
    four_stage_table = pd.DataFrame(four_stage_rows)

    new_prediction_path = REPORT_DIR / "subj11_run99_temporal_prior_v2_predictions.csv"
    landing_csv = REPORT_DIR / "landing_event_same_model_comparison.csv"
    four_stage_csv = REPORT_DIR / "four_stage_same_model_comparison.csv"
    report_path = REPORT_DIR / "CROSS_VIDEO_REPORT.json"
    new.to_csv(new_prediction_path, index=False, encoding="utf-8-sig")
    landing_table.to_csv(landing_csv, index=False, encoding="utf-8-sig")
    four_stage_table.to_csv(four_stage_csv, index=False, encoding="utf-8-sig")

    report = {
        "status": "FROZEN_CROSS_VIDEO_COMPARISON_COMPLETED_NO_RETRAINING",
        "experiment": EXPERIMENT.name,
        "training_or_refitting_performed": False,
        "threshold_or_parameter_tuning_performed": False,
        "landing_event_same_frozen_stack": landing,
        "four_stage_same_frozen_method": {
            "method": "video_phase_decoder_v2_temporal_prior",
            "training_dataset": str(TRAINING_DATASET),
            "training_dataset_sha256": sha256(TRAINING_DATASET),
            "Subj05_run81": {"prior": old_prior, "metrics": old_metrics},
            "Subj11_run99": {"prior": new_prior, "metrics": new_metrics},
            "macro_f1_change": new_metrics["macro_f1"] - old_metrics["macro_f1"],
        },
        "interpretation": {
            "landing_event": (
                "At +/-2 frames the event F1 changes only slightly; the landing event "
                "detector does not show a collapse on the second video."
            ),
            "four_stage": (
                "The same speed-conditioned temporal prior loses substantial framewise "
                "Macro F1 on Subj11/run99. The target speed 2.75 m/s is above the frozen "
                "training speeds and maps to the nearest 2.40 m/s prior."
            ),
        },
        "scope_note": (
            "Subj11/run99 was not used by the landing stack, but was previously part of "
            "the separate historical locked four-stage evaluation. This is a diagnostic "
            "same-method comparison, not a newly untouched project-wide benchmark."
        ),
        "outputs": {
            "new_four_stage_predictions": str(new_prediction_path),
            "landing_comparison_csv": str(landing_csv),
            "four_stage_comparison_csv": str(four_stage_csv),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "landing_f1_within_2": {
                    trial: values["within_2_frames"]["event_f1"]
                    for trial, values in landing.items()
                },
                "four_stage_macro_f1": {
                    "Subj05_run81": old_metrics["macro_f1"],
                    "Subj11_run99": new_metrics["macro_f1"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
