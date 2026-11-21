from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def training_thresholds(dataset_path: Path, target_speed: float) -> tuple[float, tuple[float, float, float], int]:
    """Return medians learned only from Reznick train+validation strides."""
    dataset = np.load(dataset_path)
    speeds = np.asarray(dataset["event_speeds"], dtype=float)
    splits = dataset["event_splits"].astype(str)
    available = np.unique(speeds[np.isin(splits, ["train", "validation"])])
    selected_speed = float(available[np.argmin(np.abs(available - target_speed))])
    event_indices = np.flatnonzero(
        np.isin(splits, ["train", "validation"])
        & np.isclose(speeds, selected_speed)
    )
    fractions = []
    for event_index in event_indices:
        indices = np.flatnonzero(dataset["sample_event_index"] == event_index)
        labels = dataset["y"][indices]
        length = len(labels)
        fractions.append([
            int(np.flatnonzero(labels == 2)[0]) / length,
            int(np.flatnonzero(labels == 3)[0]) / length,
            int(np.flatnonzero(labels == 4)[0]) / length,
        ])
    medians = tuple(float(value) for value in np.median(fractions, axis=0))
    return selected_speed, medians, len(event_indices)


def metrics(table: pd.DataFrame, prediction_column: str) -> dict:
    labels = [1, 2, 3, 4]
    truth = table["ground_truth_label"].to_numpy(dtype=int)
    prediction = table[prediction_column].to_numpy(dtype=int)
    report = classification_report(
        truth,
        prediction,
        labels=labels,
        target_names=[STAGE_NAMES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": int(len(table)),
        "accuracy": float(np.mean(truth == prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro")),
        "weighted_f1": float(f1_score(truth, prediction, labels=labels, average="weighted")),
        "per_stage": {
            STAGE_NAMES[label]: {
                key: float(report[STAGE_NAMES[label]][key])
                for key in ("precision", "recall", "f1-score")
            }
            for label in labels
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate the training-derived temporal-prior v2 decoder")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=root / "video_v1/test_outputs/twente_subj05_run81_pilot/Subj05_run81_side_10s_to_22s_twente_force_comparison.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "video_v1/test_outputs/twente_subj05_run81_pilot/Subj05_run81_side_10s_to_22s_auto_phase_summary.csv",
    )
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=root / "model_v1/data/runscope_phase_frames_v1.npz",
    )
    parser.add_argument("--speed", type=float, default=2.25)
    parser.add_argument("--clip-offset-s", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=33.0)
    parser.add_argument("--target-macro-f1", type=float, default=0.85)
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison)
    summary = pd.read_csv(args.summary).set_index("stride_id")
    selected_speed, thresholds, training_stride_count = training_thresholds(
        args.training_dataset, args.speed
    )
    landing_end, propulsion_start, recovery_start = thresholds

    predictions = []
    fractions = []
    for row in comparison.itertuples(index=False):
        stride = summary.loc[row.stride_id]
        stride_start_s = args.clip_offset_s + float(stride.start_frame_inclusive) / args.fps
        duration_s = (
            float(stride.end_frame_exclusive) - float(stride.start_frame_inclusive)
        ) / args.fps
        fraction = float(np.clip((float(row.video_time_s) - stride_start_s) / duration_s, 0.0, 0.999999))
        if fraction < landing_end:
            label = 1
        elif fraction < propulsion_start:
            label = 2
        elif fraction < recovery_start:
            label = 3
        else:
            label = 4
        fractions.append(fraction)
        predictions.append(label)

    comparison["cycle_fraction"] = fractions
    comparison["v2_predicted_label"] = predictions
    comparison["v2_predicted_stage"] = [STAGE_NAMES[label] for label in predictions]
    comparison["v2_correct"] = comparison["v2_predicted_label"].eq(
        comparison["ground_truth_label"]
    )

    original = metrics(comparison, "predicted_label")
    improved = metrics(comparison, "v2_predicted_label")
    output_dir = args.comparison.parent / "temporal_prior_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "twente_temporal_prior_v2_comparison.csv"
    confusion_path = output_dir / "twente_temporal_prior_v2_confusion_matrix.csv"
    report_path = output_dir / "twente_temporal_prior_v2_metrics.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        improved["confusion_matrix"],
        index=[f"true_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
        columns=[f"pred_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
    ).to_csv(confusion_path, encoding="utf-8-sig")

    report = {
        "model_version": "video_phase_decoder_v2_temporal_prior",
        "method": "nearest-speed median stage boundaries from Reznick train+validation only",
        "target_video_speed_m_s": args.speed,
        "nearest_training_speed_m_s": selected_speed,
        "training_strides_used_for_thresholds": training_stride_count,
        "threshold_cycle_fractions": {
            "landing_end": landing_end,
            "propulsion_start": propulsion_start,
            "recovery_start": recovery_start,
        },
        "original_ml_video_adapter": original,
        "temporal_prior_v2": improved,
        "macro_f1_change": improved["macro_f1"] - original["macro_f1"],
        "target_macro_f1": args.target_macro_f1,
        "meets_target": improved["macro_f1"] >= args.target_macro_f1,
        "test_label_leakage_control": (
            "The v2 thresholds were calculated from Reznick train+validation labels; "
            "Twente force labels were used only for final evaluation."
        ),
        "limitations": [
            "This external evaluation still contains one participant, one trial, and one speed.",
            "The temporal prior is robust but less personalized than a well-calibrated kinematic model.",
            "A separate multi-participant Twente holdout is required before claiming generalization.",
        ],
        "outputs": {
            "comparison_csv": str(comparison_path),
            "confusion_matrix_csv": str(confusion_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
