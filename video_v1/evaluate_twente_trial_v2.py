from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from evaluate_twente_force_f1 import force_strides, ground_truth_label
from evaluate_twente_temporal_prior_v2 import metrics, training_thresholds


STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate temporal-prior v2 on another Twente trial")
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("strides_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--clip-offset-s", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=33.0)
    parser.add_argument("--sync-offset-s", type=float, required=True)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=root / "model_v1/data/runscope_phase_frames_v1.npz",
    )
    args = parser.parse_args()

    datastr = loadmat(args.mat_file, simplify_cells=True, variable_names=["Datastr"])["Datastr"]
    force_rate = float(datastr["Force"]["FrameRate"])
    body_mass = float(datastr["Info"]["subjMass"])
    trial = str(datastr["Info"]["Trial"])
    force_events = {
        side: force_strides(
            datastr["Force"][side.capitalize() + "ForceData"],
            force_rate,
            body_mass,
        )
        for side in ("left", "right")
    }
    nearest_speed, boundaries, training_count = training_thresholds(
        args.training_dataset, args.speed
    )
    landing_end, propulsion_start, recovery_start = boundaries

    strides = pd.read_csv(args.strides_csv)
    strides = strides[
        strides["accepted_stride"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    rows = []
    for stride in strides.itertuples(index=False):
        side = str(stride.tracked_side)
        start = int(stride.start_frame_inclusive)
        end = int(stride.end_frame_exclusive)
        duration_s = (end - start) / args.fps
        sample_count = int(round(duration_s * 100.0))
        for sample_index in range(sample_count):
            local_time = sample_index / 100.0
            fraction = min(0.999999, local_time / duration_s)
            if fraction < landing_end:
                prediction = 1
            elif fraction < propulsion_start:
                prediction = 2
            elif fraction < recovery_start:
                prediction = 3
            else:
                prediction = 4
            video_time = args.clip_offset_s + start / args.fps + local_time
            sensor_time = video_time + args.sync_offset_s
            truth = ground_truth_label(sensor_time, force_events[side])
            if truth is None:
                continue
            rows.append({
                "stride_id": str(stride.stride_id),
                "anatomical_side": side,
                "source_frame_nearest": start + int(round(local_time * args.fps)),
                "video_time_s": video_time,
                "sensor_time_s": sensor_time,
                "cycle_fraction": fraction,
                "v2_predicted_label": prediction,
                "v2_predicted_stage": STAGE_NAMES[prediction],
                "ground_truth_label": int(truth),
                "ground_truth_stage": STAGE_NAMES[int(truth)],
                "v2_correct": prediction == truth,
            })
    comparison = pd.DataFrame(rows)
    result = metrics(comparison, "v2_predicted_label")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / f"{trial}_v2_force_comparison.csv"
    confusion_path = args.output_dir / f"{trial}_v2_confusion_matrix.csv"
    report_path = args.output_dir / f"{trial}_v2_metrics.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        result["confusion_matrix"],
        index=[f"true_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
        columns=[f"pred_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
    ).to_csv(confusion_path, encoding="utf-8-sig")
    report = {
        "model_version": "video_phase_decoder_v2_temporal_prior",
        "participant": "Subj05",
        "trial": trial,
        "target_speed_m_s": args.speed,
        "nearest_training_speed_m_s": nearest_speed,
        "outside_training_speed_range": not (1.8 <= args.speed <= 2.4),
        "training_strides_used_for_thresholds": training_count,
        "threshold_cycle_fractions": {
            "landing_end": landing_end,
            "propulsion_start": propulsion_start,
            "recovery_start": recovery_start,
        },
        "sync_offset_s": args.sync_offset_s,
        "sync_method": "video ankle-x versus synchronized 3D lateral-malleolus-x trajectory correlation",
        "accepted_video_strides": int(len(strides)),
        "force_strides_available": {side: len(events) for side, events in force_events.items()},
        "metrics": result,
        "target_macro_f1": 0.85,
        "meets_target": result["macro_f1"] >= 0.85,
        "test_label_leakage_control": (
            "Thresholds use Reznick train+validation only. This trial's force labels are evaluation-only."
        ),
        "outputs": {
            "comparison_csv": str(comparison_path),
            "confusion_matrix_csv": str(confusion_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
