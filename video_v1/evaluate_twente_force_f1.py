from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import classification_report, confusion_matrix, f1_score


STAGE_NAMES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def smooth(values: np.ndarray, window: int = 11) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def phase23_boundary(ap_force: np.ndarray, start: int, end: int) -> tuple[int, float]:
    """Find braking-to-propulsion boundary after orienting braking positive."""
    y = smooth(ap_force, 11)
    lo = start + max(2, int(round((end - start) * 0.12)))
    hi = end - max(2, int(round((end - start) * 0.12)))
    crossings = [i for i in range(lo, hi) if y[i] >= 0 and y[i + 1] < 0]
    candidates = crossings or list(range(lo, hi + 1))
    best: tuple[float, int, float] | None = None
    for index in candidates:
        pre = y[start : index + 1]
        post = y[index + 1 : end + 1]
        pre_ratio = float(np.mean(pre >= 0))
        post_ratio = float(np.mean(post <= 0))
        balance = min(len(pre), len(post)) / max(len(pre), len(post))
        score = 0.45 * pre_ratio + 0.45 * post_ratio + 0.10 * balance
        consistency = (pre_ratio + post_ratio) / 2.0
        if best is None or score > best[0]:
            best = (score, index, consistency)
    if best is None:
        raise RuntimeError("Could not locate braking-to-propulsion boundary")
    return best[1], best[2]


def force_strides(force: np.ndarray, frame_rate: float, body_mass_kg: float) -> list[dict]:
    lowpass = butter(4, 30.0, fs=frame_rate, output="sos")
    ap = sosfiltfilt(lowpass, force[:, 0])
    vertical = sosfiltfilt(lowpass, force[:, 1])
    contact = vertical > 20.0
    starts = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
    ends = np.flatnonzero(~contact & np.r_[False, contact[:-1]])
    valid_contacts: list[tuple[int, int]] = []
    minimum_peak_force = body_mass_kg * 9.81 * 0.50
    for start in starts:
        following = ends[ends > start]
        if not len(following):
            continue
        end = int(following[0])
        duration = (end - start) / frame_rate
        peak = float(np.max(vertical[start:end]))
        if 0.15 <= duration <= 0.50 and peak >= minimum_peak_force:
            valid_contacts.append((int(start), end))

    strides: list[dict] = []
    # Twente AP force is negative during braking and positive during propulsion,
    # opposite to the Reznick convention used to train RunScope.
    oriented_ap = -ap
    for (initial_contact, toe_off), (next_contact, _) in zip(
        valid_contacts[:-1], valid_contacts[1:]
    ):
        boundary, consistency = phase23_boundary(
            oriented_ap, initial_contact, toe_off - 1
        )
        strides.append(
            {
                "initial_contact_s": initial_contact / frame_rate,
                "landing_end_s": initial_contact / frame_rate + 0.030,
                "propulsion_start_s": boundary / frame_rate,
                "toe_off_s": toe_off / frame_rate,
                "next_contact_s": next_contact / frame_rate,
                "ap_sign_consistency": consistency,
            }
        )
    return strides


def ground_truth_label(time_s: float, strides: list[dict]) -> int | None:
    for stride in strides:
        if stride["initial_contact_s"] <= time_s < stride["next_contact_s"]:
            if time_s < stride["landing_end_s"]:
                return 1
            if time_s < stride["propulsion_start_s"]:
                return 2
            if time_s < stride["toe_off_s"]:
                return 3
            return 4
    return None


def calculate_metrics(table: pd.DataFrame) -> dict:
    labels = [1, 2, 3, 4]
    truth = table["ground_truth_label"].to_numpy(dtype=int)
    prediction = table["predicted_label"].to_numpy(dtype=int)
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
            name: {
                key: float(report[name][key])
                for key in ("precision", "recall", "f1-score")
            }
            for name in [STAGE_NAMES[label] for label in labels]
        },
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def build_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
    source_stem: str,
    clip_offset_s: float,
    source_fps: float,
    sync_offset_s: float,
    force_events: dict[str, list[dict]],
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, stride in summary.iterrows():
        tracked_side = str(stride["tracked_side_coco"])
        anatomical_side = str(stride["anatomical_side_in_this_clip"])
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        feature_path = output_dir / (
            f"{source_stem}_{tracked_side}_stride_{start}_{end}"
            "_features_and_prediction.csv"
        )
        features = pd.read_csv(feature_path)
        for _, sample in features.iterrows():
            video_time = clip_offset_s + start / source_fps + float(sample["time_s"])
            sensor_time = video_time + sync_offset_s
            truth = ground_truth_label(sensor_time, force_events[anatomical_side])
            if truth is None:
                continue
            rows.append(
                {
                    "stride_id": stride["stride_id"],
                    "tracked_side": tracked_side,
                    "anatomical_side": anatomical_side,
                    "video_time_s": video_time,
                    "sensor_time_s": sensor_time,
                    "predicted_label": int(sample["phase_label"]),
                    "predicted_stage": STAGE_NAMES[int(sample["phase_label"])],
                    "ground_truth_label": int(truth),
                    "ground_truth_stage": STAGE_NAMES[int(truth)],
                    "predicted_probability": float(sample["predicted_probability"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RunScope video phases against Twente force data")
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("phase_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--clip-offset-s", type=float, required=True)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--sync-offset-s", type=float, default=0.0285)
    args = parser.parse_args()

    datastr = loadmat(
        args.mat_file, simplify_cells=True, variable_names=["Datastr"]
    )["Datastr"]
    force_rate = float(datastr["Force"]["FrameRate"])
    body_mass = float(datastr["Info"]["subjMass"])
    force_events = {
        side: force_strides(
            datastr["Force"][side.capitalize() + "ForceData"],
            force_rate,
            body_mass,
        )
        for side in ("left", "right")
    }
    summary = pd.read_csv(args.phase_summary_csv)
    source_stem = args.phase_summary_csv.name.replace("_auto_phase_summary.csv", "")

    calibrated = build_comparison(
        summary,
        args.output_dir,
        source_stem,
        args.clip_offset_s,
        args.source_fps,
        args.sync_offset_s,
        force_events,
    )
    uncalibrated = build_comparison(
        summary,
        args.output_dir,
        source_stem,
        args.clip_offset_s,
        args.source_fps,
        0.0,
        force_events,
    )
    calibrated_metrics = calculate_metrics(calibrated)
    uncalibrated_metrics = calculate_metrics(uncalibrated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / f"{source_stem}_twente_force_comparison.csv"
    confusion_path = args.output_dir / f"{source_stem}_twente_force_confusion_matrix.csv"
    report_path = args.output_dir / f"{source_stem}_twente_force_f1.json"
    calibrated.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        calibrated_metrics["confusion_matrix"],
        index=[f"true_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
        columns=[f"pred_{STAGE_NAMES[label]}" for label in (1, 2, 3, 4)],
    ).to_csv(confusion_path, encoding="utf-8-sig")

    report = {
        "status": "force_ground_truth_evaluation",
        "participant": "Subj05",
        "trial": "run_81",
        "video_clip_offset_s": args.clip_offset_s,
        "source_fps": args.source_fps,
        "force_frame_rate_hz": force_rate,
        "sync_offset_s": args.sync_offset_s,
        "sync_offset_method": "common offset maximizing mean correlation between video ankle-x and synchronized 3D lateral-malleolus-x trajectories",
        "sync_mean_absolute_correlation": 0.990750881774537,
        "label_definition": {
            "landing": "first 30 ms after vertical GRF exceeds 20 N",
            "braking": "end of landing window to AP GRF zero crossing",
            "propulsion": "AP GRF zero crossing to vertical GRF toe-off",
            "recovery": "toe-off to next same-foot initial contact",
        },
        "force_strides_available": {
            side: len(events) for side, events in force_events.items()
        },
        "calibrated_metrics": calibrated_metrics,
        "zero_offset_sensitivity_metrics": uncalibrated_metrics,
        "outputs": {
            "sample_comparison_csv": str(comparison_path),
            "confusion_matrix_csv": str(confusion_path),
        },
        "limitations": [
            "This is one participant and one treadmill speed, not a final generalization score.",
            "The 28.5 ms synchronization offset was estimated from the same video using an independent marker trajectory.",
            "The RunScope video adapter uses monocular 2D proxies while the model was trained on laboratory 3D kinematics.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
