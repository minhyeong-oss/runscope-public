from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


PAIRS = {
    "left_heel": ("LCAL",),
    "right_heel": ("RCAL",),
    "left_foot_index": ("L1MT", "L5MT"),
    "right_foot_index": ("R1MT", "R5MT"),
}


def interpolate_finite(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        raise RuntimeError("Insufficient finite marker samples")
    return np.interp(time, time[finite], values[finite])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MediaPipe feet against Twente 3D markers")
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("twente_mat", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--clip-offset-s", type=float, default=10.0)
    parser.add_argument("--pass-correlation", type=float, default=0.90)
    args = parser.parse_args()

    video = pd.read_csv(args.landmarks_csv)
    datastr = loadmat(
        args.twente_mat, simplify_cells=True, variable_names=["Datastr"]
    )["Datastr"]
    marker = datastr["Marker"]
    labels = list(marker["DataLabel"])
    data = np.asarray(marker["MarkerData"], dtype=float)
    marker_rate = float(marker["FrameRate"])
    marker_time = np.arange(data.shape[2], dtype=float) / marker_rate
    video_time = args.clip_offset_s + video["time_s"].to_numpy(dtype=float)
    midhip = (
        video["left_hip_x"].interpolate(limit_direction="both")
        + video["right_hip_x"].interpolate(limit_direction="both")
    ).to_numpy(dtype=float) / 2.0

    video_signals = {}
    marker_signals = {}
    for video_name, marker_names in PAIRS.items():
        video_signals[video_name] = video[f"{video_name}_x"].interpolate(
            limit_direction="both"
        ).to_numpy(dtype=float) - midhip
        components = [data[labels.index(name), 0] for name in marker_names]
        foot_marker = np.nanmean(np.vstack(components), axis=0)
        sacrum = data[labels.index("SACR"), 0]
        marker_signals[video_name] = interpolate_finite(marker_time, foot_marker - sacrum)

    candidates = []
    for offset in np.arange(-0.100, 0.1001, 0.0005):
        correlations = {}
        for name in PAIRS:
            aligned = np.interp(video_time + offset, marker_time, marker_signals[name])
            value = abs(float(np.corrcoef(video_signals[name], aligned)[0, 1]))
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite trajectory correlation for {name}")
            correlations[name] = value
        candidates.append((float(np.mean(list(correlations.values()))), float(offset), correlations))
    mean_correlation, offset, correlations = max(candidates, key=lambda item: item[0])
    heel_mean = float(np.mean([correlations["left_heel"], correlations["right_heel"]]))
    toe_mean = float(np.mean([
        correlations["left_foot_index"], correlations["right_foot_index"]
    ]))
    report = {
        "experiment": "pose_foot_keypoints_v1",
        "participant": "Subj05",
        "trial": str(datastr["Info"]["Trial"]),
        "validation_scope": "development-only pilot; locked test subjects not used",
        "sync_offset_s": offset,
        "trajectory_absolute_pearson": correlations,
        "heel_mean_absolute_pearson": heel_mean,
        "foot_index_mean_absolute_pearson": toe_mean,
        "all_foot_keypoints_mean_absolute_pearson": mean_correlation,
        "pass_threshold": args.pass_correlation,
        "passes_to_next_experiment": mean_correlation >= args.pass_correlation,
        "notes": [
            "Heel is compared with CAL marker relative to SACR.",
            "Foot index is compared with the mean of 1MT and 5MT relative to SACR.",
            "Absolute correlation evaluates trajectory shape, not metric position error.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
