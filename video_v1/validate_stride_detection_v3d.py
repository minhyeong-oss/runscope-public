from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, savgol_filter


def nearest_unique_matches(
    detected: np.ndarray, reference: np.ndarray, tolerance_frames: float
) -> list[tuple[float, float, float]]:
    candidates = sorted(
        (abs(float(d) - float(r)), float(d), float(r))
        for d in detected
        for r in reference
        if abs(float(d) - float(r)) <= tolerance_frames
    )
    used_detected: set[float] = set()
    used_reference: set[float] = set()
    matches: list[tuple[float, float, float]] = []
    for error, detected_frame, reference_frame in candidates:
        if detected_frame in used_detected or reference_frame in used_reference:
            continue
        used_detected.add(detected_frame)
        used_reference.add(reference_frame)
        matches.append((detected_frame, reference_frame, error))
    return sorted(matches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate video heel-strike proxies against synchronized V3D heel kinematics"
    )
    parser.add_argument("events_csv", type=Path)
    parser.add_argument("v3d_mat", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--motion", default="jogging")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--mocap-fps", type=float, default=120.0)
    parser.add_argument("--tolerance-video-frames", type=float, default=6.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    root_candidates = [key for key in loadmat(args.v3d_mat) if not key.startswith("__")]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected one MATLAB root variable, found {root_candidates}")
    dataset = loadmat(args.v3d_mat, simplify_cells=True)[root_candidates[0]]["move"]
    motion_names = [str(name) for name in dataset["motions_list"]]
    motion_index = motion_names.index(args.motion)
    start_30, end_30 = [int(value) for value in dataset["flags30"][motion_index]]
    start_120, end_120 = [int(value) for value in dataset["flags120"][motion_index]]

    marker_names = [str(name) for name in dataset["markerName"]]
    marker_location = np.asarray(dataset["markerLocation"], dtype=float)
    reference_by_anatomical_side: dict[str, np.ndarray] = {}
    for anatomical_side, marker_name in (("left", "LHEE"), ("right", "RHEE")):
        marker_index = marker_names.index(marker_name)
        # MATLAB flags are one-based and inclusive. Vertical position is axis 2.
        heel_vertical = marker_location[start_120 - 1 : end_120, marker_index, 2]
        smoothed = savgol_filter(heel_vertical, 15, 3)
        minima, _ = find_peaks(-smoothed, distance=60, prominence=5)
        reference_by_anatomical_side[anatomical_side] = minima / (
            args.mocap_fps / args.video_fps
        )

    events = pd.read_csv(args.events_csv)
    events = events[events["accepted_event"].astype(str).str.lower().isin(["true", "1"])].copy()
    detected_by_tracked_side = {
        side: events.loc[events["tracked_side"] == side, "frame_index"].to_numpy(dtype=float)
        for side in ("left", "right")
    }

    mappings = {
        "direct": {"left": "left", "right": "right"},
        "swapped": {"left": "right", "right": "left"},
    }
    mapping_results: dict[str, dict] = {}
    all_mapping_matches: dict[str, list[dict]] = {}
    for mapping_name, mapping in mappings.items():
        match_rows: list[dict] = []
        reference_total = 0
        for tracked_side, anatomical_side in mapping.items():
            reference = reference_by_anatomical_side[anatomical_side]
            reference_total += len(reference)
            for detected_frame, reference_frame, error in nearest_unique_matches(
                detected_by_tracked_side[tracked_side],
                reference,
                args.tolerance_video_frames,
            ):
                match_rows.append({
                    "mapping": mapping_name,
                    "tracked_side": tracked_side,
                    "anatomical_side": anatomical_side,
                    "detected_video_frame": detected_frame,
                    "reference_video_frame": reference_frame,
                    "absolute_error_frames": error,
                    "absolute_error_ms": error / args.video_fps * 1000.0,
                })
        errors = [row["absolute_error_frames"] for row in match_rows]
        mapping_results[mapping_name] = {
            "matches": len(match_rows),
            "detected_events": int(sum(len(values) for values in detected_by_tracked_side.values())),
            "reference_events": int(reference_total),
            "mae_video_frames": float(np.mean(errors)) if errors else None,
            "mae_ms": float(np.mean(errors) / args.video_fps * 1000.0) if errors else None,
        }
        all_mapping_matches[mapping_name] = match_rows

    chosen_mapping = min(
        mapping_results,
        key=lambda name: (
            -mapping_results[name]["matches"],
            mapping_results[name]["mae_video_frames"]
            if mapping_results[name]["mae_video_frames"] is not None
            else float("inf"),
        ),
    )
    chosen_matches = pd.DataFrame(all_mapping_matches[chosen_mapping])
    stem = args.events_csv.stem.replace("_auto_heel_strikes", "")
    matches_path = args.output_dir / f"{stem}_v3d_event_validation.csv"
    report_path = args.output_dir / f"{stem}_v3d_event_validation.json"
    chosen_matches.to_csv(matches_path, index=False, encoding="utf-8-sig")

    report = {
        "events_csv": str(args.events_csv),
        "v3d_mat": str(args.v3d_mat),
        "motion": args.motion,
        "synchronization": {
            "video_30hz_flags_one_based_inclusive": [start_30, end_30],
            "mocap_120hz_flags_one_based_inclusive": [start_120, end_120],
        },
        "reference_heel_minima_video_frames": {
            side: [round(float(value), 2) for value in values]
            for side, values in reference_by_anatomical_side.items()
        },
        "mapping_comparison": mapping_results,
        "chosen_mapping": chosen_mapping,
        "interpretation": (
            "COCO left/right labels are swapped relative to anatomical V3D labels in this side-view clip."
            if chosen_mapping == "swapped"
            else "COCO left/right labels match anatomical V3D labels in this clip."
        ),
        "scope_warning": (
            "This validates heel-strike/stride event timing against heel kinematics; "
            "it does not validate the four force-defined running phases."
        ),
        "outputs": {"matches_csv": str(matches_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(chosen_matches.to_string(index=False))


if __name__ == "__main__":
    main()
