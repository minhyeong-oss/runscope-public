from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_visual_contact_v1"
DATA_ROOT = ROOT / "data/landing_video_new_v1/extracted"
POSE_ROOT = DATA_ROOT / "03_markerless_data"
MARKER_ROOT = DATA_ROOT / "02_markerbased_data"

POSE_RATE = 30.0
MARKER_RATE = 120.0
SYNC_SEARCH_FRAMES = 150
SYNC_QUALITY_MIN = 0.60
CONTACT_HEIGHT_M = 0.025
SEED = 20261109

TRAIN_SUBJECTS = ["02", "04", "05", "08", "11", "13", "19", "21"]
VALIDATION_SUBJECTS = ["15", "16"]
LOCKED_TEST_SUBJECTS = ["07", "14"]
EXCLUDED_SYNC_SUBJECTS = ["06"]
EXPECTED_PAIRED = sorted(
    TRAIN_SUBJECTS + VALIDATION_SUBJECTS + LOCKED_TEST_SUBJECTS + EXCLUDED_SYNC_SUBJECTS
)

SELECTED_JOINTS = [5, 6, 11, 12, 13, 14, 15, 16]
JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
PER_JOINT_FEATURES = ["x", "y", "vx", "vy", "ax", "ay", "confidence"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_trc(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = lines[2].split("\t")
    rate = float(metadata[0])
    marker_count = int(metadata[3])
    marker_names = [item for item in lines[3].split("\t")[2:] if item]
    if len(marker_names) != marker_count:
        raise RuntimeError(f"TRC marker count mismatch: {path}")
    data = np.genfromtxt(
        path,
        delimiter="\t",
        skip_header=5,
        usecols=range(2 + marker_count * 3),
    )
    time = data[:, 1]
    markers = {
        name: data[:, 2 + index * 3:2 + (index + 1) * 3]
        for index, name in enumerate(marker_names)
    }
    return time, markers, rate


def interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = values.astype(np.float64, copy=True)
    indices = np.arange(len(output))
    good = valid & np.isfinite(output)
    if good.sum() < 2:
        return np.nan_to_num(output)
    output[~good] = np.interp(indices[~good], indices[good], output[good])
    return output


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(
        path,
        usecols=["frame", "person", "keypoint", "x", "y", "confidence"],
    )
    table = table[table["person"] == 0].copy()
    frames = int(table["frame"].max()) + 1
    coordinates = np.zeros((frames, 17, 2), dtype=np.float32)
    confidence = np.zeros((frames, 17), dtype=np.float32)
    for keypoint in range(17):
        rows = table[table["keypoint"] == keypoint].sort_values("frame")
        source_frames = rows["frame"].to_numpy(dtype=int)
        coordinates[source_frames, keypoint, 0] = rows["x"].to_numpy(dtype=np.float32)
        coordinates[source_frames, keypoint, 1] = rows["y"].to_numpy(dtype=np.float32)
        confidence[source_frames, keypoint] = rows["confidence"].to_numpy(dtype=np.float32)
        for axis in range(2):
            valid = confidence[:, keypoint] > 0.05
            coordinates[:, keypoint, axis] = interpolate(coordinates[:, keypoint, axis], valid)
    return coordinates, confidence


def highpass_standardize(values: np.ndarray) -> np.ndarray:
    values = interpolate(values, np.isfinite(values))
    values = savgol_filter(values, 15, 2, mode="interp")
    trend = pd.Series(values).rolling(61, center=True, min_periods=1).mean().to_numpy()
    values = values - trend
    return (values - values.mean()) / max(values.std(), 1e-6)


def overlap_correlation(pose: np.ndarray, marker: np.ndarray, lag: int) -> float:
    if lag >= 0:
        length = min(len(pose), len(marker) - lag)
        if length <= 0:
            return -1.0
        pose_slice, marker_slice = pose[:length], marker[lag:lag + length]
    else:
        length = min(len(marker), len(pose) + lag)
        if length <= 0:
            return -1.0
        pose_slice, marker_slice = pose[-lag:-lag + length], marker[:length]
    return float(np.mean(pose_slice * marker_slice))


def estimate_alignment(
    coordinates: np.ndarray,
    markers: dict[str, np.ndarray],
) -> tuple[int, bool, float, list[float]]:
    pose_signals = [
        highpass_standardize(coordinates[:, keypoint, 1]) for keypoint in (15, 16)
    ]
    marker_signals = [
        highpass_standardize(-markers[f"{side}ANK"][:, 1][::4]) for side in ("L", "R")
    ]
    best: tuple[float, int, bool, list[float]] | None = None
    for swapped in (False, True):
        ordered = marker_signals[::-1] if swapped else marker_signals
        for lag in range(-SYNC_SEARCH_FRAMES, SYNC_SEARCH_FRAMES + 1):
            feet = [
                overlap_correlation(pose, marker, lag)
                for pose, marker in zip(pose_signals, ordered)
            ]
            score = float(np.mean(feet))
            if best is None or score > best[0]:
                best = (score, lag, swapped, feet)
    assert best is not None
    return best[1], best[2], best[0], best[3]


def contact_onsets(
    markers: dict[str, np.ndarray],
    side: str,
    rate: float,
) -> tuple[np.ndarray, float]:
    height = np.nanmin(
        np.column_stack([markers[f"{side}HEE"][:, 1], markers[f"{side}TOE"][:, 1]]),
        axis=1,
    )
    height = interpolate(height, np.isfinite(height))
    height = pd.Series(height).rolling(7, center=True, min_periods=1).mean().to_numpy()
    floor = float(np.percentile(height, 2))
    low = height < floor + CONTACT_HEIGHT_M
    candidates = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    accepted: list[int] = []
    for onset in candidates:
        if onset < 12 or onset + 18 >= len(low):
            continue
        if low[onset - 8:onset].mean() > 0.25:
            continue
        if low[onset:onset + 15].mean() < 0.70:
            continue
        if accepted and onset - accepted[-1] < int(0.35 * rate):
            continue
        accepted.append(int(onset))
    return np.asarray(accepted, dtype=np.int64), floor


def extract_features(
    coordinates: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    mid_hip = (coordinates[:, 11] + coordinates[:, 12]) / 2.0
    mid_shoulder = (coordinates[:, 5] + coordinates[:, 6]) / 2.0
    scale = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    fallback = np.linalg.norm(coordinates[:, 5] - coordinates[:, 6], axis=1)
    scale = np.where(scale > 5.0, scale, fallback)
    scale = interpolate(scale, scale > 5.0)
    scale = np.clip(scale, np.percentile(scale, 5), np.percentile(scale, 95))

    selected = coordinates[:, SELECTED_JOINTS]
    normalized = (selected - mid_hip[:, None, :]) / scale[:, None, None]
    for joint in range(normalized.shape[1]):
        for axis in range(2):
            normalized[:, joint, axis] = savgol_filter(
                normalized[:, joint, axis], 5, 2, mode="interp"
            )
    velocity = np.gradient(normalized, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    selected_confidence = confidence[:, SELECTED_JOINTS]

    blocks = []
    names = []
    for index, joint_name in enumerate(JOINT_NAMES):
        block = np.column_stack([
            normalized[:, index, 0], normalized[:, index, 1],
            velocity[:, index, 0], velocity[:, index, 1],
            acceleration[:, index, 0], acceleration[:, index, 1],
            selected_confidence[:, index],
        ])
        blocks.append(block)
        names.extend([f"{joint_name}_{feature}" for feature in PER_JOINT_FEATURES])
    return np.column_stack(blocks).astype(np.float32), names


def process_subject(subject: str, output_dir: Path) -> dict:
    pose_path = POSE_ROOT / subject / "side/Walk_8km.csv"
    marker_path = MARKER_ROOT / subject / "Walk_8km.trc"
    coordinates, confidence = load_pose(pose_path)
    marker_time, markers, marker_rate = parse_trc(marker_path)
    if abs(marker_rate - MARKER_RATE) > 1e-6:
        raise RuntimeError(f"Unexpected marker rate for {subject}: {marker_rate}")

    lag, swapped, sync_correlation, foot_correlations = estimate_alignment(coordinates, markers)
    events = []
    side_counts = {}
    floor_height = {}
    for side in ("L", "R"):
        onsets, floor = contact_onsets(markers, side, marker_rate)
        pose_frames = np.rint(marker_time[onsets] * POSE_RATE).astype(int) - lag
        pose_frames = pose_frames[(pose_frames >= 0) & (pose_frames < len(coordinates))]
        events.extend(pose_frames.tolist())
        side_counts[side] = int(len(pose_frames))
        floor_height[side] = floor
    events = np.asarray(sorted(set(events)), dtype=np.int64)
    labels = np.zeros(len(coordinates), dtype=np.uint8)
    labels[events] = 1
    features, feature_names = extract_features(coordinates, confidence)
    frame_quality = confidence[:, SELECTED_JOINTS].mean(axis=1).astype(np.float32)

    output_path = output_dir / f"subject_{subject}.npz"
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        events=events,
        frame_quality=frame_quality,
        feature_names=np.asarray(feature_names),
        subject=np.asarray(subject),
        pose_rate=np.asarray(POSE_RATE),
        alignment_lag_frames=np.asarray(lag),
        marker_sides_swapped=np.asarray(swapped),
    )
    return {
        "subject": subject,
        "pose_frames": int(len(coordinates)),
        "pose_duration_s": float(len(coordinates) / POSE_RATE),
        "marker_frames": int(len(marker_time)),
        "marker_duration_s": float(marker_time[-1]),
        "alignment_lag_frames": int(lag),
        "alignment_lag_s": float(lag / POSE_RATE),
        "marker_sides_swapped_for_alignment": bool(swapped),
        "sync_correlation": float(sync_correlation),
        "foot_correlations": [float(value) for value in foot_correlations],
        "contact_events": int(len(events)),
        "side_contact_events": side_counts,
        "floor_height_m": floor_height,
        "mean_frame_quality": float(frame_quality.mean()),
        "artifact": str(output_path),
        "artifact_sha256": sha256(output_path),
    }


def main() -> None:
    marker_subjects = sorted(path.parent.name for path in MARKER_ROOT.glob("*/Walk_8km.trc"))
    pose_subjects = sorted(path.parent.parent.name for path in POSE_ROOT.glob("*/side/Walk_8km.csv"))
    paired = sorted(set(marker_subjects) & set(pose_subjects))
    if paired != EXPECTED_PAIRED:
        raise RuntimeError(f"Unexpected paired subject set: {paired}")

    artifact_dir = EXPERIMENT / "artifacts/sequences"
    report_dir = EXPERIMENT / "reports"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [process_subject(subject, artifact_dir) for subject in paired]
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "subject_data_quality.csv", index=False, encoding="utf-8-sig")

    observed_excluded = sorted(
        table.loc[table["sync_correlation"] < SYNC_QUALITY_MIN, "subject"].astype(str).str.zfill(2)
    )
    if observed_excluded != EXCLUDED_SYNC_SUBJECTS:
        raise RuntimeError(
            f"Fixed sync QA expected {EXCLUDED_SYNC_SUBJECTS}, observed {observed_excluded}"
        )
    eligible = sorted(set(paired) - set(EXCLUDED_SYNC_SUBJECTS))
    assigned = sorted(TRAIN_SUBJECTS + VALIDATION_SUBJECTS + LOCKED_TEST_SUBJECTS)
    if eligible != assigned or set(TRAIN_SUBJECTS) & set(LOCKED_TEST_SUBJECTS):
        raise RuntimeError("Frozen split does not match eligible subjects")

    marker_zip = ROOT / "data/landing_video_new_v1/zenodo_19720803/02_markerbased_data.zip"
    pose_zip = ROOT / "data/landing_video_new_v1/zenodo_19720803/03_markerless_data.zip"
    manifest = {
        "experiment": "landing_visual_contact_v1",
        "source_doi": "https://doi.org/10.5281/zenodo.19720803",
        "seed_recorded_before_training": SEED,
        "target_definition": "kinematic visual-contact onset from synchronized heel/toe markers",
        "not_force_plate_ground_truth": True,
        "raw_movie_files_available_in_public_archive": False,
        "pose_input": "30 fps side-view video-derived YOLO COCO-17 keypoints",
        "marker_supervision": "120 Hz OptiTrack TRC heel/toe vertical position",
        "paired_subjects": paired,
        "eligible_subjects": eligible,
        "excluded_before_training": {
            "subjects": EXCLUDED_SYNC_SUBJECTS,
            "rule": f"sync_correlation < {SYNC_QUALITY_MIN:.2f}",
        },
        "split": {
            "train": TRAIN_SUBJECTS,
            "validation": VALIDATION_SUBJECTS,
            "locked_test": LOCKED_TEST_SUBJECTS,
            "development_to_test": "10:2 participants",
        },
        "test_policy": "No threshold tuning, architecture changes, or retraining after test access.",
        "contact_rule": {
            "foot_height": "minimum of heel and toe vertical marker positions",
            "floor_estimator": "subject/foot 2nd percentile",
            "height_threshold_m": CONTACT_HEIGHT_M,
            "minimum_precontact_air_frames_at_120hz": 8,
            "minimum_postcontact_support_ratio": 0.70,
        },
        "raw_archives": {
            "marker_zip_sha256": sha256(marker_zip),
            "pose_zip_sha256": sha256(pose_zip),
        },
        "subjects": rows,
    }
    (EXPERIMENT / "artifacts/DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "paired": len(paired),
        "eligible": len(eligible),
        "excluded": EXCLUDED_SYNC_SUBJECTS,
        "train": TRAIN_SUBJECTS,
        "validation": VALIDATION_SUBJECTS,
        "locked_test": LOCKED_TEST_SUBJECTS,
        "events": int(table.loc[table.subject.isin(eligible), "contact_events"].sum()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
