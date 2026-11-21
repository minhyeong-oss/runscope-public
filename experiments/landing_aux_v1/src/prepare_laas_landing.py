from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_aux_v1"
DATA_ROOT = ROOT / "data/landing_aux_v1/extracted/laas/c3d"
VENDOR = ROOT / "data/landing_aux_v1/vendor"
sys.path.insert(0, str(VENDOR))

import ezc3d  # noqa: E402


SEQUENCE_LENGTH = 64
SEQUENCE_DURATION_S = 0.48
LANDING_WINDOW_S = 0.030
FORCE_THRESHOLD_N = 20.0
MIN_FLIGHT_S = 0.100
SUBJECTS = ["PKFC", "PKLP", "PKLT", "PKMT", "PKRY"]

FEATURE_NAMES = [
    "tracked_heel_x", "tracked_heel_y",
    "tracked_foot_index_x", "tracked_foot_index_y",
    "tracked_ankle_x", "tracked_ankle_y",
    "tracked_knee_x", "tracked_knee_y",
    "tracked_hip_x", "tracked_hip_y",
    "tracked_heel_visibility", "tracked_toe_visibility",
    "opposite_heel_x", "opposite_heel_y",
    "opposite_foot_index_x", "opposite_foot_index_y",
    "opposite_ankle_x", "opposite_ankle_y",
    "opposite_knee_x", "opposite_knee_y",
    "opposite_hip_x", "opposite_hip_y",
    "opposite_heel_visibility", "opposite_toe_visibility",
    "trunk_x", "trunk_y",
]


def stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def interpolate(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(positions, dtype=float)
    if finite.sum() == 1:
        return np.full_like(positions, values[finite][0], dtype=float)
    indices = np.arange(len(values), dtype=float)
    filled = np.interp(indices, indices[finite], values[finite])
    return np.interp(positions, indices, filled)


def marker_signal(
    points: np.ndarray,
    residuals: np.ndarray,
    labels: list[str],
    names: tuple[str, ...],
    axis: int,
    positions: np.ndarray,
) -> np.ndarray:
    signals = []
    for name in names:
        index = labels.index(name)
        values = points[axis, index].astype(float).copy()
        values[residuals[index] < 0] = np.nan
        signals.append(interpolate(values, positions))
    return np.nanmean(np.vstack(signals), axis=0)


def marker_visibility(
    residuals: np.ndarray,
    labels: list[str],
    names: tuple[str, ...],
    positions: np.ndarray,
) -> np.ndarray:
    indices = np.arange(residuals.shape[1], dtype=float)
    valid = []
    for name in names:
        signal = (residuals[labels.index(name)] >= 0).astype(float)
        valid.append(np.interp(positions, indices, signal))
    return np.clip(np.nanmean(np.vstack(valid), axis=0), 0.0, 1.0)


def side_markers(side: str) -> dict[str, tuple[str, ...]]:
    return {
        "heel": (f"{side}CAL",),
        "foot_index": (f"{side}MFH1", f"{side}MFH5"),
        "ankle": (f"{side}LM",),
        "knee": (f"{side}MFE", f"{side}LFE"),
        "hip": (f"{side}GT",),
    }


def projected_features(
    points: np.ndarray,
    residuals: np.ndarray,
    labels: list[str],
    positions: np.ndarray,
    tracked_side: str,
) -> np.ndarray:
    opposite_side = "R" if tracked_side == "L" else "L"
    hip_x = np.nanmean(
        np.vstack([
            marker_signal(points, residuals, labels, ("LGT",), 0, positions),
            marker_signal(points, residuals, labels, ("RGT",), 0, positions),
        ]),
        axis=0,
    )
    hip_z = np.nanmean(
        np.vstack([
            marker_signal(points, residuals, labels, ("LGT",), 2, positions),
            marker_signal(points, residuals, labels, ("RGT",), 2, positions),
        ]),
        axis=0,
    )
    shoulder_x = np.nanmean(
        np.vstack([
            marker_signal(points, residuals, labels, ("LA",), 0, positions),
            marker_signal(points, residuals, labels, ("RA",), 0, positions),
        ]),
        axis=0,
    )
    shoulder_z = np.nanmean(
        np.vstack([
            marker_signal(points, residuals, labels, ("LA",), 2, positions),
            marker_signal(points, residuals, labels, ("RA",), 2, positions),
        ]),
        axis=0,
    )
    torso = np.clip(np.hypot(shoulder_x - hip_x, shoulder_z - hip_z), 150.0, None)

    # Parkour actions face the travel direction. Normalizing it matches the existing
    # side-view feature convention where forward is positive screen x.
    span = max(2, int(round(len(positions) * 0.15)))
    travel = float(np.nanmean(hip_x[-span:]) - np.nanmean(hip_x[:span]))
    direction = float(np.sign(travel) or 1.0)

    columns: list[np.ndarray] = []
    for side in (tracked_side, opposite_side):
        markers = side_markers(side)
        for joint in ("heel", "foot_index", "ankle", "knee", "hip"):
            joint_x = marker_signal(points, residuals, labels, markers[joint], 0, positions)
            joint_z = marker_signal(points, residuals, labels, markers[joint], 2, positions)
            columns.append(direction * (joint_x - hip_x) / torso)
            # Image coordinates grow downwards, whereas C3D z grows upwards.
            columns.append((hip_z - joint_z) / torso)
        columns.append(marker_visibility(residuals, labels, markers["heel"], positions))
        columns.append(marker_visibility(residuals, labels, markers["foot_index"], positions))
    columns.append(direction * (shoulder_x - hip_x) / torso)
    columns.append((hip_z - shoulder_z) / torso)
    output = np.column_stack(columns).astype(np.float32)
    if output.shape[1] != len(FEATURE_NAMES) or not np.isfinite(output).all():
        raise RuntimeError("Invalid projected feature matrix")
    return output


def contact_onsets(force: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    vertical = np.maximum(np.asarray(force[2], dtype=float), 0.0)
    smoothing = max(3, int(round(0.005 * rate)))
    smoothed = np.convolve(vertical, np.ones(smoothing) / smoothing, mode="same")
    contact = smoothed > FORCE_THRESHOLD_N
    onsets = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
    return onsets, contact


def choose_foot(
    points: np.ndarray,
    residuals: np.ndarray,
    labels: list[str],
    point_index: float,
    cop_xy: np.ndarray,
) -> tuple[str, float, float]:
    distances = []
    for side in ("L", "R"):
        names = (f"{side}CAL", f"{side}MFH1", f"{side}MFH5")
        x = marker_signal(points, residuals, labels, names, 0, np.asarray([point_index]))[0]
        y = marker_signal(points, residuals, labels, names, 1, np.asarray([point_index]))[0]
        z = marker_signal(points, residuals, labels, names, 2, np.asarray([point_index]))[0]
        distances.append((float(np.hypot(x - cop_xy[0], y - cop_xy[1])), side, float(z)))
    distance, side, height = min(distances)
    return side, distance, height


def load_trial(path: Path) -> dict:
    c3d = ezc3d.c3d(str(path), extract_forceplat_data=True)
    points4 = np.asarray(c3d["data"]["points"], dtype=float)
    labels = [str(label).strip() for label in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    return {
        "points": points4[:3],
        "residuals": points4[3],
        "labels": labels,
        "point_rate": float(c3d["parameters"]["POINT"]["RATE"]["value"][0]),
        "analog_rate": float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0]),
        "platforms": c3d["data"].get("platform", [])[:2],
    }


def sequence_positions(start_s: float, rate: float) -> np.ndarray:
    times = start_s + np.linspace(0.0, SEQUENCE_DURATION_S, SEQUENCE_LENGTH)
    return times * rate


def main() -> None:
    paths = sorted(
        path for path in DATA_ROOT.rglob("*.c3d")
        if path.stem[:2] in {"sv", "kv"}
    )
    sequences: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    subjects: list[str] = []
    trials: list[str] = []
    sides: list[str] = []
    kinds: list[str] = []
    relative_times: list[np.ndarray] = []
    metadata: list[dict] = []

    for path in paths:
        trial = load_trial(path)
        points = trial["points"]
        residuals = trial["residuals"]
        point_rate = trial["point_rate"]
        analog_rate = trial["analog_rate"]
        duration_s = points.shape[2] / point_rate
        subject = path.stem.split("_")[-1]
        if subject not in SUBJECTS:
            raise RuntimeError(f"Unexpected subject {subject}")

        all_onsets_s: list[float] = []
        candidates: list[dict] = []
        for platform_index, platform in enumerate(trial["platforms"]):
            onsets, contact = contact_onsets(platform["force"], analog_rate)
            all_onsets_s.extend(float(onset / analog_rate) for onset in onsets)
            pre_samples = int(round(MIN_FLIGHT_S * analog_rate))
            post_samples = int(round(LANDING_WINDOW_S * analog_rate))
            for onset in onsets:
                if onset < pre_samples:
                    continue
                if contact[onset - pre_samples:onset].any():
                    continue
                if contact[onset:min(len(contact), onset + post_samples)].mean() < 0.80:
                    continue
                onset_s = float(onset / analog_rate)
                if onset_s < 0.10 or duration_s - onset_s < 0.36:
                    continue
                cop = np.asarray(platform["center_of_pressure"], dtype=float)
                cop_end = min(cop.shape[1], onset + post_samples)
                cop_xy = np.nanmedian(cop[:2, onset:cop_end], axis=1)
                side, distance, foot_height = choose_foot(
                    points, residuals, trial["labels"], onset_s * point_rate, cop_xy
                )
                if distance > 250.0 or foot_height > 250.0:
                    continue
                candidates.append({
                    "onset_s": onset_s,
                    "platform": platform_index,
                    "side": side,
                    "distance_mm": distance,
                    "foot_height_mm": foot_height,
                })

        # Two platforms can occasionally report the same foot within a few ms.
        # Keep only the spatially closer observation in that case.
        kept: list[dict] = []
        for candidate in sorted(candidates, key=lambda value: (value["onset_s"], value["distance_mm"])):
            duplicate = next(
                (
                    existing for existing in kept
                    if existing["side"] == candidate["side"]
                    and abs(existing["onset_s"] - candidate["onset_s"]) < 0.050
                ),
                None,
            )
            if duplicate is None:
                kept.append(candidate)
            elif candidate["distance_mm"] < duplicate["distance_mm"]:
                kept.remove(duplicate)
                kept.append(candidate)

        for event_index, candidate in enumerate(sorted(kept, key=lambda value: value["onset_s"])):
            onset_s = float(candidate["onset_s"])
            pre_s = 0.10 + 0.18 * stable_fraction(f"{path.stem}:{event_index}:{candidate['side']}")
            start_s = onset_s - pre_s
            positions = sequence_positions(start_s, point_rate)
            relative = positions / point_rate - onset_s
            features = projected_features(
                points, residuals, trial["labels"], positions, str(candidate["side"])
            )
            target = ((relative >= 0.0) & (relative < LANDING_WINDOW_S)).astype(np.int64)
            sequences.append(features)
            labels_out.append(target)
            subjects.append(subject)
            trials.append(path.stem)
            sides.append(str(candidate["side"]))
            kinds.append("landing")
            relative_times.append(relative.astype(np.float32))
            metadata.append({"trial": path.stem, "subject": subject, **candidate})

        # Add one hard negative from each sufficiently long trial. The chosen window
        # is maximally far from every force onset, so static support and preparation
        # frames do not become false landing labels.
        if duration_s >= SEQUENCE_DURATION_S:
            possible = np.linspace(0.0, duration_s - SEQUENCE_DURATION_S, 101)
            if all_onsets_s:
                distances = np.asarray([
                    min(abs((start + SEQUENCE_DURATION_S / 2.0) - onset) for onset in all_onsets_s)
                    for start in possible
                ])
                negative_start = float(possible[int(np.argmax(distances))])
                far_enough = float(distances.max()) >= 0.26
            else:
                negative_start = float(possible[len(possible) // 2])
                far_enough = True
            if far_enough:
                side = "L" if stable_fraction(path.stem) < 0.5 else "R"
                positions = sequence_positions(negative_start, point_rate)
                sequences.append(projected_features(
                    points, residuals, trial["labels"], positions, side
                ))
                labels_out.append(np.zeros(SEQUENCE_LENGTH, dtype=np.int64))
                subjects.append(subject)
                trials.append(path.stem)
                sides.append(side)
                kinds.append("negative")
                relative_times.append(np.full(SEQUENCE_LENGTH, np.nan, dtype=np.float32))

    if not sequences:
        raise RuntimeError("No usable LAAS sequences found")
    output_dir = EXPERIMENT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "laas_landing_sequences_64.npz"
    x = np.stack(sequences).astype(np.float32)
    y = np.stack(labels_out).astype(np.int64)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        subjects=np.asarray(subjects),
        trials=np.asarray(trials),
        sides=np.asarray(sides),
        kinds=np.asarray(kinds),
        relative_times=np.stack(relative_times),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    summary = {
        "experiment": "landing_aux_v1",
        "source": "LAAS Parkour Dataset (safety vault and kong vault only)",
        "purpose": "auxiliary landing/contact pretraining; not final running validation",
        "output": str(output_path),
        "trials_read": len(paths),
        "sequences": len(sequences),
        "landing_sequences": int(np.sum(np.asarray(kinds) == "landing")),
        "negative_sequences": int(np.sum(np.asarray(kinds) == "negative")),
        "landing_frames": int(y.sum()),
        "non_landing_frames": int((y == 0).sum()),
        "subjects": {subject: int(np.sum(np.asarray(subjects) == subject)) for subject in SUBJECTS},
        "feature_names": FEATURE_NAMES,
        "force_threshold_n": FORCE_THRESHOLD_N,
        "minimum_precontact_no_force_s": MIN_FLIGHT_S,
        "landing_window_s": LANDING_WINDOW_S,
        "sequence_duration_s": SEQUENCE_DURATION_S,
        "locked_twente_test_subjects_accessed": False,
        "event_metadata": metadata,
    }
    (output_dir / "DATASET_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
