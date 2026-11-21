from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_event_v3"
SOURCE = ROOT / "experiments/landing_onset_transfer_v1/artifacts/twente_development_onset_sequences_64.npz"
MATCHED = ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"
TRIAL_QUALITY = ROOT / "experiments/boundary_regression_v1/artifacts/development_trial_quality.csv"
SPLIT = ROOT / "data/twente_sensor/SPLIT_LOCK_v1.json"
PROTOCOL = EXPERIMENT / "config/protocol.json"
EXTERNAL_VIDEO_DIR = Path(
    os.environ.get("RUNSCOPE_EXTERNAL_VIDEO_DIR", ROOT / "private_inputs")
).expanduser().resolve()
EXTERNAL_VIDEOS = [EXTERNAL_VIDEO_DIR / "test2.MOV", EXTERNAL_VIDEO_DIR / "test3.MOV"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_tier(correlation: float) -> str:
    if correlation >= 0.70:
        return "high"
    if correlation >= 0.30:
        return "medium"
    return "low"


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    source = np.load(SOURCE)
    matched = pd.read_csv(MATCHED).iloc[source["source_rows"].astype(int)].reset_index(drop=True)
    trial_quality = pd.read_csv(TRIAL_QUALITY)

    subjects = source["subjects"].astype(str)
    runs = source["runs"].astype(int)
    sides = source["sides"].astype(str)
    expected_development = set(split["development_subjects"])
    forbidden = set(split["test_subjects"])
    if set(subjects) != expected_development:
        raise RuntimeError("Development subject set changed")
    if set(subjects) & forbidden:
        raise RuntimeError("Forbidden historical test subject found")
    if len(source["x"]) != len(matched):
        raise RuntimeError("Sequence/source-row mismatch")

    sequence_length = int(source["x"].shape[1])
    duration_s = float(protocol["sequence_duration_s"])
    sample_interval_s = duration_s / (sequence_length - 1)
    onset = source["onset_indices"].astype(int)
    sample_index = np.arange(sequence_length, dtype=float)[None, :]
    sigma_samples = protocol["soft_target_sigma_ms"] / 1000.0 / sample_interval_s
    soft_target = np.exp(-0.5 * ((sample_index - onset[:, None]) / sigma_samples) ** 2)
    soft_target /= np.clip(soft_target.sum(axis=1, keepdims=True), 1e-12, None)
    hard_target = np.zeros((len(onset), sequence_length), dtype=np.uint8)
    hard_target[np.arange(len(onset)), onset] = 1
    display_samples = int(round(protocol["landing_display_window_ms"] / 1000.0 / sample_interval_s))
    landing_window_target = np.zeros_like(hard_target)
    for row, event_index in enumerate(onset):
        landing_window_target[row, event_index:min(sequence_length, event_index + display_samples + 1)] = 1

    quality_lookup = trial_quality.set_index(["subject", "run"])
    sync_correlation = np.asarray(
        [float(quality_lookup.loc[(subject, run), "sync_correlation"]) for subject, run in zip(subjects, runs)],
        dtype=np.float32,
    )
    pose_detection_rate = np.asarray(
        [float(quality_lookup.loc[(subject, run), "pose_detection_rate"]) for subject, run in zip(subjects, runs)],
        dtype=np.float32,
    )
    tiers = np.asarray([quality_tier(float(value)) for value in sync_correlation])
    high_quality_mask = tiers == "high"

    artifacts = EXPERIMENT / "artifacts"
    reports = EXPERIMENT / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    output = artifacts / "landing_event_development_v3.npz"
    np.savez_compressed(
        output,
        x=source["x"].astype(np.float32),
        hard_event_target=hard_target,
        soft_event_target=soft_target.astype(np.float32),
        landing_window_target=landing_window_target,
        onset_indices=onset,
        subjects=subjects,
        runs=runs,
        sides=sides,
        feature_names=source["feature_names"],
        target_ic_offset_s=source["target_ic_offset_s"].astype(np.float32),
        sync_correlation=sync_correlation,
        pose_detection_rate=pose_detection_rate,
        quality_tier=tiers,
        high_quality_mask=high_quality_mask,
        sample_interval_s=np.asarray(sample_interval_s, dtype=np.float64),
    )

    audit = matched[["subject", "run", "side", "detected_stride_duration_s", "mean_tracked_foot_visibility"]].copy()
    audit["onset_sequence_index"] = onset
    audit["target_ic_offset_s"] = source["target_ic_offset_s"]
    audit["sync_correlation"] = sync_correlation
    audit["pose_detection_rate"] = pose_detection_rate
    audit["quality_tier"] = tiers
    audit["primary_clean_label"] = high_quality_mask
    audit.to_csv(reports / "sequence_label_quality_audit.csv", index=False, encoding="utf-8-sig")

    trial_rows = []
    for (subject, run), group in audit.groupby(["subject", "run"]):
        trial_rows.append(
            {
                "subject": subject,
                "run": int(run),
                "sequences": int(len(group)),
                "sync_correlation": float(group["sync_correlation"].iloc[0]),
                "pose_detection_rate": float(group["pose_detection_rate"].iloc[0]),
                "quality_tier": str(group["quality_tier"].iloc[0]),
                "onset_index_median": float(group["onset_sequence_index"].median()),
                "ic_offset_ms_mean": float(group["target_ic_offset_s"].mean() * 1000.0),
                "ic_offset_ms_std": float(group["target_ic_offset_s"].std() * 1000.0),
            }
        )
    pd.DataFrame(trial_rows).to_csv(
        reports / "trial_quality_summary.csv", index=False, encoding="utf-8-sig"
    )

    tolerance_samples = {
        str(int(value)): int(round(value / 1000.0 / sample_interval_s))
        for value in protocol["event_evaluation_tolerances_ms"]
    }
    external = []
    for path in EXTERNAL_VIDEOS:
        external.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256(path) if path.exists() else None,
                "role": "EXTERNAL_INFERENCE_ONLY_EXCLUDED_FROM_TRAINING",
            }
        )
    tier_counts = {
        tier: int(np.sum(tiers == tier)) for tier in ("high", "medium", "low")
    }
    manifest = {
        "status": "LANDING_EVENT_V3_PREPARATION_COMPLETE_NO_MODEL_TRAINED",
        "source_dataset": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "output_dataset": str(output),
        "output_sha256": sha256(output),
        "sequences": int(len(onset)),
        "subjects": sorted(set(subjects)),
        "runs": sorted(set(int(value) for value in runs)),
        "sequence_length": sequence_length,
        "features": int(source["x"].shape[2]),
        "sample_interval_ms": sample_interval_s * 1000.0,
        "soft_target_sigma_ms": protocol["soft_target_sigma_ms"],
        "landing_display_window_ms": protocol["landing_display_window_ms"],
        "event_tolerance_samples": tolerance_samples,
        "quality_tier_sequence_counts": tier_counts,
        "primary_high_quality_sequences": int(high_quality_mask.sum()),
        "secondary_quality_sequences": int((~high_quality_mask).sum()),
        "historical_test_subjects_accessed": False,
        "external_video_exclusions": external,
        "training_performed": False,
        "known_metric_correction": (
            "Previous landing_onset_transfer_v1 +/-frame values used 7.62 ms sequence samples. "
            "V3 reports tolerances in milliseconds; +/-30 ms and +/-60 ms approximate one and two 33 fps video frames."
        ),
    }
    (artifacts / "DATASET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
