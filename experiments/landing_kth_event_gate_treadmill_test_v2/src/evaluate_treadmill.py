from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_SOURCE = (
    ROOT
    / "experiments/landing_kth_event_gate_treadmill_test_v1/src/evaluate_treadmill.py"
)
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_treadmill_test_v2"
VIDEO = ROOT / "test_videos/twente_locked_test/Subj11_run99_side_10s_to_22s.mp4"
POSE_ROOT = ROOT / "video_v1/test_outputs/twente_locked_test/Subj11_run99"
POSE_CSV = POSE_ROOT / "Subj11_run99_side_10s_to_22s_yolo_landmarks.csv"
POSE_QUALITY = POSE_ROOT / "Subj11_run99_side_10s_to_22s_yolo_pose_quality.json"
LOCKED_FORCE_TRUTH = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/final_v3/locked_test_predictions_and_force_truth.csv"
)
SUBJECT = "Subj11"
RUN = 99


def load_base():
    spec = importlib.util.spec_from_file_location("treadmill_test_v2_base", BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def locked_force_truth(duration_frames: int) -> tuple[np.ndarray, pd.DataFrame]:
    table = pd.read_csv(LOCKED_FORCE_TRUTH)
    table = table[(table["subject"] == SUBJECT) & (table["run"] == RUN)].copy()
    landing = (
        table[table["ground_truth_label"] == 1]
        .sort_values(["stride_id", "video_time_s"])
        .groupby(["stride_id", "side"], as_index=False)
        .first()
    )
    landing["force_initial_contact_source_time_s"] = landing["video_time_s"]
    landing["force_initial_contact_clip_time_s"] = (
        landing["force_initial_contact_source_time_s"] - base.CLIP_SOURCE_OFFSET_S
    )
    landing = landing[
        (landing["force_initial_contact_clip_time_s"] >= 0)
        & (
            landing["force_initial_contact_clip_time_s"]
            < duration_frames / base.POSE_RATE
        )
    ].copy()
    landing["force_initial_contact_frame_30hz"] = np.rint(
        landing["force_initial_contact_clip_time_s"] * base.POSE_RATE
    ).astype(int)
    events = np.asarray(
        sorted(set(landing["force_initial_contact_frame_30hz"].astype(int))),
        dtype=np.int64,
    )
    return events, landing


def main() -> None:
    base.EXPERIMENT = EXPERIMENT
    base.VIDEO = VIDEO
    base.POSE_CSV = POSE_CSV
    base.POSE_QUALITY = POSE_QUALITY
    base.FORCE_MATCHED = LOCKED_FORCE_TRUTH
    base.SUBJECT = SUBJECT
    base.RUN = RUN
    base.force_truth = locked_force_truth
    base.main()


if __name__ == "__main__":
    main()
