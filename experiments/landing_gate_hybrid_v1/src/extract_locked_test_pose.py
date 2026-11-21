from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_gate_hybrid_v1"
SOURCE = ROOT / "experiments/boundary_regression_v1/src/extract_development_pose.py"
MODEL = ROOT / "experiments/pose_foot_keypoints_v1/models/pose_landmarker_full.task"
VIDEO_ROOT = ROOT / "data/twente_rgb/side_running/Videos_anonymized"
TRIALS = [("Subj04", 81), ("Subj04", 99), ("Subj11", 63), ("Subj11", 81), ("Subj11", 99)]
START_S = 10.0
END_S = 22.0


def load_source():
    spec = importlib.util.spec_from_file_location("locked_pose_extractor", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    source = load_source()
    output_root = EXPERIMENT / "artifacts/raw_pose"
    output_root.mkdir(parents=True, exist_ok=True)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    reports = []
    for subject, run in TRIALS:
        video_path = VIDEO_ROOT / subject / f"run_{run}_side_anonymized.avi"
        output_path = output_root / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            table = source.extract_clip(video_path, landmarker, START_S, END_S)
        table.to_csv(output_path, index=False, encoding="utf-8-sig")
        detected = int(table["pose_detected"].sum())
        report = {
            "subject": subject,
            "run": run,
            "frames": int(len(table)),
            "detected_frames": detected,
            "detection_rate": detected / max(1, len(table)),
            "output": str(output_path),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    summary = {
        "experiment": "landing_gate_hybrid_v1",
        "operation": "inference-only pose extraction on the historical locked test videos",
        "training_performed": False,
        "model_or_threshold_selection_performed": False,
        "clips": reports,
        "mean_detection_rate": float(np.mean([item["detection_rate"] for item in reports])),
    }
    path = EXPERIMENT / "artifacts/LOCKED_TEST_POSE_EXTRACTION.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
