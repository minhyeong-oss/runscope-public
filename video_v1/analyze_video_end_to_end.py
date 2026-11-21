from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], environment: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="One-command offline RunScope video analysis prototype"
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--speed", type=float, required=True, help="Measured or known running speed in m/s")
    parser.add_argument("--output-dir", type=Path, default=root / "video_v1/outputs")
    parser.add_argument(
        "--side-mapping", choices=["direct", "swapped"], default="direct",
        help="Use swapped only when camera/mocap validation shows COCO sides are reversed.",
    )
    parser.add_argument(
        "--motion-mode", choices=["auto", "overground", "treadmill"], default="auto",
        help="Use treadmill when the runner stays nearly fixed in the image.",
    )
    parser.add_argument(
        "--facing", choices=["auto", "left", "right"], default="auto",
        help="Runner facing direction; useful for treadmill recordings.",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.85,
        help="Minimum interval lower-body confidence used by stride detection.",
    )
    parser.add_argument(
        "--decoder",
        choices=["temporal-prior-v2", "kinematic-ml-v1"],
        default="temporal-prior-v2",
        help="Four-stage decoder. v2 is the validated default for side-view treadmill video.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_video.stem
    landmarks = args.output_dir / f"{stem}_yolo_landmarks.csv"
    strides = args.output_dir / f"{stem}_auto_strides.csv"
    phase_summary = args.output_dir / f"{stem}_auto_phase_summary.csv"

    environment = os.environ.copy()
    environment.setdefault("YOLO_CONFIG_DIR", "/private/tmp/ultralytics")
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/runscope_mpl")
    environment.setdefault("LOKY_MAX_CPU_COUNT", "8")
    python = sys.executable
    run([
        python, str(root / "video_v1/extract_pose_yolo.py"),
        str(args.input_video), str(args.output_dir),
    ], environment)
    quality_path = args.output_dir / f"{stem}_yolo_pose_quality.json"
    pose_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    source_fps = float(pose_quality["fps"])
    run([
        python, str(root / "video_v1/detect_strides.py"),
        str(landmarks), str(args.output_dir),
        "--mode", args.motion_mode,
        "--facing", args.facing,
        "--fps", str(source_fps),
        "--min-confidence", str(args.min_confidence),
    ], environment)
    run([
        python, str(root / "video_v1/run_auto_phase_batch.py"),
        str(args.input_video), str(landmarks), str(strides),
        "--output-dir", str(args.output_dir),
        "--speed", str(args.speed),
        "--side-mapping", args.side_mapping,
        "--decoder", args.decoder,
    ], environment)
    run([
        python, str(root / "video_v1/render_full_phase_video.py"),
        str(args.input_video), str(landmarks), str(phase_summary), str(args.output_dir),
    ], environment)
    run([
        python, str(root / "video_v1/make_four_stage_storyboards.py"),
        str(args.input_video), str(landmarks), str(phase_summary), str(args.output_dir),
    ], environment)
    run([
        python, str(root / "video_v1/generate_video_assessment.py"),
        str(args.input_video), str(phase_summary), str(args.output_dir),
    ], environment)

    manifest = {
        "status": "complete_experimental_offline_analysis",
        "input_video": str(args.input_video),
        "provided_speed_m_s": args.speed,
        "side_mapping": args.side_mapping,
        "motion_mode": args.motion_mode,
        "facing": args.facing,
        "source_fps": source_fps,
        "minimum_stride_confidence": args.min_confidence,
        "decoder": args.decoder,
        "speed_in_model_training_range": 1.8 <= args.speed <= 2.4,
        "outputs": {
            "landmarks_csv": str(landmarks),
            "strides_csv": str(strides),
            "phase_summary_csv": str(phase_summary),
            "full_overlay_video": str(args.output_dir / f"{stem}_full_two_foot_phase_overlay.mp4"),
            "full_timeline_csv": str(args.output_dir / f"{stem}_full_phase_timeline.csv"),
            "full_report_json": str(args.output_dir / f"{stem}_full_phase_report.json"),
            "four_stage_scenes": str(args.output_dir / f"{stem}_four_stage_scenes"),
            "technical_assessment_csv": str(args.output_dir / f"{stem}_technical_assessment.csv"),
            "technical_assessment_report_ko": str(args.output_dir / f"{stem}_technical_assessment_KO.md"),
        },
        "limitations": [
            "The pipeline requires a completed stride and is offline, not real-time.",
            "The supplied speed must be measured or known; monocular video has no metric scale.",
            "Four-stage output is not force-label validated on BML-MoVi.",
        ],
    }
    manifest_path = args.output_dir / f"{stem}_end_to_end_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
