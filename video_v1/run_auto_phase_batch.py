from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run every accepted automatically detected stride through the phase prototype"
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("strides_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "video_v1/outputs")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.04,
        help="Nominal speed in m/s. BML-MoVi Subject 1 jogging is approximately 1.04 m/s.",
    )
    parser.add_argument(
        "--side-mapping",
        choices=["direct", "swapped"],
        default="direct",
        help="Map COCO tracked-side labels to anatomical sides.",
    )
    parser.add_argument(
        "--decoder",
        choices=["temporal-prior-v2", "kinematic-ml-v1"],
        default="temporal-prior-v2",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strides = pd.read_csv(args.strides_csv)
    accepted = strides[
        strides["accepted_stride"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    summary_rows: list[dict] = []
    for _, stride in accepted.iterrows():
        tracked_side = str(stride["tracked_side"])
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        command = [
            sys.executable,
            str(root / "video_v1/run_phase_prototype.py"),
            str(args.source_video),
            str(args.landmarks_csv),
            "--output-dir", str(args.output_dir),
            "--side", tracked_side,
            "--stride-start", str(start),
            "--stride-end", str(end),
            "--speed", str(args.speed),
            "--decoder", args.decoder,
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        output_stem = f"{args.source_video.stem}_{tracked_side}_stride_{start}_{end}"
        report_path = args.output_dir / f"{output_stem}_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        ranges = {row["phase_label"]: row for row in report["phase_ranges"]}
        anatomical_side = (
            ("right" if tracked_side == "left" else "left")
            if args.side_mapping == "swapped"
            else tracked_side
        )
        summary_rows.append({
            "stride_id": stride["stride_id"],
            "tracked_side_coco": tracked_side,
            "anatomical_side_in_this_clip": anatomical_side,
            "start_frame_inclusive": start,
            "end_frame_exclusive": end,
            "duration_s": report["stride_duration_s"],
            "nominal_speed_m_s": args.speed,
            "landing_start_ms": ranges[1]["start_ms"],
            "braking_start_ms": ranges[2]["start_ms"],
            "propulsion_start_ms": ranges[3]["start_ms"],
            "recovery_start_ms": ranges[4]["start_ms"],
            "mean_predicted_probability": report["mean_predicted_probability"],
            "report_json": str(report_path),
            "phase_overlay_video": report["outputs"]["overlay_video"],
        })

    summary = pd.DataFrame(summary_rows).sort_values("start_frame_inclusive")
    stem = args.source_video.stem
    summary_path = args.output_dir / f"{stem}_auto_phase_summary.csv"
    report_path = args.output_dir / f"{stem}_auto_phase_batch_report.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    batch_report = {
        "status": "experimental_out_of_domain_prototype",
        "accepted_strides_processed": int(len(summary)),
        "nominal_speed_m_s": args.speed,
        "decoder": args.decoder,
        "speed_source": "single approximate BML-MoVi V3D jogging speed for this demonstration",
        "coco_side_mapping": args.side_mapping,
        "warning": (
            "The phase outputs are model estimates, not ground-truth accuracy. "
            "Only heel-strike timing was validated against V3D kinematics."
        ),
        "outputs": {"phase_summary_csv": str(summary_path)},
    }
    report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")
    print(json.dumps(batch_report, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
