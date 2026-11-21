from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PHASE_NAMES = {1: "LANDING", 2: "BRAKING", 3: "PROPULSION", 4: "RECOVERY"}
PHASE_COLORS = {
    1: (30, 205, 245),
    2: (70, 70, 225),
    3: (75, 185, 85),
    4: (215, 130, 55),
}


def point(row: pd.Series, name: str, width: int, height: int) -> tuple[int, int]:
    return int(row[f"{name}_x"] * width), int(row[f"{name}_y"] * height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render both-foot four-stage predictions on the full source video"
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("phase_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--review-threshold", type=float, default=0.60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    landmarks = pd.read_csv(args.landmarks_csv)
    summary = pd.read_csv(args.phase_summary_csv)
    timeline: dict[str, list[dict | None]] = {
        "left": [None] * frame_count,
        "right": [None] * frame_count,
    }
    for _, stride in summary.iterrows():
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        tracked_side = str(stride["tracked_side_coco"])
        anatomical_side = str(stride["anatomical_side_in_this_clip"])
        feature_path = args.output_dir / (
            f"{args.source_video.stem}_{tracked_side}_stride_{start}_{end}"
            "_features_and_prediction.csv"
        )
        predictions = pd.read_csv(feature_path)
        for frame in range(start, min(end, frame_count)):
            prediction_index = int(
                np.clip(round((frame - start) / fps * 100.0), 0, len(predictions) - 1)
            )
            prediction = predictions.iloc[prediction_index]
            confidence = float(prediction["predicted_probability"])
            timeline[anatomical_side][frame] = {
                "stride_id": str(stride["stride_id"]),
                "tracked_side": tracked_side,
                "phase_label": int(prediction["phase_label"]),
                "phase_name": str(prediction["phase_name"]).upper(),
                "confidence": confidence,
                "qc_status": "AUTO_OK" if confidence >= args.review_threshold else "REVIEW_TRANSITION",
            }

    stem = args.source_video.stem
    video_path = args.output_dir / f"{stem}_full_two_foot_phase_overlay.mp4"
    timeline_path = args.output_dir / f"{stem}_full_phase_timeline.csv"
    report_path = args.output_dir / f"{stem}_full_phase_report.json"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    cap = cv2.VideoCapture(str(args.source_video))
    timeline_rows: list[dict] = []
    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            break
        landmark_row = landmarks.iloc[frame_index]
        cv2.rectangle(frame, (0, 0), (width, 112), (18, 18, 18), -1)

        # Central body context remains neutral; each leg is colored independently.
        neutral_connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
        ]
        for first, second in neutral_connections:
            cv2.line(
                frame,
                point(landmark_row, first, width, height),
                point(landmark_row, second, width, height),
                (210, 210, 210), 3, cv2.LINE_AA,
            )

        row = {"frame_index": frame_index, "time_s": frame_index / fps}
        for anatomical_side, x_origin in (("left", 18), ("right", width // 2 + 8)):
            state = timeline[anatomical_side][frame_index]
            if state is None:
                title = f"{anatomical_side.upper()} FOOT: NO ACCEPTED STRIDE"
                color = (150, 150, 150)
                confidence_text = ""
                tracked_side = ""
                phase_label: int | str = ""
                phase_name = ""
                confidence: float | str = ""
                qc_status = "OUTSIDE_ACCEPTED_STRIDE"
                stride_id = ""
            else:
                phase_label = state["phase_label"]
                phase_name = state["phase_name"]
                confidence = state["confidence"]
                qc_status = state["qc_status"]
                stride_id = state["stride_id"]
                tracked_side = state["tracked_side"]
                color = PHASE_COLORS[int(phase_label)]
                title = f"{anatomical_side.upper()} FOOT: {phase_label} {phase_name}"
                confidence_text = f"p={confidence:.2f}  {qc_status}"
                for first, second in (
                    (f"{tracked_side}_hip", f"{tracked_side}_knee"),
                    (f"{tracked_side}_knee", f"{tracked_side}_ankle"),
                ):
                    cv2.line(
                        frame,
                        point(landmark_row, first, width, height),
                        point(landmark_row, second, width, height),
                        color, 7, cv2.LINE_AA,
                    )
                    cv2.circle(frame, point(landmark_row, second, width, height), 7, color, -1)
            cv2.putText(
                frame, title, (x_origin, 37), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, confidence_text, (x_origin, 67), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (225, 225, 225), 1, cv2.LINE_AA,
            )
            row.update({
                f"{anatomical_side}_stride_id": stride_id,
                f"{anatomical_side}_tracked_side_coco": tracked_side,
                f"{anatomical_side}_phase_label": phase_label,
                f"{anatomical_side}_phase_name": phase_name,
                f"{anatomical_side}_confidence": confidence,
                f"{anatomical_side}_qc_status": qc_status,
            })
        cv2.putText(
            frame,
            "EXPERIMENTAL: 2D proxy, 1.04 m/s is outside the trained 1.8-2.4 m/s range",
            (18, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
        writer.write(frame)
        timeline_rows.append(row)
    cap.release()
    writer.release()

    timeline_table = pd.DataFrame(timeline_rows)
    timeline_table.to_csv(timeline_path, index=False, encoding="utf-8-sig")
    active_mask = (
        timeline_table["left_phase_label"].astype(str).ne("")
        | timeline_table["right_phase_label"].astype(str).ne("")
    )
    review_mask = (
        timeline_table["left_qc_status"].eq("REVIEW_TRANSITION")
        | timeline_table["right_qc_status"].eq("REVIEW_TRANSITION")
    )
    report = {
        "status": "experimental_out_of_domain_prototype",
        "source_video": str(args.source_video),
        "frames": len(timeline_table),
        "fps": fps,
        "accepted_strides": int(len(summary)),
        "frames_with_at_least_one_phase": int(active_mask.sum()),
        "coverage_percent": float(active_mask.mean() * 100.0),
        "frames_flagged_near_low_confidence_transition": int(review_mask.sum()),
        "review_threshold": args.review_threshold,
        "qc_definition": (
            "AUTO_OK and REVIEW_TRANSITION describe technical model confidence only; "
            "they are not good/bad running-form labels."
        ),
        "outputs": {
            "full_overlay_video": str(video_path),
            "frame_timeline_csv": str(timeline_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
