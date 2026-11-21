from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

from make_four_stage_storyboards import CONNECTIONS, runner_crop


PHASE_NAMES = {1: "LANDING", 2: "BRAKING", 3: "PROPULSION", 4: "RECOVERY"}
UNKNOWN_COLOR = (145, 145, 145)
NEUTRAL_COLOR = (205, 205, 205)


def draw_unknown_form_scene(
    frame,
    landmark_row: pd.Series,
    width: int,
    height: int,
    tracked_side: str,
    phase: int,
    phase_score: float,
    stride_id: str,
):
    for first, second in CONNECTIONS:
        first_point = (
            int(landmark_row[f"{first}_x"] * width),
            int(landmark_row[f"{first}_y"] * height),
        )
        second_point = (
            int(landmark_row[f"{second}_x"] * width),
            int(landmark_row[f"{second}_y"] * height),
        )
        active_leg = first.startswith(tracked_side) and second.startswith(tracked_side)
        color = UNKNOWN_COLOR if active_leg else NEUTRAL_COLOR
        thickness = 7 if active_leg else 3
        cv2.line(frame, first_point, second_point, color, thickness, cv2.LINE_AA)
        if active_leg:
            cv2.circle(frame, second_point, 7, color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 105), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"STAGE {phase} {PHASE_NAMES[phase]} | PHASE SCORE {phase_score:.1f}/100",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FORM STATUS: UNKNOWN (no labeled good/bad rule)",
        (18, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        UNKNOWN_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{stride_id} | gray = cannot judge running-form quality",
        (18, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create per-stage screenshots with honest phase scores and form status"
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("phase_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    scene_dir = args.output_dir / f"{args.source_video.stem}_scored_stage_screenshots"
    scene_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.phase_summary_csv)
    landmarks = pd.read_csv(args.landmarks_csv)
    metrics = json.loads((root / "model_v1/reports/metrics.json").read_text(encoding="utf-8"))
    lab_f1 = metrics["test_ml_frame_metrics"]["per_stage"]

    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    score_rows: list[dict] = []

    for _, stride in summary.iterrows():
        stride_id = str(stride["stride_id"])
        tracked_side = str(stride["tracked_side_coco"])
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        report = json.loads(Path(str(stride["report_json"])).read_text(encoding="utf-8"))
        feature_path = args.output_dir / (
            f"{args.source_video.stem}_{tracked_side}_stride_{start}_{end}"
            "_features_and_prediction.csv"
        )
        features = pd.read_csv(feature_path)
        crop_x0, crop_y0, crop_x1, crop_y1 = runner_crop(
            landmarks, start, end, width, height
        )
        images: list[Image.Image] = []
        for phase_range in report["phase_ranges"]:
            phase = int(phase_range["phase_label"])
            phase_values = features.loc[
                features["phase_label"] == phase, "predicted_probability"
            ]
            phase_score = float(phase_values.mean() * 100.0)
            midpoint_ms = (
                float(phase_range["start_ms"]) + float(phase_range["end_ms"])
            ) / 2.0
            frame_index = min(end - 1, start + int(round(midpoint_ms / 1000.0 * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index}")
            frame = draw_unknown_form_scene(
                frame,
                landmarks.iloc[frame_index],
                width,
                height,
                tracked_side,
                phase,
                phase_score,
                stride_id,
            )
            frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            crop_height, crop_width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (crop_width, min(105, crop_height)), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"STAGE {phase} {PHASE_NAMES[phase]} | SCORE {phase_score:.1f}/100",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "FORM: UNKNOWN - gray",
                (10, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.50, UNKNOWN_COLOR, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{stride_id} | no good/bad ground truth",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
            )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            screenshot_path = scene_dir / f"{stride_id}_{tracked_side}_stage_{phase}_{PHASE_NAMES[phase].lower()}.jpg"
            image.save(screenshot_path, quality=92)
            images.append(image)
            f1_key = {1: "landing_window", 2: "braking", 3: "propulsion", 4: "recovery"}[phase]
            score_rows.append({
                "stride_id": stride_id,
                "tracked_side": tracked_side,
                "stage": phase,
                "stage_name": PHASE_NAMES[phase].lower(),
                "representative_frame": frame_index,
                "phase_score_0_100": phase_score,
                "form_status": "UNKNOWN",
                "form_color": "gray",
                "lab_3d_test_f1": float(lab_f1[f1_key]["f1-score"]),
                "current_video_f1": "NOT_AVAILABLE_NO_GROUND_TRUTH",
                "screenshot": str(screenshot_path),
            })

        board_width = max(image.width for image in images)
        board_height = max(image.height for image in images)
        board = Image.new("RGB", (board_width * 2, board_height * 2), (20, 20, 20))
        for index, image in enumerate(images):
            board.paste(image, ((index % 2) * board_width, (index // 2) * board_height))
        board.save(scene_dir / f"{stride_id}_{tracked_side}_scored_four_stage_board.jpg", quality=92)

    cap.release()
    scores = pd.DataFrame(score_rows)
    scores_path = scene_dir / "stage_scores_and_f1.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    aggregate = scores.groupby(["stage", "stage_name"]).agg(
        video_phase_score_mean=("phase_score_0_100", "mean"),
        video_phase_score_min=("phase_score_0_100", "min"),
        video_phase_score_max=("phase_score_0_100", "max"),
        lab_3d_test_f1=("lab_3d_test_f1", "first"),
        screenshots=("screenshot", "count"),
    ).reset_index()
    aggregate_path = scene_dir / "four_stage_score_summary.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "strides": int(summary.shape[0]),
        "stage_screenshots": int(scores.shape[0]),
        "form_status": "UNKNOWN",
        "reason": "The public video has no good/bad form labels or force-derived stage ground truth.",
        "stage_scores_csv": str(scores_path),
        "aggregate_scores_csv": str(aggregate_path),
        "scene_dir": str(scene_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
