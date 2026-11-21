from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageOps


PHASE_NAMES_KO = {1: "1 LANDING", 2: "2 BRAKING", 3: "3 PROPULSION", 4: "4 RECOVERY"}
PHASE_COLORS = {
    1: (30, 205, 245),
    2: (70, 70, 225),
    3: (75, 185, 85),
    4: (215, 130, 55),
}
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def runner_crop(
    landmarks: pd.DataFrame,
    start: int,
    end: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return one stable crop around the tracked runner for the whole stride."""
    names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    ]
    section = landmarks.iloc[start:end]
    xs: list[float] = []
    ys: list[float] = []
    for name in names:
        confident = section[f"{name}_confidence"] >= 0.20
        xs.extend(section.loc[confident, f"{name}_x"].dropna().tolist())
        ys.extend(section.loc[confident, f"{name}_y"].dropna().tolist())
    if not xs or not ys:
        return 0, 0, width, height
    pad_x = 0.055
    pad_y = 0.045
    x0 = max(0, int((min(xs) - pad_x) * width))
    x1 = min(width, int((max(xs) + pad_x) * width))
    y0 = max(0, int((min(ys) - pad_y) * height))
    y1 = min(height, int((max(ys) + pad_y) * height))
    return x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)


def draw_stage(
    frame, landmarks: pd.Series, width: int, height: int,
    tracked_side: str, anatomical_side: str, phase: int, stride_id: str,
):
    color = PHASE_COLORS[phase]
    for first, second in CONNECTIONS:
        first_point = (
            int(landmarks[f"{first}_x"] * width),
            int(landmarks[f"{first}_y"] * height),
        )
        second_point = (
            int(landmarks[f"{second}_x"] * width),
            int(landmarks[f"{second}_y"] * height),
        )
        active_leg = first.startswith(tracked_side) and second.startswith(tracked_side)
        line_color = color if active_leg else (210, 210, 210)
        thickness = 7 if active_leg else 3
        cv2.line(frame, first_point, second_point, line_color, thickness, cv2.LINE_AA)
        if active_leg:
            cv2.circle(frame, second_point, 7, color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 76), (18, 18, 18), -1)
    cv2.putText(
        frame, f"{anatomical_side.upper()} FOOT - {PHASE_NAMES_KO[phase]}",
        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"{stride_id} / representative frame / experimental prototype",
        (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 225, 225), 1, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create four-stage scene boards and GIFs for each stride")
    parser.add_argument("source_video", type=Path)
    parser.add_argument("landmarks_csv", type=Path)
    parser.add_argument("phase_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    story_dir = args.output_dir / f"{args.source_video.stem}_four_stage_scenes"
    story_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    summary = pd.read_csv(args.phase_summary_csv)
    landmarks = pd.read_csv(args.landmarks_csv)
    index_rows: list[dict] = []

    for _, stride in summary.iterrows():
        stride_id = str(stride["stride_id"])
        tracked_side = str(stride["tracked_side_coco"])
        anatomical_side = str(stride["anatomical_side_in_this_clip"])
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        report_path = Path(str(stride["report_json"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        crop_x0, crop_y0, crop_x1, crop_y1 = runner_crop(
            landmarks, start, end, width, height
        )
        crop_width = crop_x1 - crop_x0
        crop_height = crop_y1 - crop_y0
        images: list[Image.Image] = []
        representative_frames: list[int] = []
        for phase_range in report["phase_ranges"]:
            phase = int(phase_range["phase_label"])
            midpoint_ms = (float(phase_range["start_ms"]) + float(phase_range["end_ms"])) / 2.0
            frame_index = min(end - 1, start + int(round(midpoint_ms / 1000.0 * fps)))
            representative_frames.append(frame_index)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index}")
            frame = draw_stage(
                frame, landmarks.iloc[frame_index], width, height,
                tracked_side, anatomical_side, phase, stride_id,
            )
            frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            cv2.rectangle(frame, (0, 0), (crop_width, min(76, crop_height)), (18, 18, 18), -1)
            cv2.putText(
                frame, f"{anatomical_side.upper()} FOOT - {PHASE_NAMES_KO[phase]}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, PHASE_COLORS[phase], 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{stride_id} / tracked runner crop / experimental prototype",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (225, 225, 225), 1, cv2.LINE_AA,
            )
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        board = Image.new("RGB", (crop_width * 2, crop_height * 2), "white")
        for index, image in enumerate(images):
            board.paste(image, ((index % 2) * crop_width, (index // 2) * crop_height))
        base = f"{stride_id}_{anatomical_side}_frames_{start}_{end}"
        board_path = story_dir / f"{base}_four_stage_board.jpg"
        gif_path = story_dir / f"{base}_four_stage.gif"
        board.save(board_path, quality=92)
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=[800, 1100, 900, 1200], loop=0,
        )
        index_rows.append({
            "stride_id": stride_id,
            "anatomical_side": anatomical_side,
            "start_frame": start,
            "end_frame": end,
            "landing_frame": representative_frames[0],
            "braking_frame": representative_frames[1],
            "propulsion_frame": representative_frames[2],
            "recovery_frame": representative_frames[3],
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "storyboard_jpg": str(board_path),
            "four_stage_gif": str(gif_path),
        })
    cap.release()
    index_path = story_dir / "four_stage_scene_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    contact_sheet_path = story_dir / "all_strides_contact_sheet.jpg"
    if index_rows:
        cell_width, cell_height = 320, 460
        columns = 3
        rows = (len(index_rows) + columns - 1) // columns
        contact_sheet = Image.new(
            "RGB", (columns * cell_width, rows * cell_height), (24, 24, 24)
        )
        for index, row in enumerate(index_rows):
            board_image = Image.open(row["storyboard_jpg"]).convert("RGB")
            thumbnail = ImageOps.contain(board_image, (cell_width - 12, cell_height - 12))
            x = (index % columns) * cell_width + (cell_width - thumbnail.width) // 2
            y = (index // columns) * cell_height + (cell_height - thumbnail.height) // 2
            contact_sheet.paste(thumbnail, (x, y))
        contact_sheet.save(contact_sheet_path, quality=90)
    print(json.dumps({
        "strides_rendered": len(index_rows),
        "story_dir": str(story_dir),
        "index_csv": str(index_path),
        "contact_sheet": str(contact_sheet_path),
    }, indent=2))


if __name__ == "__main__":
    main()
