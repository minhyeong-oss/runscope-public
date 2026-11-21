from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "test_videos/twente/Subj05_run81_side_10s_to_22s.mp4"
OUTPUT_DIR = ROOT / "video_v1/test_outputs/twente_subj05_run81_pilot"
LANDMARKS = OUTPUT_DIR / "Subj05_run81_side_10s_to_22s_yolo_landmarks.csv"
COMPARISON = OUTPUT_DIR / "temporal_prior_v2/twente_temporal_prior_v2_comparison.csv"
REPORT = OUTPUT_DIR / "temporal_prior_v2/twente_temporal_prior_v2_metrics.json"
RESULT_DIR = OUTPUT_DIR / "user_four_stage_result_v2"

STRIDE_ID = "stride_009"
CLIP_OFFSET_S = 10.0
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

STAGES = {
    1: ("LANDING", "착지"),
    2: ("BRAKING", "제동"),
    3: ("PROPULSION", "추진"),
    4: ("RECOVERY", "회복"),
}

CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

GREEN = (41, 214, 112)
GRAY = (158, 166, 178)
WHITE = (242, 245, 247)
PANEL = (18, 23, 29)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def choose_representative(group: pd.DataFrame) -> pd.Series:
    """Choose the temporal middle of the correctly classified stage samples."""
    ordered = group.sort_values("video_time_s").reset_index(drop=True)
    center_time = float(ordered["video_time_s"].median())
    index = (ordered["video_time_s"] - center_time).abs().idxmin()
    return ordered.loc[index]


def crop_box(landmarks: pd.DataFrame, frame_indices: list[int], width: int, height: int):
    names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    ]
    section = landmarks.iloc[max(0, min(frame_indices) - 3): min(len(landmarks), max(frame_indices) + 4)]
    xs, ys = [], []
    for name in names:
        good = section[f"{name}_confidence"] >= 0.20
        xs.extend(section.loc[good, f"{name}_x"].dropna().tolist())
        ys.extend(section.loc[good, f"{name}_y"].dropna().tolist())
    x0 = max(0, int((min(xs) - 0.045) * width))
    x1 = min(width, int((max(xs) + 0.045) * width))
    y0 = max(0, int((min(ys) - 0.045) * height))
    y1 = min(height, int((max(ys) + 0.055) * height))
    return x0, y0, x1, y1


def draw_skeleton(image: Image.Image, row: pd.Series, full_width: int, full_height: int, offset):
    draw = ImageDraw.Draw(image)
    x0, y0 = offset
    for first, second in CONNECTIONS:
        if row[f"{first}_confidence"] < 0.20 or row[f"{second}_confidence"] < 0.20:
            continue
        p1 = (int(row[f"{first}_x"] * full_width) - x0, int(row[f"{first}_y"] * full_height) - y0)
        p2 = (int(row[f"{second}_x"] * full_width) - x0, int(row[f"{second}_y"] * full_height) - y0)
        draw.line([p1, p2], fill=GREEN, width=8)
        draw.ellipse([p1[0] - 6, p1[1] - 6, p1[0] + 6, p1[1] + 6], fill=GREEN)
        draw.ellipse([p2[0] - 6, p2[1] - 6, p2[0] + 6, p2[1] + 6], fill=GREEN)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(COMPARISON)
    landmarks = pd.read_csv(LANDMARKS)
    metrics = json.loads(REPORT.read_text(encoding="utf-8"))["temporal_prior_v2"]
    stage_f1 = {
        1: metrics["per_stage"]["landing"]["f1-score"],
        2: metrics["per_stage"]["braking"]["f1-score"],
        3: metrics["per_stage"]["propulsion"]["f1-score"],
        4: metrics["per_stage"]["recovery"]["f1-score"],
    }

    stride = comparison[comparison["stride_id"] == STRIDE_ID].copy()
    selected = []
    for stage in STAGES:
        correct = stride[
            (stride["ground_truth_label"] == stage)
            & (stride["v2_predicted_label"] == stage)
        ]
        if correct.empty:
            raise RuntimeError(f"No correct representative sample for stage {stage}")
        selected.append(choose_representative(correct))

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_indices = [
        max(0, min(len(landmarks) - 1, int(round((float(row["video_time_s"]) - CLIP_OFFSET_S) * fps))))
        for row in selected
    ]
    x0, y0, x1, y1 = crop_box(landmarks, frame_indices, width, height)

    card_width, image_height, text_height = 420, 500, 205
    card_height = image_height + text_height
    cards = []
    manifest = []
    for stage, row, frame_index in zip(STAGES, selected, frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(frame_rgb[y0:y1, x0:x1]).convert("RGB")
        draw_skeleton(crop, landmarks.iloc[frame_index], width, height, (x0, y0))
        crop.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
        fitted = Image.new("RGB", (card_width, image_height), (8, 12, 16))
        fitted.paste(crop, ((card_width - crop.width) // 2, (image_height - crop.height) // 2))

        card = Image.new("RGB", (card_width, card_height), PANEL)
        card.paste(fitted, (0, 0))
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card_width - 1, card_height - 1), outline=GREEN, width=6)
        english, korean = STAGES[stage]
        draw.text((18, 515), f"{stage}. {english} · {korean}", font=font(27), fill=WHITE)
        draw.text((18, 555), "단계 판정: 일치 ✓", font=font(23), fill=GREEN)
        draw.text((18, 589), "자세 Good/Bad: 미평가", font=font(21), fill=GRAY)
        draw.text(
            (18, 623),
            f"v2 단계 F1 {stage_f1[stage]:.3f}  |  2.2m/s 학습 사전값 적용",
            font=font(18),
            fill=WHITE,
        )
        draw.text(
            (18, 657),
            f"영상 {float(row['video_time_s']):.3f}s  |  frame {frame_index}",
            font=font(17),
            fill=GRAY,
        )
        individual_path = RESULT_DIR / f"stage_{stage}_{english.lower()}_v2.jpg"
        card.save(individual_path, quality=94)
        cards.append(card)
        manifest.append({
            "stage": stage,
            "stage_name": english.lower(),
            "frame": frame_index,
            "video_time_s": float(row["video_time_s"]),
            "phase_ground_truth_match": True,
            "running_form_status": "NOT_EVALUATED",
            "stage_f1": float(stage_f1[stage]),
            "image": str(individual_path),
        })
    cap.release()

    header_height = 138
    board = Image.new("RGB", (card_width * 4, header_height + card_height), (10, 14, 19))
    draw = ImageDraw.Draw(board)
    draw.text((28, 20), "RunScope v2 · 트레드밀 러닝 4단계 분석", font=font(37), fill=WHITE)
    draw.text(
        (28, 72),
        "초록 관절선 = 단계 판정이 힘판 정답과 일치  |  회색 문구 = 자세 품질은 아직 미평가",
        font=font(24),
        fill=GRAY,
    )
    draw.text(
        (28, 106),
        f"Twente Subj05 run_81 · {STRIDE_ID} · 개선 Macro F1 {metrics['macro_f1']:.3f}",
        font=font(20),
        fill=GREEN,
    )
    for index, card in enumerate(cards):
        board.paste(card, (index * card_width, header_height))
    board_path = RESULT_DIR / "RunScope_Twente_four_stage_user_view_v2.jpg"
    board.save(board_path, quality=95)
    manifest_path = RESULT_DIR / "four_stage_user_view.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "board": str(board_path),
        "individual_scenes": [item["image"] for item in manifest],
        "manifest": str(manifest_path),
        "stride": STRIDE_ID,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
