from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
REPORT_DIR = EXPERIMENT / "reports"
VIDEO = ROOT / "test_videos/twente_locked_test/Subj11_run99_side_10s_to_22s.mp4"
LANDMARKS = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/Subj11_run99"
    / "Subj11_run99_side_10s_to_22s_yolo_landmarks.csv"
)
PREDICTIONS = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/final_v3"
    / "locked_test_predictions_and_force_truth.csv"
)

SUBJECT = "Subj11"
RUN = 99
CLIP_OFFSET_S = 10.0
REPRESENTATIVE_STRIDE = "stride_006"
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

GREEN = (47, 210, 126)
RED = (255, 91, 100)
GRAY = (157, 167, 179)
WHITE = (241, 244, 248)
YELLOW = (250, 196, 70)
BACKGROUND = (9, 13, 18)
PANEL = (17, 23, 30)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def nearest_frame_index(video_time_s: float, fps: float, frame_count: int) -> int:
    return max(0, min(frame_count - 1, int(round((video_time_s - CLIP_OFFSET_S) * fps))))


def choose_temporal_middle(group: pd.DataFrame) -> pd.Series:
    center = float(group["video_time_s"].median())
    return group.loc[(group["video_time_s"] - center).abs().idxmin()]


def crop_box(landmarks: pd.DataFrame, frame_indices: list[int], width: int, height: int):
    names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle",
    ]
    indices: set[int] = set()
    for frame_index in frame_indices:
        indices.update(range(max(0, frame_index - 2), min(len(landmarks), frame_index + 3)))
    section = landmarks.iloc[sorted(indices)]
    xs: list[float] = []
    ys: list[float] = []
    for name in names:
        valid = section[f"{name}_confidence"] >= 0.20
        xs.extend(section.loc[valid, f"{name}_x"].dropna().tolist())
        ys.extend(section.loc[valid, f"{name}_y"].dropna().tolist())
    if not xs or not ys:
        return 0, 0, width, height
    x0 = max(0, int((min(xs) - 0.055) * width))
    x1 = min(width, int((max(xs) + 0.055) * width))
    y0 = max(0, int((min(ys) - 0.055) * height))
    y1 = min(height, int((max(ys) + 0.060) * height))
    return x0, y0, x1, y1


def draw_skeleton(
    image: Image.Image,
    row: pd.Series,
    full_width: int,
    full_height: int,
    offset: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0 = offset
    for first, second in CONNECTIONS:
        if row[f"{first}_confidence"] < 0.20 or row[f"{second}_confidence"] < 0.20:
            continue
        p1 = (
            int(row[f"{first}_x"] * full_width) - x0,
            int(row[f"{first}_y"] * full_height) - y0,
        )
        p2 = (
            int(row[f"{second}_x"] * full_width) - x0,
            int(row[f"{second}_y"] * full_height) - y0,
        )
        draw.line([p1, p2], fill=color, width=8)
        for point in (p1, p2):
            draw.ellipse(
                [point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6],
                fill=color,
            )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(PREDICTIONS)
    samples = all_rows[(all_rows["subject"] == SUBJECT) & (all_rows["run"] == RUN)].copy()
    if samples.empty:
        raise RuntimeError(f"No prediction rows for {SUBJECT} run {RUN}")
    landmarks = pd.read_csv(LANDMARKS, encoding="utf-8-sig")

    labels = list(STAGES)
    report = classification_report(
        samples["ground_truth_label"],
        samples["predicted_label"],
        labels=labels,
        target_names=[STAGES[label][0].lower() for label in labels],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "experiment": EXPERIMENT.name,
        "video": str(VIDEO),
        "subject": SUBJECT,
        "run": RUN,
        "speed_m_s": float(samples["speed_m_s"].iloc[0]),
        "sample_count": int(len(samples)),
        "stride_count": int(samples["stride_id"].nunique()),
        "accuracy": float(accuracy_score(samples["ground_truth_label"], samples["predicted_label"])),
        "macro_f1": float(
            f1_score(
                samples["ground_truth_label"],
                samples["predicted_label"],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_stage": {name: report[name] for name in ["landing", "braking", "propulsion", "recovery"]},
        "representative_stride": REPRESENTATIVE_STRIDE,
        "evaluation_note": (
            "Frozen four-stage v3 predictions compared with force-derived stage truth. "
            "No retraining, refitting, or threshold tuning was performed for this test."
        ),
    }

    stride = samples[samples["stride_id"] == REPRESENTATIVE_STRIDE].copy()
    selected: list[pd.Series] = []
    for stage in labels:
        stage_rows = stride[stride["ground_truth_label"] == stage]
        if stage_rows.empty:
            raise RuntimeError(f"No ground-truth samples for stage {stage} in {REPRESENTATIVE_STRIDE}")
        selected.append(choose_temporal_middle(stage_rows))

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_indices = [
        nearest_frame_index(float(row["video_time_s"]), fps, frame_count) for row in selected
    ]
    x0, y0, x1, y1 = crop_box(landmarks, frame_indices, width, height)

    card_width, image_height, text_height = 430, 510, 238
    card_height = image_height + text_height
    cards: list[Image.Image] = []
    manifest = []
    for stage, sample, frame_index in zip(labels, selected, frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(frame_rgb[y0:y1, x0:x1]).convert("RGB")
        correct = bool(sample["correct"])
        verdict_color = GREEN if correct else RED
        draw_skeleton(crop, landmarks.iloc[frame_index], width, height, (x0, y0), verdict_color)
        crop.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
        fitted = Image.new("RGB", (card_width, image_height), BACKGROUND)
        fitted.paste(crop, ((card_width - crop.width) // 2, (image_height - crop.height) // 2))

        card = Image.new("RGB", (card_width, card_height), PANEL)
        card.paste(fitted, (0, 0))
        draw = ImageDraw.Draw(card)
        draw.rectangle((0, 0, card_width - 1, card_height - 1), outline=verdict_color, width=6)
        english, korean = STAGES[stage]
        predicted_stage = STAGES[int(sample["predicted_label"])][1]
        stage_f1 = float(report[english.lower()]["f1-score"])
        draw.text((18, 524), f"{stage}. {english} · {korean}", font=font(27), fill=WHITE)
        if correct:
            draw.text((18, 565), "단계 판정: 정답 일치 ✓", font=font(23), fill=GREEN)
        else:
            draw.text(
                (18, 565),
                f"단계 판정: 오답 → {predicted_stage}",
                font=font(23),
                fill=RED,
            )
        draw.text((18, 602), "자세 Good/Bad: 정답 없음 · 미평가", font=font(20), fill=GRAY)
        draw.text((18, 638), f"전체 영상 단계 F1 {stage_f1:.3f}", font=font(20), fill=WHITE)
        draw.text(
            (18, 674),
            f"영상 {float(sample['video_time_s']):.3f}s  |  frame {frame_index}",
            font=font(18),
            fill=GRAY,
        )
        draw.text(
            (18, 708),
            f"힘판 정답 {korean}  |  모델 출력 {predicted_stage}",
            font=font(18),
            fill=WHITE,
        )
        card_path = REPORT_DIR / f"four_stage_{stage}_{english.lower()}.png"
        card.save(card_path)
        cards.append(card)
        manifest.append(
            {
                "stage": stage,
                "stage_name": english.lower(),
                "frame": frame_index,
                "video_time_s": float(sample["video_time_s"]),
                "ground_truth_label": int(sample["ground_truth_label"]),
                "predicted_label": int(sample["predicted_label"]),
                "phase_prediction_correct": correct,
                "running_form_status": "NOT_EVALUATED_NO_FORM_GROUND_TRUTH",
                "stage_f1": stage_f1,
                "image": str(card_path),
            }
        )
    cap.release()

    header_height = 176
    board = Image.new("RGB", (card_width * 4, header_height + card_height), BACKGROUND)
    draw = ImageDraw.Draw(board)
    draw.text((28, 18), "RunScope · 다른 트레드밀 영상 4단계 고정 테스트", font=font(37), fill=WHITE)
    draw.text(
        (28, 68),
        "초록 = 힘판 단계 정답과 일치  |  빨강 = 단계 오답  |  회색 = 자세 Good/Bad 정답 없음",
        font=font(23),
        fill=GRAY,
    )
    draw.text(
        (28, 106),
        f"Twente {SUBJECT} run{RUN} · 2.75 m/s (9.9 km/h) · {metrics['stride_count']}보폭 · 재학습 없음",
        font=font(21),
        fill=WHITE,
    )
    draw.text(
        (28, 140),
        f"4단계 Macro F1 {metrics['macro_f1']:.3f}  |  Accuracy {metrics['accuracy']:.3f}  |  대표 {REPRESENTATIVE_STRIDE}",
        font=font(21),
        fill=YELLOW,
    )
    for index, card in enumerate(cards):
        board.paste(card, (index * card_width, header_height))

    board_path = REPORT_DIR / "subj11_run99_four_stage_force_test.png"
    manifest_path = REPORT_DIR / "FOUR_STAGE_MANIFEST.json"
    metrics_path = REPORT_DIR / "FOUR_STAGE_REPORT.json"
    board.save(board_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "board": str(board_path),
                "metrics": str(metrics_path),
                "manifest": str(manifest_path),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "stage_f1": {
                    name: values["f1-score"] for name, values in metrics["per_stage"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
