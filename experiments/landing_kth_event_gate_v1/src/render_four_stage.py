from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_v1"
SOURCE_EXPERIMENT = ROOT / "experiments/landing_kth_adaptation_v1"
PREP_SOURCE = SOURCE_EXPERIMENT / "src/prepare_kth_sequences.py"
GATE_SOURCE = EXPERIMENT / "src/train_event_gate.py"
POSE_MODEL = ROOT / "video_v1/models/yolo11n-pose.pt"
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

STEM = "person01_running_d1"
PHASES = {
    1: ("LANDING", "착지", 0.00),
    2: ("BRAKING", "제동", 0.15),
    3: ("PROPULSION", "추진", 0.40),
    4: ("RECOVERY", "회복", 0.70),
}
PHASE_RGB = {
    1: (247, 190, 51),
    2: (239, 83, 80),
    3: (55, 201, 112),
    4: (69, 144, 238),
}
GREEN = (49, 205, 113)
GRAY = (160, 169, 181)
WHITE = (242, 245, 247)
PANEL = (18, 23, 29)
BACKGROUND = (10, 14, 19)

CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module("kth_four_stage_prep", PREP_SOURCE)
gate_source = load_module("kth_four_stage_gate", GATE_SOURCE)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def source_frame_indices(ranges: list[tuple[int, int]]) -> list[int]:
    output = []
    for start, end in ranges:
        output.extend(range(start, end + 1))
    return output


def gate_probabilities(bundle, gate_bundle) -> tuple[np.ndarray, np.ndarray]:
    events = bundle["teacher_events"].astype(int)
    x = np.stack([
        gate_source.context_features(
            bundle["features"], bundle["teacher_probability"], int(event)
        )
        for event in events
    ])
    return events, gate_bundle["model"].predict_proba(x)[:, 1]


def select_stride(events: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[int, int, int]:
    best = None
    for index in range(len(events) - 2):
        candidate = events[index:index + 3]
        gaps = np.diff(candidate)
        if not np.all((gaps >= 8) & (gaps <= 16)):
            continue
        scores = probabilities[index:index + 3]
        if not np.all(scores >= threshold):
            continue
        ranking = (float(scores.min()), float(scores.mean()))
        if best is None or ranking > best[0]:
            best = (ranking, tuple(int(value) for value in candidate))
    if best is None:
        raise RuntimeError("No three consecutive gated contacts form a plausible stride")
    return best[1]


def primary_pose(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or not len(result.boxes) or result.keypoints is None:
        raise RuntimeError("Pose was not detected in a selected stage frame")
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    xy = result.keypoints.xy.detach().cpu().numpy()
    if result.keypoints.conf is None:
        confidence = np.ones(xy.shape[:2], dtype=np.float32)
    else:
        confidence = result.keypoints.conf.detach().cpu().numpy()
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    selected = int(np.argmax(areas))
    return xy[selected], confidence[selected], boxes[selected]


def draw_pose(
    frame: np.ndarray,
    xy: np.ndarray,
    confidence: np.ndarray,
    tracked_side: str,
    phase: int,
) -> np.ndarray:
    output = frame.copy()
    phase_bgr = tuple(reversed(PHASE_RGB[phase]))
    active = {11, 13, 15} if tracked_side == "left" else {12, 14, 16}
    for start, end in CONNECTIONS:
        if min(confidence[start], confidence[end]) < 0.12:
            continue
        p1 = tuple(np.rint(xy[start]).astype(int))
        p2 = tuple(np.rint(xy[end]).astype(int))
        is_active = start in active and end in active
        color = phase_bgr if is_active else (205, 205, 205)
        # KTH is only 160x120. Keep source-space marks thin so upscaling does
        # not hide the runner behind oversized joints.
        thickness = 2 if is_active else 1
        radius = 2 if is_active else 1
        cv2.line(output, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.circle(output, p1, radius, color, -1, cv2.LINE_AA)
        cv2.circle(output, p2, radius, color, -1, cv2.LINE_AA)
    return output


def crop_runner(frame: np.ndarray, box: np.ndarray) -> Image.Image:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    box_width = max(x1 - x0, 1.0)
    box_height = max(y1 - y0, 1.0)
    x0 = max(0, int(x0 - box_width * 0.28))
    x1 = min(width, int(x1 + box_width * 0.28))
    y0 = max(0, int(y0 - box_height * 0.12))
    y1 = min(height, int(y1 + box_height * 0.12))
    crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)


def main() -> None:
    sequence_path = SOURCE_EXPERIMENT / f"artifacts/sequences/{STEM}.npz"
    bundle = np.load(sequence_path)
    gate_bundle = joblib.load(EXPERIMENT / "artifacts/landing_kth_event_gate_v1.joblib")
    events, probabilities = gate_probabilities(bundle, gate_bundle)
    landing, opposite_contact, next_same_foot = select_stride(
        events, probabilities, float(gate_bundle["threshold"])
    )
    stride_length = next_same_foot - landing
    stage_frames_30hz = {
        phase: int(round(landing + fraction * stride_length))
        for phase, (_, _, fraction) in PHASES.items()
    }

    annotations = prep.parse_annotations()
    ranges = annotations[STEM]
    mapping = source_frame_indices(ranges)
    source_frames = {
        phase: mapping[min(len(mapping) - 1, int(round(frame * 25.0 / 30.0)))]
        for phase, frame in stage_frames_30hz.items()
    }
    video_path = prep.locate_video(STEM)
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    for phase in PHASES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_frames[phase])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read source frame {source_frames[phase]}")
        frames.append(frame)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    pose_model = YOLO(str(POSE_MODEL))
    results = pose_model.predict(
        source=frames,
        imgsz=640,
        conf=0.10,
        iou=0.50,
        device="cpu",
        batch=4,
        stream=False,
        verbose=False,
    )
    poses = [primary_pose(result) for result in results]
    landing_xy = poses[0][0]
    tracked_side = "left" if landing_xy[15, 1] >= landing_xy[16, 1] else "right"
    landing_probability = float(probabilities[np.where(events == landing)[0][0]])

    card_width, image_height, info_height = 360, 465, 190
    cards = []
    metadata = []
    for phase, frame, result_pose in zip(PHASES, frames, poses):
        english, korean, fraction = PHASES[phase]
        xy, confidence, box = result_pose
        overlay = draw_pose(frame, xy, confidence, tracked_side, phase)
        crop = crop_runner(overlay, box)
        crop = ImageOps.contain(crop, (card_width, image_height), Image.Resampling.LANCZOS)
        image_panel = Image.new("RGB", (card_width, image_height), (7, 10, 14))
        image_panel.paste(crop, ((card_width - crop.width) // 2, (image_height - crop.height) // 2))

        card = Image.new("RGB", (card_width, image_height + info_height), PANEL)
        card.paste(image_panel, (0, 0))
        draw = ImageDraw.Draw(card)
        color = PHASE_RGB[phase]
        draw.rectangle((0, 0, card_width - 1, image_height + info_height - 1), outline=color, width=5)
        draw.text((16, image_height + 14), f"{phase}. {english} · {korean}", font=font(25), fill=color)
        if phase == 1:
            draw.text(
                (16, image_height + 53),
                f"착지 게이트 통과  {landing_probability:.3f}",
                font=font(21), fill=GREEN,
            )
        else:
            draw.text(
                (16, image_height + 53),
                f"스트라이드 {fraction * 100:.0f}% · 운동학 추정",
                font=font(20), fill=GRAY,
            )
        draw.text((16, image_height + 88), "자세 Good/Bad · 미평가", font=font(20), fill=GRAY)
        draw.text(
            (16, image_height + 122),
            f"30Hz frame {stage_frames_30hz[phase]}  |  원본 {source_frames[phase]}",
            font=font(17), fill=WHITE,
        )
        draw.text(
            (16, image_height + 151),
            f"원본 시간 {source_frames[phase] / fps:.2f}s  |  {tracked_side.upper()} FOOT",
            font=font(17), fill=GRAY,
        )
        cards.append(card)
        metadata.append({
            "phase": phase,
            "english": english,
            "korean": korean,
            "stride_fraction": fraction,
            "resampled_frame_30hz": stage_frames_30hz[phase],
            "source_frame_25hz": source_frames[phase],
            "source_time_s": source_frames[phase] / fps,
            "status": "gate_accepted" if phase == 1 else "kinematic_stride_fraction_estimate",
        })

    header_height = 150
    footer_height = 88
    board = Image.new(
        "RGB", (card_width * 4, header_height + image_height + info_height + footer_height), BACKGROUND
    )
    draw = ImageDraw.Draw(board)
    draw.text((26, 18), "RunScope · KTH 러닝 영상 4단계 출력", font=font(36), fill=WHITE)
    draw.text(
        (26, 66),
        f"{STEM} · 검증 영상 · 한 스트라이드 {landing}→{next_same_foot} frame · Event Gate",
        font=font(22), fill=GREEN,
    )
    draw.text(
        (26, 104),
        "색 관절선 = 단계 구분이며 자세가 Good이라는 뜻은 아님",
        font=font(21), fill=GRAY,
    )
    for index, card in enumerate(cards):
        board.paste(card, (index * card_width, header_height))
    footer_y = header_height + image_height + info_height
    draw.text(
        (26, footer_y + 18),
        "1단계 착지 = 학습 모델 판정  |  2~4단계 = 착지 사이 시간비율 기반 대표 장면  |  힘판 정답 없음",
        font=font(21), fill=WHITE,
    )
    draw.text(
        (26, footer_y + 51),
        f"게이트 기준 {float(gate_bundle['threshold']):.2f} · 반대발 접촉 후보 {opposite_contact} · KTH 공식 테스트 미사용",
        font=font(18), fill=GRAY,
    )

    report_dir = EXPERIMENT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{STEM}_four_stage_output.png"
    board.save(output, optimize=True)
    report = {
        "source_video": str(video_path),
        "source_dataset_role": "KTH official validation split",
        "landing_model": "landing_kth_event_gate_v1",
        "landing_gate_threshold": float(gate_bundle["threshold"]),
        "landing_gate_probability": landing_probability,
        "landing_frame_30hz": landing,
        "opposite_contact_frame_30hz": opposite_contact,
        "next_same_foot_frame_30hz": next_same_foot,
        "tracked_side_from_landing_pose": tracked_side,
        "stages": metadata,
        "limitations": [
            "Only landing is selected by the trained event gate.",
            "Braking, propulsion, and recovery frames use fixed stride fractions for visualization.",
            "No force-plate or manual four-stage ground truth is available for KTH.",
            "Skeleton color identifies phase, not running-form quality.",
        ],
        "output": str(output),
    }
    (report_dir / f"{STEM}_four_stage_output.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
