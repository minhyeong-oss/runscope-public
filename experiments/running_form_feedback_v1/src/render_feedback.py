from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = EXPERIMENT / "reports"
DEFAULT_LANDMARKS = ROOT / (
    "video_v1/test_outputs/twente_subj05_run81_pilot/"
    "Subj05_run81_side_10s_to_22s_yolo_landmarks.csv"
)
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

STAGES = {
    1: ("LANDING", "착지"),
    2: ("BRAKING", "제동"),
    3: ("PROPULSION", "추진"),
    4: ("RECOVERY", "회복"),
}
STATUS_COLORS = {
    "GOOD": (38, 214, 112),
    "IMPROVE": (240, 76, 84),
    "UNKNOWN": (151, 160, 174),
}
STATUS_CSS = {
    "GOOD": "#26d670",
    "IMPROVE": "#f04c54",
    "UNKNOWN": "#97a0ae",
}
WHITE = (242, 245, 247)
MUTED = (174, 183, 194)
PANEL = (17, 22, 29)
BACKGROUND = (8, 12, 17)
SKELETON_GRAY = (205, 211, 219)

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


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def wrap_korean(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    face: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    width: int,
    line_height: int,
    max_lines: int | None = None,
) -> int:
    lines = wrap_korean(text, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += line_height
    return y


def crop_box(
    landmarks: pd.DataFrame,
    frame_indices: list[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    names = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    xs: list[float] = []
    ys: list[float] = []
    for frame_index in frame_indices:
        section = landmarks.iloc[max(0, frame_index - 2) : min(len(landmarks), frame_index + 3)]
        for name in names:
            good = section[f"{name}_confidence"] >= 0.25
            xs.extend(section.loc[good, f"{name}_x"].dropna().tolist())
            ys.extend(section.loc[good, f"{name}_y"].dropna().tolist())
    if not xs or not ys:
        return 0, 0, width, height
    x0 = max(0, int((min(xs) - 0.035) * width))
    x1 = min(width, int((max(xs) + 0.035) * width))
    y0 = max(0, int((min(ys) - 0.035) * height))
    y1 = min(height, int((max(ys) + 0.045) * height))
    return x0, y0, x1, y1


def local_point(
    row: pd.Series,
    joint: str,
    full_width: int,
    full_height: int,
    offset: tuple[int, int],
) -> tuple[int, int]:
    x0, y0 = offset
    return (
        int(float(row[f"{joint}_x"]) * full_width) - x0,
        int(float(row[f"{joint}_y"]) * full_height) - y0,
    )


def draw_skeleton(
    image: Image.Image,
    row: pd.Series,
    tracked_side: str,
    color: tuple[int, int, int],
    full_width: int,
    full_height: int,
    offset: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    for first, second in CONNECTIONS:
        if float(row[f"{first}_confidence"]) < 0.25 or float(row[f"{second}_confidence"]) < 0.25:
            continue
        p1 = local_point(row, first, full_width, full_height, offset)
        p2 = local_point(row, second, full_width, full_height, offset)
        draw.line([p1, p2], fill=SKELETON_GRAY, width=5)

    active = [
        f"{tracked_side}_shoulder",
        f"{tracked_side}_hip",
        f"{tracked_side}_knee",
        f"{tracked_side}_ankle",
    ]
    for first, second in zip(active, active[1:]):
        if float(row[f"{first}_confidence"]) < 0.25 or float(row[f"{second}_confidence"]) < 0.25:
            continue
        p1 = local_point(row, first, full_width, full_height, offset)
        p2 = local_point(row, second, full_width, full_height, offset)
        draw.line([p1, p2], fill=color, width=10)
        for point_xy in (p1, p2):
            draw.ellipse(
                [point_xy[0] - 7, point_xy[1] - 7, point_xy[0] + 7, point_xy[1] + 7],
                fill=color,
                outline=WHITE,
                width=2,
            )


def measurement_label(stage: int, row: pd.Series) -> str:
    if stage == 1:
        return f"정강이 수직 기울기 {float(row['tibia_tilt_deg']):.1f}°"
    if stage == 2:
        return f"보폭 내 최대 무릎 굽힘 {float(row['knee_flexion_deg']):.1f}°"
    if stage == 3:
        return f"고관절 신전 대리값 {float(row['hip_extension_proxy_deg']):.1f}°"
    return f"무릎 굽힘 {float(row['knee_flexion_deg']):.1f}° · 참고값"


def render_card(
    frame_rgb: Any,
    landmark_row: pd.Series,
    measurement: pd.Series,
    assessment: dict[str, Any],
    stage_f1: float,
    crop: tuple[int, int, int, int],
    full_width: int,
    full_height: int,
) -> Image.Image:
    card_width, image_height, text_height = 480, 620, 405
    card = Image.new("RGB", (card_width, image_height + text_height), PANEL)
    x0, y0, x1, y1 = crop
    scene = Image.fromarray(frame_rgb[y0:y1, x0:x1]).convert("RGB")
    color = STATUS_COLORS[assessment["status"]]
    draw_skeleton(
        scene,
        landmark_row,
        str(measurement["tracked_side"]),
        color,
        full_width,
        full_height,
        (x0, y0),
    )
    scene.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
    fitted = Image.new("RGB", (card_width, image_height), BACKGROUND)
    fitted.paste(scene, ((card_width - scene.width) // 2, (image_height - scene.height) // 2))
    card.paste(fitted, (0, 0))

    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, card.width - 1, card.height - 1), outline=color, width=7)
    english, korean = STAGES[int(measurement["stage"])]
    y = image_height + 18
    draw.text((20, y), f"{int(measurement['stage'])}. {english} · {korean}", font=font(28), fill=WHITE)
    y += 43
    draw.rounded_rectangle((18, y, 194, y + 42), radius=12, fill=color)
    draw.text((32, y + 4), assessment["status_ko"], font=font(24), fill=(8, 12, 17))
    y += 56
    draw.text((20, y), assessment["headline_ko"], font=font(23), fill=color)
    y += 38
    draw.text((20, y), measurement_label(int(measurement["stage"]), measurement), font=font(20), fill=WHITE)
    y += 33
    y = draw_wrapped(draw, (20, y), assessment["evidence_ko"], font(19), MUTED, 35, 28, 2)
    y += 5
    y = draw_wrapped(draw, (20, y), assessment["action_ko"], font(18), WHITE, 39, 27, 3)
    if assessment["remediation_links"]:
        link = assessment["remediation_links"][0]
        draw.text((20, min(y + 4, card.height - 78)), "개선 영상 ▶ Learn2Run · DPT", font=font(18), fill=color)
        draw.text((20, min(y + 32, card.height - 48)), "youtube.com/watch?v=vLh2nCNMtPA", font=font(15), fill=MUTED)
    draw.text((20, card.height - 28), f"4단계 인식 F1 {stage_f1:.3f} · 자세 F1 아님", font=font(14), fill=MUTED)
    return card


def write_html(
    output_path: Path,
    report: dict[str, Any],
    stage_images: list[Path],
) -> None:
    cards: list[str] = []
    for assessment, image_path in zip(report["assessments"], stage_images):
        links = ""
        if assessment["remediation_links"]:
            rendered_links = []
            for link in assessment["remediation_links"]:
                rendered_links.append(
                    f'<a href="{html.escape(link["url"])}" target="_blank" rel="noopener">'
                    f'{html.escape(link["title"])} · {html.escape(link["channel"])}</a>'
                )
            links = '<div class="links"><strong>개선 영상</strong>' + "".join(rendered_links) + "</div>"
        stage_number = int(assessment["stage"])
        cards.append(
            f"""
            <article class="card {assessment['status'].lower()}">
              <img src="{html.escape(image_path.name)}" alt="{stage_number}단계 {html.escape(assessment['stage_name_ko'])} 분석">
              <div class="body">
                <span class="status">{html.escape(assessment['status_ko'])}</span>
                <h2>{stage_number}. {html.escape(assessment['stage_name_ko'])}</h2>
                <h3>{html.escape(assessment['headline_ko'])}</h3>
                <p>{html.escape(assessment['evidence_ko'])}</p>
                <p>{html.escape(assessment['action_ko'])}</p>
                {links}
              </div>
            </article>
            """
        )

    evidence_links = "".join(
        f'<li><a href="{html.escape(item["url"])}" target="_blank" rel="noopener">'
        f'{html.escape(item["title"])}</a></li>'
        for item in report["evidence"]
    )
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RunScope running-form feedback v1</title>
  <style>
    :root {{ color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif; }}
    body {{ margin: 0; background: #080c11; color: #f2f5f7; }}
    main {{ max-width: 1500px; margin: auto; padding: 32px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    .subtitle, .note {{ color: #aeb7c2; line-height: 1.55; }}
    .legend {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 18px 0; }}
    .legend span, .status {{ border-radius: 999px; padding: 7px 13px; font-weight: 800; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
    .card {{ background: #11161d; border: 3px solid #97a0ae; border-radius: 16px; overflow: hidden; }}
    .card.good {{ border-color: #26d670; }} .card.improve {{ border-color: #f04c54; }}
    .card img {{ display: block; width: 100%; height: auto; }}
    .body {{ padding: 20px; }} h2 {{ margin: 18px 0 8px; }} h3 {{ margin: 8px 0; }}
    p {{ line-height: 1.55; color: #d2d8df; }}
    .good .status {{ background: #26d670; color: #080c11; }}
    .improve .status {{ background: #f04c54; color: #080c11; }}
    .unknown .status {{ background: #97a0ae; color: #080c11; }}
    .links {{ display: grid; gap: 8px; padding-top: 10px; }}
    a {{ color: #7cc4ff; }} footer {{ margin-top: 28px; border-top: 1px solid #2b3440; padding-top: 20px; }}
    @media (max-width: 850px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
  </style>
</head>
<body><main>
  <header>
    <h1>RunScope · 측면 러닝 4단계 자세 피드백</h1>
    <p class="subtitle">근거 기반 스크리닝 · {report['usable_strides']}보폭 집계 · 의료 진단 아님</p>
    <div class="legend">
      <span style="background:#26d670;color:#080c11">양호</span>
      <span style="background:#f04c54;color:#080c11">개선 권장</span>
      <span style="background:#97a0ae;color:#080c11">판정 불가</span>
    </div>
    <p class="note">4단계 인식 Macro F1 {report['phase_recognition']['macro_f1']:.3f}. 이 수치는 자세 Good/Bad 성능이 아닙니다. 자세 정답 라벨이 없어 자세 F1은 계산하지 않습니다.</p>
  </header>
  <section class="grid">{''.join(cards)}</section>
  <footer>
    <p><strong>한계:</strong> {html.escape(report['disclaimer_ko'])} 측면 영상에서는 후면 지표인 무릎 외반, 골반 하강, 발뒤꿈치 외번을 판정하지 않습니다.</p>
    <h2>판정 근거</h2><ul>{evidence_links}</ul>
  </footer>
</main></body></html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render RunScope four-stage form feedback")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_path = args.report_dir / "FORM_FEEDBACK.json"
    measurements_path = args.report_dir / "stride_measurements.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    measurements = pd.read_csv(measurements_path)
    landmarks = pd.read_csv(args.landmarks)

    representative = measurements[
        measurements["stride_id"] == report["representative_stride"]
    ].sort_values("stage")
    if representative["stage"].nunique() != 4:
        raise RuntimeError("Representative stride does not contain four stages")

    video_path = Path(report["video"])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_indices = representative["frame_index"].astype(int).tolist()
    crop = crop_box(landmarks, frame_indices, width, height)

    per_stage_f1 = report["phase_recognition"]["per_stage_f1"]
    stage_f1 = {
        1: per_stage_f1["landing"],
        2: per_stage_f1["braking"],
        3: per_stage_f1["propulsion"],
        4: per_stage_f1["recovery"],
    }

    cards: list[Image.Image] = []
    stage_images: list[Path] = []
    for _, measurement in representative.iterrows():
        stage = int(measurement["stage"])
        assessment = report["assessments"][stage - 1]
        frame_index = int(measurement["frame_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        card = render_card(
            frame_rgb=frame_rgb,
            landmark_row=landmarks.iloc[frame_index],
            measurement=measurement,
            assessment=assessment,
            stage_f1=float(stage_f1[stage]),
            crop=crop,
            full_width=width,
            full_height=height,
        )
        image_path = args.report_dir / f"stage_{stage}_{STAGES[stage][0].lower()}_feedback.png"
        card.save(image_path, optimize=True)
        stage_images.append(image_path)
        cards.append(card)
    cap.release()

    header_height = 205
    card_width, card_height = cards[0].size
    board = Image.new("RGB", (card_width * 4, header_height + card_height), BACKGROUND)
    draw = ImageDraw.Draw(board)
    draw.text((32, 24), "RunScope · 측면 러닝 4단계 자세 피드백", font=font(40), fill=WHITE)
    draw.text(
        (32, 78),
        f"근거 기반 스크리닝 · {report['usable_strides']}보폭 집계 · 의료 진단 아님",
        font=font(25),
        fill=MUTED,
    )
    draw.text((32, 120), "초록 양호", font=font(23), fill=STATUS_COLORS["GOOD"])
    draw.text((185, 120), "빨강 개선 권장", font=font(23), fill=STATUS_COLORS["IMPROVE"])
    draw.text((405, 120), "회색 판정 불가", font=font(23), fill=STATUS_COLORS["UNKNOWN"])
    draw.text(
        (32, 158),
        f"4단계 Macro F1 {report['phase_recognition']['macro_f1']:.3f} · 자세 Good/Bad 정답이 없어 자세 F1은 계산하지 않음",
        font=font(21),
        fill=WHITE,
    )
    for index, card in enumerate(cards):
        board.paste(card, (index * card_width, header_height))
    board_path = args.report_dir / "running_form_feedback_board.png"
    board.save(board_path, optimize=True)

    html_path = args.report_dir / "running_form_feedback.html"
    write_html(html_path, report, stage_images)
    print(
        json.dumps(
            {
                "board": str(board_path),
                "html": str(html_path),
                "stage_images": [str(path) for path in stage_images],
                "representative_stride": report["representative_stride"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
