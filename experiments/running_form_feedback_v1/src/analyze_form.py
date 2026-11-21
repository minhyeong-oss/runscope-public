from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "test_videos/twente/Subj05_run81_side_10s_to_22s.mp4"
DEFAULT_LANDMARKS = ROOT / (
    "video_v1/test_outputs/twente_subj05_run81_pilot/"
    "Subj05_run81_side_10s_to_22s_yolo_landmarks.csv"
)
DEFAULT_COMPARISON = ROOT / (
    "video_v1/test_outputs/twente_subj05_run81_pilot/temporal_prior_v2/"
    "twente_temporal_prior_v2_comparison.csv"
)
DEFAULT_PHASE_METRICS = ROOT / (
    "video_v1/test_outputs/twente_subj05_run81_pilot/temporal_prior_v2/"
    "twente_temporal_prior_v2_metrics.json"
)
DEFAULT_CONFIG = EXPERIMENT / "config/evidence_and_remediation.json"
DEFAULT_REPORT_DIR = EXPERIMENT / "reports"

MIN_STRIDES = 5
MIN_CONFIDENCE = 0.50
LANDING_GOOD_MEDIAN_TIBIA_DEG = 5.0
LANDING_GOOD_STRIDE_TIBIA_DEG = 8.0
LANDING_GOOD_CONSISTENCY = 0.80
BRAKING_LOW_KNEE_FLEXION_DEG = 40.0

STAGES = {
    1: ("landing", "착지"),
    2: ("braking", "제동"),
    3: ("propulsion", "추진"),
    4: ("recovery", "회복"),
}


@dataclass(frozen=True)
class Assessment:
    stage: int
    stage_name: str
    stage_name_ko: str
    status: str
    status_ko: str
    headline_ko: str
    evidence_ko: str
    action_ko: str
    problem_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def point(row: pd.Series, joint: str, width: int, height: int) -> np.ndarray:
    return np.array([float(row[f"{joint}_x"]) * width, float(row[f"{joint}_y"]) * height])


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the smaller 2D angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator <= 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def line_tilt_from_vertical(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    if float(np.linalg.norm(delta)) <= 1e-9:
        return float("nan")
    return math.degrees(math.atan2(abs(float(delta[0])), abs(float(delta[1]))))


def minimum_confidence(row: pd.Series, joints: list[str]) -> float:
    return min(float(row[f"{joint}_confidence"]) for joint in joints)


def video_metadata(video_path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, frames, width, height


def nearest_landmark_row(
    landmarks: pd.DataFrame, video_time_s: float, clip_offset_s: float, fps: float
) -> tuple[int, pd.Series]:
    index = int(round((video_time_s - clip_offset_s) * fps))
    index = max(0, min(len(landmarks) - 1, index))
    return index, landmarks.iloc[index]


def select_stage_row(group: pd.DataFrame, stage: int) -> pd.Series | None:
    stage_rows = group[group["v2_predicted_label"] == stage].sort_values("video_time_s")
    if stage_rows.empty:
        return None
    return stage_rows.iloc[len(stage_rows) // 2]


def stride_measurements(
    comparison: pd.DataFrame,
    landmarks: pd.DataFrame,
    fps: float,
    width: int,
    height: int,
    clip_offset_s: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for stride_id, group in comparison.groupby("stride_id", sort=True):
        tracked_side = str(group["tracked_side"].iloc[0])
        if tracked_side not in {"left", "right"}:
            continue

        for stage in STAGES:
            stage_group = group[group["v2_predicted_label"] == stage].sort_values("video_time_s")
            if stage_group.empty:
                continue

            candidates: list[dict[str, Any]] = []
            for _, sample in stage_group.iterrows():
                frame_index, row = nearest_landmark_row(
                    landmarks, float(sample["video_time_s"]), clip_offset_s, fps
                )
                hip = point(row, f"{tracked_side}_hip", width, height)
                knee = point(row, f"{tracked_side}_knee", width, height)
                ankle = point(row, f"{tracked_side}_ankle", width, height)
                shoulder = point(row, f"{tracked_side}_shoulder", width, height)
                knee_flexion = 180.0 - joint_angle(hip, knee, ankle)
                tibia_tilt = line_tilt_from_vertical(knee, ankle)
                hip_extension_proxy = 180.0 - joint_angle(shoulder, hip, knee)
                confidence = minimum_confidence(
                    row,
                    [
                        f"{tracked_side}_shoulder",
                        f"{tracked_side}_hip",
                        f"{tracked_side}_knee",
                        f"{tracked_side}_ankle",
                    ],
                )
                candidates.append(
                    {
                        "sample": sample,
                        "frame_index": frame_index,
                        "knee_flexion_deg": knee_flexion,
                        "tibia_tilt_deg": tibia_tilt,
                        "hip_extension_proxy_deg": hip_extension_proxy,
                        "joint_confidence": confidence,
                    }
                )

            if stage == 2:
                chosen = max(candidates, key=lambda item: item["knee_flexion_deg"])
            elif stage == 3:
                chosen = candidates[-1]
            else:
                chosen = candidates[len(candidates) // 2]

            sample = chosen.pop("sample")
            records.append(
                {
                    "stride_id": stride_id,
                    "tracked_side": tracked_side,
                    "stage": stage,
                    "stage_name": STAGES[stage][0],
                    "video_time_s": float(sample["video_time_s"]),
                    **chosen,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {
        "median": float(clean.median()),
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
    }


def unknown_assessment(stage: int, reason: str) -> Assessment:
    english, korean = STAGES[stage]
    return Assessment(
        stage=stage,
        stage_name=english,
        stage_name_ko=korean,
        status="UNKNOWN",
        status_ko="판정 불가",
        headline_ko="현재 영상만으로 확정하지 않음",
        evidence_ko=reason,
        action_ko="회색은 오류가 아니라 근거 또는 영상 조건이 부족하다는 뜻입니다.",
    )


def classify_landing(rows: pd.DataFrame) -> Assessment:
    stage = 1
    if len(rows) < MIN_STRIDES:
        return unknown_assessment(stage, f"사용 가능한 보폭 {len(rows)}개: 최소 {MIN_STRIDES}개 필요")
    confidence = float(rows["joint_confidence"].median())
    if confidence < MIN_CONFIDENCE:
        return unknown_assessment(stage, f"관련 관절 중앙 신뢰도 {confidence:.2f}: {MIN_CONFIDENCE:.2f} 미만")
    stats = summarize(rows["tibia_tilt_deg"])
    consistency = float((rows["tibia_tilt_deg"] <= LANDING_GOOD_STRIDE_TIBIA_DEG).mean())
    if stats["median"] <= LANDING_GOOD_MEDIAN_TIBIA_DEG and consistency >= LANDING_GOOD_CONSISTENCY:
        return Assessment(
            stage=stage,
            stage_name="landing",
            stage_name_ko="착지",
            status="GOOD",
            status_ko="양호",
            headline_ko="정강이가 수직에 가깝게 반복됨",
            evidence_ko=(
                f"{len(rows)}보폭 중앙값 {stats['median']:.1f}° · "
                f"8° 이내 {consistency * 100:.0f}%"
            ),
            action_ko="현재 패턴을 유지하되 통증과 피로가 생기면 별도로 확인하세요.",
        )
    return Assessment(
        stage=stage,
        stage_name="landing",
        stage_name_ko="착지",
        status="IMPROVE",
        status_ko="개선 권장",
        headline_ko="착지 정강이 기울기 재확인 권장",
        evidence_ko=(
            f"{len(rows)}보폭 중앙값 {stats['median']:.1f}° · "
            f"8° 이내 {consistency * 100:.0f}%"
        ),
        action_ko="보폭을 억지로 줄이기보다 짧은 구간에서 자연스럽게 발이 몸 가까이 닿는지 확인하세요.",
        problem_code="landing_tibia_alignment",
    )


def classify_braking(rows: pd.DataFrame) -> Assessment:
    stage = 2
    if len(rows) < MIN_STRIDES:
        return unknown_assessment(stage, f"사용 가능한 보폭 {len(rows)}개: 최소 {MIN_STRIDES}개 필요")
    confidence = float(rows["joint_confidence"].median())
    if confidence < MIN_CONFIDENCE:
        return unknown_assessment(stage, f"관련 관절 중앙 신뢰도 {confidence:.2f}: {MIN_CONFIDENCE:.2f} 미만")
    stats = summarize(rows["knee_flexion_deg"])
    if stats["median"] < BRAKING_LOW_KNEE_FLEXION_DEG:
        return Assessment(
            stage=stage,
            stage_name="braking",
            stage_name_ko="제동",
            status="IMPROVE",
            status_ko="개선 권장",
            headline_ko="지지기 무릎 굽힘이 작은 편",
            evidence_ko=(
                f"{len(rows)}보폭 최대 굽힘 중앙값 {stats['median']:.1f}° "
                f"(IQR {stats['q1']:.1f}–{stats['q3']:.1f}°)"
            ),
            action_ko="짧은 러닝 구간에서 부드러운 착지와 자연스러운 보폭을 연습하고 다시 촬영하세요.",
            problem_code="limited_stance_knee_flexion",
        )
    return Assessment(
        stage=stage,
        stage_name="braking",
        stage_name_ko="제동",
        status="GOOD",
        status_ko="양호",
        headline_ko="지지기 무릎 굽힘이 기준 범위에 도달",
        evidence_ko=(
            f"{len(rows)}보폭 최대 굽힘 중앙값 {stats['median']:.1f}° "
            f"(IQR {stats['q1']:.1f}–{stats['q3']:.1f}°)"
        ),
        action_ko="현재 패턴을 유지하고 동일한 촬영 조건에서 변화를 비교하세요.",
    )


def classify_propulsion(rows: pd.DataFrame) -> Assessment:
    if len(rows) < MIN_STRIDES:
        return unknown_assessment(3, f"사용 가능한 보폭 {len(rows)}개: 최소 {MIN_STRIDES}개 필요")
    stats = summarize(rows["hip_extension_proxy_deg"])
    return unknown_assessment(
        3,
        (
            f"고관절 신전 대리값 중앙값 {stats['median']:.1f}°를 측정했지만, "
            "속도·개인차를 반영한 합의 절단값이 없음"
        ),
    )


def classify_recovery(rows: pd.DataFrame) -> Assessment:
    if len(rows) < MIN_STRIDES:
        return unknown_assessment(4, f"사용 가능한 보폭 {len(rows)}개: 최소 {MIN_STRIDES}개 필요")
    return unknown_assessment(4, "측면 2D 회복 단계에 적용할 검증된 v1 Good/Bad 규칙을 선택하지 않음")


def attach_remediation(
    assessments: list[Assessment], config: dict[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    remediation = config.get("remediation", {})
    for assessment in assessments:
        item = assessment.as_dict()
        links: list[dict[str, Any]] = []
        if assessment.status == "IMPROVE" and assessment.problem_code:
            links = remediation.get(assessment.problem_code, [])
        item["remediation_links"] = links
        output.append(item)
    return output


def choose_representative_stride(measurements: pd.DataFrame) -> str:
    candidates = measurements.groupby("stride_id")["stage"].nunique()
    complete = candidates[candidates == 4].index.tolist()
    if not complete:
        raise RuntimeError("No stride contains all four predicted stages")
    return complete[len(complete) // 2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidence-informed side-view running-form screen")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--phase-metrics", type=Path, default=DEFAULT_PHASE_METRICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--clip-offset-s", type=float, default=10.0)
    parser.add_argument("--representative-stride", default="stride_009")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fps, frame_count, width, height = video_metadata(args.video)
    landmarks = pd.read_csv(args.landmarks)
    comparison = pd.read_csv(args.comparison)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    phase_metrics = json.loads(args.phase_metrics.read_text(encoding="utf-8"))["temporal_prior_v2"]

    measurements = stride_measurements(
        comparison=comparison,
        landmarks=landmarks,
        fps=fps,
        width=width,
        height=height,
        clip_offset_s=args.clip_offset_s,
    )
    if measurements.empty:
        raise RuntimeError("No stride measurements were produced")

    assessments = [
        classify_landing(measurements[measurements["stage"] == 1]),
        classify_braking(measurements[measurements["stage"] == 2]),
        classify_propulsion(measurements[measurements["stage"] == 3]),
        classify_recovery(measurements[measurements["stage"] == 4]),
    ]
    representative_stride = args.representative_stride
    if representative_stride not in set(measurements["stride_id"]):
        representative_stride = choose_representative_stride(measurements)

    report = {
        "schema_version": "1.0",
        "system": "RunScope running-form feedback v1",
        "video": str(args.video.resolve()),
        "landmarks": str(args.landmarks.resolve()),
        "comparison": str(args.comparison.resolve()),
        "clip_offset_s": args.clip_offset_s,
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        },
        "usable_strides": int(measurements["stride_id"].nunique()),
        "representative_stride": representative_stride,
        "phase_recognition": {
            "macro_f1": float(phase_metrics["macro_f1"]),
            "per_stage_f1": {
                stage_name: float(values["f1-score"])
                for stage_name, values in phase_metrics["per_stage"].items()
            },
            "note_ko": "4단계 인식 F1이며 자세 양호/개선 권장 F1이 아닙니다.",
        },
        "form_feedback_has_ground_truth": False,
        "form_feedback_f1": None,
        "form_feedback_f1_note_ko": "자세 Good/Bad 정답 라벨이 없어 F1을 계산하지 않습니다.",
        "assessments": attach_remediation(assessments, config),
        "quality_gate": {
            "minimum_strides": MIN_STRIDES,
            "minimum_median_joint_confidence": MIN_CONFIDENCE,
        },
        "disclaimer_ko": config["disclaimer_ko"],
        "evidence": config["evidence"],
    }

    measurements_path = args.output_dir / "stride_measurements.csv"
    report_path = args.output_dir / "FORM_FEEDBACK.json"
    measurements.to_csv(measurements_path, index=False)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "measurements": str(measurements_path),
                "strides": report["usable_strides"],
                "statuses": {
                    item["stage_name"]: item["status"] for item in report["assessments"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
