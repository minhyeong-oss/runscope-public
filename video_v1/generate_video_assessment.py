from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def percent_difference(left: float, right: float) -> float:
    denominator = (abs(left) + abs(right)) / 2.0
    return abs(left - right) / denominator * 100.0 if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a technical running-video assessment")
    parser.add_argument("source_video", type=Path)
    parser.add_argument("phase_summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.phase_summary_csv)
    rows: list[dict] = []
    for _, stride in summary.iterrows():
        duration_ms = float(stride["duration_s"]) * 1000.0
        landing_ms = float(stride["braking_start_ms"] - stride["landing_start_ms"])
        braking_ms = float(stride["propulsion_start_ms"] - stride["braking_start_ms"])
        propulsion_ms = float(stride["recovery_start_ms"] - stride["propulsion_start_ms"])
        recovery_ms = duration_ms - float(stride["recovery_start_ms"])
        tracked_side = str(stride["tracked_side_coco"])
        start = int(stride["start_frame_inclusive"])
        end = int(stride["end_frame_exclusive"])
        features_path = args.output_dir / (
            f"{args.source_video.stem}_{tracked_side}_stride_{start}_{end}"
            "_features_and_prediction.csv"
        )
        features = pd.read_csv(features_path)
        technical_status = (
            "GOOD_EXTRACTION"
            if float(stride["mean_predicted_probability"]) >= 0.85
            and 0.65 <= float(stride["duration_s"]) <= 1.15
            and min(landing_ms, braking_ms, propulsion_ms, recovery_ms) >= 0
            else "REVIEW_REQUIRED"
        )
        rows.append({
            "stride_id": stride["stride_id"],
            "anatomical_side": stride["anatomical_side_in_this_clip"],
            "start_frame": start,
            "end_frame": end,
            "stride_duration_ms": duration_ms,
            "cadence_estimate_spm": 120.0 / float(stride["duration_s"]),
            "landing_duration_ms": landing_ms,
            "braking_duration_ms": braking_ms,
            "propulsion_duration_ms": propulsion_ms,
            "recovery_duration_ms": recovery_ms,
            "braking_percent_stride": braking_ms / duration_ms * 100.0,
            "propulsion_percent_stride": propulsion_ms / duration_ms * 100.0,
            "recovery_percent_stride": recovery_ms / duration_ms * 100.0,
            "hip_range_proxy_deg": float(features["hip_flexion_deg"].max() - features["hip_flexion_deg"].min()),
            "max_knee_flexion_proxy_deg": float(features["knee_flexion_deg"].max()),
            "mean_model_probability": float(stride["mean_predicted_probability"]),
            "technical_status": technical_status,
        })
    assessment = pd.DataFrame(rows)
    stem = args.source_video.stem
    csv_path = args.output_dir / f"{stem}_technical_assessment.csv"
    report_path = args.output_dir / f"{stem}_technical_assessment_KO.md"
    assessment.to_csv(csv_path, index=False, encoding="utf-8-sig")

    side_summary = assessment.groupby("anatomical_side").agg(
        strides=("stride_id", "count"),
        median_stride_ms=("stride_duration_ms", "median"),
        median_cadence_spm=("cadence_estimate_spm", "median"),
        median_braking_ms=("braking_duration_ms", "median"),
        median_propulsion_ms=("propulsion_duration_ms", "median"),
        median_recovery_ms=("recovery_duration_ms", "median"),
        median_knee_proxy_deg=("max_knee_flexion_proxy_deg", "median"),
        mean_probability=("mean_model_probability", "mean"),
    ).reset_index()
    review_count = int((assessment["technical_status"] == "REVIEW_REQUIRED").sum())
    lines = [
        "# RunScope 영상 기술 평가",
        "",
        f"- 분석 영상: `{args.source_video}`",
        f"- 자동 분석 stride: {len(assessment)}개",
        f"- Good extraction: {len(assessment) - review_count}개",
        f"- Review required: {review_count}개",
        "- 주의: 아래 평가는 영상 처리 및 운동학적 일관성 평가이며 의료 진단이나 부상 위험 판정이 아니다.",
        "",
        "## 발별 요약",
        "",
        "| 발 | stride 수 | stride 중앙값(ms) | 케이던스 추정(spm) | 버티기(ms) | 밀어내기(ms) | 회수(ms) | 모델확률 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, side in side_summary.iterrows():
        lines.append(
            f"| {side['anatomical_side']} | {int(side['strides'])} | "
            f"{side['median_stride_ms']:.0f} | {side['median_cadence_spm']:.1f} | "
            f"{side['median_braking_ms']:.0f} | {side['median_propulsion_ms']:.0f} | "
            f"{side['median_recovery_ms']:.0f} | {side['mean_probability']:.3f} |"
        )
    lines += ["", "## 좌우 비교", ""]
    side_index = side_summary.set_index("anatomical_side")
    if {"left", "right"}.issubset(side_index.index):
        left = side_index.loc["left"]
        right = side_index.loc["right"]
        comparisons = [
            ("stride 시간", "median_stride_ms", "ms"),
            ("케이던스 추정", "median_cadence_spm", "spm"),
            ("버티기 시간", "median_braking_ms", "ms"),
            ("밀어내기 시간", "median_propulsion_ms", "ms"),
            ("회수 시간", "median_recovery_ms", "ms"),
            ("최대 무릎 굴곡 2D proxy", "median_knee_proxy_deg", "deg"),
        ]
        lines += [
            "| 지표 | 왼발 | 오른발 | 좌우 차이율 |",
            "|---|---:|---:|---:|",
        ]
        for label, column, unit in comparisons:
            left_value = float(left[column])
            right_value = float(right[column])
            lines.append(
                f"| {label} | {left_value:.1f} {unit} | {right_value:.1f} {unit} | "
                f"{percent_difference(left_value, right_value):.1f}% |"
            )
    else:
        lines.append("양쪽 발의 유효 stride가 모두 있어야 좌우 비교를 계산할 수 있다.")
    lines += [
        "",
        "## 판정 행동강령",
        "",
        "- `GOOD_EXTRACTION`: 자동 결과표에 포함해도 되는 기술적 품질이다.",
        "- `REVIEW_REQUIRED`: 원영상, 관절 가림, 단계 전환 프레임을 사람이 확인한다.",
        "- 좌우 차이는 관찰값으로 보고하며 현재 단계에서 정상·비정상 판정 기준으로 사용하지 않는다.",
        "- 실제 자세 Good/Bad 판정은 전문가 라벨 또는 임상·부상 기준 데이터로 별도 학습해야 한다.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"assessment_csv={csv_path}")
    print(f"assessment_report={report_path}")


if __name__ == "__main__":
    main()
