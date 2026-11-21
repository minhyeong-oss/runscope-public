from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_failure_audit_v1"
PROTOCOL_PATH = EXPERIMENT / "config/protocol.json"
BOUNDARY_SOURCE = ROOT / "experiments/boundary_regression_v1/src/train_boundary_regression.py"
DEV_POSE_ROOT = ROOT / "experiments/boundary_regression_v1/artifacts/raw_pose"
TEST_POSE_ROOT = ROOT / "experiments/landing_gate_hybrid_v1/artifacts/raw_pose"
DEV_MATCHED = ROOT / "experiments/boundary_regression_v1/artifacts/development_matched_strides.csv"
DEV_SOURCE = ROOT / "experiments/landing_onset_transfer_v1/artifacts/twente_development_onset_sequences_64.npz"
DEV_OOF = ROOT / "experiments/landing_event_v3/reports/development_oof_predictions_v3.npz"
TEST_PREDICTIONS = ROOT / "experiments/landing_event_v3/reports/historical_test_predictions.csv"
SPLIT_PATH = ROOT / "data/twente_sensor/SPLIT_LOCK_v1.json"

RUNS = (63, 81, 99)
HISTORICAL_TRIALS = (
    ("Subj04", 81),
    ("Subj04", 99),
    ("Subj11", 63),
    ("Subj11", 81),
    ("Subj11", 99),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mat_path(subject: str, run: int, split_name: str) -> Path:
    if split_name == "historical_reference":
        return ROOT / f"data/twente_sensor/{subject}/locked_test_extracted/{subject}/{subject}_run_{run}.mat"
    if subject == "Subj05":
        return ROOT / f"data/twente_sensor/{subject}/extracted_run{run}/{subject}/{subject}_run_{run}.mat"
    return ROOT / f"data/twente_sensor/{subject}/development_extracted/{subject}/{subject}_run_{run}.mat"


def prediction_key(subject: str, run: int, side: str, source_time: float) -> tuple[str, int, str, float]:
    return subject, int(run), side, round(float(source_time), 6)


def build_prediction_lookup(protocol: dict) -> tuple[dict, dict]:
    interval_s = 0.48 / 63.0
    dev_matched = pd.read_csv(DEV_MATCHED)
    dev_source = np.load(DEV_SOURCE)
    dev_oof = np.load(DEV_OOF)
    source_rows = dev_source["source_rows"].astype(int)
    if len(source_rows) != len(dev_oof["ridge_alpha1000"]):
        raise RuntimeError("Development prediction/source length mismatch")
    dev_lookup = {}
    for sequence_index, source_row in enumerate(source_rows):
        row = dev_matched.iloc[int(source_row)]
        error_ms = abs(
            int(dev_oof["ridge_alpha1000"][sequence_index])
            - int(dev_oof["truth_onset"][sequence_index])
        ) * interval_s * 1000.0
        dev_lookup[prediction_key(row.subject, row.run, row.side, row.detected_start_source_time_s)] = {
            "absolute_error_ms": float(error_ms),
            "prediction_source": "development_subject_disjoint_oof_ridge_alpha1000",
        }

    test_lookup = {}
    test = pd.read_csv(TEST_PREDICTIONS)
    for row in test.itertuples(index=False):
        test_lookup[prediction_key(row.subject, row.run, row.side, row.detected_start_source_time_s)] = {
            "absolute_error_ms": float(row.absolute_error_ms),
            "prediction_source": "frozen_ridge_alpha1000_historical_reference",
        }
    return dev_lookup, test_lookup


def valid_candidates(boundary, pose: pd.DataFrame, side: str, duration_min: float, duration_max: float) -> list[dict]:
    peaks, _, _ = boundary.detect_contacts(pose, side)
    fps = float(pose["fps"].iloc[0])
    candidates = []
    for first, second in zip(peaks[:-1], peaks[1:]):
        duration_s = (int(second) - int(first)) / fps
        if duration_min <= duration_s <= duration_max:
            candidates.append(
                {
                    "side": side,
                    "first_frame": int(first),
                    "second_frame": int(second),
                    "source_time_s": float(pose["source_time_s"].iloc[int(first)]),
                    "duration_s": float(duration_s),
                }
            )
    return candidates


def nearest_candidate(candidates: list[dict], sensor_ic_s: float, sync_offset_s: float, tolerance_s: float):
    eligible = []
    for candidate in candidates:
        sensor_time = candidate["source_time_s"] + sync_offset_s
        distance_s = abs(sensor_time - sensor_ic_s)
        if distance_s <= tolerance_s:
            eligible.append((distance_s, candidate))
    return min(eligible, key=lambda item: item[0]) if eligible else None


def pose_quality_near_contact(pose: pd.DataFrame, side: str, video_ic_s: float, radius: int) -> tuple[float, float]:
    times = pose["source_time_s"].to_numpy(dtype=float)
    center = int(np.argmin(np.abs(times - video_ic_s)))
    start = max(0, center - radius)
    stop = min(len(pose), center + radius + 1)
    window = pose.iloc[start:stop]
    detected_fraction = float(window["pose_detected"].mean())
    visibility = float(
        np.nanmean(
            np.concatenate(
                [
                    window[f"{side}_heel_visibility"].to_numpy(dtype=float),
                    window[f"{side}_foot_index_visibility"].to_numpy(dtype=float),
                ]
            )
        )
    )
    return detected_fraction, visibility


def trial_rows(
    boundary,
    protocol: dict,
    subject: str,
    run: int,
    split_name: str,
    prediction_lookup: dict,
) -> list[dict]:
    thresholds = protocol["fixed_thresholds"]
    duration_min, duration_max = thresholds["valid_stride_duration_s"]
    pose_root = TEST_POSE_ROOT if split_name == "historical_reference" else DEV_POSE_ROOT
    pose_path = pose_root / f"{subject}_run{run}_side_10s_to_22s_mediapipe.csv"
    pose = pd.read_csv(pose_path)
    datastr = loadmat(
        mat_path(subject, run, split_name), simplify_cells=True, variable_names=["Datastr"]
    )["Datastr"]
    sync_offset_s, sync_correlation = boundary.synchronization_offset(datastr["Marker"], pose)
    force_rate = float(datastr["Force"]["FrameRate"])
    mass = float(datastr["Info"]["subjMass"])
    video_start_sensor_s = float(pose["source_time_s"].min()) + sync_offset_s
    video_end_sensor_s = float(pose["source_time_s"].max()) + sync_offset_s
    candidate_by_side = {
        side: valid_candidates(boundary, pose, side, duration_min, duration_max)
        for side in ("left", "right")
    }
    rows = []
    for side in ("left", "right"):
        opposite = "right" if side == "left" else "left"
        force_events = boundary.force_strides(
            datastr["Force"][side.capitalize() + "ForceData"], force_rate, mass
        )
        eligible_event_index = 0
        for source_event_index, event in enumerate(force_events):
            force_duration_s = float(event["next_contact_s"] - event["initial_contact_s"])
            if not duration_min <= force_duration_s <= duration_max:
                continue
            if event["initial_contact_s"] < video_start_sensor_s:
                continue
            if event["next_contact_s"] > video_end_sensor_s:
                continue
            eligible_event_index += 1
            video_ic_s = float(event["initial_contact_s"] - sync_offset_s)
            pose_fraction, foot_visibility = pose_quality_near_contact(
                pose,
                side,
                video_ic_s,
                int(thresholds["pose_window_frames_each_side"]),
            )
            same = nearest_candidate(
                candidate_by_side[side],
                float(event["initial_contact_s"]),
                sync_offset_s,
                float(thresholds["candidate_force_match_tolerance_s"]),
            )
            other = nearest_candidate(
                candidate_by_side[opposite],
                float(event["initial_contact_s"]),
                sync_offset_s,
                float(thresholds["candidate_force_match_tolerance_s"]),
            )
            prediction = None
            if same is not None:
                candidate = same[1]
                prediction = prediction_lookup.get(
                    prediction_key(subject, run, side, candidate["source_time_s"])
                )
            pose_miss = (
                pose_fraction < float(thresholds["pose_detected_fraction_min"])
                or foot_visibility < float(thresholds["tracked_foot_visibility_min"])
            )
            foot_swap = same is None and other is not None
            sync_low = abs(sync_correlation) < float(thresholds["sync_correlation_low_threshold"])
            stride_miss = same is None or prediction is None
            model_error = (
                prediction is not None
                and prediction["absolute_error_ms"] > float(thresholds["landing_success_tolerance_ms"])
            )
            if pose_miss:
                cause = "POSE_MISS"
            elif foot_swap:
                cause = "FOOT_SWAP"
            elif sync_low:
                cause = "SYNC_LOW"
            elif stride_miss:
                cause = "STRIDE_MISS"
            elif model_error:
                cause = "MODEL_ERROR"
            else:
                cause = "PASS"
            rows.append(
                {
                    "split": split_name,
                    "subject": subject,
                    "run": int(run),
                    "side": side,
                    "eligible_force_event_index": eligible_event_index,
                    "source_force_event_index": int(source_event_index),
                    "force_initial_contact_s": float(event["initial_contact_s"]),
                    "force_next_contact_s": float(event["next_contact_s"]),
                    "force_stride_duration_s": force_duration_s,
                    "sync_offset_s": float(sync_offset_s),
                    "sync_correlation": float(sync_correlation),
                    "pose_detected_fraction_near_ic": pose_fraction,
                    "tracked_foot_visibility_near_ic": foot_visibility,
                    "same_side_candidate_found": same is not None,
                    "same_side_candidate_source_time_s": (
                        float(same[1]["source_time_s"]) if same else np.nan
                    ),
                    "same_side_candidate_distance_ms": float(same[0] * 1000.0) if same else np.nan,
                    "opposite_side_candidate_found": other is not None,
                    "opposite_side_candidate_source_time_s": (
                        float(other[1]["source_time_s"]) if other else np.nan
                    ),
                    "opposite_side_candidate_distance_ms": float(other[0] * 1000.0) if other else np.nan,
                    "frozen_prediction_found": prediction is not None,
                    "absolute_error_ms": prediction["absolute_error_ms"] if prediction else np.nan,
                    "prediction_source": prediction["prediction_source"] if prediction else "",
                    "flag_pose_miss": pose_miss,
                    "flag_foot_swap": foot_swap,
                    "flag_sync_low": sync_low,
                    "flag_stride_miss": stride_miss,
                    "flag_model_error": bool(model_error),
                    "primary_cause": cause,
                }
            )
    return rows


def summarize(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    order = ["POSE_MISS", "FOOT_SWAP", "SYNC_LOW", "STRIDE_MISS", "MODEL_ERROR", "PASS"]
    cause_rows = []
    for split_name in ("development_oof", "historical_reference", "all"):
        subset = events if split_name == "all" else events[events["split"] == split_name]
        denominator = len(subset)
        for cause in order:
            count = int((subset["primary_cause"] == cause).sum())
            cause_rows.append(
                {
                    "split": split_name,
                    "primary_cause": cause,
                    "events": count,
                    "eligible_force_events": denominator,
                    "fraction": count / denominator if denominator else np.nan,
                }
            )
    by_trial = (
        events.groupby(["split", "subject", "run"], as_index=False)
        .agg(
            eligible_force_events=("primary_cause", "size"),
            pose_miss=("flag_pose_miss", "sum"),
            foot_swap=("flag_foot_swap", "sum"),
            sync_low=("flag_sync_low", "sum"),
            stride_miss=("flag_stride_miss", "sum"),
            model_error=("flag_model_error", "sum"),
            pass_events=("primary_cause", lambda s: int((s == "PASS").sum())),
            predicted_events=("frozen_prediction_found", "sum"),
            sync_correlation=("sync_correlation", "first"),
        )
    )
    by_trial["end_to_end_prediction_coverage"] = (
        by_trial["predicted_events"] / by_trial["eligible_force_events"]
    )
    by_trial["end_to_end_pass_rate"] = by_trial["pass_events"] / by_trial["eligible_force_events"]
    summary = {}
    for split_name in ("development_oof", "historical_reference", "all"):
        subset = events if split_name == "all" else events[events["split"] == split_name]
        predicted = subset[subset["frozen_prediction_found"]]
        within_30ms = int((predicted["absolute_error_ms"] <= 30.0).sum())
        trusted_passes = int((subset["primary_cause"] == "PASS").sum())
        summary[split_name] = {
            "eligible_force_events": int(len(subset)),
            "predicted_events": int(subset["frozen_prediction_found"].sum()),
            "end_to_end_prediction_coverage": float(subset["frozen_prediction_found"].mean()),
            "predictions_within_30ms": within_30ms,
            "end_to_end_timing_success_rate": float(within_30ms / len(subset)),
            "trusted_pass_events": trusted_passes,
            "end_to_end_trusted_pass_rate": float(trusted_passes / len(subset)),
            "conditional_30ms_accuracy_given_prediction": float(
                (predicted["absolute_error_ms"] <= 30.0).mean()
            ) if len(predicted) else None,
            "primary_cause_counts": {
                cause: int((subset["primary_cause"] == cause).sum()) for cause in order
            },
            "raw_flag_counts": {
                "POSE_MISS": int(subset["flag_pose_miss"].sum()),
                "FOOT_SWAP": int(subset["flag_foot_swap"].sum()),
                "SYNC_LOW": int(subset["flag_sync_low"].sum()),
                "STRIDE_MISS": int(subset["flag_stride_miss"].sum()),
                "MODEL_ERROR": int(subset["flag_model_error"].sum()),
            },
            "critical_flag_overlaps": {
                "FOOT_SWAP_and_SYNC_LOW": int(
                    (subset["flag_foot_swap"] & subset["flag_sync_low"]).sum()
                ),
                "FOOT_SWAP_without_SYNC_LOW": int(
                    (subset["flag_foot_swap"] & ~subset["flag_sync_low"]).sum()
                ),
                "STRIDE_MISS_and_SYNC_LOW": int(
                    (subset["flag_stride_miss"] & subset["flag_sync_low"]).sum()
                ),
                "STRIDE_MISS_without_SYNC_LOW": int(
                    (subset["flag_stride_miss"] & ~subset["flag_sync_low"]).sum()
                ),
            },
        }
    return pd.DataFrame(cause_rows), by_trial, summary


def write_markdown(summary: dict, protocol_hash: str) -> str:
    lines = [
        "# Landing 실패 원인 0단계 감사 보고서",
        "",
        f"- 적용 프로토콜 SHA-256: `{protocol_hash}`",
        "- 개발 자료는 참가자 분리 OOF Ridge 예측, 역사적 참조 자료는 동결 Ridge 예측을 사용했다.",
        "- 기존 모델 재학습·임계값 재조정은 수행하지 않았다.",
        "",
        "## 핵심 결과",
        "",
        "| 구분 | 힘판 보폭 | 예측 연결 | coverage | ±30 ms 예측 | timing 성공률 | 신뢰 PASS | 신뢰 PASS율 | 예측 조건부 ±30 ms 정확도 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("development_oof", "개발 OOF"), ("historical_reference", "역사적 참조"), ("all", "전체")):
        item = summary[key]
        lines.append(
            f"| {label} | {item['eligible_force_events']} | {item['predicted_events']} | "
            f"{item['end_to_end_prediction_coverage']:.3f} | {item['predictions_within_30ms']} | "
            f"{item['end_to_end_timing_success_rate']:.3f} | {item['trusted_pass_events']} | "
            f"{item['end_to_end_trusted_pass_rate']:.3f} | {item['conditional_30ms_accuracy_given_prediction']:.3f} |"
        )
    lines.extend(["", "## 주원인 개수", "", "| 구분 | POSE_MISS | FOOT_SWAP | SYNC_LOW | STRIDE_MISS | MODEL_ERROR | PASS |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key, label in (("development_oof", "개발 OOF"), ("historical_reference", "역사적 참조"), ("all", "전체")):
        counts = summary[key]["primary_cause_counts"]
        lines.append(
            f"| {label} | {counts['POSE_MISS']} | {counts['FOOT_SWAP']} | {counts['SYNC_LOW']} | "
            f"{counts['STRIDE_MISS']} | {counts['MODEL_ERROR']} | {counts['PASS']} |"
        )
    all_item = summary["all"]
    raw = all_item["raw_flag_counts"]
    overlap = all_item["critical_flag_overlaps"]
    lines.extend(
        [
            "",
            "## 핵심 진단",
            "",
            f"- 기존 조건부 ±30 ms 정확도는 {all_item['conditional_30ms_accuracy_given_prediction']:.3f}로 재현됐지만, 힘판 전체 분모의 timing 성공률은 {all_item['end_to_end_timing_success_rate']:.3f}이다.",
            f"- 전체 {all_item['eligible_force_events']}건 중 저동기화 flag는 {raw['SYNC_LOW']}건이고, 이 중 {overlap['STRIDE_MISS_and_SYNC_LOW']}건은 예측 미연결과 겹친다.",
            f"- FOOT_SWAP flag {raw['FOOT_SWAP']}건 중 {overlap['FOOT_SWAP_and_SYNC_LOW']}건이 저동기화와 겹친다. 저동기화가 아닌 FOOT_SWAP은 {overlap['FOOT_SWAP_without_SYNC_LOW']}건뿐이므로 현재 단계에서 좌우발 교환이 주병목이라고 결론 내리지 않는다.",
            f"- 저동기화가 아닌 STRIDE_MISS는 {overlap['STRIDE_MISS_without_SYNC_LOW']}건이다. 동기화 개선 뒤에도 남는 후보 검출·매칭 문제로 본다.",
            f"- 강한 pose 누락 flag는 {raw['POSE_MISS']}건뿐이지만, visibility가 높은 잘못된 좌표는 이 규칙으로 검출되지 않는다. BioCV의 투영 정답 비교가 필요한 이유다.",
            "",
            "## 해석 제한",
            "",
            "- 이 감사는 이미 생성된 자료에 대한 후향적 원인 분해이며 새로운 미접촉 테스트가 아니다.",
            "- `primary_cause`는 고정 우선순위에 따른 하나의 대표 원인이다. 중복 가능성은 `flag_*` 열로 확인해야 한다.",
            "- `timing 성공률`은 연결된 예측 중 ±30 ms인 사건을 전체 힘판 분모로 나눈 값이다. `신뢰 PASS율`은 여기에 앞선 포즈·발·동기화 품질 gate까지 통과한 사건만 센다.",
            "- 포즈 visibility와 동기화 상관 임계값은 원인 진단용 운영 정의이며 생체역학적 절대 기준이 아니다.",
            "- 역사적 참조의 Subj04·Subj11 결과로 모델을 재선택하거나 임계값을 변경하면 안 된다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    boundary = load_module("landing_failure_audit_boundary", BOUNDARY_SOURCE)
    dev_lookup, test_lookup = build_prediction_lookup(protocol)
    rows = []
    for subject in split["development_subjects"]:
        for run in RUNS:
            rows.extend(trial_rows(boundary, protocol, subject, run, "development_oof", dev_lookup))
    for subject, run in HISTORICAL_TRIALS:
        rows.extend(trial_rows(boundary, protocol, subject, run, "historical_reference", test_lookup))
    events = pd.DataFrame(rows)
    cause_summary, trial_summary, summary = summarize(events)
    reports = EXPERIMENT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    event_path = reports / "force_event_failure_audit.csv"
    cause_path = reports / "primary_cause_summary.csv"
    trial_path = reports / "trial_end_to_end_summary.csv"
    json_path = reports / "AUDIT_SUMMARY.json"
    md_path = reports / "AUDIT_REPORT_KO.md"
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    cause_summary.to_csv(cause_path, index=False, encoding="utf-8-sig")
    trial_summary.to_csv(trial_path, index=False, encoding="utf-8-sig")
    protocol_hash = sha256(PROTOCOL_PATH)
    report = {
        "status": "LANDING_FAILURE_AUDIT_V1_COMPLETE_NO_TRAINING_NO_RETUNING",
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_hash,
        "summary": summary,
        "outputs": {
            "force_event_failure_audit_csv": str(event_path),
            "primary_cause_summary_csv": str(cause_path),
            "trial_end_to_end_summary_csv": str(trial_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(write_markdown(summary, protocol_hash), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
