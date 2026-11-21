from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(os.environ.get("RUNSCOPE_ROOT", Path(__file__).resolve().parents[1])).resolve()
MODEL_DIR = ROOT / "model_v1"
ARTIFACTS = MODEL_DIR / "artifacts"
DATA_DIR = MODEL_DIR / "data"
REPORTS = MODEL_DIR / "reports"
OUTPUTS = ROOT / "outputs"
STREAMING = Path(
    os.environ.get("RUNSCOPE_REZNICK_STREAMING", ROOT / "data/reznick/Streaming.mat")
).expanduser().resolve()
EVENTS = OUTPUTS / "RunScope_Reznick_phase_events.csv"

STAGE_NAMES = {
    1: "landing_window",
    2: "braking",
    3: "propulsion",
    4: "recovery",
}
FEATURE_NAMES = [
    "nominal_speed_m_s",
    "cycle_fraction",
    "hip_flexion_deg",
    "knee_flexion_deg",
    "ankle_flexion_deg",
    "pelvis_tilt_deg",
    "hip_velocity_dps",
    "knee_velocity_dps",
    "ankle_velocity_dps",
    "pelvis_velocity_dps",
    "hip_accel_dps2",
    "knee_accel_dps2",
    "ankle_accel_dps2",
    "pelvis_accel_dps2",
]


def constrained_decode(probabilities: np.ndarray) -> np.ndarray:
    """Decode one stride with the allowed order 1, 2, 3, 4."""
    eps = 1e-12
    logp = np.log(np.clip(probabilities, eps, 1.0))
    n, states = logp.shape
    score = np.full((n, states), -np.inf)
    back = np.full((n, states), -1, dtype=np.int16)
    score[0, 0] = logp[0, 0]
    for t in range(1, n):
        for state in range(states):
            candidates = [(score[t - 1, state], state)]
            if state > 0:
                candidates.append((score[t - 1, state - 1], state - 1))
            best_score, previous = max(candidates, key=lambda item: item[0])
            score[t, state] = best_score + logp[t, state]
            back[t, state] = previous
    state = 3 if np.isfinite(score[-1, 3]) else int(np.nanargmax(score[-1]))
    decoded = np.empty(n, dtype=np.int8)
    decoded[-1] = state + 1
    for t in range(n - 1, 0, -1):
        state = int(back[t, state])
        decoded[t - 1] = state + 1
    return decoded


def build_frame_dataset() -> dict[str, np.ndarray]:
    events = pd.read_csv(EVENTS, encoding="utf-8-sig")
    events = events[
        (events["extraction_quality"] == "good_extraction_case")
        & (events["four_stage_complete"].astype(str).str.lower() == "true")
    ].copy()

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_event_index: list[np.ndarray] = []
    event_ids: list[str] = []
    event_subjects: list[str] = []
    event_splits: list[str] = []
    event_speeds: list[float] = []
    event_conditions: list[str] = []
    event_lengths: list[int] = []
    event_true_stage3_s: list[float] = []
    event_true_stage4_s: list[float] = []

    with h5py.File(STREAMING, "r") as h5:
        root = h5["Streaming"]
        for _, row in events.sort_values("event_id").iterrows():
            subject = str(row["subject_id"])
            condition = str(row["condition_code"])
            side = "L" if str(row["side"]).lower() == "left" else "R"
            group = root[subject]["Run"][condition]
            start = int(row["source_hs_frame_100hz"]) - 1
            duration_s = float(row["next_initial_contact_s"])
            n_frames = int(round(duration_s * 100))
            if n_frames < 30:
                continue

            arrays = []
            for name in ("HipAngles", "KneeAngles", "AnkleAngles", "PelvisAngles"):
                values = np.asarray(group["jointAngles"][f"{side}{name}"], dtype=float)[0]
                arrays.append(values)
            total_frames = min(len(values) for values in arrays)
            end = min(total_frames, start + n_frames)
            if start < 0 or end - start < 30:
                continue

            angles = np.column_stack([values[start:end] for values in arrays])
            if not np.all(np.isfinite(angles)):
                continue
            velocities = np.gradient(angles, 0.01, axis=0)
            accelerations = np.gradient(velocities, 0.01, axis=0)
            local_time = np.arange(len(angles), dtype=float) / 100.0
            actual_duration = len(angles) / 100.0
            cycle_fraction = np.clip(local_time / max(0.01, actual_duration), 0.0, 0.999999)
            speed = float(row["nominal_speed_m_s"])
            x = np.column_stack([
                np.full(len(angles), speed),
                cycle_fraction,
                angles,
                velocities,
                accelerations,
            ]).astype(np.float32)

            stage3_s = float(row["stage3_propulsion_start_s"])
            stage4_s = float(row["stage4_recovery_start_s"])
            landing_end_s = min(stage3_s - 0.01, 0.03)
            if not (0.015 <= landing_end_s < stage3_s < stage4_s < actual_duration + 0.03):
                continue
            y = np.full(len(angles), 4, dtype=np.int8)
            y[local_time < stage4_s] = 3
            y[local_time < stage3_s] = 2
            y[local_time < landing_end_s] = 1
            if set(np.unique(y).tolist()) != {1, 2, 3, 4}:
                continue

            event_index = len(event_ids)
            features.append(x)
            labels.append(y)
            sample_event_index.append(np.full(len(y), event_index, dtype=np.int32))
            event_ids.append(str(row["event_id"]))
            event_subjects.append(subject)
            event_splits.append(str(row["fixed_split"]))
            event_speeds.append(speed)
            event_conditions.append(condition)
            event_lengths.append(len(y))
            event_true_stage3_s.append(stage3_s)
            event_true_stage4_s.append(stage4_s)

    dataset = {
        "X": np.concatenate(features),
        "y": np.concatenate(labels),
        "sample_event_index": np.concatenate(sample_event_index),
        "event_ids": np.asarray(event_ids),
        "event_subjects": np.asarray(event_subjects),
        "event_splits": np.asarray(event_splits),
        "event_speeds": np.asarray(event_speeds, dtype=np.float32),
        "event_conditions": np.asarray(event_conditions),
        "event_lengths": np.asarray(event_lengths, dtype=np.int16),
        "event_true_stage3_s": np.asarray(event_true_stage3_s, dtype=np.float32),
        "event_true_stage4_s": np.asarray(event_true_stage4_s, dtype=np.float32),
        "feature_names": np.asarray(FEATURE_NAMES),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "runscope_phase_frames_v1.npz", **dataset)
    return dataset


def split_mask(dataset: dict[str, np.ndarray], split_names: set[str]) -> np.ndarray:
    event_mask = np.isin(dataset["event_splits"], list(split_names))
    return event_mask[dataset["sample_event_index"]]


def predict_constrained(model, dataset: dict[str, np.ndarray], split_names: set[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    event_indices = np.flatnonzero(np.isin(dataset["event_splits"], list(split_names))).tolist()
    sample_indices = []
    predictions = []
    for event_index in event_indices:
        idx = np.flatnonzero(dataset["sample_event_index"] == event_index)
        proba = model.predict_proba(dataset["X"][idx])
        predictions.append(constrained_decode(proba))
        sample_indices.append(idx)
    return np.concatenate(sample_indices), np.concatenate(predictions), event_indices


def baseline_thresholds(dataset: dict[str, np.ndarray], development_splits: set[str]) -> dict[float, tuple[float, float, float]]:
    by_speed: dict[float, list[tuple[float, float, float]]] = defaultdict(list)
    for event_index in np.flatnonzero(np.isin(dataset["event_splits"], list(development_splits))):
        idx = np.flatnonzero(dataset["sample_event_index"] == event_index)
        y = dataset["y"][idx]
        length = len(y)
        first2 = int(np.flatnonzero(y == 2)[0]) / length
        first3 = int(np.flatnonzero(y == 3)[0]) / length
        first4 = int(np.flatnonzero(y == 4)[0]) / length
        by_speed[float(dataset["event_speeds"][event_index])].append((first2, first3, first4))
    return {speed: tuple(np.median(values, axis=0).tolist()) for speed, values in by_speed.items()}


def predict_baseline(dataset: dict[str, np.ndarray], split_names: set[str], thresholds: dict[float, tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    event_indices = np.flatnonzero(np.isin(dataset["event_splits"], list(split_names))).tolist()
    sample_indices = []
    predictions = []
    speeds = np.asarray(sorted(thresholds), dtype=float)
    for event_index in event_indices:
        idx = np.flatnonzero(dataset["sample_event_index"] == event_index)
        fraction = dataset["X"][idx, FEATURE_NAMES.index("cycle_fraction")]
        speed = float(dataset["event_speeds"][event_index])
        nearest = float(speeds[np.argmin(abs(speeds - speed))])
        b1, b2, b3 = thresholds[nearest]
        pred = np.full(len(idx), 4, dtype=np.int8)
        pred[fraction < b3] = 3
        pred[fraction < b2] = 2
        pred[fraction < b1] = 1
        sample_indices.append(idx)
        predictions.append(pred)
    return np.concatenate(sample_indices), np.concatenate(predictions), event_indices


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_stage": classification_report(
            y_true,
            y_pred,
            labels=[1, 2, 3, 4],
            target_names=[STAGE_NAMES[i] for i in [1, 2, 3, 4]],
            output_dict=True,
            zero_division=0,
        ),
    }


def event_timing_rows(dataset: dict[str, np.ndarray], y_pred: np.ndarray, sample_indices: np.ndarray, event_indices: list[int], model_name: str) -> list[dict]:
    rows = []
    offset = 0
    for event_index in event_indices:
        idx = np.flatnonzero(dataset["sample_event_index"] == event_index)
        count = len(idx)
        pred = y_pred[offset: offset + count]
        offset += count
        pred3 = np.flatnonzero(pred == 3)
        pred4 = np.flatnonzero(pred == 4)
        if not len(pred3) or not len(pred4):
            continue
        p3_s = float(pred3[0]) / 100.0
        p4_s = float(pred4[0]) / 100.0
        true3 = float(dataset["event_true_stage3_s"][event_index])
        true4 = float(dataset["event_true_stage4_s"][event_index])
        rows.append({
            "model": model_name,
            "event_id": str(dataset["event_ids"][event_index]),
            "subject_id": str(dataset["event_subjects"][event_index]),
            "condition_code": str(dataset["event_conditions"][event_index]),
            "speed_m_s": float(dataset["event_speeds"][event_index]),
            "true_stage3_s": true3,
            "pred_stage3_s": p3_s,
            "stage3_error_ms": (p3_s - true3) * 1000.0,
            "true_stage4_s": true4,
            "pred_stage4_s": p4_s,
            "stage4_error_ms": (p4_s - true4) * 1000.0,
        })
    return rows


def timing_summary(rows: list[dict]) -> dict:
    result = {}
    for stage in (3, 4):
        errors = np.asarray([abs(row[f"stage{stage}_error_ms"]) for row in rows], dtype=float)
        result[f"stage{stage}_mae_ms"] = float(np.mean(errors))
        result[f"stage{stage}_median_ae_ms"] = float(np.median(errors))
        result[f"stage{stage}_p90_ae_ms"] = float(np.quantile(errors, 0.90))
    return result


def external_evidence() -> list[dict]:
    rows = []
    fukuchi = pd.read_csv(OUTPUTS / "RunScope_Fukuchi_gait_events.csv")
    fukuchi_good = fukuchi[fukuchi["qc_label"] == "good_extraction_case"]
    rows.append({
        "dataset_number": 3,
        "dataset": "Fukuchi RBDS",
        "role": "high_speed_phase_reference",
        "subjects": int(fukuchi_good["subject_id"].nunique()),
        "events": int(len(fukuchi_good)),
        "speed_min_m_s": float(fukuchi_good["speed_m_s"].min()),
        "speed_max_m_s": float(fukuchi_good["speed_m_s"].max()),
        "native_phase23_ground_truth": True,
        "note": "External calibration only; speeds exceed the model training range.",
    })
    king = pd.read_csv(OUTPUTS / "RunScope_King_four_stage_events.csv")
    king_good = king[king["good_detection_20ms"].astype(str).str.lower() == "true"]
    rows.append({
        "dataset_number": 4,
        "dataset": "King event timing",
        "role": "ic_to_detection_benchmark",
        "subjects": int(king["subject_id"].nunique()),
        "events": int(len(king)),
        "speed_min_m_s": float(king["speed_m_s"].min()),
        "speed_max_m_s": float(king["speed_m_s"].max()),
        "native_phase23_ground_truth": False,
        "note": f"{len(king_good)} contacts have both IC and TO algorithm errors within 20 ms.",
    })
    reznick = pd.read_csv(OUTPUTS / "RunScope_Reznick_phase_events.csv", encoding="utf-8-sig")
    reznick_good = reznick[reznick["extraction_quality"] == "good_extraction_case"]
    rows.append({
        "dataset_number": 5,
        "dataset": "Reznick continuous locomotion",
        "role": "primary_frame_level_training",
        "subjects": int(reznick_good["subject_id"].nunique()),
        "events": int(len(reznick_good)),
        "speed_min_m_s": float(reznick_good["nominal_speed_m_s"].min()),
        "speed_max_m_s": float(reznick_good["nominal_speed_m_s"].max()),
        "native_phase23_ground_truth": True,
        "note": "100 Hz joint angles with 1000 Hz force-derived phase boundaries.",
    })
    baker = pd.read_csv(OUTPUTS / "RunScope_Baker_phase_events.csv")
    baker_good = baker[baker["qc_label"] == "good_extraction_case"]
    rows.append({
        "dataset_number": 7,
        "dataset": "Baker running environments",
        "role": "environment_external_validation",
        "subjects": int(baker_good["subject_id"].nunique()),
        "events": int(len(baker_good)),
        "speed_min_m_s": float(baker_good["median_speed_m_s"].min()),
        "speed_max_m_s": float(baker_good["median_speed_m_s"].max()),
        "native_phase23_ground_truth": True,
        "note": "Force-based outdoor/treadmill timing reference; recovery is only partially observed.",
    })
    return rows


def main() -> None:
    for directory in (ARTIFACTS, DATA_DIR, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    dataset = build_frame_dataset()

    train_mask = split_mask(dataset, {"train"})
    validation_mask = split_mask(dataset, {"validation"})
    test_mask = split_mask(dataset, {"test"})
    configs = [
        {"max_leaf_nodes": 15, "max_depth": 6, "learning_rate": 0.08, "max_iter": 180, "l2_regularization": 1.0},
        {"max_leaf_nodes": 31, "max_depth": 8, "learning_rate": 0.08, "max_iter": 220, "l2_regularization": 1.5},
        {"max_leaf_nodes": 63, "max_depth": 10, "learning_rate": 0.06, "max_iter": 260, "l2_regularization": 2.0},
    ]
    tuning = []
    best = None
    for index, config in enumerate(configs, start=1):
        model = HistGradientBoostingClassifier(
            **config,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=20261104,
        )
        model.fit(dataset["X"][train_mask], dataset["y"][train_mask])
        idx, pred, _ = predict_constrained(model, dataset, {"validation"})
        score = float(f1_score(dataset["y"][idx], pred, average="macro"))
        tuning.append({"candidate": index, "validation_macro_f1": score, **config, "iterations_used": int(model.n_iter_)})
        if best is None or score > best[0]:
            best = (score, config)
    assert best is not None

    development_mask = train_mask | validation_mask
    final_model = HistGradientBoostingClassifier(
        **best[1],
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=20261104,
    )
    final_model.fit(dataset["X"][development_mask], dataset["y"][development_mask])

    test_idx, test_pred, test_events = predict_constrained(final_model, dataset, {"test"})
    test_true = dataset["y"][test_idx]
    ml_frame = frame_metrics(test_true, test_pred)
    ml_timing_rows = event_timing_rows(dataset, test_pred, test_idx, test_events, "ml_v1")
    ml_timing = timing_summary(ml_timing_rows)

    thresholds = baseline_thresholds(dataset, {"train", "validation"})
    base_idx, base_pred, base_events = predict_baseline(dataset, {"test"}, thresholds)
    baseline_frame = frame_metrics(dataset["y"][base_idx], base_pred)
    baseline_timing_rows = event_timing_rows(dataset, base_pred, base_idx, base_events, "timing_baseline")
    baseline_timing = timing_summary(baseline_timing_rows)

    model_bundle = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "stage_names": STAGE_NAMES,
        "landing_window_s": 0.03,
        "decoder": "monotonic_1_2_3_4",
        "training_speed_range_m_s": [1.8, 2.4],
        "dataset_numbers": [3, 4, 5, 7],
        "primary_training_dataset": 5,
        "version": "0.1.0",
    }
    joblib.dump(model_bundle, ARTIFACTS / "runscope_phase_model_v1.joblib", compress=3)
    (ARTIFACTS / "feature_schema.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "units": {
            "nominal_speed_m_s": "m/s",
            "cycle_fraction": "0-1",
            "*_flexion_deg": "degrees",
            "pelvis_tilt_deg": "degrees",
            "*_velocity_dps": "degrees/second",
            "*_accel_dps2": "degrees/second^2",
        },
        "phase_labels": STAGE_NAMES,
        "landing_window_definition": "first 30 ms after source heel strike",
        "valid_speed_range_m_s": [1.8, 2.4],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = confusion_matrix(test_true, test_pred, labels=[1, 2, 3, 4])
    with (REPORTS / "test_confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *[STAGE_NAMES[i] for i in [1, 2, 3, 4]]])
        for label, row in zip([1, 2, 3, 4], cm):
            writer.writerow([STAGE_NAMES[label], *row.tolist()])
    pd.DataFrame(ml_timing_rows + baseline_timing_rows).to_csv(REPORTS / "test_event_timing.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tuning).to_csv(REPORTS / "validation_tuning.csv", index=False, encoding="utf-8-sig")
    evidence = external_evidence()
    pd.DataFrame(evidence).to_csv(REPORTS / "dataset_roles_3457.csv", index=False, encoding="utf-8-sig")

    split_counts = {}
    for split in ("train", "validation", "test"):
        event_indices = np.flatnonzero(dataset["event_splits"] == split)
        split_counts[split] = {
            "subjects": sorted(set(dataset["event_subjects"][event_indices].tolist())),
            "events": int(len(event_indices)),
            "frames": int(np.sum(np.isin(dataset["sample_event_index"], event_indices))),
            "class_counts": {STAGE_NAMES[int(key)]: int(value) for key, value in Counter(dataset["y"][np.isin(dataset["sample_event_index"], event_indices)]).items()},
        }
    metrics = {
        "model_version": "0.1.0",
        "selected_hyperparameters": best[1],
        "selected_validation_macro_f1": best[0],
        "iterations_used_final": int(final_model.n_iter_),
        "split_counts": split_counts,
        "test_ml_frame_metrics": ml_frame,
        "test_ml_event_timing": ml_timing,
        "test_baseline_frame_metrics": baseline_frame,
        "test_baseline_event_timing": baseline_timing,
        "ppt_target_macro_f1": 0.85,
        "meets_ppt_f1_target": ml_frame["macro_f1"] >= 0.85,
        "limitations": [
            "Training frames come from dataset 5 because datasets 3, 4, and 7 do not share the same frame-level joint-angle schema.",
            "Dataset 3 and most dataset 7 speeds exceed the 1.8-2.4 m/s training range.",
            "The 30 ms landing window is an operational label around an instantaneous heel-strike event.",
            "Evaluation is offline and uses a completed stride to calculate cycle_fraction.",
            "Quality labels describe extraction reliability, not whether a runner has good or bad form.",
        ],
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# RunScope 첫 모델 v0.1

## 결과

- 목적: 측면 영상에서 계산한 관절각 계열로 달리기 4단계를 분류
- 테스트 참가자: {', '.join(split_counts['test']['subjects'])}
- 테스트 stride: {split_counts['test']['events']:,}건
- 테스트 프레임: {split_counts['test']['frames']:,}개
- ML macro F1: {ml_frame['macro_f1']:.4f}
- 타이밍 baseline macro F1: {baseline_frame['macro_f1']:.4f}
- 2→3 경계 MAE: {ml_timing['stage3_mae_ms']:.2f} ms
- 3→4 경계 MAE: {ml_timing['stage4_mae_ms']:.2f} ms
- PPT 목표 F1 ≥ 0.85: {'통과' if metrics['meets_ppt_f1_target'] else '미달'}

## 데이터 역할

- 3번 Fukuchi: 2.5–4.5 m/s 고속 phase 기준. 모델의 속도 외삽 점검에 사용
- 4번 King: IC/TO 검출 시각 기준. phase 2→3 정답은 없음
- 5번 Reznick: 100 Hz 관절각과 1000 Hz force 경계. 모델의 직접 학습·검증·테스트에 사용
- 7번 Baker: 실외와 트레드밀 force phase 기준. 환경 외부 검증에 사용

## 단계 라벨

1. landing_window: source heel strike 뒤 30 ms
2. braking: landing window 끝부터 AP force 0교차까지
3. propulsion: AP force 0교차부터 toe-off까지
4. recovery: toe-off부터 같은 발 다음 heel strike까지

## 사용 범위

- 현재 보증 속도 범위: 1.8–2.4 m/s
- 입력: 속도, cycle fraction, 골반·고관절·무릎·발목 시상면 각도와 1·2차 미분
- 영상 적용 시 MediaPipe landmark에서 동일한 각도 열을 생성해야 함
- 결과의 good/review는 추출 신뢰도이며 러닝 자세의 의학적 판정이 아님

## 다음 검증

3번의 2.5–4.5 m/s와 7번의 실외 조건은 현재 모델의 직접 테스트 범위 밖입니다. 같은 feature schema를 만든 뒤 별도 외부 검증해야 합니다.
"""
    (OUTPUTS / "RunScope_model_v1_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "model": str(ARTIFACTS / "runscope_phase_model_v1.joblib"),
        "events": int(len(dataset["event_ids"])),
        "frames": int(len(dataset["y"])),
        "test_macro_f1": ml_frame["macro_f1"],
        "baseline_macro_f1": baseline_frame["macro_f1"],
        "stage3_mae_ms": ml_timing["stage3_mae_ms"],
        "stage4_mae_ms": ml_timing["stage4_mae_ms"],
        "meets_target": metrics["meets_ppt_f1_target"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
