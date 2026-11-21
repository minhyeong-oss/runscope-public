from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/twente_expanded_alignment_v1"
PROTOCOL_PATH = EXPERIMENT / "config/protocol.json"
AUDIT_WINDOWS = (
    ROOT
    / "experiments/twente_full_duration_sync_audit_v1/reports/window_sync_audit_full.csv"
)
BOUNDARY_SOURCE = (
    ROOT / "experiments/boundary_regression_v1/src/train_boundary_regression.py"
)
ARTIFACTS = EXPERIMENT / "artifacts"
REPORTS = EXPERIMENT / "reports"


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("runscope_boundary_v1", BOUNDARY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BOUNDARY_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_weights(subjects: pd.Series) -> np.ndarray:
    counts = subjects.value_counts()
    raw = subjects.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    return raw * (len(raw) / raw.sum())


def included_windows(audit: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    gate = protocol["window_inclusion"]
    lo, hi = gate["offset_scan_limits_ms"]
    margin = float(gate["offset_boundary_margin_ms"])
    corr_min = float(gate["minimum_marker_pose_correlation"])
    table = audit.copy()
    table["included"] = (
        (table["mean_abs_correlation"] >= corr_min)
        & (table["offset_ms"] > float(lo) + margin)
        & (table["offset_ms"] < float(hi) - margin)
    )
    table["exclusion_reason"] = np.where(
        table["mean_abs_correlation"] < corr_min,
        "LOW_CORRELATION",
        np.where(
            (table["offset_ms"] <= float(lo) + margin)
            | (table["offset_ms"] >= float(hi) - margin),
            "OFFSET_SCAN_BOUNDARY",
            "INCLUDED",
        ),
    )
    return table


def build_dataset(boundary, windows: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    window_rows: list[dict] = []
    allowed_subjects = set(protocol["development_subjects"])
    forbidden = set(protocol["forbidden_subjects"])
    if allowed_subjects & forbidden:
        raise RuntimeError("Development and forbidden subjects overlap")
    selected = windows[windows["included"]].copy()
    if set(selected["subject"]) & forbidden:
        raise RuntimeError("Forbidden subject present in selected audit windows")

    for item in selected.itertuples(index=False):
        subject = str(item.subject)
        run = int(item.run)
        pose = pd.read_csv(item.pose_csv)
        datastr = loadmat(
            boundary.mat_path(subject, run),
            simplify_cells=True,
            variable_names=["Datastr"],
        )["Datastr"]
        force_rate = float(datastr["Force"]["FrameRate"])
        mass = float(datastr["Info"]["subjMass"])
        sync_offset_s = float(item.offset_ms) / 1000.0
        matches_before_dedup = 0
        candidate_strides = 0
        for side in ("left", "right"):
            events = boundary.force_strides(
                datastr["Force"][side.capitalize() + "ForceData"], force_rate, mass
            )
            peaks, _, direction = boundary.detect_contacts(pose, side)
            used: set[int] = set()
            for first, second in zip(peaks[:-1], peaks[1:]):
                duration = (second - first) / float(pose["fps"].iloc[0])
                if not 0.45 <= duration <= 1.20:
                    continue
                candidate_strides += 1
                detected_sensor_time = (
                    float(pose["source_time_s"].iloc[first]) + sync_offset_s
                )
                match = boundary.match_force_event(detected_sensor_time, events, used)
                if match is None:
                    continue
                event_index, event = match
                force_duration = float(event["next_contact_s"] - event["initial_contact_s"])
                if not 0.45 <= force_duration <= 1.20:
                    continue
                used.add(event_index)
                matches_before_dedup += 1
                timing_error_s = float(event["initial_contact_s"] - detected_sensor_time)
                features = boundary.sample_features(
                    pose, int(first), int(second), side, direction, boundary.SPEEDS[run]
                )
                rows.append(
                    {
                        "subject": subject,
                        "run": run,
                        "side": side,
                        "window_start_s": float(item.window_start_s),
                        "window_end_s": float(item.window_end_s),
                        "force_event_index": int(event_index),
                        "force_initial_contact_s": float(event["initial_contact_s"]),
                        "detected_start_source_time_s": float(
                            pose["source_time_s"].iloc[first]
                        ),
                        "sync_offset_s": sync_offset_s,
                        "sync_correlation": float(item.mean_abs_correlation),
                        "candidate_force_timing_error_s": timing_error_s,
                        "target_ic_offset_s": timing_error_s,
                        "target_propulsion_offset_s": float(
                            event["propulsion_start_s"] - detected_sensor_time
                        ),
                        "target_recovery_offset_s": float(
                            event["toe_off_s"] - detected_sensor_time
                        ),
                        "target_next_ic_offset_s": float(
                            event["next_contact_s"] - detected_sensor_time
                        ),
                        **features,
                    }
                )
        window_rows.append(
            {
                "subject": subject,
                "run": run,
                "window_start_s": float(item.window_start_s),
                "window_end_s": float(item.window_end_s),
                "sync_offset_s": sync_offset_s,
                "sync_correlation": float(item.mean_abs_correlation),
                "candidate_strides": candidate_strides,
                "matches_before_dedup": matches_before_dedup,
                "pose_detection_rate": float(pose["pose_detected"].mean()),
            }
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No matched strides were built")
    raw["abs_candidate_force_timing_error_s"] = raw[
        "candidate_force_timing_error_s"
    ].abs()
    raw = raw.sort_values(
        [
            "subject",
            "run",
            "side",
            "force_event_index",
            "sync_correlation",
            "abs_candidate_force_timing_error_s",
        ],
        ascending=[True, True, True, True, False, True],
    )
    dedup_keys = ["subject", "run", "side", "force_event_index"]
    raw["duplicate_force_event"] = raw.duplicated(dedup_keys, keep="first")
    deduped = raw[~raw["duplicate_force_event"]].copy().reset_index(drop=True)
    return deduped, pd.DataFrame(window_rows)


def evaluate(boundary, dataset: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target_names = list(boundary.TARGETS)
    metadata = {
        "subject",
        "run",
        "side",
        "window_start_s",
        "window_end_s",
        "force_event_index",
        "force_initial_contact_s",
        "detected_start_source_time_s",
        "sync_offset_s",
        "sync_correlation",
        "candidate_force_timing_error_s",
        "abs_candidate_force_timing_error_s",
        "duplicate_force_event",
        *target_names,
    }
    feature_names = [column for column in dataset.columns if column not in metadata]
    subjects = sorted(dataset["subject"].unique().tolist())
    cv_rows: list[dict] = []
    for held_out in subjects:
        train = dataset[dataset["subject"] != held_out]
        validation = dataset[dataset["subject"] == held_out]
        baseline = boundary.baseline_predict(train, validation)
        baseline_scores = boundary.score_predictions(validation, baseline)
        cv_rows.append(
            {
                "held_out_subject": held_out,
                "candidate": "timing_baseline",
                "train_strides": int(len(train)),
                "validation_strides": int(len(validation)),
                **baseline_scores["per_stage_f1"],
                "macro_f1": baseline_scores["macro_f1"],
            }
        )
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        weights = subject_weights(train["subject"])
        model.fit(
            train[feature_names].to_numpy(dtype=float),
            train[target_names].to_numpy(dtype=float),
            ridge__sample_weight=weights,
        )
        predicted = model.predict(validation[feature_names].to_numpy(dtype=float))
        scores = boundary.score_predictions(validation, predicted)
        cv_rows.append(
            {
                "held_out_subject": held_out,
                "candidate": "ridge_alpha10_subject_balanced",
                "train_strides": int(len(train)),
                "validation_strides": int(len(validation)),
                **scores["per_stage_f1"],
                "macro_f1": scores["macro_f1"],
            }
        )
    cv = pd.DataFrame(cv_rows)
    summary = (
        cv.groupby("candidate")
        .agg(
            participants=("held_out_subject", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            landing_f1_mean=("landing", "mean"),
            braking_f1_mean=("braking", "mean"),
            propulsion_f1_mean=("propulsion", "mean"),
            recovery_f1_mean=("recovery", "mean"),
        )
        .reset_index()
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    weights = subject_weights(dataset["subject"])
    model.fit(
        dataset[feature_names].to_numpy(dtype=float),
        dataset[target_names].to_numpy(dtype=float),
        ridge__sample_weight=weights,
    )
    artifact = {
        "experiment": "twente_expanded_alignment_v1",
        "candidate": "ridge_alpha10_subject_balanced",
        "feature_names": feature_names,
        "target_names": target_names,
        "development_subjects_present": subjects,
        "forbidden_subjects": protocol["forbidden_subjects"],
        "model": model,
    }
    return cv, summary, artifact


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    boundary = load_boundary_module()
    audit = pd.read_csv(AUDIT_WINDOWS)
    windows = included_windows(audit, protocol)
    windows.to_csv(REPORTS / "window_inclusion_decisions.csv", index=False, encoding="utf-8-sig")
    dataset, window_quality = build_dataset(boundary, windows, protocol)
    dataset_path = ARTIFACTS / "expanded_development_matched_strides.csv"
    quality_path = REPORTS / "included_window_quality.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    window_quality.to_csv(quality_path, index=False, encoding="utf-8-sig")

    cv, summary, artifact = evaluate(boundary, dataset, protocol)
    cv_path = REPORTS / "development_loso_cv.csv"
    summary_path = REPORTS / "development_loso_summary.csv"
    model_path = ARTIFACTS / "expanded_boundary_ridge.joblib"
    cv.to_csv(cv_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    joblib.dump(artifact, model_path, compress=3)

    participant_counts = dataset.groupby("subject").size().astype(int).to_dict()
    run_counts = dataset.groupby("run").size().astype(int).to_dict()
    report = {
        "status": "DEVELOPMENT_ONLY_EXPANSION_COMPLETE_HISTORICAL_TEST_UNTOUCHED",
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "audit_source": str(AUDIT_WINDOWS),
        "forbidden_subjects_accessed": False,
        "force_events_used_to_choose_offsets": False,
        "windows_total": int(len(windows)),
        "windows_included": int(windows["included"].sum()),
        "windows_excluded": int((~windows["included"]).sum()),
        "matched_strides_after_deduplication": int(len(dataset)),
        "development_subjects_present": sorted(dataset["subject"].unique().tolist()),
        "development_subjects_absent": sorted(
            set(protocol["development_subjects"]) - set(dataset["subject"].unique())
        ),
        "participant_stride_counts": {str(k): int(v) for k, v in participant_counts.items()},
        "run_stride_counts": {str(k): int(v) for k, v in run_counts.items()},
        "model_artifact": str(model_path),
        "model_sha256": sha256(model_path),
        "loso_summary": summary.to_dict(orient="records"),
        "interpretation": "Development-only LOSO after an audit-informed expansion. This is not a new untouched-test result.",
    }
    report_path = REPORTS / "EXPANSION_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Twente audited expansion v1",
        "",
        f"- 포함 구간: {report['windows_included']} / {report['windows_total']}",
        f"- 중복 제거 후 보폭: {report['matched_strides_after_deduplication']}",
        f"- 실제 포함 참가자: {', '.join(report['development_subjects_present'])}",
        f"- 제외된 개발 참가자: {', '.join(report['development_subjects_absent']) or '없음'}",
        "- Subj04·11 및 test2·test3 접근: 없음",
        "- offset 선택에 힘판 이벤트 사용: 없음",
        "",
        "## Development-only LOSO",
        "",
        summary.to_markdown(index=False),
        "",
        "이 수치는 확장 개발 LOSO이며 새로운 untouched test 성능이 아니다.",
    ]
    (REPORTS / "EXPANSION_REPORT_KO.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
