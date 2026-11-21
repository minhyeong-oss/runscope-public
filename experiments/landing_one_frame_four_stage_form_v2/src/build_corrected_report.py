from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_one_frame_four_stage_form_v2"
REPORTS = EXPERIMENT / "reports"
V1_REPORT = (
    ROOT
    / "experiments/one_frame_four_stage_form_v1/reports/ONE_FRAME_FOUR_STAGE_FORM_REPORT.json"
)
V1_BOUNDARIES = (
    ROOT / "experiments/one_frame_four_stage_form_v1/reports/one_frame_boundary_events.csv"
)
LOCKED = (
    ROOT
    / "video_v1/test_outputs/twente_locked_test/final_v3/locked_test_predictions_and_force_truth.csv"
)
STAGES = {1: "landing", 2: "braking", 3: "propulsion", 4: "recovery"}


def frame_metrics(table: pd.DataFrame) -> dict:
    truth = table["ground_truth_label"].to_numpy(dtype=int)
    prediction = table["predicted_label"].to_numpy(dtype=int)
    labels = list(STAGES)
    classification = classification_report(
        truth,
        prediction,
        labels=labels,
        target_names=[STAGES[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": int(len(table)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro")),
        "per_stage_f1": {
            STAGES[label]: float(classification[STAGES[label]]["f1-score"])
            for label in labels
        },
    }


def landing_boundary_metrics(table: pd.DataFrame) -> dict:
    valid = table["absolute_error_ms"].notna()
    expected = int(len(table))
    predicted = int(valid.sum())
    hits = int(table.loc[valid, "within_one_frame"].sum())
    precision = hits / predicted if predicted else 0.0
    recall = hits / expected if expected else 0.0
    event_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "expected_events": expected,
        "predicted_events": predicted,
        "one_frame_hits": hits,
        "precision": precision,
        "recall": recall,
        "event_f1": event_f1,
        "mae_ms": float(table.loc[valid, "absolute_error_ms"].mean()),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    v1 = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    locked = pd.read_csv(LOCKED)
    boundaries = pd.read_csv(V1_BOUNDARIES)
    landing_boundaries = boundaries[boundaries["boundary_into_stage"] == 1].copy()

    per_trial = {}
    for (subject, run), group in locked.groupby(["subject", "run"]):
        trial_name = f"{subject}_run{int(run)}"
        landing_group = landing_boundaries[
            (landing_boundaries["subject"] == subject)
            & (landing_boundaries["run"] == int(run))
        ]
        posture = next(
            item for item in v1["trial_posture_feedback"] if item["trial"] == trial_name
        )
        per_trial[trial_name] = {
            "landing_boundary_one_frame": landing_boundary_metrics(landing_group),
            "four_stage_frame_classification": frame_metrics(group),
            "posture_assessments": posture["assessments"],
            "posture_f1": None,
            "board": posture["board"],
            "html": posture["html"],
        }

    report = {
        "status": "CORRECTED_EVALUATION_POLICY_COMPLETE",
        "policy": {
            "landing_initial_contact": "event F1 with ±1 native video frame tolerance",
            "braking_propulsion_recovery": "ordinary frame classification F1; no tolerance",
            "four_stage_overall": "ordinary framewise Macro F1; no tolerance",
            "posture": "GOOD/IMPROVE/UNKNOWN screening; no ground-truth F1",
        },
        "landing_event_v3": v1["landing_event"],
        "frozen_four_stage_landing_boundary_one_frame": landing_boundary_metrics(
            landing_boundaries
        ),
        "frozen_four_stage_frame_classification": frame_metrics(locked),
        "per_trial": per_trial,
        "removed_metric": (
            "The former four-boundary ±1-frame Macro F1 is not used in v2 because braking, "
            "propulsion, and recovery are evaluated as frame intervals, not tolerant events."
        ),
        "test_refit_or_retuning": False,
        "historical_reference_warning": (
            "Subj04/Subj11 are reused historical references, not a new untouched benchmark."
        ),
    }
    output = REPORTS / "CORRECTED_POLICY_REPORT.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    phase = report["frozen_four_stage_frame_classification"]
    landing = report["frozen_four_stage_landing_boundary_one_frame"]
    result = f"""# RunScope 정정 평가 결과

## 적용 기준

- Landing 최초 접지: **±1프레임 이벤트 F1**
- 제동·추진·회복: **허용오차 없는 프레임 분류 F1**
- 자세 Good/Bad: 정답 라벨이 없어 F1 없음

## 전체 결과

- Landing-event v3 역사적 참조 ±1프레임 F1: **{v1['landing_event']['historical_reference']['one_frame_event_f1']:.3f}**
- Frozen 4단계 시스템 Landing 경계 ±1프레임 F1: **{landing['event_f1']:.3f}**
- 4단계 프레임 Macro F1: **{phase['macro_f1']:.3f}**
- 착지 구간 프레임 F1: **{phase['per_stage_f1']['landing']:.3f}**
- 제동 구간 프레임 F1: **{phase['per_stage_f1']['braking']:.3f}**
- 추진 구간 프레임 F1: **{phase['per_stage_f1']['propulsion']:.3f}**
- 회복 구간 프레임 F1: **{phase['per_stage_f1']['recovery']:.3f}**

`±1프레임`은 Landing 최초 접지에만 적용한다. 기존 v1의 4개 경계 평균
`0.804`는 정정 정책에서 사용하지 않는다.
"""
    (REPORTS / "RESULT_KO.md").write_text(result, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output),
                "landing_event_v3_one_frame_f1": v1["landing_event"][
                    "historical_reference"
                ]["one_frame_event_f1"],
                "four_stage_landing_boundary_one_frame_f1": landing["event_f1"],
                "four_stage_frame_macro_f1": phase["macro_f1"],
                "per_stage_frame_f1": phase["per_stage_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
