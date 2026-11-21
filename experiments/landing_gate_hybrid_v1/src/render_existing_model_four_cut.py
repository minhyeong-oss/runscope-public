from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "video_v1/test_outputs/twente_locked_test/final_v3/LOCKED_TEST_REPORT.json"
COMPARISON = ROOT / "video_v1/test_outputs/twente_locked_test/final_v3/locked_test_predictions_and_force_truth.csv"
POSE_ROOT = ROOT / "experiments/landing_gate_hybrid_v1/artifacts/raw_pose"
OUTPUT = ROOT / "experiments/landing_gate_hybrid_v1/reports/existing_model_four_cut.jpg"
VISUAL_ROOT = Path(
    os.environ.get("RUNSCOPE_VISUAL_ROOT", ROOT / "visualizations")
).expanduser().resolve()
VISUAL = VISUAL_ROOT / "existing-model-four-cut.html"
STAGES = {1: "LANDING", 2: "BRAKING", 3: "PROPULSION", 4: "RECOVERY"}
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
]


def representative_stride(table: pd.DataFrame) -> tuple[tuple, list[pd.Series]]:
    candidates = []
    for key, group in table.groupby(["subject", "run", "stride_id", "side"]):
        reps = []
        for stage in (1, 2, 3, 4):
            part = group[group["predicted_label"] == stage]
            if part.empty:
                break
            midpoint = (part["video_time_s"].min() + part["video_time_s"].max()) / 2.0
            reps.append(part.iloc[(part["video_time_s"] - midpoint).abs().argmin()])
        if len(reps) != 4:
            continue
        # Prefer an honest example that exposes the weak landing stage while the
        # other representative phases match force truth.
        if bool(reps[0]["correct"]) or not all(bool(row["correct"]) for row in reps[1:]):
            continue
        candidates.append((float(group["correct"].mean()), key, reps))
    if not candidates:
        raise RuntimeError("No suitable four-stage stride found")
    _, key, reps = max(candidates, key=lambda item: item[0])
    return key, reps


def pose_point(row: pd.Series, name: str, width: int, height: int) -> tuple[int, int] | None:
    x, y = row.get(f"{name}_x"), row.get(f"{name}_y")
    visibility = row.get(f"{name}_visibility", 1.0)
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(visibility):
        return None
    if float(visibility) < 0.35:
        return None
    return int(round(float(x) * width)), int(round(float(y) * height))


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = report["overall_metrics"]
    table = pd.read_csv(COMPARISON)
    (subject, run, stride_id, side), reps = representative_stride(table)
    video_path = ROOT / f"data/twente_rgb/side_running/Videos_anonymized/{subject}/run_{int(run)}_side_anonymized.avi"
    pose_path = POSE_ROOT / f"{subject}_run{int(run)}_side_10s_to_22s_mediapipe.csv"
    pose = pd.read_csv(pose_path).interpolate(limit_direction="both")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    pose_rows = []
    for rep in reps:
        source_time = float(rep["video_time_s"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(source_time * fps)))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame at {source_time}")
        index = int((pose["source_time_s"] - source_time).abs().argmin())
        frames.append(frame)
        pose_rows.append(pose.iloc[index])
    capture.release()

    height, width = frames[0].shape[:2]
    all_points = []
    joint_names = sorted({name for pair in CONNECTIONS for name in pair} | {"nose"})
    for row in pose_rows:
        for name in joint_names:
            point = pose_point(row, name, width, height)
            if point is not None:
                all_points.append(point)
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x0 = max(0, min(x_values) - int(0.10 * width))
    x1 = min(width, max(x_values) + int(0.10 * width))
    y0 = max(0, min(y_values) - int(0.08 * height))
    y1 = min(height, max(y_values) + int(0.08 * height))

    panel_width, panel_height = 300, 470
    header_height, footer_height = 74, 72
    image_height = panel_height - header_height - footer_height
    panels = []
    for stage, (frame, row, rep) in enumerate(zip(frames, pose_rows, reps), start=1):
        correct = bool(rep["correct"])
        color = (74, 220, 105) if correct else (65, 74, 238)
        for first, second in CONNECTIONS:
            p1 = pose_point(row, first, width, height)
            p2 = pose_point(row, second, width, height)
            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, color, 6, cv2.LINE_AA)
                cv2.circle(frame, p1, 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, p2, 6, color, -1, cv2.LINE_AA)
        crop = frame[y0:y1, x0:x1]
        scale = min(panel_width / crop.shape[1], image_height / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.full((panel_height, panel_width, 3), (17, 20, 25), dtype=np.uint8)
        px = (panel_width - resized.shape[1]) // 2
        py = header_height + (image_height - resized.shape[0]) // 2
        panel[py:py + resized.shape[0], px:px + resized.shape[1]] = resized
        stage_f1 = metrics["per_stage"][STAGES[stage].lower()]["f1-score"]
        cv2.putText(panel, f"{stage}  {STAGES[stage]}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 242, 246), 2, cv2.LINE_AA)
        cv2.putText(panel, f"F1  {stage_f1:.3f}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
        status = "MATCH" if correct else "MISMATCH"
        truth_name = STAGES[int(rep["ground_truth_label"])]
        cv2.putText(panel, status, (15, panel_height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"Force truth: {truth_name}", (15, panel_height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (194, 199, 207), 1, cv2.LINE_AA)
        panels.append(panel)

    top = 72
    canvas = np.full((panel_height + top, panel_width * 4, 3), (11, 14, 18), dtype=np.uint8)
    canvas[top:, :] = np.hstack(panels)
    cv2.putText(canvas, "RUNSCOPE  |  EXISTING 4-STAGE MODEL", (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 242, 246), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Locked test Macro F1  {metrics['macro_f1']:.3f}", (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (194, 199, 207), 1, cv2.LINE_AA)
    legend = "GREEN = prediction matches force truth    RED = mismatch"
    cv2.putText(canvas, legend, (690, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (194, 199, 207), 1, cv2.LINE_AA)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), canvas, [cv2.IMWRITE_JPEG_QUALITY, 84])

    encoded = base64.b64encode(OUTPUT.read_bytes()).decode("ascii")
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    fragment = f'''<div id="runscope-four-cut" style="display:grid;gap:calc(var(--font-size-base)*.5);">
  <h1>Existing model: four-stage force-validated result</h1>
  <img src="data:image/jpeg;base64,{encoded}" alt="Four side-view running frames for landing, braking, propulsion, and recovery. Landing is marked mismatch in red with F1 0.464. The other phases are marked match in green with F1 values 0.825, 0.820, and 0.932. Overall macro F1 is 0.760." style="display:block;width:100%;height:auto;" />
  <div class="viz-row text-small text-muted" aria-label="Example details">
    <span>{subject} · {int(run) / 10:.1f} km/h label · {stride_id} · {side} foot</span>
    <span>Green: force-truth match · Red: mismatch</span>
  </div>
</div>
'''
    VISUAL.write_text(fragment, encoding="utf-8")
    print(json.dumps({
        "image": str(OUTPUT),
        "visual": str(VISUAL),
        "subject": subject,
        "run": int(run),
        "stride_id": stride_id,
        "side": side,
        "macro_f1": metrics["macro_f1"],
        "stage_f1": {STAGES[stage].lower(): metrics["per_stage"][STAGES[stage].lower()]["f1-score"] for stage in STAGES},
        "representative_correct": [bool(row["correct"]) for row in reps],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
