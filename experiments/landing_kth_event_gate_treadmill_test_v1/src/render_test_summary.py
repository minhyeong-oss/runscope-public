from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_kth_event_gate_treadmill_test_v1"
REPORT = EXPERIMENT / "reports/TEST_REPORT.json"
OUTPUTS = EXPERIMENT / "reports/frozen_test_outputs.npz"

BACKGROUND = "#0B0F14"
PANEL = "#121820"
FOREGROUND = "#F1F5F9"
MUTED = "#9AA6B2"
GREEN = "#35D07F"
RED = "#FF5A62"
YELLOW = "#F6C344"
BLUE = "#4A90E2"


def matched_pairs(truth: np.ndarray, prediction: np.ndarray, tolerance: int):
    unmatched = set(range(len(truth)))
    pairs = []
    false_predictions = []
    for predicted in prediction:
        candidates = [
            index for index in unmatched
            if abs(int(truth[index]) - int(predicted)) <= tolerance
        ]
        if not candidates:
            false_predictions.append(int(predicted))
            continue
        chosen = min(candidates, key=lambda index: abs(int(truth[index]) - int(predicted)))
        unmatched.remove(chosen)
        pairs.append((int(truth[chosen]), int(predicted)))
    missed_truth = [int(truth[index]) for index in sorted(unmatched)]
    return pairs, false_predictions, missed_truth


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    bundle = np.load(OUTPUTS)
    truth = bundle["force_truth_events"].astype(int)
    prediction = bundle["gate_accepted_events"].astype(int)
    teacher = bundle["teacher_events"].astype(int)
    pairs, false_predictions, missed_truth = matched_pairs(truth, prediction, tolerance=1)

    figure = plt.figure(figsize=(15, 7.2), dpi=160, facecolor=BACKGROUND)
    grid = figure.add_gridspec(2, 1, height_ratios=[2.2, 1.1], hspace=0.34)
    timeline = figure.add_subplot(grid[0])
    bars = figure.add_subplot(grid[1])
    for axis in (timeline, bars):
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=MUTED)
        for spine in axis.spines.values():
            spine.set_color("#27313D")

    truth_seconds = truth / 30.0
    prediction_seconds = prediction / 30.0
    teacher_seconds = teacher / 30.0
    timeline.scatter(truth_seconds, np.full_like(truth_seconds, 2.0), marker="|", s=260, linewidths=3, color=BLUE, label="Force truth")
    timeline.scatter(teacher_seconds, np.full_like(teacher_seconds, 1.0), marker="|", s=230, linewidths=2.5, color=MUTED, label="v2 candidates")
    timeline.scatter(prediction_seconds, np.full_like(prediction_seconds, 0.0), marker="|", s=260, linewidths=3, color=GREEN, label="Gate accepted")
    for true_frame, predicted_frame in pairs:
        timeline.plot(
            [true_frame / 30.0, predicted_frame / 30.0], [2.0, 0.0],
            color=GREEN, alpha=0.28, linewidth=1.0,
        )
    if false_predictions:
        timeline.scatter(
            np.asarray(false_predictions) / 30.0,
            np.full(len(false_predictions), -0.18),
            marker="x", s=55, linewidths=2, color=RED, label="False positive (±1 frame)",
        )
    if missed_truth:
        timeline.scatter(
            np.asarray(missed_truth) / 30.0,
            np.full(len(missed_truth), 2.18),
            marker="x", s=55, linewidths=2, color=YELLOW, label="Missed truth (±1 frame)",
        )
    timeline.set_xlim(0, max(12.0, float(max(truth.max(), prediction.max()) / 30.0 + 0.25)))
    timeline.set_ylim(-0.55, 2.55)
    timeline.set_yticks([0, 1, 2], ["Gate accepted", "v2 candidate", "Force truth"])
    timeline.set_xlabel("Clip time (s)", color=FOREGROUND)
    video_label = Path(report["video"]).stem.replace("_side_10s_to_22s", "")
    timeline.set_title(
        f"Frozen treadmill landing test · {video_label}",
        color=FOREGROUND, fontsize=17, fontweight="bold", loc="left", pad=14,
    )
    timeline.text(
        0.0, 1.04,
        f"{report['force_truth']['events']} force contacts · "
        f"{report['predicted_events']['accepted_after_gate']} predictions · "
        "no retraining or threshold tuning",
        transform=timeline.transAxes, color=MUTED, fontsize=10,
    )
    timeline.grid(axis="x", color="#27313D", linewidth=0.7, alpha=0.65)
    legend = timeline.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)

    labels = ["±1 frame\n33 ms", "±2 frames\n67 ms", "±3 frames\n100 ms"]
    keys = ["within_1_frames", "within_2_frames", "within_3_frames"]
    values = [report["event_gate_metrics"][key]["event_f1"] for key in keys]
    positions = np.arange(len(values))
    bars.barh(positions, values, color=[GREEN, BLUE, BLUE], height=0.54)
    bars.set_yticks(positions, labels)
    bars.set_xlim(0, 1.0)
    bars.invert_yaxis()
    bars.set_xlabel("Event F1", color=FOREGROUND)
    bars.grid(axis="x", color="#27313D", linewidth=0.7, alpha=0.65)
    for position, value in zip(positions, values):
        metrics = report["event_gate_metrics"][keys[position]]
        bars.text(
            min(value + 0.018, 0.95), position,
            f"F1 {value:.3f}  P {metrics['precision']:.3f}  R {metrics['recall']:.3f}",
            va="center", color=FOREGROUND, fontsize=10,
        )
    delta = (
        report["event_gate_metrics"]["within_1_frames"]["event_f1"]
        - report["teacher_metrics"]["within_1_frames"]["event_f1"]
    )
    bars.text(
        0.99, 0.02,
        f"KTH Gate passed {report['predicted_events']['accepted_after_gate']}/"
        f"{report['predicted_events']['teacher_before_gate']} candidates → "
        f"ΔF1 vs frozen v2 = {delta:+.3f}",
        transform=bars.transAxes, ha="right", va="bottom", color=YELLOW, fontsize=10,
    )

    output = EXPERIMENT / "reports/treadmill_landing_force_test_summary.png"
    figure.savefig(output, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
