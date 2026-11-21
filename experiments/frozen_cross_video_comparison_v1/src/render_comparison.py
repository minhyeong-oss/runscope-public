from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
REPORT_DIR = EXPERIMENT / "reports"
REPORT = REPORT_DIR / "CROSS_VIDEO_REPORT.json"
OUTPUT = REPORT_DIR / "frozen_cross_video_comparison.png"

COLORS = {"Subj05_run81": "#4C94DF", "Subj11_run99": "#35C985"}
LABELS = {"Subj05_run81": "Subj05 · 8.1 km/h", "Subj11_run99": "Subj11 · 9.9 km/h"}


def annotate(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color="#F2F5F8",
            fontsize=11,
            fontweight="bold",
        )


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    trials = ["Subj05_run81", "Subj11_run99"]
    landing = report["landing_event_same_frozen_stack"]
    four_stage = report["four_stage_same_frozen_method"]

    plt.rcParams.update({
        "font.family": "Apple SD Gothic Neo",
        "axes.facecolor": "#121820",
        "figure.facecolor": "#0A0F15",
        "axes.edgecolor": "#34404D",
        "axes.labelcolor": "#DCE3EA",
        "xtick.color": "#AEB9C5",
        "ytick.color": "#AEB9C5",
        "text.color": "#F2F5F8",
        "grid.color": "#2B3540",
    })
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.34, wspace=0.22)
    ax_event = fig.add_subplot(grid[0, 0])
    ax_macro = fig.add_subplot(grid[0, 1])
    ax_stage = fig.add_subplot(grid[1, :])

    fig.suptitle("동일 고정 모델 · 다른 트레드밀 영상 비교", fontsize=23, fontweight="bold", y=0.992)
    fig.text(
        0.5,
        0.946,
        "재학습·재조정 없음  |  착지 이벤트와 4단계 프레임 분류는 서로 다른 평가 과제",
        ha="center",
        color="#AEB9C5",
        fontsize=13,
    )

    x = np.arange(2)
    width = 0.34
    event_bars = []
    for index, trial in enumerate(trials):
        values = [
            landing[trial]["within_1_frame"]["event_f1"],
            landing[trial]["within_2_frames"]["event_f1"],
        ]
        bars = ax_event.bar(x + (index - 0.5) * width, values, width, color=COLORS[trial], label=LABELS[trial])
        event_bars.append(bars)
        annotate(ax_event, bars)
    ax_event.set_title("착지 이벤트 F1", fontsize=16, loc="left")
    ax_event.set_xticks(x, ["±1 frame\n33 ms", "±2 frames\n67 ms"])
    ax_event.set_ylim(0, 1.07)
    ax_event.set_ylabel("Event F1")
    ax_event.grid(axis="y", alpha=0.55)
    ax_event.legend(frameon=False, loc="lower right")

    macro_values = [four_stage[trial]["metrics"]["macro_f1"] for trial in trials]
    macro_bars = ax_macro.bar(
        np.arange(2),
        macro_values,
        color=[COLORS[trial] for trial in trials],
        width=0.58,
    )
    annotate(ax_macro, macro_bars)
    ax_macro.axhline(0.85, color="#F6C64A", linestyle="--", linewidth=1.5)
    ax_macro.text(1.42, 0.86, "목표 0.85", ha="right", va="bottom", color="#F6C64A", fontsize=10)
    ax_macro.set_title("4단계 동일 temporal-prior Macro F1", fontsize=16, loc="left")
    ax_macro.set_xticks(np.arange(2), [LABELS[trial] for trial in trials])
    ax_macro.set_ylim(0, 1.07)
    ax_macro.set_ylabel("Macro F1")
    ax_macro.grid(axis="y", alpha=0.55)

    stages = ["landing", "braking", "propulsion", "recovery"]
    stage_labels = ["착지", "제동", "추진", "회복"]
    sx = np.arange(len(stages))
    stage_bars = []
    for index, trial in enumerate(trials):
        values = [four_stage[trial]["metrics"]["per_stage"][stage]["f1-score"] for stage in stages]
        bars = ax_stage.bar(
            sx + (index - 0.5) * width,
            values,
            width,
            color=COLORS[trial],
            label=LABELS[trial],
        )
        stage_bars.append(bars)
        annotate(ax_stage, bars)
    ax_stage.set_title("4단계별 F1 · 동일한 속도 조건부 경계 규칙", fontsize=16, loc="left")
    ax_stage.set_xticks(sx, stage_labels)
    ax_stage.set_ylim(0, 1.09)
    ax_stage.set_ylabel("Stage F1")
    ax_stage.grid(axis="y", alpha=0.55)
    ax_stage.legend(frameon=False, loc="lower right")
    ax_stage.text(
        0.0,
        -0.23,
        "주의: 9.9 km/h 영상은 학습 속도 범위를 벗어나 가장 가까운 8.64 km/h(2.40 m/s) prior를 사용",
        transform=ax_stage.transAxes,
        color="#F6C64A",
        fontsize=11,
    )

    for ax in (ax_event, ax_macro, ax_stage):
        ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.975, top=0.865, bottom=0.11)
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor())
    print(OUTPUT)


if __name__ == "__main__":
    main()
