from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/landing_visual_contact_v2"
V1_EXPERIMENT = ROOT / "experiments/landing_visual_contact_v1"
V1_TRAIN_SOURCE = V1_EXPERIMENT / "src/train_and_test.py"
V1_PREP_SOURCE = V1_EXPERIMENT / "src/prepare_dataset.py"

CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

BACKGROUND = "#0D1117"
PANEL = "#131A23"
FOREGROUND = "#E6EDF3"
MUTED = "#8B949E"
GREEN = "#2ECC71"
RED = "#FF5A5F"
GRAY = "#9AA4B2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module("landing_v1_train_visual", V1_TRAIN_SOURCE)
prep = load_module("landing_v1_prep_visual", V1_PREP_SOURCE)


def match_pairs(truth: np.ndarray, prediction: np.ndarray) -> list[dict]:
    pairs = []
    for true_frame in truth:
        if not len(prediction):
            pairs.append({"truth": int(true_frame), "prediction": None, "error": None})
            continue
        nearest = int(prediction[np.argmin(np.abs(prediction - true_frame))])
        pairs.append({
            "truth": int(true_frame),
            "prediction": nearest,
            "error": int(nearest - true_frame),
        })
    return pairs


def choose_examples() -> list[dict]:
    manifest = json.loads(
        (EXPERIMENT / "reports/MODEL_MANIFEST.json").read_text(encoding="utf-8")
    )
    bundle = np.load(
        EXPERIMENT / "reports/development_oof_predictions.npz", allow_pickle=True
    )
    subjects = bundle["subjects"].astype(str).tolist()
    probabilities = {
        subject: np.asarray(probability, dtype=np.float32)
        for subject, probability in zip(subjects, bundle["probabilities"])
    }
    truths = {
        subject: np.asarray(truth, dtype=np.int64)
        for subject, truth in zip(subjects, bundle["truths"])
    }
    threshold = float(manifest["selected_threshold"])
    distance = int(manifest["selected_minimum_peak_distance_frames"])

    records = []
    for subject in subjects:
        prediction = train.predicted_events(probabilities[subject], threshold, distance)
        for pair in match_pairs(truths[subject], prediction):
            frame = pair["truth"]
            if frame < 100 or frame >= len(probabilities[subject]) - 100:
                continue
            pair.update({
                "subject": subject,
                "probability": float(probabilities[subject][
                    pair["prediction"] if pair["prediction"] is not None else frame
                ]),
            })
            records.append(pair)

    def first(predicate, preferred_subjects: list[str]) -> dict:
        for subject in preferred_subjects:
            for record in records:
                if record["subject"] == subject and predicate(record):
                    return record.copy()
        raise RuntimeError("No example satisfies selection rule")

    exact = first(lambda item: item["error"] == 0, ["19", "13", "16"])
    one_frame = first(lambda item: abs(item["error"]) == 1, ["11", "05", "08"])
    near = first(lambda item: abs(item["error"]) == 2, ["02", "04", "21", "15"])
    missed = first(lambda item: abs(item["error"]) >= 5, ["15", "02", "21"])
    examples = [
        {**exact, "title": "EXACT HIT", "status": "GOOD", "color": GREEN},
        {**one_frame, "title": "WITHIN 1 FRAME", "status": "GOOD", "color": GREEN},
        {**near, "title": "2-FRAME NEAR MISS", "status": "UNCERTAIN", "color": GRAY},
        {**missed, "title": "MISSED CONTACT", "status": "BAD", "color": RED},
    ]
    return examples


def draw_skeleton(
    axis: plt.Axes,
    coordinates: np.ndarray,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for start, end in CONNECTIONS:
        axis.plot(
            [coordinates[start, 0], coordinates[end, 0]],
            [coordinates[start, 1], coordinates[end, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1],
        s=20 if linewidth > 2 else 11,
        color=color, alpha=alpha, edgecolors="none", zorder=3,
    )


def main() -> None:
    examples = choose_examples()
    figure, axes = plt.subplots(1, 4, figsize=(16, 6.4), dpi=150)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.025, right=0.985, top=0.83, bottom=0.13, wspace=0.08)
    figure.suptitle(
        "RunScope Landing v2 — subject-disjoint OOF examples (30 fps)",
        color=FOREGROUND, fontsize=17, fontweight="bold", y=0.95,
    )
    figure.text(
        0.5, 0.895,
        "Gray ghost = marker-derived true contact pose   |   Colored skeleton = model-selected frame",
        ha="center", color=MUTED, fontsize=10,
    )

    selection_records = []
    for index, (axis, example) in enumerate(zip(axes, examples), start=1):
        subject = example["subject"]
        pose_path = prep.POSE_ROOT / subject / "side/Walk_8km.csv"
        coordinates, _ = prep.load_pose(pose_path)
        true_frame = int(example["truth"])
        predicted_frame = int(example["prediction"])
        true_pose = coordinates[true_frame]
        predicted_pose = coordinates[predicted_frame]

        combined = np.vstack([true_pose, predicted_pose])
        minimum = combined.min(axis=0)
        maximum = combined.max(axis=0)
        width = max(maximum[0] - minimum[0], 1.0)
        height = max(maximum[1] - minimum[1], 1.0)
        x_padding = width * 0.20
        y_padding = height * 0.10

        axis.set_facecolor(PANEL)
        draw_skeleton(axis, true_pose, GRAY, linewidth=2.0, alpha=0.38)
        draw_skeleton(axis, predicted_pose, example["color"], linewidth=4.0, alpha=0.95)
        foot = predicted_pose[15 if predicted_pose[15, 1] >= predicted_pose[16, 1] else 16]
        axis.scatter(
            [foot[0]], [foot[1]], s=170, facecolors="none",
            edgecolors=example["color"], linewidths=3.0, zorder=5,
        )
        floor_y = max(true_pose[15, 1], true_pose[16, 1]) + height * 0.04
        axis.plot(
            [minimum[0] - x_padding, maximum[0] + x_padding],
            [floor_y, floor_y], color=MUTED, linewidth=1.2, alpha=0.45,
        )

        axis.set_xlim(minimum[0] - x_padding, maximum[0] + x_padding)
        axis.set_ylim(maximum[1] + y_padding, minimum[1] - y_padding)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

        error_frames = int(example["error"])
        error_ms = error_frames * 1000.0 / 30.0
        axis.text(
            0.04, 0.97, f"{index} · {example['title']}",
            transform=axis.transAxes, va="top", color=example["color"],
            fontsize=11, fontweight="bold",
        )
        axis.text(
            0.04, 0.89, f"Subject {subject}  ·  {example['status']}",
            transform=axis.transAxes, va="top", color=FOREGROUND, fontsize=9.5,
        )
        axis.text(
            0.04, 0.08,
            f"True {true_frame}  |  Pred {predicted_frame}\n"
            f"Error {error_frames:+d} frame ({error_ms:+.0f} ms)\n"
            f"Probability {example['probability']:.2f}",
            transform=axis.transAxes, va="bottom", color=FOREGROUND,
            fontsize=9.5, linespacing=1.35,
        )
        selection_records.append({
            "panel": index,
            "title": example["title"],
            "status": example["status"],
            "subject": subject,
            "truth_frame": true_frame,
            "predicted_frame": predicted_frame,
            "error_frames": error_frames,
            "error_ms": error_ms,
            "probability": example["probability"],
        })

    figure.text(
        0.5, 0.055,
        "Green = correct within ±1 frame   ·   Gray = outside strict tolerance but within ±2   ·   Red = failed localization",
        ha="center", color=FOREGROUND, fontsize=10,
    )
    output = EXPERIMENT / "reports/v2_landing_examples_four_cut.png"
    figure.savefig(output, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    (EXPERIMENT / "reports/v2_landing_examples_four_cut.json").write_text(
        json.dumps(selection_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
