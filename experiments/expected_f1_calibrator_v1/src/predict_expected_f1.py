from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate expected Macro F1 from frozen-model confidence")
    parser.add_argument("inference_report", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.inference_report.read_text(encoding="utf-8"))
    artifact = joblib.load(args.artifact)
    confidence = float(report["inference"]["mean_max_probability"]["hybrid"])
    estimate = float(artifact["model"].predict([[confidence]])[0])
    error = float(artifact["empirical_absolute_error_90"])
    lower = max(0.0, estimate - error)
    upper = min(1.0, estimate + error)
    inside_range = bool(
        artifact["training_feature_min"] <= confidence <= artifact["training_feature_max"]
    )
    result = {
        "status": "EXPECTED_F1_ESTIMATE_NOT_OBSERVED_F1",
        "source_inference_report": str(args.inference_report),
        "test_video_used_for_calibrator_training": False,
        "hybrid_mean_max_probability": confidence,
        "confidence_within_calibrator_training_range": inside_range,
        "predicted_expected_macro_f1": estimate,
        "empirical_90_percent_range": {"lower": lower, "upper": upper},
        "meaning": (
            "Estimated performance from prior labeled trials with similar model confidence. "
            "It is not a direct F1 measurement because the new video has no ground truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
