from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def constrained_decode(probabilities: np.ndarray) -> np.ndarray:
    logp = np.log(np.clip(probabilities, 1e-12, 1.0))
    n, states = logp.shape
    score = np.full((n, states), -np.inf)
    back = np.full((n, states), -1, dtype=np.int16)
    score[0, 0] = logp[0, 0]
    for t in range(1, n):
        for state in range(states):
            candidates = [(score[t - 1, state], state)]
            if state > 0:
                candidates.append((score[t - 1, state - 1], state - 1))
            best, previous = max(candidates, key=lambda item: item[0])
            score[t, state] = best + logp[t, state]
            back[t, state] = previous
    state = 3
    decoded = np.empty(n, dtype=np.int8)
    decoded[-1] = state + 1
    for t in range(n - 1, 0, -1):
        state = int(back[t, state])
        decoded[t - 1] = state + 1
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser(description="RunScope v0.1 four-stage phase inference")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "artifacts" / "runscope_phase_model_v1.joblib")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    data = pd.read_csv(args.input_csv)
    missing = [name for name in bundle["feature_names"] if name not in data.columns]
    if missing:
        raise SystemExit(f"missing input columns: {missing}")
    x = data[bundle["feature_names"]].to_numpy(dtype=np.float32)
    labels = constrained_decode(bundle["model"].predict_proba(x))
    data["phase_label"] = labels
    data["phase_name"] = [bundle["stage_names"][int(label)] for label in labels]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
