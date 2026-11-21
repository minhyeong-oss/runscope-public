# RunScope 4-stage phase model v0.1

This research prototype classifies one completed running stride into four ordered stages.

1. landing window
2. braking
3. propulsion
4. recovery

The model uses datasets 3, 4, 5, and 7 only. Dataset 5 supplies compatible frame-level joint-angle training examples. Datasets 3, 4, and 7 supply speed, event-timing, and environment evidence.

## Locked-test result

- Participant-held-out test: AB04 and AB10, 353 strides, 28,623 frames
- Macro F1: 0.9585 (timing-only baseline: 0.8558)
- Braking-to-propulsion boundary MAE: 16.53 ms
- Propulsion-to-recovery boundary MAE: 23.35 ms

See `MODEL_CARD.md` for the exact stage definitions, data roles, split policy, and limitations.

## Train

```bash
LOKY_MAX_CPU_COUNT=8 python train_runscope_v1.py
```

## Inference

Prepare a CSV with the columns listed in `artifacts/feature_schema.json`, then run:

```bash
python predict_phase_csv.py input.csv output.csv
```

The first model supports offline completed-stride analysis at 1.8–2.4 m/s. Do not use it for medical diagnosis or as a good/bad running-form classifier.
