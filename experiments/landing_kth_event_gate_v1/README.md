# landing_kth_event_gate_v1

This is a separate development model; it does not overwrite the KTH TCN
adaptation or either visual-contact v1/v2 artifact.

The frozen v2 TCN proposes candidate landing peaks. A small ML gate then uses
only local pose features around each peak to decide whether to retain it. Gate
targets come from the KTH teacher/ankle consensus prepared in
`landing_kth_adaptation_v1`.

- Training: accepted KTH official-training videos only.
- Model/threshold selection: accepted KTH official-validation videos only.
- KTH official test people: untouched.
- Previous Zenodo locked-test participants: untouched.

Reported F1 is pseudo-label agreement, not force-plate or manual-contact F1.

## Result

- Selected gate: histogram gradient boosting, depth 2, L2 0.1
- Training teacher events: 357 (208 consensus positives)
- Validation teacher events: 485 (335 consensus positives)
- Accept-all frozen-v2 baseline F1: 0.817
- Event-gate validation pseudo-label F1: 0.912
- Improvement: +0.095
- Validation participant F1 range: 0.820--0.980

The KTH official test split remains untouched, so this is a development/
validation result rather than an independent test score.

```bash
.venv_yolo/bin/python \
  experiments/landing_kth_event_gate_v1/src/train_event_gate.py
```
