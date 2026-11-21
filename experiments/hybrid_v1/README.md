# Hybrid v1

Independent ensemble of `boundary_regression_v1` and `temporal_tcn_v1`.
Neither component is overwritten.

The evaluation first creates out-of-fold predictions from both components. For
each held-out subject, the ensemble rule is selected using only the other
subjects' out-of-fold predictions, then applied to that subject. This prevents
choosing the blend with the evaluated person's labels.

```bash
.venv_yolo/bin/python experiments/hybrid_v1/src/evaluate_hybrid.py
```
