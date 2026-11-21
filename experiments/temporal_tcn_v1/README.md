# Temporal TCN v1

Independent framewise four-stage classifier. It does not overwrite the
baseline, the foot-keypoint experiment, or boundary regression v1.

The input is a 64-step pose sequence spanning 120 ms before a video contact
proxy through the next same-foot proxy. A small temporal convolutional network
predicts landing, braking, propulsion, and recovery at each step. A monotonic
decoder enforces the biomechanical order recovery -> landing -> braking ->
propulsion -> recovery.

Only development subjects are used. Subj04 and Subj11 are forbidden.

```bash
.venv_yolo/bin/python experiments/temporal_tcn_v1/src/prepare_sequences.py
.venv_yolo/bin/python experiments/temporal_tcn_v1/src/train_tcn.py
```
