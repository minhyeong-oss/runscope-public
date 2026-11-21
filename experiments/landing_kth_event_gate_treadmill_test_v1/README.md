# landing_kth_event_gate_treadmill_test_v1

Fixed-parameter external treadmill-video test of the frozen
`landing_kth_event_gate_v1` stack.

- Video: Twente Subj05 run81, sagittal treadmill view, 10--22 s clip
- Ground truth: force-platform synchronized initial-contact times
- Teacher: frozen `landing_visual_contact_v2`
- Gate: frozen `landing_kth_event_gate_v1`
- No training, refitting, decoder search, or threshold tuning on Twente

This Twente trial was not used to train either model in the tested stack. It
has been used by older, separate RunScope experiments, so the report describes
it as an external test for the current stack rather than a globally unseen
repository benchmark.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LOKY_MAX_CPU_COUNT=1 .venv_yolo/bin/python \
  experiments/landing_kth_event_gate_treadmill_test_v1/src/evaluate_treadmill.py
```

## Result

- Force-truth contacts: 29
- Predicted contacts: 31
- Event F1 within 1 frame (33 ms): 0.867
- Event F1 within 2 frames (67 ms): 0.933
- Pose detection rate: 100%
- KTH gate accepted all 31 v2 candidates, so its delta on this treadmill trial
  was 0.000. The strong result came from the frozen v2 candidate detector.
