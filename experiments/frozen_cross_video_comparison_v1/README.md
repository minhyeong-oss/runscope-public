# frozen_cross_video_comparison_v1

Apples-to-apples comparison of the two treadmill videos without overwriting
the original test outputs.

The comparison contains two separate tasks:

1. Landing-event detection with the same frozen
   `landing_visual_contact_v2 + landing_kth_event_gate_v1` stack.
2. Four-stage frame classification with the same frozen
   `video_phase_decoder_v2_temporal_prior` method.

No model was retrained, refitted, or threshold-tuned for either test video.
The temporal-prior decoder selects the nearest speed available in the frozen
Reznick train+validation data. Subj11/run99 is 2.75 m/s, outside that dataset's
available maximum, so it uses the nearest 2.40 m/s prior.

```bash
MPLCONFIGDIR=/private/tmp/matplotlib OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv_yolo/bin/python \
  experiments/frozen_cross_video_comparison_v1/src/evaluate_same_model.py

MPLCONFIGDIR=/private/tmp/matplotlib \
  .venv_yolo/bin/python \
  experiments/frozen_cross_video_comparison_v1/src/render_comparison.py
```

## Main result

- Landing event F1 at ±2 frames: 0.933 (Subj05/run81) vs 0.918
  (Subj11/run99).
- Four-stage temporal-prior Macro F1: 0.898 vs 0.618.
- The second result is an observed speed/participant domain-shift failure, not
  evidence that the landing-event detector collapsed.
