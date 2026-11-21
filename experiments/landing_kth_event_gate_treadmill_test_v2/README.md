# landing_kth_event_gate_treadmill_test_v2

Second fixed-parameter treadmill-video test of the frozen
`landing_visual_contact_v2` + `landing_kth_event_gate_v1` stack.

- Video: Twente Subj11 run99, sagittal treadmill view, 10--22 s clip
- Different participant and speed from the first Subj05 run81 test
- Ground truth: first force-synchronized `landing` sample in each stride
- No training, refitting, decoder search, or threshold tuning
- The first treadmill test remains stored separately in `..._test_v1`

Subj11 belongs to a historical locked-test set that older RunScope models have
already evaluated. It is nevertheless external to the current Zenodo+KTH
training stack, so this is a frozen additional-video check rather than a claim
of a newly untouched project-wide benchmark.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LOKY_MAX_CPU_COUNT=1 .venv_yolo/bin/python \
  experiments/landing_kth_event_gate_treadmill_test_v2/src/evaluate_treadmill.py
```

## Landing result

- Force-truth contacts: 31
- v2 candidates: 32
- KTH gate accepted: 30
- Event F1 within 1 frame (33 ms): 0.787
- Event F1 within 2 frames (67 ms): 0.918
- At 67 ms the gate reduced false positives from 4 to 2, improving F1 from
  0.889 to 0.918.

## Separate four-stage result

The older frozen four-stage v3 predictions were also compared with the
force-derived stage labels for the same video. This is a separate model and
metric from the landing-event gate above.

```bash
MPLCONFIGDIR=/private/tmp/matplotlib OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv_yolo/bin/python \
  experiments/landing_kth_event_gate_treadmill_test_v2/src/render_four_stage_force_test.py
```

- Macro F1: 0.709
- Accuracy: 0.832
- Landing F1: 0.359
- Braking F1: 0.762
- Propulsion F1: 0.820
- Recovery F1: 0.896

The green/red skeleton on the result board indicates whether the stage
prediction matches the force-derived stage truth. It is not a clinical
Good/Bad running-form label; form quality remains unknown because the dataset
does not provide that annotation.
