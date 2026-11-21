# landing_kth_adaptation_v1

This experiment adapts the landing-event model to previously unused, real
running videos from the official KTH Human Actions dataset. It is isolated
from `landing_visual_contact_v1` and `landing_visual_contact_v2`; neither
experiment is overwritten.

## Data and split

- Source: <https://www.csc.kth.se/cvap/actions/>
- Input: the KTH `running.zip` AVI videos (25 fps, 160 x 120)
- Train: KTH official training people 11--18
- Validation: KTH official validation people 19, 20, 21, 23, 24, 25, 01, 04
- Untouched: KTH official test people 22, 02, 03, 05, 06, 07, 08, 09, 10
- Also forbidden: the prior Zenodo synchronized-data locked-test people 07
  and 14. KTH participant identifiers are dataset-local, so KTH person14 is a
  different individual and remains in KTH's official training split.

The KTH archive has video/action labels but no force plate or contact-event
labels. Consequently, the frame targets are conservative pseudo-labels: a
frozen v2 teacher peak is accepted only when it is within three 30-Hz frames
of a foot/ankle vertical extremum. Validation scores are teacher/kinematic
pseudo-label agreement, **not** force-ground-truth F1.

## Commands

```bash
unzip -q data/landing_video_kth_v1/running.zip \
  -d data/landing_video_kth_v1/extracted

YOLO_CONFIG_DIR=/private/tmp/ultralytics \
.venv_yolo/bin/python \
  experiments/landing_kth_adaptation_v1/src/prepare_kth_sequences.py

.venv_yolo/bin/python \
  experiments/landing_kth_adaptation_v1/src/train_kth_adaptation.py
```

The first pass uses all four KTH recording conditions for the official train
and validation people. The official KTH test subjects are not decoded, tuned,
or trained on.

## Result

- Processed videos: 64
- Accepted after pose/consensus quality gates: 54 (train 23, validation 31)
- Consensus pseudo-events: 543
- TCN validation pseudo-label F1: 0.779 at +/-1 frame
- Frozen-v2 baseline on the same pseudo-labels: 0.817

The TCN fine-tuning result is retained as a documented negative result. It is
not promoted over the frozen v2 model.
