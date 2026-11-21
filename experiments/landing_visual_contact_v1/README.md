# landing_visual_contact_v1

This experiment trains a treadmill-running landing/contact-onset model from the
new Zenodo dual-system dataset. The model input is the 30 fps side-view YOLO
pose sequence. The supervision is a synchronized 120 Hz OptiTrack heel/toe
contact proxy.

Important scope:

- The public archive contains video-derived YOLO pose CSV files, not the raw
  smartphone movie files.
- The target is visual/kinematic contact onset. It is not force-plate contact.
- Participant IDs are disjoint across train, validation, and locked test.
- The locked test is evaluated once after model and decoding parameters freeze.

## Frozen split

- Train: `02, 04, 05, 08, 11, 13, 19, 21`
- Validation: `15, 16`
- Locked test: `07, 14`
- Excluded before training: `06` because cross-system synchronization
  correlation is below the fixed 0.60 quality threshold.

The development/test ratio is 10:2, the closest whole-participant split to
80:20 for 12 eligible participants. Validation participants remain inside the
development partition.

## Run

```bash
.venv_yolo/bin/python experiments/landing_visual_contact_v1/src/prepare_dataset.py
.venv_yolo/bin/python experiments/landing_visual_contact_v1/src/train_and_test.py
```

Primary metrics are event F1 with a +/-1 video-frame tolerance and matched
onset MAE in milliseconds.
