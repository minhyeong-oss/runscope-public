# Boundary regression v1

This is an isolated RunScope experiment. It does not overwrite or refit the
existing v1/v2/v3 models.

## Purpose

The locked v3 evaluation showed that the 30 ms landing stage was most sensitive
to video heel-strike timing error. This experiment uses MediaPipe heel and foot
index trajectories to learn force-derived stage boundary times directly from
development subjects.

## Data rule

- Development only: Subj05, Subj06, Subj07, Subj08, Subj09, Subj10, Subj12
- Never accessed by this experiment: Subj04, Subj11
- Model selection: subject-disjoint leave-one-subject-out validation

## Run

```bash
.venv_yolo/bin/python experiments/boundary_regression_v1/src/extract_development_pose.py
.venv_yolo/bin/python experiments/boundary_regression_v1/src/train_boundary_regression.py
```

Raw MediaPipe CSV files are reproducible and excluded from Git. The selected
model, cross-validation tables, manifest, and concise report are versioned.
