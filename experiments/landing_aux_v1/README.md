# landing_aux_v1

This experiment trains a separate force-supervised landing gate without
overwriting the existing four-stage models.

## Scope

- Source: LAAS Parkour Dataset, `sv` and `kv` trials only.
- Ground truth: each foot-force-platform onset after at least 100 ms below 20 N.
- Positive label: the first 30 ms after contact onset.
- Split: leave-one-subject-out across PKFC, PKLP, PKLT, PKMT, and PKRY.
- Locked Twente subjects `Subj04` and `Subj11` are not read during preparation,
  model selection, or training.

This is auxiliary landing/contact pretraining. It is not a claim of final
treadmill-running performance because parkour landing differs from running.

## Reproduce

```bash
PYTHONPATH=data/landing_aux_v1/vendor \
  python experiments/landing_aux_v1/src/prepare_laas_landing.py
python experiments/landing_aux_v1/src/train_landing_aux.py
```

The final binary model is saved to
`artifacts/landing_aux_laas_v1.pt`. Integration with the existing four-stage
running model is a separate, subsequent experiment.
