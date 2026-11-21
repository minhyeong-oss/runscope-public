# landing_visual_contact_v2

This experiment preserves `landing_visual_contact_v1` and addresses the first
run's subject-domain shift without using its locked test participants for
training or tuning.

Changes from v1:

- robust per-video feature normalization at inference and training time;
- leave-one-participant-out predictions across all 10 development subjects;
- decoder selection from development out-of-fold predictions;
- final model trained on development participants only;
- v1 locked-test participants `07` and `14` remain forbidden and are not read.

The reported score is development OOF performance. It is not a new untouched
test result. A new independent labeled video source is required for another
honest test.

## Result

The selected decoder uses threshold `0.34` and a minimum event distance of 10
frames. Subject-disjoint development OOF results are:

| Event tolerance | F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| +/-1 frame (33 ms) | 0.705 | 0.712 | 0.698 |
| +/-2 frames (67 ms) | 0.896 | 0.904 | 0.887 |
| +/-3 frames (100 ms) | 0.957 | 0.967 | 0.948 |

The +/-1-frame result remains below the project target. The +/-2-frame result
exceeds 0.85, but it must be presented as a 67 ms visual-contact tolerance, not
as force-plate F1.

```bash
.venv_yolo/bin/python experiments/landing_visual_contact_v2/src/train_loso.py
```
