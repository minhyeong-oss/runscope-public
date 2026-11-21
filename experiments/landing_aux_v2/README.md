# landing_aux_v2

This experiment preserves `landing_aux_v1` and replaces its highly imbalanced
per-frame objective with explicit contact-onset localization.

The model first decides whether a contact onset exists in a 0.48 s sequence,
then localizes one onset frame. The four samples (about 30 ms) beginning at that
frame are decoded as the landing stage.

All validation is leave-one-athlete-out on the LAAS auxiliary dataset. Locked
Twente test subjects `Subj04` and `Subj11` remain unread.
