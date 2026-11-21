# landing_onset_transfer_v1

Video-domain landing-onset model for real use without a force plate.

1. Initialize the temporal onset localizer with the LAAS force-labeled model.
2. Fine-tune only on the seven Twente development subjects whose RGB pose and
   force-platform initial contacts are synchronized.
3. Validate with subject-disjoint leave-one-subject-out folds.
4. Keep `Subj04` and `Subj11` out of training and model selection.

At deployment the user supplies only a side-view running video. The model can
predict a visual contact onset and a 30 ms landing window, but true F1 cannot be
computed for that user video without independent ground truth.
