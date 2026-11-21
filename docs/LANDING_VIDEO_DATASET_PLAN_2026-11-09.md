# Landing video dataset plan — 2026-11-09

## Selected new source

**A Synchronized Dual-System Marker-based and Markerless Motion Capture Dataset
for Gait and Fundamental Fitness Activities**, Zenodo v3:
https://doi.org/10.5281/zenodo.19720803

This source was selected after excluding sources already used in RunScope
(BioCV, Twente, BML-MoVi, Quintic, LAAS, Stryder, and the Figshare 3D motion
dataset).

The release provides:

- 21 participants;
- running plus six other activities;
- 30 fps smartphone-video-derived YOLO pose coordinates from two views;
- OptiTrack marker trajectories at 120 Hz;
- openly downloadable archives;
- 485.8 MB markerless pose CSV data and 654.2 MB marker-based data.

The metadata and processing-code archives were downloaded and their published
MD5 checksums verified locally. Raw downloads live under
`data/landing_video_new_v1/` and are excluded from Git.

After downloading the full release, archive inspection showed that the public
markerless archive does **not** contain the raw smartphone movie files. It
contains the YOLO keypoints previously extracted from those videos. Therefore
this release supports pose-sequence training but cannot produce original-image
four-cut screenshots.

## Why it is useful

There is no force plate, so this source cannot provide a force-threshold landing
label. It can provide a **visual-contact** target: the first synchronized video
frame associated with the heel/toe marker reaching its contact-height region
after downward motion. The 120 Hz marker trajectory gives a better pseudo-label
than estimating contact from the 30 fps image alone.

This target must not be reported as force-plate F1. Evaluation names will use
`visual_contact_f1`, `within_one_video_frame`, and `onset_mae_ms`.

## Frozen split before training

Only 13 participants contain both the 8 km/h side pose sequence and the 120 Hz
marker trial. One pair (participant 06) failed the fixed synchronization quality
threshold before training. The resulting split is:

1. Train: `02, 04, 05, 08, 11, 13, 19, 21`.
2. Validation: `15, 16`.
3. Locked test: `07, 14`.
4. Excluded before training: `06` because synchronization correlation was below
   0.60.
5. Never fine-tune or change thresholds using locked-test participants.
6. Keep left- and right-foot contacts from the same participant in the same
   split.

## Label generation

1. Synchronize the 120 Hz marker trajectory to each 30 fps video using the
   release alignment.
2. Find candidate contacts independently for each foot from heel/toe vertical
   position, downward velocity, and subsequent low-velocity support.
3. Project each candidate onto the nearest video frame.
4. Reject occluded, clipped, turning, or ambiguous contacts.
5. Manually inspect a stratified sample from every development participant.
6. Use an `unknown` label where visual contact cannot be determined reliably.

Because 30 fps is approximately 33 ms per frame, the primary metric will be
contact-event F1 with a ±1-frame tolerance. Exact 30 ms force-window F1 is not a
valid claim for this dataset.

## Secondary source

The KTH Human Action Dataset provides running and jogging clips for 25 people in
four scenes, but only at 160×120 and 25 fps with no pose or force labels:
https://www.csc.kth.se/cvap/actions/

It may be used later for coarse running-motion augmentation, not as the landing
ground-truth source.
