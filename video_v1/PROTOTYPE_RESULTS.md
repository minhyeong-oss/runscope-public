# RunScope video prototype results

## Pose extraction

- Video frames: 206
- Detected frames: 206
- Detection rate: 100%
- Left hip/knee/ankle mean confidence: 0.997 / 0.994 / 0.971
- Right hip/knee/ankle mean confidence: 0.998 / 0.996 / 0.975

## Phase output

The automatic detector accepted five same-foot strides and ran all five through the phase model with an approximate nominal speed of 1.04 m/s.

| Anatomical side | Frames | Duration | Landing | Braking | Propulsion | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| Left | 36-64 | 0.933 s | 0-30 ms | 30-440 ms | 440-500 ms | 500-930 ms |
| Right | 104-129 | 0.833 s | 0-30 ms | 30-350 ms | 350-450 ms | 450-830 ms |
| Left | 116-142 | 0.867 s | 0-30 ms | 30-380 ms | 380-470 ms | 470-870 ms |
| Right | 129-157 | 0.933 s | 0-30 ms | 30-380 ms | 380-490 ms | 490-930 ms |
| Left | 142-170 | 0.933 s | 0-30 ms | 30-360 ms | 360-490 ms | 490-930 ms |

## Automatic event validation

- Accepted video heel-strike proxies: 9
- Matched against synchronized V3D heel minima: 9/9
- Mean absolute timing error: 1.78 video frames, or 59 ms at 30 Hz
- COCO tracked-side labels were swapped relative to anatomical V3D labels in this side-view clip
- This is a kinematic heel-strike timing check. It does not validate force-defined braking, propulsion, or toe-off boundaries.

## Full-video integration

- Total source frames: 206
- Frames with at least one accepted foot-phase result: 94 (45.6%)
- Automatically accepted strides: 5
- Frames below the 0.60 transition-confidence threshold: 5
- Left and right foot phases are displayed simultaneously when their accepted strides overlap.
- Frames outside accepted strides are intentionally left unclassified instead of forcing a result.
- Each accepted stride now has a four-panel representative-stage image and an animated four-stage GIF.
- A technical assessment report calculates phase durations, estimated cadence, left/right differences, and extraction QC.
- A separate treadmill mode no longer requires pelvis translation and instead uses the ankle's forward-and-low position relative to the body.

## Sensitivity check

For the first stride, changing the joint features while keeping speed and cycle fraction fixed moved the stage boundaries:

- Original: stage 3 at 350 ms, stage 4 at 470 ms
- All joint features set to zero: stage 3 at 320 ms, stage 4 at 410 ms
- Joint feature sequence reversed: stage 3 at 290 ms, stage 4 at 460 ms
- Joint features held at their medians: stage 3 at 330 ms, stage 4 at 390 ms

The joint lines affect the output, but completed-stride timing remains a strong input. These numbers are a behavior check, not video accuracy, because this video has no force-derived phase labels.
