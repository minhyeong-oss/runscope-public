# RunScope video prototype v0.1

This folder connects a side-view running video to the existing RunScope four-stage model.

## Current result

- Input: BML-MoVi Subject 1, PG1 side-view jogging clip
- Pose backend: YOLO11n-pose, COCO-17
- Pose detection: 206 / 206 frames
- Mean lower-body keypoint confidence: 0.97 or higher
- Automatically accepted strides: 5 (frames 36-64, 104-129, 116-142, 129-157, 142-170)
- Heel-strike validation: 9/9 detected events matched synchronized V3D heel minima
- Heel-strike timing MAE: 1.78 video frames (59 ms at 30 Hz)
- Output: colored skeleton video, 100 Hz feature/prediction CSV, phase ranges, angle plot
- Full-video output: simultaneous left/right phase overlay and frame-level QC timeline

The output is an offline, out-of-domain prototype. It is not a new accuracy result.

## Pipeline

1. Extract 17 video keypoints with YOLO pose.
   When more than one person is present, keep one runner using temporal
   center-and-box-area association rather than selecting the highest-confidence
   person independently in every frame.
2. Smooth shoulder, hip, knee, and ankle trajectories.
3. Detect heel-strike proxies from direction-aware ankle position relative to the pelvis.
4. Reject turnarounds, non-side-view frames, low-confidence keypoints, and implausible durations.
5. Pair accepted same-foot events into strides.
6. Calculate 2D hip and knee angles plus ankle and pelvis proxies.
7. Resample each completed stride to 100 Hz.
8. Build the existing model's 14 input columns.
9. Predict phases in the fixed order landing, braking, propulsion, recovery.
10. Draw the predicted phase on the skeleton.

## Reproduce

Run the complete offline pipeline with one command:

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics \
MPLCONFIGDIR=/private/tmp/runscope_mpl \
  .venv_yolo/bin/python video_v1/analyze_video_end_to_end.py \
  /absolute/path/to/side_view_running.mp4 \
  --speed 2.0 \
  --motion-mode treadmill \
  --facing right \
  --min-confidence 0.60 \
  --side-mapping direct
```

`--speed` must be measured or known. Use `--side-mapping swapped` only when an external reference confirms that COCO left/right labels are reversed.
For treadmill recordings, explicitly set `--motion-mode treadmill` and the runner's `--facing` direction. The treadmill detector uses the ankle's forward-and-low position relative to the pelvis and hip, because pelvis translation is nearly zero.
The source FPS is read from the pose-extraction report and passed into stride detection automatically. Use a relaxed `--min-confidence` only for obstructed footage after visually checking the tracked-runner overlay.

Extract pose landmarks:

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics \
  .venv_yolo/bin/python video_v1/extract_pose_yolo.py \
  /path/to/Subject_1_jogging_PG1_side.mp4 \
  video_v1/outputs
```

Run one stride through the phase model:

```bash
MPLCONFIGDIR=/private/tmp/runscope_mpl LOKY_MAX_CPU_COUNT=8 \
  .venv_yolo/bin/python video_v1/run_phase_prototype.py \
  /path/to/Subject_1_jogging_PG1_side.mp4 \
  video_v1/outputs/Subject_1_jogging_PG1_side_yolo_landmarks.csv \
  --side left --stride-start 104 --stride-end 130 --speed 1.0314
```

Detect strides automatically:

```bash
MPLCONFIGDIR=/private/tmp/runscope_mpl \
  .venv_yolo/bin/python video_v1/detect_strides.py \
  video_v1/outputs/Subject_1_jogging_PG1_side_yolo_landmarks.csv \
  video_v1/outputs
```

Process every accepted stride:

```bash
MPLCONFIGDIR=/private/tmp/runscope_mpl LOKY_MAX_CPU_COUNT=8 \
  .venv_yolo/bin/python video_v1/run_auto_phase_batch.py \
  /path/to/Subject_1_jogging_PG1_side.mp4 \
  video_v1/outputs/Subject_1_jogging_PG1_side_yolo_landmarks.csv \
  video_v1/outputs/Subject_1_jogging_PG1_side_auto_strides.csv \
  --speed 1.04 --side-mapping swapped
```

## Interpretation boundary

- The source speed is approximately 1.03-1.05 m/s, below the model's 1.8-2.4 m/s training range.
- The model was trained on laboratory 3D angles; this adapter supplies monocular 2D angles.
- COCO-17 has no heel or foot-index landmarks, so ankle flexion is a shank-orientation proxy.
- Pelvis tilt is represented by a trunk-lean proxy.
- BML-MoVi does not provide force-derived four-stage labels. The phase colors cannot be scored as ground truth accuracy.
- The synchronized V3D heel trajectory validates only the automatic event timing, not the four phase boundaries.
- In this clip, COCO left/right tracking labels are swapped relative to the anatomical V3D labels; the batch summary records both.
- The model uses completed-stride `cycle_fraction`, so this version is offline rather than real-time.

For a scientifically validated video model, obtain synchronized video and force labels or collect and label a small calibration set, then retrain on video-derived features.
