# RunScope

RunScope is a research prototype for analysing fixed, side-view treadmill-running video. It extracts pose landmarks, detects strides, estimates four gait phases (`Landing`, `Braking`, `Propulsion`, and `Recovery`), and produces conservative form-feedback candidates.

> This repository contains source code and protocol documentation only. It does **not** include participant videos, raw biomechanical data, derived landmark tables, trained weights, model checkpoints, or private user recordings.

The public repository intentionally does not redistribute research datasets or private media, even where a source dataset may permit redistribution under attribution. See [DATA_AND_PRIVACY.md](DATA_AND_PRIVACY.md).

## Project status

- Personal, non-commercial research prototype
- Intended for software and biomechanics experimentation
- Not a medical device and not suitable for diagnosis, injury prediction, or treatment decisions
- Predictions on an unlabeled user video are inference results, not a new F1 score
- Historical experiment notes are retained for transparency; they are not claims of universal performance

## Repository layout

```text
model_v1/      classical four-stage baseline and CSV inference
video_v1/      pose extraction, stride detection, rendering, and evaluation code
experiments/   isolated ML/DL experiments and preregistered protocols
docs/          product requirements and dataset plans
data/          source attribution and split metadata only (no dataset files)
```

Generated `artifacts/`, `reports/`, videos, pose tables, checkpoints, and dataset files are intentionally excluded.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The research scripts expect user-supplied data under `data/` or paths provided through their command-line arguments. The legacy Reznick baseline also accepts:

```bash
export RUNSCOPE_REZNICK_STREAMING=/path/to/Streaming.mat
```

Optional private-video paths used by historical experiments can be placed under `private_inputs/` or configured with environment variables documented in the relevant source file. Do not commit those files.

## Typical entry points

```bash
# Inspect command-line options before supplying your own video and model files.
python video_v1/analyze_video_end_to_end.py --help
python video_v1/extract_pose_yolo.py --help
python model_v1/predict_phase_csv.py --help
```

Some scripts depend on model artifacts produced by training scripts. No pretrained RunScope artifact is distributed in this public repository.

## Data and evaluation policy

- Follow each dataset's original license and access conditions.
- Split train/validation/test data by participant, not by frame or stride.
- Never tune on a locked test participant.
- Report force-defined, visually annotated, and unlabeled-video results separately.
- Do not report F1 for a video without independent ground truth.
- Treat `GOOD / IMPROVE / UNKNOWN` as conservative screening states, not clinical labels.

See [the Korean PRD](docs/PRD_RunScope_KO.md) and [the experiment rules](EXPERIMENT_LAYOUT.md) for the design and evaluation boundaries.

## License

Original RunScope code in this repository is released under the [GNU Affero General Public License v3.0](LICENSE) to align conservatively with the Ultralytics AGPL dependency used by parts of the video pipeline. Dataset files, private media, model weights, and third-party software are not covered by the RunScope license. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

