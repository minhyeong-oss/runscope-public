# Third-party notices

RunScope source code can call third-party libraries and work with separately obtained datasets. Those components retain their own licenses; the repository's MIT License does not relicense them.

## Runtime libraries

- **Ultralytics** (`ultralytics`, YOLO pose): AGPL-3.0 or an Ultralytics Enterprise license, depending on use. See <https://www.ultralytics.com/license>.
- **MediaPipe**: Apache License 2.0. See <https://github.com/google-ai-edge/mediapipe>.
- **PyTorch**: BSD-style license. See <https://github.com/pytorch/pytorch>.
- **scikit-learn**: BSD 3-Clause License. See <https://github.com/scikit-learn/scikit-learn>.
- **OpenCV**: Apache License 2.0. See <https://github.com/opencv/opencv>.
- **NumPy, SciPy, pandas, Matplotlib, Pillow, h5py, joblib, ezc3d**: each package remains subject to its upstream license.

No third-party source code or pretrained model weights are vendored here.

## Research datasets

No dataset files are redistributed. Source metadata under `data/` is provided only for attribution and reproducibility. Users must obtain data from the official provider and comply with its current license, ethics, privacy, and access terms.

The Twente source metadata references Zenodo records including:

- RGB front/side video: <https://doi.org/10.5281/zenodo.6644593>
- Matching wearable/biomechanical data: <https://doi.org/10.5281/zenodo.6457662>
- Processed participant files: <https://doi.org/10.5281/zenodo.7422031>

Do not publish identifiable participant media or private user recordings without explicit permission.

