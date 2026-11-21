# Twente RGB treadmill dataset

- Title: RGB data [front and side views] of the wearable system validation study
- Creator: Huawei Wang, University of Twente
- DOI: https://doi.org/10.5281/zenodo.6644593
- License: CC BY 4.0
- Archive: `Videos_anonymized.rar`
- Expected size: 17,770,674,024 bytes
- Expected MD5: `aba38a34afb7d2e3ba1a5e9f76e06fa7`
- Views: synchronized frontal and left-side RGB recordings
- Matching sensor dataset: https://doi.org/10.5281/zenodo.6457662
- Individual processed-subject files: https://doi.org/10.5281/zenodo.7422031

## RunScope extraction plan

1. Verify archive size and MD5.
2. List subject folders and trial filenames before extraction.
3. Extract only `*_side_anonymized.avi` running trials at 6.3, 8.8, and 9.9 km/h.
4. Match each video to its processed subject/trial sensor file.
5. Use vertical GRF for initial contact and toe-off.
6. Use anterior-posterior GRF zero crossing for the braking-to-propulsion boundary.
7. Align video frames with the synchronized sensor timeline.
8. Generate participant-disjoint train/validation/test splits.

Healthy-participant trials are reference data, not direct Good/Bad form labels.
