# Data, privacy, and release policy

This policy describes what may be committed to the public RunScope repository. It is intentionally more restrictive than some source-dataset licenses.

## Public repository contents

The public repository may contain:

- original RunScope source code;
- configuration files that contain no secrets or private paths;
- protocol, model-card, and product documentation;
- dataset citations, checksums, and participant-disjoint split metadata; and
- synthetic examples that cannot be linked to a real participant.

## Material excluded from public distribution

Do not commit or publish the following through this repository:

- raw or processed research-dataset files;
- participant or user videos, still frames, audio, or identifiable images;
- landmark tables, force-plate records, motion-capture files, or derived per-participant arrays;
- trained weights, checkpoints, serialized estimators, or embeddings unless a separate license review explicitly approves them;
- API keys, access tokens, cookies, passwords, private URLs, credentials, or `.env` files;
- local absolute paths, personal email addresses, or machine-specific configuration; or
- results whose source-media license does not permit publication.

The `.gitignore` file blocks common forms of these files, but contributors must still inspect every staged change before pushing.

## Dataset terms

RunScope does not grant rights to any external dataset. Obtain each dataset from its official provider and follow its current license, citation, ethics, privacy, and access conditions.

Some datasets may permit redistribution with attribution, while others may restrict it or provide no explicit license. Regardless, this repository's release policy is to distribute citations and reproducibility metadata only—not the dataset files themselves.

## Private recordings

Private user recordings must remain outside the repository. They may be processed locally only with the recorder's permission and must not be used to identify, diagnose, embarrass, or harm a person. Delete them when they are no longer needed under the applicable consent and retention policy.

## Models and external services

No RunScope model weights are distributed here. The source code does not require a hosted inference API or an API key. Installing packages or obtaining an upstream pose model may contact package or model-hosting services; those downloads remain subject to the provider's terms.

Parts of the video pipeline import Ultralytics. Use those parts under AGPL-3.0-compatible conditions or obtain an appropriate Ultralytics Enterprise license for proprietary or commercial deployment.

## Pre-push check

Before every public push, verify the complete Git history—not just the working tree—for secrets, media, datasets, and model artifacts. If sensitive material is ever committed, removing it in a later commit is not sufficient; rewrite the affected history and rotate any exposed credential.

