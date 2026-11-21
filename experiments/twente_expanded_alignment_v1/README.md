# Twente expanded alignment v1

This experiment starts only after the full-duration synchronization audit was
completed. It creates a new development-only training table from audited
10-second windows. Existing experiments and model artifacts are never
overwritten.

Key safeguards:

- synchronization offsets come only from video pose versus 3D markers;
- force events are used only after an offset is fixed;
- `Subj04`, `Subj11`, `test2`, and `test3` are forbidden;
- window gates and model candidates are fixed in `config/protocol.json` before
  expanded scores are generated;
- duplicate force contacts at window boundaries are removed by force-event ID;
- LOSO remains subject-disjoint and training weights are balanced by subject.
