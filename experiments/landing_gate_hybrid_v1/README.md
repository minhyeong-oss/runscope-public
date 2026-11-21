# landing_gate_hybrid_v1

Fixed-parameter integration of the LAAS-trained `landing_aux_v2` onset gate with
the historical frozen Twente v3 four-stage baseline.

- The auxiliary artifact and its 0.92 threshold are loaded without refitting.
- The test videos are used only for inference-only MediaPipe pose extraction.
- A confident gate replaces only the 30 ms landing window; all other stage
  outputs remain those of the frozen v3 baseline.
- `Subj04` and `Subj11` are never used for training, threshold selection, or
  parameter selection in this experiment.

Because those subjects were already evaluated by the historical v3 experiment,
this comparison is not described as a newly untouched benchmark.
