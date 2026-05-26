# Progress Log - STFT Consistency Fix

## Summary
STFT consistency integration was debugged and stabilized for the chunk-based pipeline.

## What Was Implemented
- Added STFT consistency projection helpers in `audio_pipeline.py`:
  - `project_to_stft_consistency_torch`
  - `project_to_stft_consistency`
  - blend helpers for raw/projected spectra
- Integrated consistency path in training/validation (`training_loop.py`):
  - `pred_spec_raw` and `pred_spec_cons` flow
  - consistency penalty in total loss
  - consistency metrics added to logs/checkpoints
- Extended training config (`train.py`) with consistency knobs:
  - `use_stft_consistency`
  - `w_consistency`
  - `w_raw_aux`
  - `consistency_apply_in_val`
  - `consistency_projection_mode`
  - `consistency_blend`
- Updated inference/eval scripts:
  - `eval_one.py`, `eval.py`, `eval_worst_pesq.py`
  - consistency diagnostics (`consistency_delta`)
- Updated artifact CSV fields in `run_artifacts.py`:
  - `train_consistency_mse`
  - `val_consistency_mse`

## Root Cause Found
Initial consistency projection caused major distortion because the STFT/ISTFT pair in projection was not perfectly matched in chunk domain (window effect mismatch).

## Fix Applied
Projection math was corrected with window compensation in the consistency operator (without changing normal synthesis path).

## Sanity Check Results (Identity Mask)
Input: identity path (`mask=1`)
- `consistency=False`:
  - RMSE: `~2.05e-8`
  - SNR: `~132.42 dB`
- `consistency=True` (after fix):
  - RMSE: `~4.97e-6`
  - SNR: `~84.74 dB`

Interpretation:
- Consistency path no longer causes catastrophic degradation.
- Reconstruction is now stable enough for training usage.

## Saved Sanity Audio
Folder: `model_development/eval_outputs/sanity_consistency_on`
- `reference_input.wav`
- `recon_consistency_on.wav`

## Current Status
- Pipeline is considered training-ready.
- Recommended conservative start:
  - `w_consistency = 0.02 ~ 0.05`
  - increase gradually based on PESQ/STOI/SI-SDR plus artifact proxies

## Next Suggested Step
Run a short A/B training/eval comparison:
1. Baseline (consistency off)
2. Consistency on (low `w_consistency`)

Then compare objective metrics and artifact proxies.
