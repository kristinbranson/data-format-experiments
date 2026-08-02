# Allen Visual Behavior Conversion

This workspace converts Allen Visual Behavior 2P active-behavior data into the decoder format expected by `train_decoder.py`.

## Outputs

- `converted_data.pkl`: full converted dataset from all locally available active sessions.
- `sample_data.pkl`: smaller dataset generated with `--max-sessions 12` for quick checks.
- `convert_data.py`: reproducible conversion script.
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and final results.
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

## Processing Summary

- Source data: local Allen Visual Behavior ophys release at `data/visual-behavior-ophys-1.1.0`.
- Sessions included: `active_behavior` experiments with an NWB file present locally.
- One decoder session equals one `ophys_experiment_id`, which matches one imaging plane and one ophys timestamp stream.
- Trials included: `go` and `catch`.
- Trials excluded: `aborted` and `auto_rewarded`.
- Neural signal: AllenSDK inferred calcium `events`, with invalid ROIs excluded by `BehaviorOphysExperiment.from_nwb_path(..., exclude_invalid_rois=True)`.
- Time axis: native `750 ms` image-presentation intervals inside each AllenSDK trial.
- Outputs:
  - `image_identity`: 16 image labels plus `gray` for omitted intervals
  - `image_change`: binary flag on change intervals only
  - `running_speed_bin`: global quintiles
  - `pupil_diameter_bin`: global quintiles
  - `trial_outcome`: hit / miss / false_alarm / correct_reject

## Final Dataset Sizes

- `converted_data.pkl`: 199 sessions, 48,655 trials, 567,895 image intervals, 38 mice.
- `sample_data.pkl`: 11 sessions, 1,708 trials, 19,270 image intervals, 4 mice.

## Re-run

```bash
python convert_data.py > conversion_full_out.txt 2>&1
python convert_data.py --max-sessions 12 --output sample_data.pkl --sample-output sample_data.pkl --sample-session-count 12 > conversion_sample_out.txt 2>&1
python train_decoder.py converted_data.pkl --verify-only --cpu > verification_full_out.txt 2>&1
python train_decoder.py sample_data.pkl --verify-only --cpu > verification_sample_out.txt 2>&1
python train_decoder.py converted_data.pkl --cpu > train_decoder_full_out.txt 2>&1
python train_decoder.py sample_data.pkl --cpu > train_decoder_sample_out.txt 2>&1
```
