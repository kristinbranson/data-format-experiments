# Converted Hippocampal Reward-Switch Dataset

This repository contains a converted decoder-ready dataset derived from the NWB files for the study **"A flexible hippocampal population code for experience relative to reward"**.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `train_decoder.py`: decoder validation/training script

## Converted format
The pickle stores a Python dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: decoder inputs per trial
- `output`: decoder targets per trial
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Inputs
1. `time_from_trial_start`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

## Outputs
1. `distance_to_reward_zone`
2. `absolute_position_bin`
3. `speed_bin`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Source processing choices
- Neural signal: deconvolved ophys activity
- Trial alignment: `trial_start`
- Trial boundaries: successive `trial_start` events
- Baseline samples with invalid task codes are excluded
- Reward outcome is derived from `Reward` event timestamps within each trial
- Reward-zone identity is inferred from physical reward-zone position along the corridor

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

Create data from source NWB files:
```bash
python3 -u convert_data.py converted_data.pkl --full
python3 -u convert_data.py sample_data.pkl --sample
```

## Notes
See `CONVERSION_NOTES.md` for detailed checks, assumptions, and validation results.
