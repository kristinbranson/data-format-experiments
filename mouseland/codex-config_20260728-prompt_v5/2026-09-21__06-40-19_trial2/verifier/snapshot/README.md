# Converted Dataset README

## Dataset
This repository contains a decoder-formatted conversion of the imaging dataset associated with the paper **"Unsupervised pretraining in biological neural networks"**.

The converted dataset is saved at:

- `/app/converted_data.pkl`

The conversion script is:

- `/app/convert_data.py`

Detailed conversion decisions, validation checks, and decoder results are documented in:

- `/app/CONVERSION_NOTES.md`

## What Was Converted
- Canonical recordings converted: `89`
- Subjects: `19`
- Total converted trials: `38,110`
- Total neural rows: `4,691,034`
- Neural signal: released deconvolved `spks` traces from the source dataset
- Time bin size: `315.457 ms` per retained imaging frame (`3.17 Hz`)

Each converted trial is aligned to **corridor entry / trial start** and retains only **corridor-running** imaging frames:

- `ft_trInd == trial`
- `ft_CorrSpc == True`
- `ft_move > 0`

## Decoder Inputs
The converted `input` arrays contain four time-varying channels in this order:

1. `time_to_sound_cue_s`
2. `day_of_training`
3. `time_since_trial_start_s`
4. `reward_availability`

## Decoder Outputs
The converted `output` arrays contain four categorical channels in this order:

1. `visual_stimulus_category`
2. `licking`
3. `position_bin`
4. `running_speed_bin`

Output categories:

- `visual_stimulus_category`: raw stimulus labels from the source files
- `licking`: `no_lick`, `lick`
- `position_bin`: `0-1m`, `1-2m`, `2-3m`, `3-4m`
- `running_speed_bin`: `q1`, `q2`, `q3`, `q4`

## File Format
`/app/converted_data.pkl` stores a Python dictionary with the following top-level keys:

- `neural`
- `input`
- `output`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

Per-trial array shapes:

- `neural[session][trial]`: `(n_neurons, n_timepoints)`
- `input[session][trial]`: `(4, n_timepoints)`
- `output[session][trial]`: `(4, n_timepoints)`

## Loading Example
```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))          # sessions
print(data["input_names"])          # decoder inputs
print(data["output_names"])         # decoder outputs
print(data["neural"][0][0].shape)   # first trial neural matrix
```

## Validation Summary
- `python /app/train_decoder.py /app/converted_data.pkl --verify-only` passed with no errors or warnings.
- Full decoder training completed successfully.
- Validation balanced accuracies:
  - `visual_stimulus_category`: `0.5454`
  - `licking`: `0.8800`
  - `position_bin`: `0.3243`
  - `running_speed_bin`: `0.3337`

## Notes
- The conversion preserves the source-session count reported in the paper: `89 recordings in 19 mice`.
- The paper does not report directly comparable decoder accuracies for these exact targets, so decoder validation here is based on format checks, raw-data sanity checks, and above-chance prediction performance.
