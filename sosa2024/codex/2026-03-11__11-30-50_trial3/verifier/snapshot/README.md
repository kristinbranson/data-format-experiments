# Hippocampal Reward-Relative Decoder Dataset

This directory contains a decoder-ready conversion of the dataset from:

- `A flexible hippocampal population code for experience relative to reward`

The converted file is:

- `converted_data.pkl`

It is designed to be consumed by:

- `python3 train_decoder.py converted_data.pkl`

## Dataset Summary

- Source data: processed NWB session files in `data/`
- Subjects: 11 mice
- Sessions: 152
- Brain region: CA1
- Total neurons: 138,678 curated cells
- Total converted trials: 12,147
- Alignment: all trials are aligned to trial start
- Time base: original frame-aligned behavioral/imaging samples at approximately `0.0645 s` per frame

## Included Variables

Inputs (`input_names`):

1. `time_from_trial_start_sec`
2. `environment`
3. `trial_number`
4. `previous_trial_rewarded`

Outputs (`output_names`):

1. `distance_to_reward_zone`
2. `absolute_position_bin`
3. `speed_bin`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Data Format

`converted_data.pkl` stores a Python dictionary with:

- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial shaped `(4, n_timepoints)`
- `output`: list of sessions, each a list of trials, each trial shaped `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Loading Example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

session0_trial0_neural = data["neural"][0][0]
session0_trial0_input = data["input"][0][0]
session0_trial0_output = data["output"][0][0]
```

## Conversion Notes

The full processing rationale, reference-code mapping, validation checks, and decoder results are documented in:

- `CONVERSION_NOTES.md`

## Validation

Files produced during validation:

- `conversion_sample_out.txt`
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `conversion_full_out.txt`
- `verification_full_out.txt`
- `train_decoder_full_out.txt`

The converted dataset passes `train_decoder.py --verify-only`, and full decoder training completed successfully.
