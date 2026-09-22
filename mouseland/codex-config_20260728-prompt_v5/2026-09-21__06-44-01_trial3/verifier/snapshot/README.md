# Converted Zhong et al. Dataset

This directory contains a decoder-ready conversion of the imaging dataset from *Unsupervised pretraining in biological neural networks*.

## Dataset summary
- Source recordings: 89 unique imaging sessions from 19 mice.
- Canonical trials retained: 38,110 / 38,110 raw trials from the deduplicated session set.
- Exported neurons: 22,163 total selected neurons, mean 249.02 neurons/session.
- Brain regions: `V1`, `mHV`, `lHV`, `aHV`.
- Alignment: trial start / corridor entry.
- Time base: native imaging frames, median frame interval `314.694 ms`.

## Files
- `converted_data.pkl`: full converted dataset.
- `sample_data.pkl`: 2-session sample conversion used for validation.
- `convert_data.py`: conversion script.
- `train_decoder.py`: validation / decoder training script provided with the task.
- `CONVERSION_NOTES.md`: detailed methods, checks, and decisions.

## Conversion choices
- Sessions are deduplicated to the 89 unique `mouse/date/block` recordings described in the paper.
- Behavior arrays are trimmed to neural frame count to match the reference code.
- Trials include corridor frames only, excluding the 2 m gray inter-trial region.
- Neural activity uses the provided deconvolved `spks` arrays directly.
- To keep the export tractable while staying close to the paper, the dataset keeps up to 64 corridor-responsive, stimulus-selective neurons per grouped visual region using the paper’s `d′`-style logic.
- `running_speed_bin` is assigned by global rank quartiles across all retained corridor frames so the four bins have equal occupancy up to integer rounding.

## Decoder variables
- Inputs: `time_to_sound_cue_s`, `day_of_training`, `time_since_trial_start_s`, `reward_available`
- Outputs: `visual_stimulus_category`, `licking`, `position_bin`, `running_speed_bin`

## Loading
```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))            # sessions
print(data["input_names"])            # decoder inputs
print(data["output_names"])           # decoder outputs
print(data["neural"][0][0].shape)     # (n_neurons, n_timepoints) for one trial
```

## Format
`converted_data.pkl` stores a dictionary with:
- `neural`: list of sessions, each a list of `(n_neurons, n_timepoints)` trial arrays
- `input`: list of sessions, each a list of `(4, n_timepoints)` trial arrays
- `output`: list of sessions, each a list of `(4, n_timepoints)` trial arrays
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Validation summary
- `python3 /app/train_decoder.py /app/converted_data.pkl --verify-only` passes with no errors or warnings.
- Full decoder training completes successfully.
- Full validation balanced accuracies:
  - `visual_stimulus_category`: `0.9217`
  - `licking`: `0.9100`
  - `position_bin`: `0.7292`
  - `running_speed_bin`: `0.6595`
