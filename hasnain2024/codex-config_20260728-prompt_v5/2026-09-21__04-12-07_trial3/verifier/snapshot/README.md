# Converted ALM Decoder Dataset

This repository now includes a converted neural-decoding dataset in:

- `/app/converted_data.pkl`

The source data come from the paper *Separating cognitive and motor processes in the behaving mouse* and were converted into a session/trial format for decoding neural activity aligned to **go cue onset**.

## Dataset Summary

- Sessions: 44
- Subjects: 14
- Brain regions: ALM only
- Kept trials: 13,762
- Kept units: 2,364
- Neural bin size: 5 ms
- Alignment window: `-2.5 s` to `+2.5 s` around go cue

Included decoder outputs:

- `lick_direction`: `left`, `right`, `none`
- `behavioral_context`: `WC`, `DR`
- `outcome`: `incorrect`, `correct`, `ignore`
- `tongue_velocity`: `low`, `high`, `not_visible`
- `paw_velocity`: `low`, `high`, `not_visible`
- `motion_energy`: `low`, `high`, `no_video`

The only decoder input is:

- `time_from_go_cue_s`

## File Format

The pickle stores a Python dictionary with the following top-level keys:

- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial shaped `(1, n_timepoints)`
- `output`: list of sessions, each a list of trials, each trial shaped `(6, n_timepoints)`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

All trials within the dataset use the same neural/input/output time dimension:

- `n_timepoints = 1000`

## Loading Example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["output_names"])
print(data["neural"][0][0].shape)   # (n_neurons, 1000)
print(data["input"][0][0].shape)    # (1, 1000)
print(data["output"][0][0].shape)   # (6, 1000)
```

## Conversion Rules

Key processing choices used in the conversion:

- ALM session/probe selection follows the reference MATLAB loader files.
- Neural spikes are aligned to `goCue`, binned at 5 ms, and smoothed with the reference causal Gaussian kernel.
- Trials with `early` licks or `stim.enable` are excluded.
- Units with excluded quality labels (`garbage`, `gabrga`, `noisy`, `real?`) are removed.
- Units with mean firing rate `<= 1 Hz` are removed.
- Sessions are required to retain at least 10 units.
- For sessions where behavior continued after neural recording ended, post-recording trials are excluded.

## Validation

The dataset was validated with:

- `python /app/train_decoder.py /app/converted_data.pkl --verify-only`
- `python /app/train_decoder.py /app/converted_data.pkl --plot-samples`

Full decoder validation on the converted dataset achieved the following validation balanced accuracies:

- `lick_direction`: `0.5917`
- `behavioral_context`: `0.8317`
- `outcome`: `0.5880`
- `tongue_velocity`: `0.6261`
- `paw_velocity`: `0.5764`
- `motion_energy`: `0.7007`

Detailed decisions, sanity checks, and validation results are documented in:

- `/app/CONVERSION_NOTES.md`
