# Converted Hippocampal Reward-Relative Dataset

This directory contains a decoder-ready conversion of the NWB dataset accompanying *A flexible hippocampal population code for experience relative to reward*.

## Dataset Summary
- Subjects: 11 mice
- Sessions: 152
- Trials: 12,216
- Curated neurons: 138,678
- Brain region: `CA1`
- Time base: behavior/imaging-aligned frame times at approximately `15.5 Hz` (`64.483627 ms` per bin)
- Temporal alignment: start of trial

## Files
- `/app/convert_data.py`: conversion script
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: 2-session sample conversion
- `/app/CONVERSION_NOTES.md`: full processing and validation record
- `/app/train_decoder_full_out.txt`: full decoder-training log

## Conversion Choices
- Neural signal: deconvolved calcium activity from `processing/ophys/Deconvolved`
- Cell filter: curated cells only, `iscell[:,0] == 1`
- Trial boundaries: `trial_start` pulse to `teleport` pulse
- Reward-zone identity: reconstructed from session scene and switch schedule
- Reward outcome: whether any reward event occurred during the trial
- Lick output: binary lick trace with artifact trials zeroed using the reference-style heuristic

## Stored Format
The pickle contains a Python dictionary with these top-level keys:

- `neural`: list of sessions, each a list of trial arrays shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trial arrays shaped `(4, n_timepoints)`
- `output`: list of sessions, each a list of trial arrays shaped `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Input Channels
In order:
1. `time_from_trial_start_s`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

## Output Channels
In order:
1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

All outputs are categorical. Trial-wise outputs are repeated across frames within each trial so every trial uses the same `(n_output, n_timepoints)` shape.

## Loading Example
```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))          # sessions
print(data['input_names'])          # decoder inputs
print(data['output_names'])         # decoder outputs
print(data['neural'][0][0].shape)   # first session, first trial
```

## Validation Snapshot
- `train_decoder.py --verify-only` passed with no errors or warnings
- Full decoder validation balanced accuracies:
  - `distance_to_reward_zone`: `0.4081`
  - `absolute_position`: `0.5084`
  - `speed`: `0.3479`
  - `lick`: `0.6059`
  - `reward_zone_location`: `0.8006`
  - `reward_outcome`: `0.5069`

See `/app/CONVERSION_NOTES.md` for the detailed rationale, sanity checks, raw-data comparisons, and review iterations.
