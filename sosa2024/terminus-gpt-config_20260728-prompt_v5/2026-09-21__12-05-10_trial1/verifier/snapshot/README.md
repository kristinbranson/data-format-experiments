# Converted Hippocampal Reward-Relative Dataset

This repository contains a converted decoder-ready dataset saved in `converted_data.pkl`.

## Source
- NWB files in `/app/data`
- Reference paper: `paper.pdf`
- Reference methods: `methods.txt`
- Reference code: `/app/code`

## Converted format
The pickle contains a dictionary with keys:
- `neural`: list of sessions, each a list of trials with arrays shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials with arrays shaped `(4, n_timepoints)`
- `output`: list of sessions/trials with arrays shaped `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Decoder inputs
1. `time_from_trial_start_s`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

## Decoder outputs
1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Processing summary
- Neural signal: `ophys/Deconvolved/plane0`
- Trials reconstructed from `trial_start` and `trial number`
- Alignment event: `trial_start`
- Sparse reward events aligned to imaging timestamps
- Reward-zone distance computed from inferred per-trial reward-zone intervals

## Files
- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `CONVERSION_NOTES.md`: detailed audit trail
