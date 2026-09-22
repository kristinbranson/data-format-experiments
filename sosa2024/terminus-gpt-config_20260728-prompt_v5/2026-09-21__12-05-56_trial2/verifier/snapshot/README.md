# Converted Hippocampal Decoder Dataset

This repository contains a converted dataset saved as `/app/converted_data.pkl` for decoder training.

## Source data
- NWB files in `/app/data`
- Reference paper: `paper.pdf`
- Reference methods: `methods.txt`
- Reference code: `/app/code`

## Conversion summary
- Sessions: 152
- Subjects: 11
- Brain region: CA1
- Neural source: NWB `ophys/Deconvolved` ROI response series
- Neuron curation: suite2p-style `iscell` filter using the binary first column
- Trial alignment: `trial_start`
- Time binning: native imaging frame bins (~64.5 ms, ~15.5 Hz)

## Data format
The pickle contains a dictionary with keys:
- `neural`: list of sessions, each a list of trials with arrays shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials with arrays shaped `(4, n_timepoints)`
- `output`: list of sessions, each a list of trials with arrays shaped `(6, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Decoder inputs
1. `time_from_trial_start_sec`
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

## Notes
- Trials are reconstructed from behavioral time series because canonical NWB trial tables are empty.
- Reward-zone location is inferred from positions where `reward_zone > 0`, which cluster near three canonical locations on the 450 cm track.
- See `CONVERSION_NOTES.md` for detailed decisions, checks, and validation results.

## Validation
- Sample and full verification completed with `train_decoder.py`
- Full decoder training completed successfully with all outputs above chance
