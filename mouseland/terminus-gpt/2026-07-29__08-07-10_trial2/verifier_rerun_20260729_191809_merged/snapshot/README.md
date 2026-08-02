# Converted Neural Decoder Dataset

This repository contains a converted dataset for training a neural decoder on the virtual corridor task from the paper *Unsupervised pretraining in biological neural networks*.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log

## Data format
The pickle contains a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: aligned decoder inputs per trial
- `output`: aligned decoder outputs per trial
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Inputs
1. `time_to_sound_cue`
2. `day_of_training`
3. `time_since_trial_start`
4. `reward_availability`

## Outputs
1. `visual_stimulus_category`
2. `licking`
3. `position_bin`
4. `running_speed_bin`

## Notes
- Trials are represented using a compact 60-bin representation.
- Neural data are deconvolved calcium-event/activity traces concatenated across planes per session.
- See `CONVERSION_NOTES.md` for full methodological details and validation.
