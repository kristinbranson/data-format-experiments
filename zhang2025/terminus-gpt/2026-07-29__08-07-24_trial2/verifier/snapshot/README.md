# Converted IBL Decoder Dataset

## Summary
This repository contains a converted neuroscience dataset saved as `converted_data.pkl` for decoder training with `train_decoder.py`.

## Contents
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Data format
The pickle stores a dictionary with:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: decoder inputs per trial
- `output`: categorical decoder outputs per trial
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Decoder task represented
Aligned to stimulus onset.

Inputs:
1. time since stimulus onset
2. trial number within block

Outputs:
1. choice (`left=0`, `right=1`)
2. prior probability of left (`0.2->0`, `0.5->1`, `0.8->2`)
3. wheel speed (`low/medium/high`)
4. whisker motion energy (`low/medium/high`)

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

## Notes
See `CONVERSION_NOTES.md` for processing details, sanity checks, and validation results.
