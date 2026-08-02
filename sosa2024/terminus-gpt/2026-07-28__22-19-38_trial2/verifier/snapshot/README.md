# Converted Hippocampal Decoder Dataset

## Overview
This repository contains a converted version of the dataset from **"A flexible hippocampal population code for experience relative to reward"** formatted for `train_decoder.py`.

## Main files
- `convert_data.py`: conversion script
- `converted_data_float32_backup.pkl`: full converted dataset used for validation/training in this session
- `sample_data.pkl`: sample converted dataset
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Data format
The converted dataset is a Python pickle containing a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials with decoder inputs
- `output`: list of sessions/trials with categorical decoder outputs
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Decoder inputs
1. time from trial start
2. environment type
3. trial number
4. previous trial outcome

## Decoder outputs
1. distance to reward zone (7 bins)
2. absolute position (5 bins)
3. speed (5 bins)
4. lick (binary)
5. reward zone location (A/B/C)
6. reward outcome (binary)

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data_float32_backup.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data_float32_backup.pkl --plot-samples --cpu
```

Convert data again:
```bash
python3 -u convert_data.py output.pkl --full
python3 -u convert_data.py sample_output.pkl --sample --show-processing
```

## Notes
See `CONVERSION_NOTES.md` for processing decisions, sanity checks, and decoder results.
