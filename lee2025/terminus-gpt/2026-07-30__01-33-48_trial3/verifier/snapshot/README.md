# Converted CA1 Geometry Dataset

This repository contains a converted version of the Lee et al. (2025) CA1 geometry dataset for decoder training.

## Dataset summary
- Subjects: 7 mice
- Sessions: 207 recording days/sessions
- Brain region: CA1
- Native sampling rate: 30 Hz
- Native session duration: 40 min
- Trials in converted data: contiguous 1-minute chunks within each session

## Converted format
The converted dataset is stored in `converted_data.pkl` as a Python dictionary with fields:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial a 9-d blocked-geometry vector
- `output`: list of sessions, each a list of trials, each trial shaped `(1, n_timepoints)` with 3x3 position-bin labels 0-8
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Conversion choices
- Neural activity uses the native binarized rising-phase calcium event traces from the source dataset.
- Position is discretized into 3x3 spatial bins over the 75x75 cm arena.
- Decoder input is the blocked partition configuration encoded as a 9-element binary vector.
- Sessions are split into contiguous 1-minute trials while preserving native frame alignment.
- Unregistered/non-finite neurons are removed on a per-session basis.

## Files
- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples --cpu
```
