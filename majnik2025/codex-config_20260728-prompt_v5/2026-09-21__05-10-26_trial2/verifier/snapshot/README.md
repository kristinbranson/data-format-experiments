# Track2p Motion-Decoding Conversion

This directory contains a converted version of the Track2p longitudinal barrel-cortex dataset for decoder benchmarking.

## Dataset summary
- Source paper: *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p*
- Subjects: 6 mice
- Sessions: 41
- Neural representation: neuropil-subtracted, Suite2p-style baseline-corrected fluorescence reconstructed from `F.npy`, `Fneu.npy`, and `ops.npy`
- Behavioral target: video-derived global motion energy
- Time binning: non-overlapping 10-frame bins at 30 Hz (`333.33 ms`)
- Trials: non-overlapping 60-second windows
- Converted trial counts:
  - 20 trials for 20-minute sessions (`jm031`, `jm032`)
  - 30 trials for 30-minute sessions (`jm038`, `jm039`, `jm040`, `jm046`)
- Total converted trials: 1090

## Files
- `/app/convert_data.py`: conversion script
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: 2-session sample conversion
- `/app/CONVERSION_NOTES.md`: detailed step-by-step record of decisions, checks, and validation

## Conversion decisions
- The released `suite2p` folders are already Track2p matched-cell outputs, so neuron identities are aligned across days within each mouse.
- Motion-energy streams with dropped camera frames are interpolated onto the imaging frame clock using `tstamps.npy`.
- Motion energy is discretized per session into 5 equal-frequency categories to satisfy the categorical decoder-output requirement.
- Decoder input is absolute time from the beginning of the session in seconds.

## Loading the converted data
```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["neural"][0][0].shape)   # (n_neurons, 180)
print(data["input"][0][0].shape)    # (1, 180)
print(data["output"][0][0].shape)   # (1, 180)
```

## Data format
`converted_data.pkl` is a dictionary with these keys:
- `neural`: list of sessions, each a list of trials, each trial `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each trial `(1, n_timepoints)`
- `output`: list of sessions, each a list of trials, each trial `(1, n_timepoints)` with integer class labels `0..4`
- `subjects`: unique mouse ids
- `subject_idx`: per-session subject index
- `brain_regions`: brain-region names
- `brain_region_idx`: per-session neuron-region index arrays
- `input_names`: `["time_from_session_start_s"]`
- `output_names`: `["motion_energy_quintile"]`
- `output_values`: names of the 5 motion-energy classes
- `metadata`: task description, timing fields, preprocessing description, and per-session summaries

## Validation
- Format verification:
  - `python /app/train_decoder.py /app/converted_data.pkl --verify-only`
- Full decoder training:
  - `python /app/train_decoder.py /app/converted_data.pkl --plot-samples`

## Key results
- Full-format verification passed with no errors or warnings.
- Full decoder training completed successfully.
- Full validation balanced accuracy for `motion_energy_quintile`: `0.2569`
- Chance level for 5 classes: `0.2000`
