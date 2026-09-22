# Converted Track2p Barrel Cortex Dataset

## Summary
This repository now includes a converted dataset at `/app/converted_data.pkl` for decoder training.

- **Neural data**: iscell-filtered barrel cortex calcium-imaging traces derived from Suite2p outputs
- **Input**: session elapsed time in seconds
- **Output**: motion energy discretized into 5 per-session quantile bins
- **Trials**: continuous sessions split into consecutive non-overlapping 60 s pseudo-trials
- **Time bin**: 10 imaging frames per bin (~333.3 ms at 30 Hz)

## Files
- `/app/convert_data.py`: conversion script
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: 2-session sample dataset
- `/app/CONVERSION_NOTES.md`: detailed processing notes and validation log
- `/app/train_decoder_full_out.txt`: full decoder training log

## Data format
The pickle contains a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each array shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each array shaped `(1, n_timepoints)` containing session time in seconds
- `output`: list of sessions/trials, each array shaped `(1, n_timepoints)` containing integer motion-energy bin labels 0-4
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Loading example
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(len(data['neural']), 'sessions')
print(data['neural'][0][0].shape, 'first trial neural shape')
```

## Conversion choices
- Used Suite2p `iscell` to filter ROIs.
- Used neuropil-corrected fluorescence and a dF/F-like normalization to approximate the paper's decoding signal.
- Averaged neural and motion signals in non-overlapping 10-frame bins to match the methods description where applicable.
- Used per-session quantile binning to create the categorical decoder target required by the task.

## Key statistics
- Sessions: 41
- Subjects: 6
- Total trials: 1081
- Total neurons across sessions: 20445
- Brain region: barrel cortex
