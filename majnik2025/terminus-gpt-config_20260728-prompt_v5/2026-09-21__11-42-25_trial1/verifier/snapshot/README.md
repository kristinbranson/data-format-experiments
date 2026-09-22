# Converted Track2p Barrel Cortex Dataset

This repository contains a converted neuroscience dataset saved as `/app/converted_data.pkl` for decoder training.

## Dataset summary
- 6 mice
- 41 sessions
- 1081 trials
- 20445 included neurons across sessions
- Brain region: barrel cortex
- Neural signal: Suite2p fluorescence-derived dF/F after neuropil correction
- Decoder input: time elapsed from session start (time-varying)
- Decoder output: motion energy discretized into 5 equal-percentile bins per session (time-varying)

## Conversion choices
- Included only ROIs with `iscell[:,0] > 0.5`
- Neuropil correction: `F - 0.7 * Fneu`
- dF/F baseline: stable running low-percentile baseline informed by Suite2p metadata
- Temporal smoothing: non-overlapping 10-frame averaging for neural and motion traces
- Trialization: contiguous 60-second windows

## Loading the data
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Data format
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of `(n_neurons, n_timepoints)` trials
- `input`: list of sessions, each a list of `(1, n_timepoints)` time-elapsed trials
- `output`: list of sessions, each a list of `(1, n_timepoints)` categorical motion-bin trials
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Validation
- Format verification: `/app/verification_full_out.txt`
- Full decoder training log: `/app/train_decoder_full_out.txt`
- Detailed conversion notes: `/app/CONVERSION_NOTES.md`
