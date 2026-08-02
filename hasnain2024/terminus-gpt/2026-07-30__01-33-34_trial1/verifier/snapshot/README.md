# Converted Randomized-Delay ALM Dataset

This repository contains a converted neuroscience dataset in decoder-compatible format.

## Main output
- `converted_data.pkl`: converted dataset aligned to **go cue onset**

## Included sessions
- 19 randomized-delay ALM sessions
- Subjects: JEB11, JEB12, JEB23, JEB24
- Session list follows the reference loading scripts from the paper code

## Data contents
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each trial shaped `(1, n_timepoints)` containing `time_from_go_cue`
- `output`: list of sessions/trials, each trial shaped `(6, n_timepoints)` containing:
  1. `lick_direction` (left/right)
  2. `behavioral_context` (WC/DR)
  3. `outcome` (incorrect/correct)
  4. `tongue_velocity` (low/high)
  5. `paw_velocity` (low/high)
  6. `motion_energy` (low/high)
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Processing summary
- Neural, behavior, and outputs aligned to **go cue onset**
- Time bin size: **75 ms**
- Early and no-response trials excluded
- Trials with all-zero neural activity excluded
- Mixed MATLAB file formats handled (`h5py` for v7.3/HDF5 and `scipy.io.loadmat` for older MAT files)
- Motion energy, tongue velocity, and paw velocity discretized per session using median thresholding

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl
```

Load in Python:
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```
