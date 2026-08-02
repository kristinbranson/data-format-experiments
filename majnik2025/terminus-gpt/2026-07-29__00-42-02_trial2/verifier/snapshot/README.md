# Converted Track2p Development Dataset

## Summary
This repository contains a converted neuroscience dataset for decoder training.
The source data are longitudinal calcium-imaging recordings from mouse barrel cortex with matched tracked neurons across days, plus synchronized motion-energy measurements from videography.

## Converted file
- `converted_data.pkl`: decoder-ready dataset

## Data organization
The converted pickle contains:
- `neural`: list of sessions, each a list of 2-minute trials, each array shaped `(n_neurons, n_timepoints)`
- `input`: elapsed time within each 2-minute block, shaped `(1, n_timepoints)`
- `output`: motion-energy class labels, shaped `(1, n_timepoints)`

## Processing summary
- Source sessions were continuous recordings, not native trials.
- Sessions were segmented into consecutive 2-minute blocks.
- Neural traces were derived from Suite2p `F.npy` and `Fneu.npy` using neuropil subtraction and baseline normalization.
- Motion energy was aligned framewise to imaging length, binned in 10-frame windows, and discretized into 5 equal-percentile bins.
- Time bin size is 333.33 ms.

## Key statistics
- Subjects: 6
- Sessions: 41
- Trials: 545
- Total neurons across sessions: 20445
- Brain region: barrel cortex L2/3

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

Load in Python:
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```
