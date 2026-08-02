# Converted Track2p Motion-Decoding Dataset

This directory contains a converted version of the Track2p developmental barrel-cortex dataset from Majnik et al. (2025), formatted for `train_decoder.py`.

## Source Dataset
- 6 mice (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`)
- 41 daily longitudinal imaging sessions
- Barrel cortex layer 2/3 calcium imaging at 30 Hz
- Synchronized videography-derived motion energy at 30 Hz
- Provided suite2p exports already contain neurons tracked across all days within each mouse

## Conversion Summary
- Neural signal: Suite2p-style neuropil-subtracted, baseline-corrected fluorescence reconstructed from `F.npy`, `Fneu.npy`, and `ops.npy`
- Behavioral signal: motion energy aligned to imaging frames using `tstamps.npy`, with missing camera frames interpolated
- Temporal binning: non-overlapping 10-frame bins (`333.33 ms`)
- Trialization: continuous recordings split into consecutive 2-minute trials
- Decoder input: elapsed time from session start
- Decoder output: motion energy normalized within session, discretized into 5 equal-percentile bins, and exported as 5 one-hot binary channels

## Output Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset used for validation
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed processing log, checks, and validation results

## Converted Dataset Structure
The pickle stores a dictionary with:
- `neural`: list of sessions, each containing trial arrays of shape `(n_neurons, 360)`
- `input`: list of sessions, each containing trial arrays of shape `(1, 360)` for `time_from_session_start_s`
- `output`: list of sessions, each containing trial arrays of shape `(5, 360)` for one-hot motion quintiles
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Key Statistics
- Sessions: 41
- Subjects: 6
- Trials: 545
- Mean neurons/session: 498.66
- Trial length: 360 bins
- Session trial counts: 10 trials for 20-minute sessions, 15 trials for 30-minute sessions

## Usage
Load the dataset:

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)
```

Validate or train:

```bash
python3 train_decoder.py converted_data.pkl --verify-only
python3 train_decoder.py converted_data.pkl --plot-samples
```

Regenerate the conversion:

```bash
python3 -u convert_data.py converted_data.pkl --full
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```
