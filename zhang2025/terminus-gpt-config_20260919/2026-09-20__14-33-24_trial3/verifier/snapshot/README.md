# IBL Brain-Wide Neural Decoder Dataset

## Overview

`converted_data.pkl` contains stimulus-aligned electrophysiology and behavior from the International Brain Laboratory Brain-Wide Map release, reformatted for `/app/train_decoder.py`.

The source cohort contains 459 sessions, 139 mice, 699 Neuropixels insertions, and 621,733 Kilosort clusters. Sessions lacking a usable whisker motion-energy stream or jointly valid trial windows were excluded. The converted dataset contains:

- **444 sessions** from **136 subjects**
- **188,925 trials**
- **599,865 neurons summed across sessions**
- **100 time bins per trial**, each 20 ms
- **281 Beryl atlas region labels**
- Alignment window **-0.5 to +1.5 s around visual stimulus onset**

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(len(data['neural']))       # 444 sessions
print(data['neural'][0][0].shape)  # (n_neurons, 100)
print(data['input'][0][0].shape)   # (2, 100)
print(data['output'][0][0].shape)  # (4, 100)
```

Neural matrices contain integer spike counts. They are stored as `uint8` losslessly (maximum observed count: 68 spikes per 20 ms bin); the supplied decoder converts each session to float32 during training.

## Dictionary Format

- `neural[session][trial]`: neuron × time spike-count matrix.
- `input[session][trial]`: 2 × time float32 matrix:
  1. time since stimulus onset, at behavior bin ends from -0.48 to +1.50 s;
  2. original trial number within the current probability block, broadcast over time.
- `output[session][trial]`: 4 × time categorical matrix:
  1. choice: left=0, right=1;
  2. probability of left: 0.2=0, 0.5=1, 0.8=2;
  3. wheel speed: low/medium/high=0/1/2;
  4. whisker motion energy: low/medium/high=0/1/2.
- `subjects` and `subject_idx`: subject names and per-session indices.
- `brain_regions` and `brain_region_idx`: global Beryl acronyms and per-neuron indices.
- `input_names`, `output_names`, `output_values`: variable names and category labels.
- `metadata`: alignment, binning, global tertile thresholds, filters, and per-session information.

## Processing Summary

1. The authoritative 459-session cohort is read from `code/code_zhang2025/data/bwm_release.csv`.
2. All listed probes from each session are merged. All Kilosort clusters are retained, matching the methods-paper executable pipeline (`qc=None`).
3. Trials require finite core events, nonzero choice, reaction time from 0.08 to 2.0 s, and feedback-to-go-cue duration no greater than 10 s.
4. Spikes are counted in half-open 20 ms bins from stimulus onset -0.5 to +1.5 s.
5. Wheel position is interpolated to 1 kHz and low-pass filtered using Brainbox; wheel speed is absolute velocity.
6. Left-camera ROI motion energy is preferred for whisker motion energy, with right-camera fallback.
7. Behavior is linearly interpolated at neural-bin end times. Trials require complete finite wheel and camera windows.
8. Wheel and whisker values are discretized using global empirical tertiles. Thresholds are saved in metadata.
9. Cluster channel atlas IDs are mapped to Beryl acronyms.

See `CONVERSION_NOTES.md` for complete rationale, discrepancies, raw-file checks, and review history. The reproducible converter is `convert_data.py`.

## Key Output Distributions

- Choice: 92,973 left; 95,952 right.
- Prior: 78,864 at 0.2; 26,561 at 0.5; 83,500 at 0.8.
- Wheel classes: 6,297,500 timepoints in each class.
- Whisker classes: 6,297,499 / 6,297,501 / 6,297,500 timepoints.

## Validation

Format validation:

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Full decoder training completed successfully on CUDA for 200 epochs. Validation balanced accuracies were:

| Output | Balanced accuracy | Uniform chance |
|---|---:|---:|
| Choice | 0.5634 | 0.5000 |
| Prior probability of left | 0.5859 | 0.3333 |
| Wheel speed | 0.5922 | 0.3333 |
| Whisker motion energy | 0.7170 | 0.3333 |

All outputs exceed chance. Training and validation gaps were below 0.008 for every output.

## Reproducing Conversion

Sample conversion:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Full conversion:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

The full conversion took approximately 5.3 minutes in the provided environment. Processing plots are named `processing_<eid>.png`.
