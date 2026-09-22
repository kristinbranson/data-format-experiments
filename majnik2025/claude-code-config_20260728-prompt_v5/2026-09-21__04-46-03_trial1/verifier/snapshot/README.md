# Track2p Barrel Cortex Dataset - Decoder Format

## Dataset Description

Longitudinal two-photon calcium imaging data from mouse barrel cortex during postnatal development (P7-P14), converted from the Track2p paper (Majnik et al., 2025, eLife).

**Source**: Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel JC, Cossart R. "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p." eLife 14:RP107540.

## Quick Start

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data: list of sessions, each a list of trials
neural_session0_trial0 = data['neural'][0][0]  # shape: (n_neurons, 180)

# Access outputs
output_session0_trial0 = data['output'][0][0]  # shape: (1, 180), values 0-4

# Subject info
print(data['subjects'])  # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
```

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice |
| Sessions | 41 total (6-7 per mouse) |
| Trials | 1090 total (20-30 per session, 60s each) |
| Neurons per mouse | 221-746 (mean: 499.7) |
| Brain region | Barrel cortex |
| Time bin | 333.3 ms (10 frames at 30 Hz) |
| Bins per trial | 180 (= 60 seconds) |

## Data Format

- **neural**: dF/F traces (baseline-corrected fluorescence), binned in 10-frame windows
- **input**: Time elapsed from session start (seconds)
- **output**: Motion energy discretized into 5 equal-percentile bins per session (0-4)
- **output_values**: `['ME_bin0', 'ME_bin1', 'ME_bin2', 'ME_bin3', 'ME_bin4']`

## Processing Pipeline

1. Neuropil correction: `F_corrected = F - 0.7 * Fneu`
2. Baseline: Suite2p maximin method (Gaussian smooth -> running min -> running max)
3. dF/F: `(F_corrected - baseline) / max(baseline, 10)`
4. Temporal binning: Average 10 consecutive frames (both neural and behavioral)
5. Motion energy alignment: Interpolate missing camera frames to neural frame times
6. Discretization: 5 equal-percentile bins per session
7. Trial splitting: 60-second non-overlapping windows

## Files

- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample (2 sessions) for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion notes and validation

## Running the Decoder

```bash
python train_decoder.py converted_data.pkl --plot-samples
```

## Decoder Results

| Output | Train Balanced Acc | Val Balanced Acc | Chance |
|--------|-------------------|------------------|--------|
| motion_energy | 0.5018 | 0.3165 | 0.2000 |
