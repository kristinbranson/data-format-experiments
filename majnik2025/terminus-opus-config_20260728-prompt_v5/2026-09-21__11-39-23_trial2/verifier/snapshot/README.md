# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from mouse barrel cortex during postnatal development (P7-P14), collected using two-photon microscopy. Neural activity was recorded from the same neurons tracked across consecutive daily sessions using the Track2p algorithm.

**Reference**: Majnik et al. (2025). "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p." eLife 14:RP107540.

## Data Summary

| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice (jm031-jm046) |
| Sessions | 41 total (6-7 per subject) |
| Trials | 1,090 total (60s each) |
| Neurons per subject | 221-746 (mean: 499.7) |
| Time bin size | 333.33 ms (10 frames at 30 Hz) |
| Timepoints per trial | 180 |
| Brain region | Barrel cortex (layer 2/3) |

## Decoder Task

**Goal**: Decode motion energy from neural activity.

- **Input**: Time elapsed from session start (seconds)
- **Output**: Motion energy discretized into 5 equal-percentile bins per session

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 180)

# Access input (time) for session 0, trial 0
input_data = data['input'][0][0]  # shape: (1, 180)

# Access output (motion energy bin) for session 0, trial 0
output_data = data['output'][0][0]  # shape: (1, 180)

# Metadata
print(data['subjects'])       # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
print(data['brain_regions'])  # ['barrel_cortex']
print(data['input_names'])    # ['time_in_session']
print(data['output_names'])   # ['motion_energy']
print(data['output_values'])  # [['ME_bin_0', ..., 'ME_bin_4']]
```

## Data Format

```python
data = {
    'neural': list of 41 sessions, each containing list of trials (n_neurons, 180)
    'input': list of 41 sessions, each containing list of trials (1, 180)
    'output': list of 41 sessions, each containing list of trials (1, 180)
    'subjects': ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
    'subject_idx': array of shape (41,) mapping sessions to subjects
    'brain_regions': ['barrel_cortex']
    'brain_region_idx': list of 41 arrays mapping neurons to brain regions
    'input_names': ['time_in_session']
    'output_names': ['motion_energy']
    'output_values': [['ME_bin_0', 'ME_bin_1', 'ME_bin_2', 'ME_bin_3', 'ME_bin_4']]
    'metadata': dict with task description, time bin size, etc.
}
```

## Processing Pipeline

1. **Neuropil correction**: Fc = F - 0.7 × Fneu
2. **Baseline correction**: Suite2p maximin method (Gaussian smooth σ=10 frames, min/max pool window=1800 frames, subtract baseline)
3. **Temporal binning**: Average 10 consecutive frames (333.33 ms bins)
4. **Motion energy binning**: Same 10-frame averaging
5. **ME discretization**: 5 equal-percentile bins per session
6. **Trial splitting**: 60-second trials (180 bins each)

## Decoder Results

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| motion_energy | 0.6211 | 0.3048 | 0.2000 |

## Files

- `converted_data.pkl` - Full converted dataset (396 MB)
- `sample_data.pkl` - Sample dataset (2 sessions, 17 MB)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `processing_*.png` - Processing visualization plots
