# Track2p Neural Decoder Dataset

## Dataset Description

Longitudinal 2-photon calcium imaging from layer 2/3 barrel cortex of developing mice (P7-P14), with simultaneous motion energy from videography. Data from Majnik et al. 2025, eLife ("Track2p").

**Decoder task**: Predict discretized motion energy (5 quintile bins) from neural activity, with time elapsed as input.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice |
| Sessions | 41 total (6-7 per mouse) |
| Neurons per mouse | 221-746 (mean 499.7) |
| Brain region | Layer 2/3 barrel cortex |
| Time bin size | 333.33 ms (10 frames at 30 Hz) |
| Trial duration | 2 minutes (360 time bins) |
| Trials per session | 10 (20-min sessions) or 15 (30-min sessions) |
| Total trials | 545 |

## How to Load

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 360)

# Access motion energy labels
output = data['output'][0][0]  # shape: (1, 360), values 0-4

# Access time input
time_input = data['input'][0][0]  # shape: (1, 360), seconds from recording start
```

## Data Format

```python
data = {
    'neural': [sessions][trials] -> (n_neurons, 360) float32  # baseline-corrected dF/F, binned
    'input': [sessions][trials] -> (1, 360) float32            # time elapsed in seconds
    'output': [sessions][trials] -> (1, 360) int64             # motion energy bin (0-4)
    'subjects': ['Mouse_A', ..., 'Mouse_F']
    'subject_idx': array of shape (41,)
    'brain_regions': ['barrel_cortex']
    'brain_region_idx': [sessions] -> array of shape (n_neurons,)
    'input_names': ['time_elapsed_s']
    'output_names': ['motion_energy']
    'output_values': [['quintile_1', ..., 'quintile_5']]
    'metadata': {
        'time_bin_size': 333.33,    # ms
        'frame_rate': 30.0,         # Hz
        'bin_size_frames': 10,
        'trial_duration_s': 120,
        ...
    }
}
```

## Processing Pipeline

1. Neuropil correction: F_corrected = F - 0.7 * F_neuropil
2. Baseline correction: Suite2p maximin method (default parameters)
3. Temporal binning: Average 10 consecutive frames
4. Motion energy: Interpolate missing frames, bin by 10, discretize into 5 quintile bins per session
5. Trial segmentation: 2-minute consecutive blocks

## Running the Decoder

```bash
# Verify data format
python3 train_decoder.py converted_data.pkl --verify-only

# Train decoder
python3 train_decoder.py converted_data.pkl --plot-samples
```

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (414 MB) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
