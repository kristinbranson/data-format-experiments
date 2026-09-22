# Hippocampal Reward-Relative Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from mouse hippocampus (CA1) during a virtual reality navigation task with hidden reward zones. Data is from the paper "A flexible hippocampal population code for experience relative to reward" (Sosa, Plitt, Giocomo 2024).

### Task
- Mice navigate a 450 cm virtual linear track
- Hidden reward zone at one of three locations: A (80-130cm), B (200-250cm), C (320-370cm)
- Reward zone switches after 30 trials
- Two environments (ENV1, ENV2)
- ~15% reward omission rate
- 14 days of imaging per mouse

### Data Statistics
- **Subjects**: 11 mice
- **Sessions**: 152 total
- **Trials**: 12,216 total (mean 80.4/session)
- **Neurons**: 138,678 total (mean 912/session)
- **Brain region**: CA1 (hippocampus)
- **Imaging rate**: ~15.5 Hz (64.5 ms time bins)
- **Indicator**: GCaMP7f

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access inputs and outputs
inputs = data['input'][0][0]   # shape: (4, n_timepoints)
outputs = data['output'][0][0] # shape: (6, n_timepoints)
```

## Data Format

### Neural Data
- `data['neural']`: List of sessions, each containing list of trials
- Each trial: `(n_neurons, n_timepoints)` array of dF/F values
- Neural signal: dF/F computed from raw fluorescence with neuropil subtraction (F - 0.7*Fneu), maximin baseline correction, and Gaussian smoothing (σ=2)

### Decoder Inputs (4 variables)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_from_trial_start | continuous, time-varying | Seconds from trial start |
| 1 | environment_type | binary, per-trial | 0=ENV1, 1=ENV2 |
| 2 | trial_number | continuous, per-trial | 0-indexed trial within session |
| 3 | previous_trial_outcome | binary, per-trial | 0=omitted, 1=rewarded |

### Decoder Outputs (6 variables, all categorical)
| Index | Name | Classes | Description |
|-------|------|---------|-------------|
| 0 | distance_to_reward_zone | 7 bins | Signed distance to nearest reward zone edge |
| 1 | absolute_position | 5 bins | Position on track (90cm bins) |
| 2 | speed | 5 bins | Running speed |
| 3 | lick | 2 (binary) | Licking behavior |
| 4 | reward_zone_location | 3 (A/B/C) | Which reward zone is active |
| 5 | reward_outcome | 2 (binary) | Whether trial was rewarded |

### Metadata
- `data['subjects']`: List of subject IDs
- `data['subject_idx']`: Session-to-subject mapping
- `data['brain_regions']`: ['CA1']
- `data['brain_region_idx']`: Neuron-to-region mapping per session
- `data['metadata']`: Task description, time bin size, alignment info

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - 2-session sample for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `train_decoder_full_out.txt` - Full decoder training results

## Decoder Results

| Output | Validation Accuracy | Chance |
|--------|-------------------|--------|
| distance_to_reward_zone | 0.264 | 0.143 |
| absolute_position | 0.356 | 0.200 |
| speed | 0.300 | 0.200 |
| lick | 0.559 | 0.500 |
| reward_zone_location | 0.780 | 0.333 |
| reward_outcome | 0.504 | 0.500 |
