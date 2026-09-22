# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging neural activity and behavioral data from Zhong et al. 2025 ("Unsupervised pretraining in biological neural networks"), converted to a standardized decoder-compatible format.

### Experiment
Mice ran through virtual reality corridors with naturalistic texture patterns (circle, leaf, rock, brick). Task mice learned to discriminate between visual textures and lick for water rewards. Unsupervised mice ran through the same corridors without rewards. Neural activity was recorded using two-photon calcium imaging across multiple visual cortex areas.

### Key Parameters
- **Recording method**: Two-photon calcium imaging (GCaMP6s)
- **Frame rate**: 3.17 Hz
- **Time bin size**: ~315.5 ms
- **Corridor**: 4m texture + 2m gray space = 6m total
- **Brain regions**: V1, mHV, lHV, aHV (from retinotopy)
- **Subjects**: 19 mice
- **Sessions**: 89 recordings
- **Trials**: 38,110 total
- **Neurons**: 5,000 per session (subsampled from 20,547-89,577)

## Data Format

The data is stored as a Python pickle file (`converted_data.pkl`) with the following structure:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Keys

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list of lists of arrays | Neural activity (n_neurons, n_timepoints) per trial |
| `input` | list of lists of arrays | Decoder inputs (4, n_timepoints) per trial |
| `output` | list of lists of arrays | Decoder outputs (4, n_timepoints) per trial |
| `subjects` | list of str | Mouse names |
| `subject_idx` | array (n_sessions,) | Index into subjects for each session |
| `brain_regions` | list of str | Brain region names |
| `brain_region_idx` | list of arrays | Region index for each neuron per session |
| `input_names` | list of str | Names of input variables |
| `output_names` | list of str | Names of output variables |
| `output_values` | list of lists | Category names for each output |
| `metadata` | dict | Additional information |

### Input Variables
1. **time_to_sound_cue**: Time relative to sound cue (seconds, negative before cue)
2. **day_of_training**: Day/session number in training sequence
3. **time_since_trial_start**: Time since corridor entry (seconds)
4. **reward_availability**: 1 if rewarded corridor, 0 if not

### Output Variables
1. **visual_stimulus**: 15 categories (circle1, circle2, leaf1, leaf2, etc.)
2. **licking**: Binary (0=not licking, 1=licking)
3. **position_bin**: 4 bins of 1m each (0-1m, 1-2m, 2-3m, 3-4m+)
4. **running_speed_bin**: 4 quartile bins (Q1 slowest to Q4 fastest)

## Usage

```python
import pickle
import numpy as np

# Load data
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (5000, n_timepoints)

# Access inputs and outputs
inputs = data['input'][0][0]   # shape: (4, n_timepoints)
outputs = data['output'][0][0] # shape: (4, n_timepoints)

# Get brain region for each neuron
regions = data['brain_region_idx'][0]  # shape: (5000,)
region_names = [data['brain_regions'][i] for i in regions]
```

## Training the Decoder

```bash
python train_decoder.py converted_data.pkl
```

## File Sizes
- `converted_data.pkl`: ~20 GB
- `sample_data.pkl`: ~6 GB (2 sessions for testing)

## Reference
Zhong, L. et al. "Unsupervised pretraining in biological neural networks." (2025)
