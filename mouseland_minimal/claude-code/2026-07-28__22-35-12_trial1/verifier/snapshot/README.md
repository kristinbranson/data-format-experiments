# Neural Decoder Dataset: Zhong et al. 2025

## Overview

This dataset contains calcium imaging data from Zhong et al. 2025 "Unsupervised pretraining in biological neural networks", converted into a standardized format for neural decoding.

Mice ran through linear virtual reality corridors with naturalistic texture patterns on the walls. The task required discriminating between visual textures, with a sound cue signaling reward availability in rewarded corridors.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Mice | 19 |
| Total trials | 38,110 |
| Total neurons (across sessions) | 4,105,393 |
| Neurons per session | 17,363 - 78,815 |
| Brain regions | V1, mHV, lHV, aHV |
| Time bins per trial | 40 |
| Time bin size | 166.67 ms |
| Corridor length | 4 m |

## Data Format

Pickle file containing a dictionary with keys:

- **neural**: List of 89 sessions, each a list of trials with shape (n_neurons, 40)
- **input**: 4 decoder input variables per trial, shape (4, 40)
- **output**: 4 decoder output variables per trial, shape (4, 40)
- **subjects**: 19 unique mouse IDs
- **brain_regions**: ['V1', 'mHV', 'lHV', 'aHV']

### Decoder Inputs
1. `time_to_sound_cue` - seconds relative to sound cue position (continuous, time-varying)
2. `day_of_training` - days since first recording for each mouse (continuous, per-session)
3. `time_since_trial_start` - seconds from corridor entry (continuous, time-varying)
4. `reward_availability` - 1 if rewarded corridor, 0 if not (binary, per-trial)

### Decoder Outputs
1. `visual_stimulus` - 15 categories: circle1/2/3, leaf1/2/3, leaf1_swap1/2, rock1/2, wood1/2/5, wood1_swap1/2
2. `licking` - binary (no_lick, lick)
3. `position` - 4 bins of 1m each (0-1m, 1-2m, 2-3m, 3-4m)
4. `running_speed` - 4 quartile bins (Q1, Q2, Q3, Q4)

## Usage

```python
import pickle

# Load data
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Validate format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```

## Processing

Neural data was processed following the reference paper and code:
1. Deconvolved calcium traces loaded from Suite2p output
2. Position-interpolated to 60 evenly-spaced bins per corridor
3. Texture area (first 40 bins = 4m) extracted
4. Only visual cortex neurons included (excluding iarea == -1 and iarea == 7)
5. Only running frames used for interpolation (VR moving)

See `CONVERSION_NOTES.md` for detailed processing decisions and validation results.

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Conversion script |
| `converted_data.pkl` | Full dataset (89 sessions) |
| `sample_data.pkl` | Sample dataset (3 sessions) |
| `CONVERSION_NOTES.md` | Detailed conversion notes |
| `train_decoder.py` | Decoder training/validation script |
