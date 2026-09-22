# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from mouse visual cortex during virtual reality corridor navigation, converted from Zhong et al. 2025 "Unsupervised pretraining in biological neural networks".

### Experiment
- 19 mice ran through 4m virtual reality corridors with naturalistic texture patterns
- Visual stimuli: leaf, circle, rock, wood (and variants)
- Sound cue at random position (0.5-3.5m) signals reward availability
- Supervised mice received water rewards; unsupervised mice did not
- Two-photon mesoscope imaging across multiple visual cortex areas

### Data
- 89 recording sessions across 19 mice
- 20,547 to 89,577 neurons per session (Suite2p deconvolved traces)
- ~38,000 trials total
- 5 brain regions: V1, mHV, aHV, lHV, unassigned
- Frame rate: ~3.18 Hz (314.69 ms per frame)

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access inputs and outputs
inputs = data['input'][0][0]   # shape: (4, n_timepoints)
outputs = data['output'][0][0] # shape: (4, n_timepoints)
```

## Data Format

### Neural
- `data['neural'][session][trial]`: (n_neurons, n_timepoints) float32 array
- Deconvolved fluorescence traces, filtered to running corridor frames

### Inputs (4 dimensions)
1. `time_to_sound_cue`: Time to sound cue in seconds (negative = before cue)
2. `day_of_training`: Day of training (0 = before learning, 1+ = after)
3. `time_since_trial_start`: Time since corridor entry in seconds
4. `reward_availability`: 1 if rewarded corridor, 0 if not

### Outputs (4 dimensions, categorical)
1. `visual_stimulus`: 15 categories (circle1, circle2, leaf1, leaf2, etc.)
2. `licking`: Binary (0 = no lick, 1 = lick)
3. `position`: 4 bins of 1m each (0-1m, 1-2m, 2-3m, 3-4m)
4. `running_speed`: 4 quartile bins (Q1-Q4)

### Metadata
- `data['subjects']`: List of 19 mouse names
- `data['subject_idx']`: Session-to-subject mapping
- `data['brain_regions']`: ['V1', 'mHV', 'aHV', 'lHV', 'unassigned']
- `data['brain_region_idx']`: Neuron-to-region mapping per session
- `data['metadata']`: Task description, time bin size, alignment info

## Key Processing Decisions

1. **Running frames only**: Only frames where VR is moving (ft_move > 0), matching the reference paper
2. **Corridor frames only**: Only frames in the 4m textured corridor (not gray space)
3. **All neurons included**: No d-prime filtering (decoder uses PCA internally)
4. **Native frame rate**: ~3.18 Hz, no temporal rebinning
5. **Speed quartiles**: Computed across all corridor frames in all sessions

## Files

- `converted_data.pkl`: Full dataset (161.65 GB)
- `sample_data.pkl`: 2-session sample (2.99 GB)
- `convert_data.py`: Conversion script
- `CONVERSION_NOTES.md`: Detailed conversion documentation

## Decoder Performance

| Output | Validation Balanced Accuracy | Chance |
|--------|-----------------------------|---------|
| visual_stimulus | 0.530 | 0.067 |
| licking | 0.859 | 0.500 |
| position | 0.408 | 0.250 |
| running_speed | 0.375 | 0.250 |

## Source

Zhong et al. 2025, "Unsupervised pretraining in biological neural networks"
