# Zhong et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset contains calcium imaging data from the paper "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025). The data has been converted to a decoder-compatible format for predicting behavioral variables from neural activity.

### Experiment
Head-fixed mice ran through virtual reality corridors with naturalistic texture patterns (e.g., leaf, circle, rock, brick). Task mice discriminated between visual stimuli with reward delivery in one corridor type. Unsupervised and naive mice ran through the same corridors without rewards.

### Data
- **89 recording sessions** from **19 mice** with GCaMP6s calcium imaging
- **~20,000-80,000 neurons** per session across visual cortex areas (V1, mHV, lHV, aHV)
- Neural activity: deconvolved calcium traces (Suite2p), position-interpolated to 60 bins per trial
- Each position bin = 0.1m, covering 4m corridor + 2m grey space = 6m total
- Time bin size: ~166.67 ms (at constant VR speed of 60 cm/s)

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 60)

# Access inputs and outputs
inputs = data['input'][0][0]   # shape: (4, 60)
outputs = data['output'][0][0] # shape: (4, 60)
```

## Data Format

### Neural
- `data['neural'][session][trial]`: (n_neurons, 60) float32 array
- Position-interpolated deconvolved calcium traces
- Neurons filtered to visual cortex only (V1, mHV, lHV, aHV)

### Inputs (decoder inputs)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_to_sound_cue | continuous, time-varying | Time in seconds to sound cue position |
| 1 | day_of_training | continuous, per-trial | Session number (0 = before learning, 1 = after learning, etc.) |
| 2 | time_since_trial_start | continuous, time-varying | Time in seconds from corridor entry |
| 3 | reward_availability | discrete, per-trial | 1 if rewarded corridor, 0 if not |

### Outputs (decoder predictions)
| Index | Name | Type | Values |
|-------|------|------|--------|
| 0 | visual_stimulus | categorical, per-trial | circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5 |
| 1 | licking | binary, time-varying | not_licking (0), licking (1) |
| 2 | position | categorical, time-varying | 0-1m (0), 1-2m (1), 2-3m (2), 3-4m (3) |
| 3 | running_speed | categorical, time-varying | Q1 (0), Q2 (1), Q3 (2), Q4 (3) |

### Metadata
- `data['subjects']`: list of 19 mouse names
- `data['subject_idx']`: session-to-subject mapping
- `data['brain_regions']`: ['V1', 'mHV', 'lHV', 'aHV']
- `data['brain_region_idx']`: neuron-to-region mapping per session

## Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Subjects | 19 |
| Total trials | 38,110 |
| Neurons per session | 17,363 - 78,815 |
| Time bins per trial | 60 |
| Time bin size | 166.67 ms |
| Brain regions | V1, mHV, lHV, aHV |

## Source
Zhong et al. (2025). "Unsupervised pretraining in biological neural networks." Nature.
