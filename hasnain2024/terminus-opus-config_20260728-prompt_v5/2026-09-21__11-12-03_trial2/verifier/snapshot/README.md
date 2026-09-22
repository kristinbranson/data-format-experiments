# Economo Lab Neural Decoder Dataset

## Dataset Description

This dataset contains electrophysiology recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context directional licking task, converted from the paper "Separating cognitive and motor processes in the behaving mouse" (Economo Lab, 2024).

### Task
Mice perform two tasks that alternate block-wise:
- **Delayed Response (DR)**: Auditory stimulus indicates reward location, followed by delay, then go cue. Mouse must lick to correct side.
- **Water Cued (WC)**: No auditory cues. Water presented at random time/location. Mouse detects and consumes water.

### Data
- **44 sessions** from **14 mice** (10 DR task, 4 randomized delay task)
- **2,497 neurons** recorded from ALM
- **13,823 trials** total
- Neural activity: firing rates (spikes/s), 10ms bins, smoothed with 15-sample causal Gaussian kernel
- Aligned to go cue onset, -2.5s to +2.5s window

## Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural_trial = data['neural'][0][0]  # shape: (n_neurons, 500)

# Access outputs
output_trial = data['output'][0][0]  # shape: (6, 500)
```

## Data Format

### Neural Data
- `data['neural']`: List of sessions, each containing list of trials
- Each trial: numpy array of shape `(n_neurons, 500)` (firing rates in Hz)

### Input (Decoder Input)
- `data['input']`: Time from go cue onset in seconds
- Shape: `(1, 500)` per trial

### Output (Decoder Targets)
- `data['output']`: 6 output variables, shape `(6, 500)` per trial

| Index | Variable | Values | Type |
|-------|----------|--------|------|
| 0 | lick_direction | 0=left, 1=right, 2=none | per-trial |
| 1 | context | 0=WC, 1=DR | per-trial |
| 2 | outcome | 0=incorrect, 1=correct, 2=ignore | per-trial |
| 3 | tongue_velocity | 0=below median, 1=above median, 2=not visible | time-varying |
| 4 | paw_velocity | 0=below median, 1=above median, 2=not visible | time-varying |
| 5 | motion_energy | 0=below median, 1=above median, 2=no video | time-varying |

### Metadata
- `data['subjects']`: List of animal IDs
- `data['subject_idx']`: Session-to-subject mapping
- `data['brain_regions']`: ['ALM']
- `data['brain_region_idx']`: Neuron-to-region mapping per session
- `data['metadata']`: Task description, timing info, processing parameters

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 44 |
| Subjects | 14 |
| Total neurons | 2,497 |
| Total trials | 13,823 |
| Time bins | 500 (10ms each) |
| Time window | -2.5s to +2.5s from go cue |
| Brain region | ALM |

## Processing Pipeline

1. Load spike data from .mat files (HDF5 and v5 formats)
2. Select ALM probe based on session metadata
3. Quality filter: exclude garbage, gabrga, noisy, real? quality labels
4. Align spikes to go cue onset
5. Bin spikes (10ms bins) and compute firing rates
6. Smooth with causal Gaussian kernel (15 samples)
7. Remove low firing rate neurons (< 0.5 Hz mean)
8. Exclude early lick and stimulation trials
9. Extract behavioral variables from Bpod data
10. Process DLC trajectories for tongue and paw velocity
11. Load and align motion energy data
12. Discretize continuous outputs with per-session 50th percentile threshold

## Files

- `converted_data.pkl` - Full converted dataset (44 sessions)
- `sample_data.pkl` - Sample dataset (2 sessions, for testing)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
