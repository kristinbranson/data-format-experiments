# Converted Dataset: Hippocampal Population Code for Experience Relative to Reward

## Source
Sosa, Plitt & Giocomo (2025). "A flexible hippocampal population code for experience relative to reward." Nature Neuroscience.
DANDI:001361

## Description
Two-photon calcium imaging data from hippocampal CA1 neurons in head-fixed mice performing a virtual reality navigation task with changing hidden reward zones. 11 mice, 152 sessions, ~12,200 trials.

## Task
Mice traverse a 450cm virtual linear track with a hidden 50cm reward zone at one of three locations (A: 80-130cm, B: 200-250cm, C: 320-370cm). Reward zone switches every other day. Reward is randomly omitted on ~15% of trials.

## Data Format
Pickle file (`converted_data.pkl`) containing a dictionary with:

### Neural Data
- `neural`: List of sessions, each containing list of trials with shape (n_neurons, n_timepoints)
- Signal: dF/F computed from raw fluorescence using maximin baseline method
- Neurons filtered by iscell and interneuron exclusion (corr(dF/F, speed) > 0.5)

### Decoder Inputs (4 variables)
1. `time_from_trial_start`: Time in seconds from trial onset (continuous, time-varying)
2. `environment_type`: 0=ENV1, 1=ENV2 (binary, per-trial)
3. `trial_number`: 0-indexed trial number (continuous, per-trial)
4. `previous_trial_outcome`: 0=omission, 1=rewarded (binary, per-trial)

### Decoder Outputs (6 variables, all categorical)
1. `distance_to_reward_zone`: 7 bins (<-50, -50 to -10, -10 to 0, 0, >0 to +10, +10 to +50, >+50 cm)
2. `absolute_position`: 5 bins of 90cm (0-90, 90-180, 180-270, 270-360, 360+ cm)
3. `speed`: 5 bins (<2, 2-10, 10-20, 20-40, >40 cm/s)
4. `lick`: Binary (0=no, 1=yes)
5. `reward_zone_location`: Per-trial (0=A, 1=B, 2=C)
6. `reward_outcome`: Per-trial (0=no, 1=yes)

### Metadata
- `subjects`: List of subject IDs
- `subject_idx`: Subject index per session
- `brain_regions`: ['CA1']
- `brain_region_idx`: Region index per neuron per session
- `metadata`: Task description, time bin size (~64.5ms), temporal alignment, etc.

## Loading
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access first session, first trial
neural = data['neural'][0][0]  # (n_neurons, n_timepoints)
inputs = data['input'][0][0]   # (4, n_timepoints)
outputs = data['output'][0][0] # (6, n_timepoints)
```

## Key Statistics
- 11 subjects, 152 sessions
- 12,216 total trials
- 138,276 total neurons (across all sessions)
- 910 ± 448 neurons per session (range: 154-2323)
- 80.4 ± 6.1 trials per session
- ~15.3% reward omission rate
- ~15.5 Hz sampling rate

## Decoder Performance
| Output | Validation Balanced Acc | Chance |
|--------|------------------------|--------|
| distance_to_reward_zone | 0.629 | 0.143 |
| absolute_position | 0.769 | 0.200 |
| speed | 0.623 | 0.200 |
| lick | 0.766 | 0.500 |
| reward_zone_location | 0.866 | 0.333 |
| reward_outcome | 0.604 | 0.500 |
