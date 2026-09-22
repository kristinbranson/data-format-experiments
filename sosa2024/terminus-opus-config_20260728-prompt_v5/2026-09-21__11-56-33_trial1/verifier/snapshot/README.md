# Sosa et al. 2025 - Neural Decoder Dataset

## Dataset Description

Converted dataset from "A flexible hippocampal population code for experience relative to reward" (Sosa, Plitt, Giocomo, Nature Neuroscience 2025).

Mice performed a virtual reality navigation task on a 450 cm linear track with hidden reward zones. Two-photon calcium imaging was used to record CA1 hippocampal neurons while mice navigated the track. The reward zone location switched between three possible positions (A, B, C) across sessions, and mice experienced two different virtual environments.

## Data Format

The converted data is stored in `/app/converted_data.pkl` as a Python dictionary with the following structure:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

- **neural**: List of 152 sessions, each containing a list of trials. Each trial is a (n_neurons, n_timepoints) array of deltaF/F values.
- **input**: Decoder inputs (4 variables):
  - `time_from_trial_start`: Time in seconds from trial start (time-varying)
  - `environment`: 0=ENV1, 1=ENV2 (per-trial)
  - `trial_number`: 0-indexed trial number (per-trial)
  - `previous_trial_outcome`: 0=omission, 1=rewarded (per-trial)
- **output**: Decoder outputs (6 variables, all categorical):
  - `distance_to_reward_zone`: 7 bins (< -50cm, -50 to -10cm, -10 to 0cm, 0cm, 0 to +10cm, +10 to +50cm, > +50cm)
  - `absolute_position`: 5 bins of 90cm each (0-450cm)
  - `speed`: 5 bins (< 2, 2-10, 10-20, 20-40, > 40 cm/s)
  - `lick`: binary (no_lick, lick)
  - `reward_zone_location`: A=0, B=1, C=2 (per-trial)
  - `reward_outcome`: no=0, yes=1 (per-trial)

### Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Total trials | 12,216 |
| Total neurons | 138,318 |
| Neurons/session | 910 ± 448 (154-2328) |
| Trials/session | 80.4 ± 6.1 |
| Time bin | ~64.5 ms (15.5 Hz) |
| Brain region | CA1 (hippocampus) |
| Omission rate | ~15% |

## Neural Data Processing

Neural activity (deltaF/F) was computed from raw fluorescence following the reference code:
1. Neuropil subtraction (coefficient = 0.7)
2. Per-trial baseline via maximin method (20s sliding window)
3. dFF = (F - baseline) / |baseline|
4. Smoothing with 2-sample Gaussian kernel

Cell filtering:
1. Manual suite2p curation (iscell[:,0] == 1)
2. Interneuron exclusion (Pearson correlation > 0.5 with running speed)

## Usage

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access first session, first trial
neural_trial = data['neural'][0][0]  # (n_neurons, n_timepoints)
input_trial = data['input'][0][0]    # (4, n_timepoints)
output_trial = data['output'][0][0]  # (6, n_timepoints)

# Get subject for each session
for si in range(len(data['neural'])):
    subject = data['subjects'][data['subject_idx'][si]]
    n_trials = len(data['neural'][si])
    n_neurons = data['neural'][si][0].shape[0]
    print(f"Session {si}: {subject}, {n_trials} trials, {n_neurons} neurons")
```

## Train Decoder

```bash
python /app/train_decoder.py /app/converted_data.pkl
```

## Reference

Sosa, M., Plitt, M.H., Giocomo, L.M. A flexible hippocampal population code for experience relative to reward. Nature Neuroscience (2025).
