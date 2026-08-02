# Sosa et al. 2025 - Neural Decoder Dataset

## Dataset Description

This dataset is converted from the paper:
> Sosa, Plitt, Giocomo. 2025. "A flexible hippocampal population code for experience relative to reward." *Nature Neuroscience*.

The data contains 2-photon calcium imaging recordings from hippocampal CA1 neurons in mice performing a virtual reality hidden reward zone task.

## Task Description

Mice run on a 450 cm virtual linear track with a hidden 50 cm reward zone at one of three possible locations (A=[80-130cm], B=[200-250cm], C=[320-370cm]). The reward zone location switches within sessions. Two visually distinct environments (ENV1, ENV2) are used. Mice must lick within the reward zone to receive a water reward. ~15-17% of trials are reward omissions.

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

- **neural**: List of 152 sessions, each containing a list of trials. Each trial is a `(n_neurons, n_timepoints)` float32 array of deconvolved calcium events.
- **input**: Decoder input variables (4 dimensions):
  - `time_from_trial_start`: Time in seconds from trial start (time-varying)
  - `environment_type`: Binary, 0=ENV1, 1=ENV2 (per-trial)
  - `trial_number`: 0-indexed trial number within session (per-trial)
  - `previous_trial_outcome`: Binary, 0=omitted, 1=rewarded (per-trial)
- **output**: Decoder output variables (6 dimensions):
  - `distance_to_reward_zone`: 7 bins (time-varying)
  - `absolute_position`: 5 equal bins over 0-450cm (time-varying)
  - `speed`: 5 bins (time-varying)
  - `lick`: Binary (time-varying)
  - `reward_zone_location`: 0=A, 1=B, 2=C (per-trial)
  - `reward_outcome`: Binary (per-trial)

### Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 mice |
| Sessions | 152 |
| Total trials | 12,216 |
| Total neurons | 138,678 |
| Neurons/session | 155-2,341 |
| Time bin | 64.48 ms (~15.5 Hz) |
| Brain region | CA1 (hippocampus) |

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (~9.4 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions, ~61 MB) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training/validation script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `verification_full_out.txt` | Full dataset verification output |
| `train_decoder_full_out.txt` | Full decoder training output |

## Usage

### Convert data
```bash
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing
```

### Verify data format
```bash
python -u train_decoder.py converted_data.pkl --verify-only
```

### Train decoder
```bash
python -u train_decoder.py converted_data.pkl
```

## Decoder Results

| Output | Val Balanced Acc | Chance |
|--------|-----------------|--------|
| distance_to_reward_zone | 0.377 | 0.143 |
| absolute_position | 0.510 | 0.200 |
| speed | 0.421 | 0.200 |
| lick | 0.698 | 0.500 |
| reward_zone_location | 0.800 | 0.333 |
| reward_outcome | 0.513 | 0.500 |

All outputs are decoded above chance level.

## Reference

Sosa, M., Plitt, M.H. & Giocomo, L.M. A flexible hippocampal population code for experience relative to reward. *Nat Neurosci* (2025). https://doi.org/10.1038/s41593-025-01985-4
