# Hippocampal Calcium Imaging Decoder Dataset

## Source
Sosa et al. 2025 — "A flexible hippocampal population code for experience relative to reward"
Nature Neuroscience, Volume 28, July 2025, pp. 1497-1509
DANDI:001361

## Dataset Description
Two-photon calcium imaging of CA1 hippocampal neurons in 11 mice performing a virtual reality navigation task with hidden reward zones. The task involves a 450 cm linear track with reward zone switches across 14 days and two environments.

## How to Load

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Data Format

```python
data['neural']        # list of 152 sessions, each a list of trials (n_neurons, T)
data['input']         # list of sessions, each a list of trials (4, T)
data['output']        # list of sessions, each a list of trials (6, T), int
data['subjects']      # ['m11', 'm12', ..., 'm7'] (11 subjects)
data['subject_idx']   # (152,) int — index into subjects per session
data['brain_regions'] # ['CA1']
data['brain_region_idx']  # list of (n_neurons,) arrays, all zeros (CA1)
data['input_names']   # ['time_from_trial_start', 'environment', 'trial_number', 'previous_trial_outcome']
data['output_names']  # ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome']
data['output_values'] # bin labels for each output
data['metadata']      # task description, time_bin_size, etc.
```

## Inputs (4 dimensions)
| Index | Name | Type | Values |
|-------|------|------|--------|
| 0 | time_from_trial_start | continuous, time-varying | seconds from trial start |
| 1 | environment | binary, per-trial | 0=ENV1, 1=ENV2 |
| 2 | trial_number | continuous, per-trial | 0-99 |
| 3 | previous_trial_outcome | binary, per-trial | 0=omitted, 1=rewarded |

## Outputs (6 dimensions, integer class labels)
| Index | Name | Classes | Description |
|-------|------|---------|-------------|
| 0 | distance_to_reward_zone | 7 | Signed distance to nearest point in 50cm reward zone |
| 1 | absolute_position | 5 | Position on 450cm track (90cm bins) |
| 2 | speed | 5 | Running speed (cm/s) |
| 3 | lick | 2 | Lick detected this frame |
| 4 | reward_zone_location | 3 | A (80-130cm), B (200-250cm), C (320-370cm) |
| 5 | reward_outcome | 2 | Whether trial was rewarded |

## Key Statistics
- 11 subjects, 152 sessions, 12,135 trials
- 909 ± 448 neurons per session (CA1)
- Time bin: 64.48 ms (~15.5 Hz imaging)
- ~85% reward rate

## Neural Processing
dF/F computed following reference: neuropil subtraction (coef=0.7), maximin baseline (20s window), smoothed (2-frame Gaussian). Interneurons excluded (speed-dFF r > 0.5). Lick sensor error trials removed (81 total).

## Files
- `converted_data.pkl` — Full dataset
- `convert_data.py` — Conversion script
- `train_decoder.py` — Decoder training script
- `CONVERSION_NOTES.md` — Detailed conversion documentation
