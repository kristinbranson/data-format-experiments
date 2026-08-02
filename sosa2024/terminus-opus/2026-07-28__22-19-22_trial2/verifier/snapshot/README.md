# Neural Decoder Dataset: Hippocampal CA1 Calcium Imaging

## Dataset Description

This dataset contains two-photon calcium imaging data from hippocampal CA1 neurons in head-fixed mice navigating a virtual reality linear track. The data is from:

> Sosa, Plitt, Giocomo (2025). "A flexible hippocampal population code for experience relative to reward." *Nature Neuroscience*.

### Experimental Task
- Mice navigate a 450 cm virtual linear track with a hidden 50 cm reward zone
- Three possible reward zone locations: A (80-130 cm), B (200-250 cm), C (320-370 cm)
- Reward zone location switches between sessions
- Two visual environments (Env1, Env2) with distinct features
- Reward randomly omitted on ~15% of trials
- 11 mice, 152 sessions, ~12,216 trials total

### Neural Data
- Two-photon calcium imaging at ~15.5 Hz
- Deconvolved calcium events (OASIS algorithm via Suite2p)
- 155-2341 neurons per session (after iscell filtering + interneuron exclusion)
- Brain region: hippocampal CA1

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list of lists of arrays | Neural activity (n_neurons × n_timepoints) per trial per session |
| `input` | list of lists of arrays | Decoder inputs (4 × n_timepoints) per trial per session |
| `output` | list of lists of arrays | Decoder outputs (6 × n_timepoints) per trial per session |
| `subjects` | list of str | Subject IDs |
| `subject_idx` | array | Subject index for each session |
| `brain_regions` | list of str | Brain region names (['CA1']) |
| `brain_region_idx` | list of arrays | Brain region index for each neuron |
| `input_names` | list of str | Names of input variables |
| `output_names` | list of str | Names of output variables |
| `output_values` | list of lists | Names of each output category |
| `metadata` | dict | Task description, time bin size, etc. |

### Decoder Inputs (4 variables)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_from_trial_start | continuous, time-varying | Seconds from trial start |
| 1 | environment_type | binary, per-trial | 0=Env1, 1=Env2 |
| 2 | trial_number | continuous, per-trial | Trial index within session |
| 3 | previous_trial_outcome | binary, per-trial | 0=omitted, 1=rewarded |

### Decoder Outputs (6 variables, all categorical)
| Index | Name | Categories | Type |
|-------|------|------------|------|
| 0 | distance_to_reward_zone | 7 bins: <-50, -50 to -10, -10 to 0, 0, 0 to +10, +10 to +50, >+50 cm | time-varying |
| 1 | absolute_position | 5 bins: 0-90, 90-180, 180-270, 270-360, 360-450 cm | time-varying |
| 2 | speed | 5 bins: <2, 2-10, 10-20, 20-40, >40 cm/s | time-varying |
| 3 | lick | 2 bins: no_lick, lick | time-varying |
| 4 | reward_zone_location | 3 bins: zone_A, zone_B, zone_C | per-trial |
| 5 | reward_outcome | 2 bins: no_reward, reward | per-trial |

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Total trials | 12,216 |
| Total neurons | 138,678 |
| Neurons/session | 155-2341 (mean: 912) |
| Trials/session | 41-100 (mean: 80) |
| Time bin | 64.48 ms (~15.5 Hz) |
| Omission rate | 15.3% |
| Reward zone balance | A: 34.3%, B: 32.8%, C: 32.9% |

## Usage

### Verify data format
```bash
python train_decoder.py converted_data.pkl --verify-only
```

### Train decoder
```bash
python train_decoder.py converted_data.pkl
```

### Regenerate converted data
```bash
# Full dataset
python -u convert_data.py converted_data.pkl --full

# Sample (2 sessions)
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (152 sessions) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `README.md` | This file |
| `cache/` | Intermediate files and logs |

## Source

- Paper: https://www.nature.com/articles/s41593-025-01985-4
- Data: https://dandiarchive.org/dandiset/001361
- Code: https://github.com/GiocomoLab/Sosa_et_al_2024
