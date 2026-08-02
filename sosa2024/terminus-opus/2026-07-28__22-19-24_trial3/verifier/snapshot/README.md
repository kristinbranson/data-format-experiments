# Neural Decoder Dataset: Sosa et al. (2025)

## Dataset Description

This dataset contains 2-photon calcium imaging data from hippocampal area CA1 during virtual reality navigation in mice. The data is from:

> Sosa, Plitt & Giocomo (2025). "A flexible hippocampal population code for experience relative to reward." *Nature Neuroscience*.

### Experiment
- Head-fixed mice navigate a 450 cm virtual linear track
- Hidden 50 cm reward zone at one of three locations (A: 80-130cm, B: 200-250cm, C: 320-370cm)
- Reward zone switches between locations across sessions
- Two environments (ENV1, ENV2) with distinct visual features
- ~15% of trials have reward omission
- 2-photon calcium imaging of CA1 neurons at ~15.5 Hz

### Data Statistics
- **Subjects**: 11 mice
- **Sessions**: 152 (12-14 per subject)
- **Total trials**: 12,216
- **Trials/session**: 80.4 ± 6.1
- **Neurons/session**: 155-2,341 (mean 912)
- **Brain region**: Hippocampal CA1
- **Time bin**: ~64.5 ms (native imaging rate)

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 5
neural = data['neural'][0][5]  # shape: (n_neurons, n_timepoints)

# Access inputs for session 0, trial 5
inputs = data['input'][0][5]  # shape: (4, n_timepoints)

# Access outputs for session 0, trial 5
outputs = data['output'][0][5]  # shape: (6, n_timepoints)
```

## Data Format

### Neural Data
- Deconvolved calcium events (OASIS deconvolution of dF/F)
- Filtered by Suite2P iscell classification
- Shape: (n_neurons, n_timepoints) per trial

### Decoder Inputs (4 variables)
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | time_from_trial_start | Time-varying | Seconds from trial start |
| 1 | environment_type | Per-trial | 0=ENV1, 1=ENV2 |
| 2 | trial_number | Per-trial | 0-indexed trial number |
| 3 | previous_trial_outcome | Per-trial | 0=omission, 1=rewarded |

### Decoder Outputs (6 variables)
| Index | Name | Type | Values |
|-------|------|------|--------|
| 0 | distance_to_reward_zone | Time-varying | 0: <-50cm, 1: -50 to -10cm, 2: -10 to 0cm, 3: 0cm (in zone), 4: >0 to +10cm, 5: +10 to +50cm, 6: >+50cm |
| 1 | absolute_position | Time-varying | 5 equal bins (0-90, 90-180, 180-270, 270-360, 360-450 cm) |
| 2 | speed | Time-varying | 0: <2cm/s, 1: 2-10cm/s, 2: 10-20cm/s, 3: 20-40cm/s, 4: >40cm/s |
| 3 | lick | Time-varying | 0: no lick, 1: lick |
| 4 | reward_zone_location | Per-trial | 0: A (80-130cm), 1: B (200-250cm), 2: C (320-370cm) |
| 5 | reward_outcome | Per-trial | 0: no reward, 1: reward |

### Other Fields
- `subjects`: List of subject IDs
- `subject_idx`: Session-to-subject mapping
- `brain_regions`: ['CA1']
- `brain_region_idx`: Neuron-to-region mapping per session
- `metadata`: Task description, time bin size, alignment info, session details

## Validation

Run the decoder to validate:
```bash
python train_decoder.py converted_data.pkl --verify-only  # Check format
python train_decoder.py converted_data.pkl               # Train decoder
```

## Conversion

To regenerate the converted data:
```bash
python convert_data.py converted_data.pkl --full          # Full conversion
python convert_data.py sample_data.pkl --sample            # Sample (2 sessions)
```

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder module
