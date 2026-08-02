# Allen Brain Observatory - Visual Behavior 2P Decoder Dataset

## Dataset Description

This dataset contains converted data from the Allen Brain Observatory Visual Behavior 2P experiments, formatted for neural decoder training. Mice performed a go/no-go visual change detection task while neural activity was recorded using two-photon calcium imaging.

**Source**: Allen Brain Observatory Visual Behavior 2P dataset (Piet et al., 2024; Allen Institute whitepaper)

## Quick Start

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access output labels
outputs = data['output'][0][0]  # shape: (5, n_timepoints)
# Row 0: image_identity (0-15)
# Row 1: image_change (0 or 1)
# Row 2: running_speed (0-4, quintile bins)
# Row 3: pupil_diameter (0-4, quintile bins)
# Row 4: trial_outcome (0=hit, 1=miss, 2=false_alarm, 3=correct_rejection)
```

## Data Format

### Structure
```python
data = {
    'neural': [[trial_array, ...], ...],    # (n_neurons, n_timepoints) per trial
    'input': [[trial_array, ...], ...],      # (0, n_timepoints) - no inputs
    'output': [[trial_array, ...], ...],     # (5, n_timepoints) per trial
    'subjects': [...],                        # list of mouse IDs
    'subject_idx': array,                     # session → subject mapping
    'brain_regions': ['VISl', 'VISp'],
    'brain_region_idx': [...],               # neuron → region mapping per session
    'input_names': [],
    'output_names': ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome'],
    'output_values': [...],                   # category names per output
    'metadata': {...}
}
```

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 202 |
| Subjects (mice) | 38 |
| Total trials | 51,992 |
| Mean trials/session | 257 |
| Mean neurons/session | 146 |
| Time bin size | 750 ms |
| Brain regions | VISp, VISl |
| Unique images | 16 |

### Output Variables
| Variable | Type | Classes | Description |
|----------|------|---------|-------------|
| image_identity | Categorical (time-varying) | 16 images | Which natural image is displayed |
| image_change | Binary (time-varying) | 2 | 1 at the stimulus bin where image identity changes |
| running_speed | Ordinal (time-varying) | 5 quintile bins | Mouse running speed on wheel |
| pupil_diameter | Ordinal (time-varying) | 5 quintile bins | Pupil diameter from eye tracking |
| trial_outcome | Categorical (static/trial) | 4 (hit/miss/FA/CR) | Behavioral outcome of the trial |

### Neural Data
- **Signal**: Pre-computed dF/F (delta fluorescence over fluorescence)
- **Temporal resolution**: 750 ms bins (one per stimulus presentation)
- **Neuron filtering**: Only valid ROIs (cells) included

### Trial Selection
- **Included**: Go trials (image change) and Catch trials (no change)
- **Excluded**: Aborted trials (premature lick) and Auto-rewarded trials

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train and evaluate decoder
python train_decoder.py converted_data.pkl --plot-samples

# Use CPU if GPU memory is insufficient
python train_decoder.py converted_data.pkl --cpu
```

## Reproducing the Conversion

```bash
# Sample conversion (2 sessions, with processing plots)
python -u convert_data.py sample_data.pkl --sample --show-processing

# Full conversion
python -u convert_data.py converted_data.pkl --full
```

## Files
| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (386 MB) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
