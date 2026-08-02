# Allen Brain Observatory - Visual Behavior 2P: Decoder-Compatible Format

## Dataset Description

This dataset contains neural and behavioral data from the Allen Brain Observatory Visual Behavior 2P project, converted to a decoder-compatible format. Mice performed a visual change detection task while two-photon calcium imaging recorded neural activity from excitatory (Slc17a7), Sst, and Vip neurons in visual cortex (VISp and VISl).

### Task
Head-fixed mice viewed natural images (250ms stimulus, 500ms gray screen) and were rewarded for licking in response to image changes. The task uses a roving baseline paradigm where images repeat before changing.

### Trial Types
- **Go trials**: Image changes, mouse should lick
- **Catch trials**: Sham change (same image), mouse should not lick
- Aborted and auto-rewarded trials are excluded

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access output data for session 0, trial 0
output = data['output'][0][0]  # shape: (5, n_timepoints)
```

## Output Format

### Structure
```python
data = {
    'neural': list of sessions, each containing list of trials (n_neurons, n_timepoints),
    'input': list of sessions (empty - no decoder inputs),
    'output': list of sessions, each containing list of trials (5, n_timepoints),
    'subjects': list of mouse IDs,
    'subject_idx': array mapping sessions to subjects,
    'brain_regions': ['VISp', 'VISl'],
    'brain_region_idx': list of arrays mapping neurons to brain regions,
    'input_names': [],
    'output_names': ['image_identity', 'image_change', 'running_speed', 'pupil_diameter', 'trial_outcome'],
    'output_values': [...],
    'metadata': {...}
}
```

### Output Variables
| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| image_identity | Categorical, time-varying | 16 natural images | Currently displayed image |
| image_change | Binary, time-varying | 0/1 | 1 during change stimulus presentation |
| running_speed | Categorical (5 bins), time-varying | 0-4 | Running speed in equal percentile bins |
| pupil_diameter | Categorical (5 bins), time-varying | 0-4 | Pupil area in equal percentile bins |
| trial_outcome | Categorical, static per trial | hit/miss/CR/FA | Behavioral outcome |

### Neural Data
- **Type**: Detected calcium events (raw, unfiltered)
- **Valid ROI filtering**: Only neurons passing quality control are included
- **Time bin**: 93.23 ms (~10.73 Hz) - common across all experiments

## Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 202 |
| Subjects (mice) | 38 |
| Total trials | 51,992 |
| Total neurons | 29,444 |
| Brain regions | VISp (29,282), VISl (162) |
| Cre lines | Slc17a7 (excitatory), Sst, Vip |
| Time bin | 93.23 ms |
| Unique images | 16 |

## Decoder Performance
| Output | Validation Balanced Acc | Chance |
|--------|------------------------|--------|
| image_identity | 0.193 | 0.063 |
| image_change | 0.608 | 0.500 |
| running_speed | 0.287 | 0.200 |
| pupil_diameter | 0.315 | 0.200 |
| trial_outcome | 0.274 | 0.250 |

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample (2 sessions) for testing
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion notes

## Running the Conversion
```bash
# Sample conversion (2 sessions)
python -u convert_data.py sample_data.pkl --sample

# Full conversion (all 202 sessions)
python -u convert_data.py converted_data.pkl --full

# With processing plots
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Running the Decoder
```bash
# Verify data format
python -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python -u train_decoder.py converted_data.pkl

# With sample plots
python -u train_decoder.py converted_data.pkl --plot-samples
```
