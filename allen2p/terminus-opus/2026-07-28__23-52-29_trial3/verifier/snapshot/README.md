# Allen Brain Observatory Visual Behavior 2P - Decoder Format

## Dataset Description

This dataset contains neural recordings from the Allen Brain Observatory Visual Behavior 2P project. Mice performed a visual change detection task where they detected changes in natural image identity and licked for water reward.

### Task
- Mice viewed a series of natural images (250ms stimulus, 500ms inter-stimulus interval)
- Images repeated multiple times before changing to a new image
- Mice licked to report image changes and received water reward for correct responses
- Trial types: Go (image change) → Hit/Miss, Catch (no change) → False Alarm/Correct Reject

### Neural Data
- Two-photon calcium imaging with GCaMP6f
- Deconvolved calcium events (not raw dF/F)
- Cell types: Excitatory (Slc17a7), Sst, Vip
- Brain regions: VISp (primary visual cortex), VISl (lateral visual area)

## Data Format

The converted data is stored in `converted_data.pkl` as a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Keys
- `neural`: List of sessions, each containing list of trials with shape (n_neurons, n_timepoints)
- `input`: Empty (no decoder inputs)
- `output`: List of sessions/trials with shape (5, n_timepoints), containing:
  - [0] image_identity: Categorical (16 images)
  - [1] image_change: Binary (0=no change, 1=change)
  - [2] running_speed: 5 percentile bins
  - [3] pupil_diameter: 5 percentile bins  
  - [4] trial_outcome: 4 categories (hit, miss, false_alarm, correct_reject)
- `subjects`: List of 38 mouse IDs
- `subject_idx`: Session-to-subject mapping
- `brain_regions`: ['VISl', 'VISp']
- `brain_region_idx`: Neuron-to-region mapping per session
- `metadata`: Task description, time bin size (93ms), etc.

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 202 |
| Total trials | 51,992 |
| Total neurons | 29,444 |
| Subjects | 38 |
| Time bin size | 93 ms |
| Timepoints/trial | 78-135 (mean ~87) |

## How to Use

### Verify format
```bash
python train_decoder.py converted_data.pkl --verify-only
```

### Train decoder
```bash
python train_decoder.py converted_data.pkl
```

### Reproduce conversion
```bash
python -u convert_data.py converted_data.pkl --full
```

## Files
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Small sample (2 sessions)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model
