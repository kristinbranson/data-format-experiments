# Track2p Neural Decoder Dataset

## Dataset Description

Converted calcium imaging data from **Majnik et al. 2025** ("Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p") for neural decoding of motion energy from barrel cortex activity.

- **Source**: 6 mice, 41 sessions (6-7 daily recordings per mouse, P7-P14)
- **Brain region**: Barrel cortex (layer 2/3)
- **Neural signal**: dF/F (baseline-corrected calcium fluorescence, Suite2p defaults)
- **Behavior**: Motion energy from videography, discretized into 5 equal-percentile bins per session
- **Time bin**: 333.33 ms (10 frames at 30 Hz, as in paper's decoding analysis)
- **Trial duration**: 60 seconds (180 time bins)

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions | 41 |
| Total trials | 1090 |
| Neurons per subject | 221, 370, 685, 746, 541, 435 |
| Time bins per trial | 180 |
| Output classes | 5 (ME quintile bins) |

## Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 180)

# Access decoder input (time in seconds)
time_input = data['input'][0][0]  # shape: (1, 180)

# Access decoder output (ME bin)
output = data['output'][0][0]  # shape: (1, 180), values 0-4

# Subject and brain region info
subjects = data['subjects']  # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
subject_idx = data['subject_idx']  # index into subjects for each session
```

## Output Format

See `CONVERSION_NOTES.md` for full details on the conversion process, decisions, and validation results.

## Decoder Training

```bash
python train_decoder.py converted_data.pkl
```

Validation balanced accuracy: ~0.31 (chance: 0.20).

## Files

- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample dataset (2 sessions)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion notes
- `train_decoder.py` - Decoder training script
