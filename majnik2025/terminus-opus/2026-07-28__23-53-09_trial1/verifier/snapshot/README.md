# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains longitudinal calcium imaging recordings from developing mouse barrel cortex (S1BF), converted from the Track2p paper (Majnik et al. 2025) into a format suitable for neural decoding.

### Source
- **Paper**: Majnik et al. 2025, "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p", eLife
- **Data**: 2-photon calcium imaging with simultaneous videography of spontaneous behavior

### Subjects
- 6 mice (jm031-jm046, labeled A-F in paper)
- Recordings from postnatal days P8-P14 (barrel cortex development)
- 41 total sessions (6-7 per mouse)

### Neural Data
- **Type**: dF/F (baseline-subtracted, neuropil-corrected fluorescence)
- **Processing**: F - 0.7*Fneu, then Suite2p maximin baseline subtraction
- **Binning**: 10-frame averaging (333.33 ms bins, from 30 Hz imaging)
- **Neurons**: 221-746 tracked neurons per mouse (same cells across all sessions)
- **Total unique neurons**: 2998

### Decoder Task
- **Input**: Time elapsed from session start (seconds)
- **Output**: Motion energy (from videography), normalized and discretized into 5 equal-percentile bins per session

### Trial Structure
- Each session split into 2-minute blocks (matching paper's CV structure)
- 10 trials per 20-min session, 15 trials per 30-min session
- 360 timepoints per trial (at 333.33 ms bins)
- 545 total trials

## How to Load

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 360)

# Access decoder input (time)
input_data = data['input'][0][0]  # shape: (1, 360)

# Access decoder output (motion energy bin)
output_data = data['output'][0][0]  # shape: (1, 360), values 0-4

# Subject and brain region info
print(data['subjects'])  # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
print(data['brain_regions'])  # ['S1BF']
```

## Output Format

See the data dictionary structure in the task description. Key fields:
- `neural`: List of sessions, each containing list of trials (n_neurons × n_timepoints)
- `input`: Time elapsed (1 × n_timepoints per trial)
- `output`: Motion energy bin (1 × n_timepoints per trial, values 0-4)
- `subjects`: ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
- `brain_regions`: ['S1BF']
- `metadata`: Processing parameters and descriptions

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions | 41 |
| Total trials | 545 |
| Neurons/subject | 221-746 |
| Total unique neurons | 2998 |
| Timepoints/trial | 360 |
| Time bin size | 333.33 ms |
| Output bins | 5 (equal percentile) |
| Output distribution | 20% per bin |

## Validation

Run `python train_decoder.py converted_data.pkl --verify-only` to validate the data format.
Run `python train_decoder.py converted_data.pkl` to train and evaluate the decoder.

## Files

- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample dataset (2 sessions, 8 MB)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
