# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains longitudinal calcium imaging data from developing mouse barrel cortex, converted from the Track2p format to a decoder-compatible format for predicting motion energy from neural activity.

**Source**: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p" (eLife 14:RP107540)

**Task**: Decode motion energy (discretized into 5 equal-percentile bins) from neural activity (dF/F) recorded in barrel cortex of developing mice.

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice (jm031-jm046) |
| Sessions | 41 total (6-7 per mouse) |
| Trials | 1090 total (20 or 30 per session) |
| Neurons per mouse | 221-746 (mean 499) |
| Trial duration | 60 seconds |
| Time bin size | 333.33 ms (10 frames at 30 Hz) |
| Timepoints per trial | 180 |
| Brain region | Barrel cortex |
| Output bins | 5 (equal-percentile per session) |

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 180)

# Access input (time in seconds) for session 0, trial 0
input_data = data['input'][0][0]  # shape: (1, 180)

# Access output (discretized motion energy) for session 0, trial 0
output_data = data['output'][0][0]  # shape: (1, 180), values 0-4

# Get subject info
print(data['subjects'])  # ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
print(data['subject_idx'])  # array of subject indices for each session
```

## Output Format

```python
data = {
    'neural': list of sessions, each containing list of trials (n_neurons x 180)
    'input': list of sessions, each containing list of trials (1 x 180) - time in seconds
    'output': list of sessions, each containing list of trials (1 x 180) - motion energy bin (0-4)
    'subjects': ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
    'subject_idx': array of subject indices for each session
    'brain_regions': ['barrel_cortex']
    'brain_region_idx': list of arrays (one per session)
    'input_names': ['time_seconds']
    'output_names': ['motion_energy']
    'output_values': [['bin_0 (lowest)', 'bin_1', 'bin_2', 'bin_3', 'bin_4 (highest)']]
    'metadata': dict with task description, time bin size, etc.
}
```

## Processing Pipeline

1. **Neural data**: Raw fluorescence (F) with neuropil subtraction (F - 0.7*Fneu), baseline corrected using Suite2p's maximin method
2. **Binning**: Both neural and behavioral data averaged in 10-frame bins (333.33 ms)
3. **Trial splitting**: Sessions split into 60-second trials
4. **Output discretization**: Motion energy discretized into 5 equal-percentile bins per session
5. **Missing frames**: Camera frame mismatches handled via interpolation

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train and evaluate decoder
python train_decoder.py converted_data.pkl --plot-samples
```

## Files

- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample dataset (2 sessions, 17 MB)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder library
- `CONVERSION_NOTES.md` - Detailed conversion notes and decisions
- `cache/` - Intermediate files and analysis outputs
