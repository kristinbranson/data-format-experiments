# Track2p Barrel Cortex - Neural Decoder Dataset

## Dataset Description
This dataset contains longitudinal 2-photon calcium imaging data from mouse barrel cortex (S1BF) during the second postnatal week of development (P7-P14). The data comes from the Track2p paper (Majnik et al. 2025, eLife), which tracked the same neurons across multiple recording days using a novel cell tracking algorithm.

**Task**: Decode motion energy (animal movement) from neural activity in barrel cortex.

**Source**: Majnik et al. (2025). "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p." eLife 14:RP107540.

## Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice (jm031-jm046) |
| Sessions | 41 total (6-7 per mouse) |
| Neurons per mouse | 221-746 (mean ~500) |
| Trials per session | 20 (20-min sessions) or 30 (30-min sessions) |
| Trial duration | 60 seconds |
| Time bin size | 333.33 ms (10 frames at 30 Hz) |
| Timepoints per trial | 180 |
| Brain region | S1BF (barrel cortex) |

## How to Load the Data

```python
import pickle
import numpy as np

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 180)

# Access input (time elapsed in seconds)
time_input = data['input'][0][0]  # shape: (1, 180)

# Access output (motion energy bin, 0-4)
me_bin = data['output'][0][0]  # shape: (1, 180)

# Subject info
print(data['subjects'])           # ['jm031', 'jm032', ...]
print(data['subject_idx'])        # array of subject indices per session
print(data['brain_regions'])      # ['S1BF']
```

## Data Format
```python
data = {
    'neural': list[list[np.ndarray]],     # [sessions][trials] -> (n_neurons, 180)
    'input': list[list[np.ndarray]],      # [sessions][trials] -> (1, 180) time in seconds
    'output': list[list[np.ndarray]],     # [sessions][trials] -> (1, 180) motion energy bin (0-4)
    'subjects': list[str],                # 6 mouse IDs
    'subject_idx': np.ndarray,            # session -> subject mapping
    'brain_regions': ['S1BF'],            # barrel cortex
    'brain_region_idx': list[np.ndarray], # all neurons -> region 0
    'input_names': ['time_elapsed_s'],
    'output_names': ['motion_energy_bin'],
    'output_values': [['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']],
    'metadata': dict                      # processing details
}
```

## Processing Pipeline
1. **Neural data**: dF/F computed from raw fluorescence (F.npy) and neuropil (Fneu.npy) using Suite2p default baseline correction (neuropil coefficient 0.7, maximin baseline with 60s window)
2. **Binning**: Both neural and behavioral data averaged in bins of 10 frames (30 Hz -> 3 Hz)
3. **Motion energy**: Interpolated for missing camera frames, then discretized into 5 equal-percentile bins per session
4. **Trials**: Sessions split into 60-second non-overlapping trials

## Running the Decoder
```bash
python train_decoder.py converted_data.pkl
```

## Files
- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - 2-session sample for testing
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script
