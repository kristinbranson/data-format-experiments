# Track2p Barrel Cortex Neural Decoder Dataset

## Dataset Description

Longitudinal two-photon calcium imaging from mouse barrel cortex (layer 2/3) during the second postnatal week (P7-P14). Neural activity from neurons tracked across all recording days is used to decode motion energy (behavioral state proxy).

**Source**: Majnik et al. 2025 - "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p" (eLife)

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 mice |
| Sessions | 41 total (6-7 per mouse) |
| Trials | 545 (2-minute blocks) |
| Neurons per mouse | 221-746 (mean ~500) |
| Time bin | 333.3 ms (10 frames at 30 Hz) |
| Bins per trial | 360 |
| Brain region | Barrel cortex |

## Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, 360)

# Access inputs (time elapsed in seconds)
time_input = data['input'][0][0]  # shape: (1, 360)

# Access outputs (motion energy bin 0-4)
me_bin = data['output'][0][0]  # shape: (1, 360)
```

## Data Format

- `neural`: dF/F traces (baseline-corrected fluorescence), binned in 10-frame averages
- `input`: Time elapsed from start of 2-minute trial block (seconds)
- `output`: Motion energy discretized into 5 equal-percentile bins (quintiles)
- `subjects`: ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
- `brain_regions`: ['barrel_cortex']

## Processing Pipeline

1. **Neuropil correction**: Fc = F - 0.7 * Fneu
2. **Baseline estimation**: Suite2p maximin method (win=60s, sig=10 frames)
3. **dF/F**: Fc - F0 (baseline subtraction)
4. **Missing frame interpolation**: Video frames interpolated to match neural frames
5. **Binning**: Average of 10 consecutive frames (~333 ms bins)
6. **Trial segmentation**: 2-minute blocks (360 bins each)
7. **Discretization**: Motion energy into 5 global quintile bins

## Decoder Performance

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| motion_energy_bin | 0.5975 | 0.4681 | 0.2000 |

## Files

- `converted_data.pkl` - Full converted dataset
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
