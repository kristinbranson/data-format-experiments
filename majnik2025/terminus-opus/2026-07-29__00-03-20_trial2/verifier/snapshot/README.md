# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains longitudinal two-photon calcium imaging recordings from mouse barrel cortex (S1BF) during the second postnatal week of development (P7-P14). Data is from the Track2p paper:

> Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel JC, Cossart R (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540.

### Experimental Setup
- **Species**: Mouse (GAD67-Cre, GCaMP8m + tdTomato)
- **Brain Region**: Barrel cortex (S1BF), Layer 2/3
- **Recording**: Two-photon calcium imaging at 30 Hz
- **FOV**: 720 × 720 μm, 512 × 512 pixels
- **Subjects**: 6 mice (jm031-jm046)
- **Sessions**: 6-7 daily recordings per mouse (41 total)
- **Session Duration**: 20-30 minutes
- **Behavior**: Spontaneous movement on treadmill, recorded via videography

### Decoder Task
**Decode motion energy from neural activity.**
- **Input**: Time elapsed from start of recording (seconds)
- **Output**: Motion energy discretized into 5 equal-percentile bins
- **Neural**: dF/F calcium traces (neuropil-corrected, baseline-corrected)

## Data Format

The converted dataset (`converted_data.pkl`) is a Python dictionary:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Keys
- `neural`: List of 41 sessions, each containing 10-15 trials of shape (n_neurons, 360)
- `input`: List of 41 sessions, each containing trials of shape (1, 360) - time in seconds
- `output`: List of 41 sessions, each containing trials of shape (1, 360) - ME bin (0-4)
- `subjects`: ['jm031', 'jm032', 'jm038', 'jm039', 'jm040', 'jm046']
- `subject_idx`: Array of shape (41,) mapping sessions to subjects
- `brain_regions`: ['S1BF']
- `brain_region_idx`: List of arrays, all zeros (single brain region)
- `input_names`: ['time_seconds']
- `output_names`: ['motion_energy']
- `output_values`: [['bin0_lowest', 'bin1', 'bin2', 'bin3', 'bin4_highest']]
- `metadata`: Dictionary with processing parameters and session info

### Key Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions | 41 |
| Total trials | 545 |
| Neurons per mouse | 221-746 (avg 500) |
| Timepoints per trial | 360 (2 min at 3 Hz) |
| Time bin size | 333.33 ms |
| Output classes | 5 (equal percentile bins) |

### Processing Pipeline
1. **Neuropil correction**: Fc = F - 0.7 × Fneu
2. **Baseline correction**: Maximin filter (Suite2p default)
3. **dF/F**: (Fc - F0) / F0
4. **Temporal binning**: Average 10 consecutive frames (30 Hz → 3 Hz)
5. **Trial splitting**: 2-minute blocks
6. **Motion energy**: Interpolated for missing frames, binned, discretized into 5 equal-percentile bins

## Files
- `converted_data.pkl` - Full converted dataset (395 MB)
- `sample_data.pkl` - Sample dataset (2 sessions, 6 MB)
- `convert_data.py` - Conversion script
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder library
- `CONVERSION_NOTES.md` - Detailed conversion notes

## Usage

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples

# Re-convert data
python -u convert_data.py converted_data.pkl --full
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Decoder Performance
| Output | Training Acc | Validation Acc | Chance |
|--------|-------------|----------------|--------|
| motion_energy | 0.7193 | 0.4633 | 0.2000 |
