# Track2p Neural Decoder Dataset

## Dataset Description

This dataset contains converted neural and behavioral data from Majnik et al. 2025 (eLife), "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p".

The data consists of two-photon calcium imaging recordings from mouse barrel cortex (layer 2/3) during the second postnatal week (P7-P14), with simultaneously recorded motion energy from videography of spontaneous behavior.

### Key Features
- **6 mice** (jm031-jm046, referred to as mice A-F in the paper)
- **41 sessions** (6-7 daily recording sessions per mouse)
- **2998 unique tracked neurons** across all mice (221-746 per mouse)
- **545 trials** (2-minute blocks)
- **Brain region**: Barrel cortex, layer 2/3

## How to Load the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access decoder input (time elapsed) for session 0, trial 0
input_data = data['input'][0][0]  # shape: (1, n_timepoints)

# Access decoder output (motion energy bins) for session 0, trial 0
output_data = data['output'][0][0]  # shape: (1, n_timepoints)
```

## Data Format

### Neural Data
- **Type**: Baseline-corrected fluorescence (dF/F) computed using Suite2p defaults
- **Processing**: F - 0.7*Fneu (neuropil correction), then maximin baseline subtraction
- **Temporal binning**: Averaged in bins of 10 consecutive frames (30 Hz -> ~3.33 Hz)
- **Shape per trial**: (n_neurons, 360) where 360 = 2 minutes / 333.33ms

### Decoder Input
- **time_elapsed_s**: Time elapsed from the start of the recording session, in seconds
- **Shape per trial**: (1, 360)

### Decoder Output
- **motion_energy_bin**: Motion energy discretized into 5 equal-percentile bins (0-4)
  - Bin 0: Lowest motion energy (least movement)
  - Bin 4: Highest motion energy (most movement)
- **Normalization**: Per-session min-max normalization before percentile binning
- **Shape per trial**: (1, 360)

### Metadata
- `time_bin_size`: 333.33 ms
- `temporal_alignment_event`: Start of recording session
- `brain_regions`: ['barrel_cortex_L2/3']

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions | 41 |
| Trials | 545 |
| Mean neurons/mouse | 499.7 (paper: 526 ± 190) |
| Time bins per trial | 360 |
| Time bin size | 333.33 ms |
| Output classes | 5 (equal-percentile bins) |

## Decoder Performance

| Output | Training Acc | Validation Acc | Chance |
|--------|-------------|---------------|--------|
| motion_energy_bin | 0.6242 | 0.3008 | 0.2000 |

## Running the Decoder

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl

# Train with sample data
python train_decoder.py sample_data.pkl
```

## Reproducing the Conversion

```bash
# Full conversion
python -u convert_data.py converted_data.pkl --full

# Sample conversion (2 sessions)
python -u convert_data.py sample_data.pkl --sample

# With processing visualizations
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Reference

Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel JC, Cossart R (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540. https://doi.org/10.7554/eLife.107540.1
