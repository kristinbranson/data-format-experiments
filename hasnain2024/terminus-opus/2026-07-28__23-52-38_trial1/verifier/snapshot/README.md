# Hasnain et al. 2024 - Neural Decoder Dataset

## Dataset Description

This dataset contains electrophysiology recordings from the anterior lateral motor cortex (ALM) of mice performing a two-context behavioral paradigm. The data is from:

> Hasnain, Birnbaum et al., "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience 2024

### Experimental Design
- **Species**: Mus musculus (house mouse)
- **Brain Region**: Anterior Lateral Motor cortex (ALM)
- **Tasks**: 
  - Delayed-Response (DR): auditory cue → delay → go cue → directional lick
  - Water-Cued (WC): water drop presented at random time/location → consume
  - Tasks alternate block-wise within each session
- **Recording**: High-density silicon probes (Neuropixels)

### Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Total sessions | 43 |
| Total subjects | 14 |
| Total neurons | 2,443 |
| Total trials | 11,981 |
| Time bins per trial | 1,001 (5ms bins, -2.5 to 2.5s from go cue) |
| Brain region | ALM |

## How to Load and Use

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data for session 0, trial 0
neural = data['neural'][0][0]  # shape: (n_neurons, n_timepoints)

# Access inputs (time from go cue)
input_data = data['input'][0][0]  # shape: (1, n_timepoints)

# Access outputs
output_data = data['output'][0][0]  # shape: (6, n_timepoints)
```

## Output Format

### Neural Data
- `data['neural']`: List of sessions, each containing list of trials
- Each trial: `(n_neurons, 1001)` array of firing rates (Hz)
- Spikes binned at 5ms, smoothed with causal Gaussian kernel (width=15 bins)

### Inputs
- `data['input']`: Time from go cue onset (seconds)
  - Shape: `(1, 1001)` per trial
  - Range: [-2.5, 2.5]

### Outputs
- `data['output']`: 6 output variables per trial
  - Shape: `(6, 1001)` per trial

| Index | Name | Values | Type |
|-------|------|--------|------|
| 0 | lick_direction | 0=left, 1=right | Per-trial |
| 1 | behavioral_context | 0=WC, 1=DR | Per-trial |
| 2 | outcome | 0=incorrect, 1=correct | Per-trial |
| 3 | tongue_velocity | 0=low (<50th pct), 1=high (>=50th pct) | Time-varying |
| 4 | paw_velocity | 0=low (<50th pct), 1=high (>=50th pct) | Time-varying |
| 5 | motion_energy | 0=low (<50th pct), 1=high (>=50th pct) | Time-varying |

### Metadata
- `data['subjects']`: List of subject IDs
- `data['subject_idx']`: Session-to-subject mapping
- `data['brain_regions']`: ['ALM']
- `data['brain_region_idx']`: Neuron-to-region mapping per session
- `data['metadata']`: Task description, timing info, session details

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset |
| `sample_data.pkl` | 2-session sample for testing |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder model implementation |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `README.md` | This file |

## Reproducing the Conversion

```bash
# Sample conversion (2 sessions)
python -u convert_data.py sample_data.pkl --sample --show-processing

# Full conversion
python -u convert_data.py converted_data.pkl --full

# Verify data format
python -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python -u train_decoder.py converted_data.pkl
```

## Decoder Performance

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| lick_direction | 0.664 | 0.500 |
| behavioral_context | 0.865 | 0.500 |
| outcome | 0.664 | 0.500 |
| tongue_velocity | 0.940 | 0.500 |
| paw_velocity | 0.572 | 0.500 |
| motion_energy | 0.797 | 0.500 |
