# ALM Electrophysiology Decoder Dataset

Converted dataset from Hasnain, Birnbaum et al, "Separating cognitive and motor processes in the behaving mouse", Nature Neuroscience 2024.

## Source Data
- **DOI**: 10.5281/zenodo.13941415
- **Species**: Mus musculus
- **Brain region**: Anterior lateral motor cortex (ALM)
- **Tasks**: Delayed-response (DR) and water-cued (WC) licking paradigms, including randomized delay variants

## Dataset Summary

| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 DR + 19 randomized delay) |
| Subjects | 14 mice |
| Total neurons | 2,457 |
| Total trials | 11,985 |
| Time bins per trial | 500 (-2.5 to +2.5 s from go cue, 10 ms bins) |
| Brain region | ALM |

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full dataset (44 sessions, ~1.5 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |

## Data Format

The pickle file contains a dictionary with:

| Key | Shape/Type | Description |
|-----|------------|-------------|
| `neural` | list of 44 lists, each: list of (n_neurons, 500) arrays | Firing rates (spks/sec) per trial |
| `input` | list of 44 arrays, each: (n_trials, 500, 1) | Time from go cue (seconds) |
| `output` | list of 44 arrays, each: (n_trials, 500, 6) | Binary behavioral outputs |
| `subjects` | list of 14 strings | Subject IDs |
| `subject_idx` | list of 44 ints | Subject index per session |
| `brain_regions` | `['ALM']` | Brain region labels |
| `brain_region_idx` | list of 44 lists of ints | Brain region index per neuron |
| `input_names` | `['time_from_gocue']` | Input variable names |
| `output_names` | list of 6 strings | Output variable names |
| `output_values` | list of 6 lists | Class labels per output |
| `metadata` | dict | Processing parameters and session info |

## Output Variables

| Index | Name | Values | Description |
|-------|------|--------|-------------|
| 0 | lick_direction | left (0), right (1) | Trial lick direction |
| 1 | context | WC (0), DR (1) | Behavioral context |
| 2 | outcome | incorrect (0), correct (1) | Trial outcome |
| 3 | tongue_velocity | low (0), high (1) | Tongue speed (50th percentile threshold) |
| 4 | paw_velocity | low (0), high (1) | Paw speed (50th percentile threshold) |
| 5 | motion_energy | low (0), high (1) | Motion energy (50th percentile threshold) |

## Processing Pipeline

1. Load MATLAB .mat files (v5 and v7.3 formats)
2. Filter clusters by quality (exclude garbage, gabrga, noisy, real?)
3. Align spike times to go cue onset
4. Bin spikes at 10 ms resolution (-2.5 to +2.5 s)
5. Smooth with causal Gaussian kernel (width=15 bins, reflect boundary)
6. Remove neurons with mean firing rate < 1 Hz
7. Exclude trials: early lick, stimulation, no-response
8. Compute tongue/paw velocity from DLC tracking data
9. Align motion energy with video-neural offset correction
10. Discretize continuous outputs at per-session 50th percentile

## Decoder Results

All outputs decoded above chance (0.5) using a neural network decoder:

| Output | Validation Balanced Accuracy |
|--------|------------------------------|
| lick_direction | 0.680 |
| context | 0.876 |
| outcome | 0.672 |
| tongue_velocity | 0.806 |
| paw_velocity | 0.563 |
| motion_energy | 0.768 |

## Usage

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access session 0, trial 5 neural data
neural_trial = data['neural'][0][5]  # shape: (n_neurons, 500)

# Access session 0 outputs
outputs = data['output'][0]  # shape: (n_trials, 500, 6)
lick_dir = outputs[:, :, 0]  # lick direction for all trials
```

## Regeneration

```bash
# Full conversion
python convert_data.py converted_data.pkl --full

# Sample (2 sessions)
python convert_data.py sample_data.pkl --sample
```
