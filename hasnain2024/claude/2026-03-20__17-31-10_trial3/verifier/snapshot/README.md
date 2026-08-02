# Neural Data Conversion: Hasnain, Birnbaum et al. (Nature Neuroscience 2024)

Converts electrophysiology and behavioral data from "Separating cognitive and motor processes in the behaving mouse" into a decoder-compatible Python pickle format.

## Source Data
- **Paper**: Hasnain, Birnbaum et al., Nature Neuroscience 2024
- **Data DOI**: 10.5281/zenodo.13941415
- **Species**: Mouse (mus musculus)
- **Brain region**: Anterior lateral motor cortex (ALM)
- **Tasks**: Delayed-response (fixed and randomized delay) and water-cued licking

## Converted Dataset Summary

| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 fixed delay + 19 randomized delay) |
| Subjects | 14 |
| Total neurons | 2457 |
| Total trials | 13762 |
| Time bins | 500 (10ms bins, -2.5 to 2.5s from go cue) |
| File size | ~1.8 GB |

## Output Format

The pickle file contains a dictionary with:
- `neural`: List of 44 sessions, each a list of trials, each a numpy array (n_neurons, 500)
- `input`: List of 44 sessions, each a list of trials, each (1, 500) array of time from go cue
- `output`: List of 44 sessions, each a list of trials, each (6, 500) int64 array
- `subjects`: List of subject names
- `subject_idx`: List mapping session index to subject
- `brain_regions`: List of brain region names
- `brain_region_idx`: List mapping session index to brain region
- `input_names`: `['time_from_go_cue']`
- `output_names`: `['lick_direction', 'behavioral_context', 'outcome', 'tongue_velocity', 'paw_velocity', 'motion_energy']`
- `output_values`: Human-readable class labels

### Output Variables
| Index | Name | Values | Type |
|-------|------|--------|------|
| 0 | lick_direction | 0=left, 1=right | Per-trial |
| 1 | behavioral_context | 0=water-cued, 1=delayed-response | Per-trial |
| 2 | outcome | 0=incorrect, 1=correct | Per-trial |
| 3 | tongue_velocity | 0=low, 1=high (session median split) | Time-varying |
| 4 | paw_velocity | 0=low, 1=high (session median split) | Time-varying |
| 5 | motion_energy | 0=low, 1=high (session median split) | Time-varying |

## Processing Pipeline

1. Load MATLAB .mat files (HDF5 v7.3 and MATLAB v5 formats)
2. Filter trials: exclude optogenetic stimulation and early lick trials
3. Filter neurons by quality (exclude garbage, noisy, real?, gabrga)
4. Bin spikes into 10ms bins, compute firing rates
5. Apply causal Gaussian smoothing (N=15, reflect boundary)
6. Remove neurons with mean firing rate <= 1 Hz
7. Extract DLC kinematics (tongue and paw velocity)
8. Load and interpolate motion energy to neural time axis
9. Discretize continuous outputs by per-session 50th percentile
10. Align all data to go cue onset

## Usage

### Convert data
```bash
# Full conversion (44 sessions)
python3 -u convert_data.py converted_data.pkl --full

# Sample conversion (2 sessions)
python3 -u convert_data.py sample_data.pkl --sample
```

### Verify and train decoder
```bash
# Verify format only
python3 -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Files
- `convert_data.py` - Main conversion script
- `train_decoder.py` - Decoder training and verification script
- `decoder.py` - Decoder model definition
- `converted_data.pkl` - Full converted dataset
- `CONVERSION_NOTES.md` - Detailed conversion decisions and statistics
- `code/` - Original MATLAB reference code
- `data/` - Source data files

## Decoder Results

| Output | Validation Balanced Acc | Chance |
|--------|------------------------|--------|
| lick_direction | 0.646 | 0.50 |
| behavioral_context | 0.840 | 0.50 |
| outcome | 0.691 | 0.50 |
| tongue_velocity | 0.773 | 0.50 |
| paw_velocity | 0.557 | 0.50 |
| motion_energy | 0.756 | 0.50 |

All outputs above chance with small train/validation gaps (no overfitting).
