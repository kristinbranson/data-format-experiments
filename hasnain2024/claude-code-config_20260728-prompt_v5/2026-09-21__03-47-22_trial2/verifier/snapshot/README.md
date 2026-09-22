# Neural Decoder Dataset: Hasnain, Birnbaum et al. (2025)

Converted from: **"Separating cognitive and motor processes in the behaving mouse"**
(Hasnain, Birnbaum et al., Nature Neuroscience 2025)

## Dataset Overview

| Statistic | Value |
|-----------|-------|
| Sessions | 44 (25 fixed-delay + 19 randomized-delay) |
| Subjects | 14 mice |
| Total neurons | 3,066 (ALM) |
| Total trials | 13,823 |
| Time bins | 500 (10 ms each, -2.5 to +2.5 s around go cue) |
| Neural data | Smoothed firing rates (Hz), causal Gaussian kernel |

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full dataset (44 sessions, ~2.1 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions, ~92 MB) |
| `convert_data.py` | Conversion script |
| `CONVERSION_NOTES.md` | Detailed conversion notes and statistics |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder module |
| `train_decoder_full_out.txt` | Full training output log |
| `verification_full_out.txt` | Data format verification output |
| `train_stats.json` | Training statistics (JSON) |

## Data Format

The pickle file contains a dictionary with:

- **`neural`**: List of 44 sessions, each a list of per-trial arrays `(n_neurons, 500)`
- **`input`**: List of sessions, each a list of `(1, 500)` arrays (time from go cue in seconds)
- **`output`**: List of sessions, each a list of `(6, 500)` integer arrays

### Output Dimensions

| Dim | Name | Values | Type |
|-----|------|--------|------|
| 0 | lick_direction | 0=left, 1=right, 2=none | Per-trial |
| 1 | behavioral_context | 0=water-cued (WC), 1=delayed-response (DR) | Per-trial |
| 2 | outcome | 0=incorrect, 1=correct, 2=ignore | Per-trial |
| 3 | tongue_velocity | 0=low, 1=high, 2=not_visible | Time-varying |
| 4 | paw_velocity | 0=low, 1=high, 2=not_visible | Time-varying |
| 5 | motion_energy | 0=low, 1=high, 2=no_video | Time-varying |

### Additional Fields

- `subjects`: List of 14 subject IDs
- `subject_idx`: Session-to-subject mapping
- `brain_regions`: `['ALM']`
- `brain_region_idx`: Per-session neuron-to-region mapping
- `input_names`, `output_names`, `output_values`: Metadata
- `metadata`: Processing parameters and source information

## Processing Pipeline

1. Load MATLAB data files (HDF5 and legacy .mat formats)
2. Filter trials: exclude early lick, stimulation, NaN goCue
3. Align spikes to go cue onset
4. Bin at 10 ms, smooth with 15-sample causal Gaussian kernel
5. Filter neurons: exclude garbage/noisy quality, require mean FR > 1 Hz
6. Require minimum 10 units per session
7. Extract behavioral outputs (lick direction, context, outcome)
8. Align video tracking (DLC) and motion energy to neural time bins
9. Discretize continuous variables at per-session 50th percentile

## Decoder Results

| Output | Validation Balanced Accuracy | Chance |
|--------|------------------------------|--------|
| lick_direction | 0.640 | 0.333 |
| behavioral_context | 0.865 | 0.500 |
| outcome | 0.611 | 0.333 |
| tongue_velocity | 0.570 | 0.333 |
| paw_velocity | 0.528 | 0.333 |
| motion_energy | 0.792 | 0.333 |

## Usage

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples

# Re-run conversion
python convert_data.py converted_data.pkl --full
python convert_data.py sample_data.pkl --sample
```

## Known Data Issues

- **Sessions 36, 43** (JEB24): Last ~30 trials have all-zero neural data (recording ended early)
- **Sessions 11, 13, 41**: No motion energy video data available (100% `no_video`)
- **JEB23_2023-10-20.mat**: Excluded as duplicate of JEB23_2023-10-19.mat
