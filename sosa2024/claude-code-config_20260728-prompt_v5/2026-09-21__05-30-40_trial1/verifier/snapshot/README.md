# Sosa et al. 2025 - NWB to Decoder Format Conversion

## Dataset
- **Paper**: "A flexible hippocampal population code for experience relative to reward" (Sosa et al. 2025)
- **Source**: DANDI:001361
- **Modality**: 2-photon calcium imaging (CA1 hippocampus)
- **Task**: Virtual reality linear track navigation with hidden reward zone switches

## Files
| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script (NWB -> decoder format) |
| `converted_data.pkl` | Full converted dataset (152 sessions, 9.84 GB) |
| `sample_data.pkl` | Sample dataset (2 sessions, for testing) |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |
| `train_decoder.py` | Decoder training script (provided) |
| `decoder.py` | Decoder model (provided) |

## Quick Start

### Sample conversion (2 sessions)
```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
```

### Full conversion (all 152 sessions)
```bash
python -u convert_data.py converted_data.pkl --full
```

### Verify data format
```bash
python -u train_decoder.py converted_data.pkl --verify-only
```

### Train decoder
```bash
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Data Format

The output pickle contains a dictionary with:

- **`neural`**: List of 152 sessions, each a list of trials. Each trial is `(n_neurons, n_timepoints)` float32 array of OASIS-deconvolved calcium events.
- **`input`**: List of sessions/trials. Each trial is `(4, n_timepoints)` float32 array:
  - `time_from_trial_start` (seconds)
  - `environment_type` (0=ENV1, 1=ENV2)
  - `trial_number` (0-indexed)
  - `previous_trial_outcome` (0=omitted, 1=rewarded)
- **`output`**: List of sessions/trials. Each trial is `(6, n_timepoints)` int64 array:
  - `distance_to_reward_zone` (7 bins)
  - `absolute_position` (5 bins of 90cm)
  - `speed` (5 bins)
  - `lick` (binary)
  - `reward_zone_location` (0=A, 1=B, 2=C)
  - `reward_outcome` (binary)

## Processing Pipeline

1. Load raw fluorescence (F) and neuropil (Fneu) from NWB
2. Neuropil subtraction: `F -= 0.7 * Fneu`
3. Add back neuropil mean per trial
4. Maximin baseline (Gaussian smooth sigma=15, min/max filter window=300)
5. dF/F = (F - baseline) / |baseline|
6. Smooth dF/F with 2-sample Gaussian
7. OASIS deconvolution (tau=0.7)
8. Filter neurons by suite2p iscell flag
9. Extract trial-level data (trial_start to teleport)

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Total neurons | 138,678 |
| Mean neurons/session | 912.4 |
| Total trials | 12,216 |
| Mean trials/session | 80.4 |
| Imaging rate | ~15.5 Hz |
| Track length | 450 cm |
| Reward omission rate | 15.3% |

## Decoder Results (Validation Balanced Accuracy)

| Output | Accuracy | Chance | Ratio |
|--------|----------|--------|-------|
| distance_to_reward_zone | 0.552 | 0.143 | 3.9x |
| absolute_position | 0.647 | 0.200 | 3.2x |
| speed | 0.591 | 0.200 | 3.0x |
| lick | 0.750 | 0.500 | 1.5x |
| reward_zone_location | 0.837 | 0.333 | 2.5x |
| reward_outcome | 0.580 | 0.500 | 1.2x |
