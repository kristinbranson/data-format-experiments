# Hasnain, Birnbaum et al (Nature Neuroscience 2024) - Decoder-Ready Dataset

## Overview

Converted dataset from "Separating cognitive and motor processes in the behaving mouse" (Hasnain, Birnbaum et al, Nature Neuroscience 2024). Contains ALM electrophysiology recordings aligned to behavioral and kinematic variables, formatted for neural decoder training.

## Dataset Statistics

| Property | Value |
|----------|-------|
| Sessions | 44 |
| Subjects | 14 |
| Total neurons | 2,456 |
| Total trials | 13,762 |
| Time bins | 500 (10 ms each) |
| Time window | [-2.5, 2.5] s from go cue |
| Brain region | ALM |

## Output Variables

| Index | Name | Values | Type |
|-------|------|--------|------|
| 0 | lick_direction | left (0), right (1), none (2) | Per-trial |
| 1 | behavioral_context | WC (0), DR (1) | Per-trial |
| 2 | outcome | incorrect (0), correct (1), ignore (2) | Per-trial |
| 3 | tongue_velocity | below_p50 (0), above_p50 (1), not_visible (2) | Time-varying |
| 4 | paw_velocity | below_p50 (0), above_p50 (1), not_visible (2) | Time-varying |
| 5 | motion_energy | below_p50 (0), above_p50 (1), no_video (2) | Time-varying |

## Processing Pipeline

1. Load spike data from MATLAB .mat files (v5 and v7.3 formats)
2. Select ALM probe(s) per session based on reference loading scripts
3. Quality filter: exclude garbage, gabrga, noisy, real? units
4. Align spike times to go cue onset
5. Bin spikes at 10 ms resolution into [-2.5, 2.5] s window
6. Smooth with causal Gaussian kernel (window=15 bins, reflect boundary)
7. Remove low firing rate units (mean FR < 1 Hz)
8. Exclude stim-enabled and early-lick trials
9. Exclude trials beyond ephys recording range (recording ended early)
10. Extract behavioral labels (lick direction, context, outcome)
11. Compute tongue/paw velocity from DLC trajectories, discretize at session median
12. Align and discretize motion energy at session median

## Usage

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --plot-samples

# Run conversion from raw data
python convert_data.py converted_data.pkl --full
```

## Decoder Performance (Validation Balanced Accuracy)

| Output | Balanced Accuracy | Chance (1/K) |
|--------|------------------|-------------|
| lick_direction | 0.632 | 0.333 |
| behavioral_context | 0.850 | 0.500 |
| outcome | 0.617 | 0.333 |
| tongue_velocity | 0.550 | 0.333 |
| paw_velocity | 0.649 | 0.333 |
| motion_energy | 0.772 | 0.500 |

## Files

- `converted_data.pkl` - Converted dataset (pickle format)
- `convert_data.py` - Conversion script
- `CONVERSION_NOTES.md` - Detailed conversion documentation
- `train_decoder.py` - Decoder training script (provided)
- `decoder.py` - Decoder model (provided)

## Reference

Hasnain, Birnbaum et al. "Separating cognitive and motor processes in the behaving mouse." Nature Neuroscience (2024).
