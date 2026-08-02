# Neural Data Conversion: Hasnain, Birnbaum et al. (2024)

Conversion pipeline for ALM neural recordings from "Separating cognitive and motor processes in the behaving mouse" (Nature Neuroscience 2024).

## Quick Start

```bash
# Convert all sessions (generates converted_data.pkl and sample_data.pkl)
python convert_data.py

# Convert sample only (3 sessions)
python convert_data.py --sample

# Verify converted data format
python -c "from convert_data import verify_data_format; import pickle; verify_data_format(pickle.load(open('converted_data.pkl','rb')))"

# Train decoder on converted data
python train_decoder.py --data_path converted_data.pkl
python train_decoder.py --data_path sample_data.pkl
```

## Source Data

- **Directory**: `data/Ephys_Behavior/` (25 sessions) and `data/RandomizedDelay_Ephys_Behavior/` (22 sessions, 2 skipped as behavior-only)
- **Format**: MATLAB .mat files (v7.3/HDF5 and v5/v7)
- **Task**: Delayed-response (DR) and water-cued (WC) directional licking

## Processing Pipeline

1. **Spike alignment** to go cue onset, time window [-2.5, +2.5] s
2. **Binning** at 5 ms (dt = 1/200 s), giving 1000 time bins per trial
3. **Causal Gaussian smoothing** (kernel width N=15 samples, half-kernel zeroed)
4. **Unit filtering**: exclude garbage/noisy units and units with mean FR < 0.5 Hz
5. **Probe filtering**: ALM probes only
6. **Trial filtering**: exclude optogenetic, early lick, and non-responding trials
7. **Kinematic extraction**: tongue velocity, paw velocity, motion energy interpolated to neural time axis
8. **Discretization**: continuous outputs binned at 50th percentile per session

## Output Format

`converted_data.pkl` contains a dictionary with:

| Key | Shape | Description |
|-----|-------|-------------|
| `neural_data` | (trials, time, neurons) | Smoothed firing rates |
| `input_data` | (trials, time, 1) | Time from go cue (s) |
| `output_data` | (trials, time, 6) | Binary labels (see below) |
| `neural_locations` | (neurons, 2) | [session_idx, unit_idx] |
| `output_labels` | (6,) | Output variable names |
| `session_ids` | (trials,) | Session index per trial |

**Output channels** (axis 2 of `output_data`):
0. Lick direction (R=1, L=0)
1. Behavioral context (DR=1, WC=0)
2. Outcome (correct=1, incorrect=0)
3. Tongue velocity (high=1, low=0)
4. Paw velocity (high=1, low=0)
5. Motion energy (high=1, low=0)

## Dataset Summary

| Metric | Full | Sample |
|--------|------|--------|
| Sessions | 45 | 3 |
| Trials | 12,293 | 888 |
| Neurons | 2,579 | 218 |
| Time bins | 1,000 | 1,000 |

## Decoder Validation (Full Dataset)

| Output | Balanced Accuracy |
|--------|-------------------|
| Lick direction | 0.633 |
| Behavioral context | 0.838 |
| Outcome | 0.653 |
| Tongue velocity | 0.995* |
| Paw velocity | 0.718 |
| Motion energy | 0.787 |

*Tongue velocity is degenerate (tongue invisible ~90% of time, threshold = 0).

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset |
| `sample_data.pkl` | 3-session sample dataset |
| `CONVERSION_NOTES.md` | Detailed processing decisions and known issues |
| `conversion_full_out.txt` | Full conversion log |
| `conversion_sample_out.txt` | Sample conversion log |
| `verification_full_out.txt` | Full data verification log |
| `verification_sample_out.txt` | Sample data verification log |
| `train_decoder_full_out.txt` | Full dataset decoder results |
| `train_decoder_sample_out.txt` | Sample dataset decoder results |

## Dependencies

- numpy, scipy, h5py, scikit-learn, pickle
