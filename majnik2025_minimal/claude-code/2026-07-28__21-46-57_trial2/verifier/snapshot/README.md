# Neural Decoder: Track2p Barrel Cortex Data

## Overview

This project converts longitudinal two-photon calcium imaging data from developing mouse barrel cortex (Majnik et al. 2025, "Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p") into a standardized format for neural decoding.

## Task

**Decode motion energy from barrel cortex neural activity** during spontaneous behavior in developing mice (postnatal days P7-P14).

- **Input**: Time elapsed from recording onset (seconds)
- **Output**: Motion energy, discretized into 5 equal-percentile bins
- **Neural data**: dF/F calcium traces from tracked neurons in barrel cortex layer 2/3

## Data

- 6 mice (jm031-jm046), 6-7 daily recording sessions each (41 sessions total)
- 221-746 tracked neurons per mouse (mean ~500)
- Imaging at 30 Hz, sessions of 20 or 30 minutes
- Data binned in 10-frame bins (~333 ms) as described in the paper's decoding analysis

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset (41 sessions, 545 trials) |
| `sample_data.pkl` | Sample dataset (6 sessions from jm031) |
| `train_decoder.py` | Decoder training and validation script |
| `decoder.py` | Decoder model and utilities |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |
| `conversion_full_out.txt` | Conversion script output (full dataset) |
| `conversion_sample_out.txt` | Conversion script output (sample dataset) |
| `verification_full_out.txt` | Verification output (full dataset) |
| `verification_sample_out.txt` | Verification output (sample dataset) |
| `train_decoder_full_out.txt` | Decoder training output (full dataset) |
| `train_decoder_sample_out.txt` | Decoder training output (sample dataset) |

## Usage

```bash
# Convert data
python convert_data.py

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu

# Train with sample data
python train_decoder.py sample_data.pkl --cpu
```

## Reference

Majnik Jure, Mantez Manon, Zangila Sofia, Bugeon Stephane, Guignard Leo, Platel Jean-Claude, Cossart Rosa (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540. https://doi.org/10.7554/eLife.107540.1
