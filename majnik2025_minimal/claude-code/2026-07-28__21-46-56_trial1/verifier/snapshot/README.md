# Neural Decoder: Track2p Barrel Cortex Motion Energy

## Overview

This project converts longitudinal calcium imaging data from Majnik et al. 2025 (Track2p) into a standardized format for neural decoding. The decoder predicts motion energy (discretized into 5 equal-percentile bins) from barrel cortex neural activity in developing mice (P7-P14).

## Reference

Majnik J, Mantez M, Zangila S, Bugeon S, Guignard L, Platel JC, Cossart R (2025). Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p. eLife 14:RP107540.

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Data conversion script |
| `converted_data.pkl` | Full converted dataset (41 sessions, 6 mice) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `train_decoder.py` | Decoder training and validation script |
| `decoder.py` | Decoder model implementation |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |
| `conversion_full_out.txt` | Full conversion output log |
| `conversion_sample_out.txt` | Sample conversion output log |
| `verification_full_out.txt` | Full data verification output |
| `verification_sample_out.txt` | Sample data verification output |
| `train_decoder_full_out.txt` | Full decoder training output |
| `train_decoder_sample_out.txt` | Sample decoder training output |

## Usage

### Convert Data
```bash
python3 convert_data.py                # Full conversion
python3 convert_data.py --sample-only  # Sample only (faster)
```

### Verify Data
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

### Train Decoder
```bash
python3 train_decoder.py converted_data.pkl --cpu        # CPU only
python3 train_decoder.py converted_data.pkl --plot-samples  # With plots
```

## Data Format

- **Neural**: dF/F from Suite2p (neuropil-corrected, baseline-corrected), binned by 10 frames
- **Input**: Time elapsed from session start (seconds)
- **Output**: Motion energy in 5 equal-percentile bins (0-4)
- **Trials**: 2-minute blocks (360 timepoints at ~3 Hz)
- **Sessions**: 41 total (6-7 per mouse, 10-15 trials each)
