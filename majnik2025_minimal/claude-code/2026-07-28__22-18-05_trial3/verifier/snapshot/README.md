# Track2p Barrel Cortex Neural Decoder

Converts longitudinal calcium imaging data from Majnik et al. 2025 (Track2p) into a standardized format for training a neural decoder that predicts motion energy from barrel cortex activity.

## Quick Start

```bash
# Convert data
python convert_data.py

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```

## Dataset

- **Source**: 6 mice, barrel cortex layer 2/3, postnatal days P7-P14
- **Neural**: dF/F traces (Suite2p baseline-corrected), binned by 10 frames (333 ms bins)
- **Output**: Motion energy discretized into 5 equal-percentile bins
- **Input**: Time elapsed from session start (seconds)
- **Trials**: 2-minute blocks (360 bins each)
- **Sessions**: 41 total (6-7 per mouse)

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Data conversion script |
| `converted_data.pkl` | Full converted dataset |
| `sample_data.pkl` | Sample dataset (2 mice) |
| `CONVERSION_NOTES.md` | Detailed processing decisions |
| `train_decoder.py` | Decoder training/validation script |

## Results

- Full data validation balanced accuracy: 0.302 (chance: 0.200)
- Performance consistent with paper's finding that behavioral representation emerges around P11
