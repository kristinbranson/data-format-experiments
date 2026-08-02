# Neural Decoder: CA1 Position Decoding from Geometric Environments

## Overview

This project converts calcium imaging data from Lee et al. (2025) into a standardized format for training neural decoders. The data comes from CA1 hippocampal recordings in mice freely exploring 10 geometrically distinct environments.

## Reference

Lee, J.Q., Keinath, A.T., Cianfarano, E., & Brandon, M.P. (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. *Neuron*, 113, 307-320.

## Dataset Summary

- **Animals**: 7 mice (QLAK-CA1-08 through QLAK-CA1-75)
- **Sessions**: 207 total (21-31 per animal)
- **Neurons**: 5,413 unique neurons tracked across sessions
- **Neural data**: Binary rising-phase calcium transients at 30 Hz
- **Trial structure**: 40-minute sessions split into 1-minute trials (39-40 trials/session)
- **Environments**: 10 geometric configurations (square, o, t, u, rectangle, +, i, l, bit donut, glenn)

## Decoder Task

- **Input to decoder**: Environment geometry as 3x3 binary matrix (9 values), indicating which spatial partitions are accessible
- **Output to predict**: Mouse position discretized into 3x3 spatial bins (9 classes), time-varying at 30 Hz
- **Neural activity**: Binary spike data (n_neurons x 1800 timepoints per trial)

## Files

- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (207 sessions)
- `sample_data.pkl` - Sample dataset (2 sessions per animal, 14 sessions total)
- `train_decoder.py` - Decoder training and evaluation script
- `decoder.py` - Decoder model implementation
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation

## Usage

```bash
# Convert data
python convert_data.py

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train and evaluate decoder
python train_decoder.py converted_data.pkl --plot-samples

# Quick test with sample data
python train_decoder.py sample_data.pkl --plot-samples
```

## Results

- Validation balanced accuracy: ~0.55 (chance = 0.11 for 9 classes)
- Training balanced accuracy: ~0.62
