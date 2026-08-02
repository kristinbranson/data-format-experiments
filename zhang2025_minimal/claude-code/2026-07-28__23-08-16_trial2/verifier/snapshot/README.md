# IBL Brain Wide Map - Neural Decoder Data Conversion

## Overview

This project converts the International Brain Laboratory (IBL) Brain Wide Map (BWM) dataset
into a standardized format for neural decoder training. The conversion follows the methods
described in:

- **Data paper**: "A brain-wide map of neural activity during complex behaviour" (IBL et al., 2024)
- **Methods paper**: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., 2025)

## Task Description

Mice rotate a wheel to indicate the location of a visual stimulus (left/right) appearing
at varying contrasts, with biased blocks altering the prior probability of stimulus location.

## Data Processing Pipeline

### Loading
- Spike-sorted neural data loaded via ONE API and ibllib
- Multiple probes within a session are merged (matching reference code)
- All spike-sorted clusters are used (no QC filtering, matching reference code default)
- Brain regions mapped using Beryl atlas mapping

### Trial Selection
- Reaction time filter: 0.08s <= RT <= 2.0s
- Exclude trials with NaN in key events (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType)
- Exclude no-choice trials (choice == 0)

### Temporal Alignment
- Aligned to stimulus onset (stimOn_times)
- Window: -0.5s to +1.5s (2s total)
- Bin size: 20ms non-overlapping bins -> 100 time steps per trial

### Behavioral Variables
- **Wheel speed**: absolute velocity, interpolated to 20ms bins
- **Whisker motion energy**: from left camera (fallback: right), interpolated to 20ms bins
- Both discretized into 3 bins (terciles) using global quantile thresholds

### Decoder Inputs (2 dimensions)
1. Time since stimulus onset (continuous, time-varying)
2. Trial number in block (continuous, per-trial, replicated across time)

### Decoder Outputs (4 dimensions)
1. Choice: binary (0=left, 1=right), per-trial
2. Prior probability of left: 3-class (0=0.2, 1=0.5, 2=0.8), per-trial
3. Wheel speed: 3-bin discretization (0=low, 1=medium, 2=high), time-varying
4. Whisker motion energy: 3-bin discretization (0=low, 1=medium, 2=high), time-varying

## Files

- `convert_data.py` - Main conversion script
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (5 sessions, for quick testing)
- `train_decoder.py` - Decoder training and validation script
- `decoder.py` - Decoder model implementation
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation results
- `conversion_sample_out.txt` - Sample conversion output log
- `verification_sample_out.txt` - Sample verification output log
- `train_decoder_sample_out.txt` - Sample decoder training output log
- `conversion_full_out.txt` - Full conversion output log
- `verification_full_out.txt` - Full verification output log
- `train_decoder_full_out.txt` - Full decoder training output log

## Usage

```bash
# Convert sample data (5 sessions)
python convert_data.py --sample --output sample_data.pkl

# Convert full dataset
python convert_data.py --output converted_data.pkl

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu

# Train decoder with plots
python train_decoder.py converted_data.pkl --plot-samples --cpu
```

## Requirements

- Python 3.10+
- ONE-api, ibllib, iblatlas, iblutil
- numpy, pandas, scipy, scikit-learn, torch, matplotlib
