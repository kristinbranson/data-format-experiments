# IBL Brain-Wide Map Neural Decoder Dataset

## Overview

This dataset converts the International Brain Laboratory (IBL) Brain-Wide Map (BWM) data into a standardized format for neural decoding. The data comes from Neuropixels recordings in mice performing a visual decision-making task.

## Task Description

Mice rotate a wheel to indicate the location of a visual stimulus (Gabor patch) presented on either side of a screen. After an initial 90 unbiased trials, stimulus probability alternates between left-biased (80:20) and right-biased (20:80) blocks.

## Data Processing

### Source Data
- IBL Brain-Wide Map release (459 sessions, 354 available in cache)
- Neuropixels recordings across the mouse brain
- Spike sorting: Kilosort 2.5 (pykilosort)

### Processing Pipeline (matching Zhang et al. 2025 reference code)
1. **Temporal alignment**: Stimulus onset (`stimOn_times`)
2. **Time window**: -0.5 to 1.5 seconds relative to stimulus onset
3. **Bin size**: 20 ms (100 time bins per trial)
4. **Neuron selection**: Well-isolated neurons (cluster label >= 1.0, matching BWM paper quality criteria)
5. **Brain region mapping**: Beryl atlas
6. **Multi-probe merging**: Probes from same session merged
7. **Trial filtering**: Reaction time 0.08-2.0s, no NaN events, no no-choice trials
8. **Behavioral data**: Wheel speed and whisker motion energy interpolated to neural bin times

### Decoder Inputs
- `time_since_stimulus_onset`: Continuous, time-varying (-0.48 to 1.5 s)
- `trial_number_in_block`: Continuous, per-trial (constant within trial)

### Decoder Outputs (all categorical)
- `choice`: Binary (0=left, 1=right)
- `prior`: 3-class (0=p(left)=0.2, 1=p(left)=0.5, 2=p(left)=0.8)
- `wheel_speed`: 3-class discretization (low/medium/high, equal-frequency bins per session)
- `whisker_motion_energy`: 3-class discretization (low/medium/high, equal-frequency bins per session)

## Files

- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Sample dataset (10 sessions)
- `train_decoder.py` - Decoder training/validation script
- `decoder.py` - Decoder model and utilities
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation

## Usage

```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu

# Convert data (requires IBL data cache)
python convert_data.py --output converted_data.pkl
python convert_data.py --max-sessions 10 --output sample_data.pkl
```

## References

1. International Brain Laboratory et al. "A brain-wide map of neural activity during complex behaviour." (2023)
2. Zhang et al. "Exploiting correlations across trials and behavioral sessions to improve neural decoding." (2025)
