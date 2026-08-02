# IBL Brain-Wide Map Neural Decoder Dataset

## Overview

This dataset converts the International Brain Laboratory (IBL) Brain-Wide Map (BWM) dataset into a standardized format for training neural decoders. The data comes from Neuropixels recordings across the mouse brain during a visual decision-making task.

## Source Data

- **Data paper**: "A brain-wide map of neural activity during complex behaviour" (IBL et al.)
- **Methods paper**: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al. 2025)
- **Dataset**: IBL BWM release (459 sessions, 699 probes, 139 subjects)

## Task Description

Mice perform a visual decision-making task where they rotate a wheel to move a visual stimulus (Gabor patch) to the center of a screen. Stimuli appear at varying contrasts and with varying prior probabilities across blocks of trials.

## Data Processing

### Alignment
- All trials aligned to **stimulus onset** (stimOn_times)
- Time window: **-0.5s to +1.5s** relative to stimulus onset (2s total)

### Binning
- Neural activity binned into **20ms non-overlapping bins** (100 bins per trial)
- Behavioral signals (wheel speed, whisker motion energy) interpolated to same bins

### Trial Filtering
Trials excluded if:
- Any required event is NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType)
- Reaction time < 0.08s or > 2.0s
- No choice made (choice == 0)

### Neural Data
- All clusters loaded (no quality filtering at caching level, matching reference code)
- Probes within same session merged
- Brain regions mapped using IBL Beryl atlas mapping

## Decoder Variables

### Inputs (to decoder)
1. **time_since_stim_onset**: Continuous time relative to stimulus onset (-0.5 to 1.5s), time-varying
2. **trial_number_in_block**: Trial position within the current probability block, per-trial

### Outputs (to predict)
1. **choice**: Binary (left=0, right=1), per-trial
2. **prior**: Categorical (0.2->0, 0.5->1, 0.8->2), per-trial
3. **wheel_speed**: Discretized into 3 bins (low/medium/high), time-varying
4. **whisker_motion_energy**: Discretized into 3 bins (low/medium/high), time-varying

## Files

- `convert_data.py`: Conversion script
- `converted_data.pkl`: Full converted dataset
- `sample_data.pkl`: Small sample (5 sessions) for testing
- `train_decoder.py`: Decoder training and validation script
- `decoder.py`: Decoder model and utilities
- `CONVERSION_NOTES.md`: Detailed conversion decisions and validation results

## Usage

```bash
# Convert data (full)
python convert_data.py --output converted_data.pkl

# Convert sample
python convert_data.py --sample --max-sessions 5

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```
