# Allen Visual Behavior 2P Neural Decoder

## Overview

This project converts the Allen Brain Observatory Visual Behavior 2P dataset into a standardized format for training neural decoders. The decoder predicts behavioral and stimulus variables from calcium imaging neural activity.

## Dataset

The Allen Visual Behavior 2P dataset contains 2-photon calcium imaging recordings from transgenic mice performing a visual change detection task. Mice detect changes in the identity of flashed natural scene images.

**Source**: Allen Institute for Brain Science, Visual Behavior 2P dataset v1.1.0

## Data Selection

- **Sessions**: Active behavior sessions only (no passive viewing)
- **Trials**: Go and Catch trials only (Aborted and Auto-rewarded excluded)
- **Neural data**: Detected calcium events (event detection output, not raw dF/F)
- **Temporal alignment**: Ophys imaging frame timestamps

## Decoder Outputs

1. **Image identity** - Categorical (16 images), time-varying
2. **Image change** - Binary (change/no_change), time-varying
3. **Running speed** - 5 equal percentile bins, time-varying
4. **Pupil diameter** - 5 equal percentile bins, time-varying
5. **Trial outcome** - Categorical (hit/miss/false_alarm/correct_reject), static per trial

## Files

- `convert_data.py` - Conversion script
- `converted_data.pkl` - Full converted dataset (202 sessions, 38 mice)
- `sample_data.pkl` - Sample dataset (9 sessions, 3 mice)
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model implementation
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation
- `conversion_full_out.txt` - Full conversion output log
- `conversion_sample_out.txt` - Sample conversion output log
- `verification_full_out.txt` - Full data verification output
- `verification_sample_out.txt` - Sample data verification output
- `train_decoder_full_out.txt` - Full decoder training output
- `train_decoder_sample_out.txt` - Sample decoder training output

## Usage

```bash
# Convert sample data
python convert_data.py --sample --output sample_data.pkl

# Convert full data
python convert_data.py --output converted_data.pkl

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Dataset Statistics

- **Sessions**: 202 experiments from 38 mice
- **Trials**: 51,992 total
- **Neuron-sessions**: 29,444 total
- **Brain regions**: VISp, VISl
- **Cell types**: Slc17a7 (excitatory), Sst (inhibitory), Vip (inhibitory)
- **Imaging rate**: ~31 Hz (single-plane), ~11 Hz (multi-plane)
- **Time bin**: ~32.26 ms (median)
