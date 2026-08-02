# Allen Brain Observatory Visual Behavior 2P - Neural Decoder

## Overview

This project converts Allen Brain Observatory Visual Behavior 2P calcium imaging data into a standardized format for training neural decoders. The decoder predicts experimental and behavioral variables from neural activity.

## Dataset

- **Source**: Allen Institute Visual Behavior 2-Photon Calcium Imaging Dataset
- **Task**: Go/No-Go change detection task with natural images
- **Imaging**: Single-plane (31 Hz) and multi-plane (11 Hz) 2-photon calcium imaging
- **Neural data**: dF/F traces (normalized change in fluorescence)

## Data Selection

- All active behavior ophys experiments with available NWB files (200 sessions)
- 38 mice, 3 cell types (Excitatory, Sst, Vip)
- Brain regions: VISp (primary visual cortex), VISl (lateral visual area)
- Trials: Go and Catch only (Aborted and Auto-rewarded excluded)

## Decoder Outputs

1. **Image identity** - Which of 16 natural images is being presented (categorical, time-varying)
2. **Image change** - Binary indicator of image change events (time-varying)
3. **Running speed** - Discretized into 5 equal percentile bins (time-varying)
4. **Pupil diameter** - Discretized into 5 equal percentile bins (time-varying)
5. **Trial outcome** - Hit/miss/false_alarm/correct_reject (static per trial)

## Files

- `convert_data.py` - Data conversion script
- `converted_data.pkl` - Full converted dataset
- `sample_data.pkl` - Small sample dataset (5 sessions)
- `train_decoder.py` - Decoder training and evaluation script
- `decoder.py` - Decoder model implementation
- `CONVERSION_NOTES.md` - Detailed conversion decisions and validation
- `conversion_sample_out.txt` - Sample conversion output
- `conversion_full_out.txt` - Full conversion output
- `verification_sample_out.txt` - Sample verification output
- `verification_full_out.txt` - Full verification output
- `train_decoder_sample_out.txt` - Sample decoder training output
- `train_decoder_full_out.txt` - Full decoder training output

## Usage

```bash
# Convert data (sample)
python convert_data.py --sample --output sample_data.pkl

# Convert data (full)
python convert_data.py --output converted_data.pkl

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples
```

## References

- Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper (whitepaper.pdf)
- "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (paper.pdf)
