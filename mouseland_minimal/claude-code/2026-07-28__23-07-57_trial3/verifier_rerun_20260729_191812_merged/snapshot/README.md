# Neural Decoder - Zhong et al. 2025

Conversion of data from "Unsupervised pretraining in biological neural networks" (Zhong et al., Nature 2025) into a standardized decoder format.

## Dataset Overview

- **89 recording sessions** from **19 mice** (GCaMP6s, two-photon mesoscope)
- **Visual cortex**: V1, medial HVAs (mHV), lateral HVAs (lHV), anterior HVAs (aHV)
- **Task**: Visual discrimination of naturalistic textures in virtual reality corridors
- **Neural data**: Deconvolved fluorescence traces (Suite2p), 20K-90K neurons per session
- **Temporal structure**: 60 spatial bins per trial (40 corridor + 20 gray space), each ~166.67 ms

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Conversion script |
| `converted_data.pkl` | Full converted dataset (89 sessions) |
| `sample_data.pkl` | Sample dataset (5 sessions) |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder model and utilities |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `conversion_full_out.txt` | Full conversion output log |
| `conversion_sample_out.txt` | Sample conversion output log |
| `verification_full_out.txt` | Full data verification output |
| `verification_sample_out.txt` | Sample data verification output |
| `train_decoder_full_out.txt` | Full decoder training output |
| `train_decoder_sample_out.txt` | Sample decoder training output |

## Decoder Inputs (4 variables, time-varying)

1. **time_to_sound_cue**: Time remaining until sound cue (seconds), positive before cue
2. **day_of_training**: Day number relative to first session of each mouse
3. **time_since_trial_start**: Elapsed time from corridor entry (seconds)
4. **reward_availability**: Binary (1=rewarded corridor, 0=unrewarded)

## Decoder Outputs (4 variables, categorical, time-varying)

1. **visual_stimulus**: Stimulus category (circle1, leaf1, leaf2, etc.)
2. **licking**: Binary lick detection per spatial bin
3. **corridor_position**: 4 equal 1-m bins + gray space
4. **running_speed**: 4 quartile bins of running speed

## Usage

```bash
# Convert data
python convert_data.py --output converted_data.pkl --sample sample_data.pkl

# Verify format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu
```

## Reference

Zhong, L., Baptista, S., Gattoni, R. et al. Unsupervised pretraining in biological neural networks. Nature 644, 741-748 (2025).
