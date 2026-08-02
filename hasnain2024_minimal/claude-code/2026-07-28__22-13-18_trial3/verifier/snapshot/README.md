# Neural Decoder Data Conversion

Converts electrophysiology and behavioral data from Hasnain, Birnbaum et al. (Nature Neuroscience 2024) "Separating cognitive and motor processes in the behaving mouse" into a standardized format for neural decoder training.

## Dataset Overview

- **Species**: Mouse (Mus musculus)
- **Brain region**: ALM (anterior lateral motor cortex)
- **Task**: Two-context directional licking (delayed-response + water-cued paradigms) and randomized delay task
- **Sessions**: 45 sessions from 14 mice
- **Total units**: 2,504
- **Total trials**: 12,293

## Quick Start

```bash
# Run the conversion
python convert_data.py

# Verify the output
python train_decoder.py converted_data.pkl --verify-only --plot-samples

# Train the decoder
python train_decoder.py converted_data.pkl --cpu --plot-samples

# Use sample data for quick testing
python train_decoder.py sample_data.pkl --cpu
```

## Decoder Variables

### Input
- `time_from_go_cue`: Time relative to go cue onset (continuous, -2.5 to 2.5 s)

### Outputs (all categorical)
- `lick_direction`: left (0) vs right (1)
- `behavioral_context`: water-cued (0) vs delayed-response (1)
- `outcome`: incorrect (0) vs correct (1)
- `tongue_velocity`: below (0) vs above (1) session median
- `paw_velocity`: below (0) vs above (1) session median
- `motion_energy`: below (0) vs above (1) session median

## Processing Details

See `CONVERSION_NOTES.md` for full documentation of processing decisions and validation results.

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Conversion script |
| `converted_data.pkl` | Full dataset (45 sessions) |
| `sample_data.pkl` | Sample dataset (3 sessions) |
| `CONVERSION_NOTES.md` | Detailed conversion documentation |
| `train_decoder.py` | Decoder training script |
