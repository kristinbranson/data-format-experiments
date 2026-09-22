# Zhong et al., 2025 - Converted Dataset for Neural Decoding

## Overview
Converted two-photon calcium imaging data from "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025) into a standardized Python dictionary format suitable for training neural decoders.

## Output Files
- **`converted_data.pkl`** - Full dataset (89 sessions, 19 mice, ~142 GB)
- **`sample_data.pkl`** - Sample dataset (2 sessions, ~3.6 GB)
- **`CONVERSION_NOTES.md`** - Detailed documentation of all conversion decisions

## Data Format
```python
data = {
    'session_id': {
        'neural_data': np.array, shape (n_trials, n_neurons, T),  # float32
        'brain_region': np.array, shape (n_neurons,),  # string labels
        'input': np.array, shape (n_trials, 4),  # float32
        'output': np.array, shape (n_trials, 4),  # float32
        'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
        'output_names': ['visual_stimulus', 'licking', 'position', 'running_speed'],
    }
}
```

### Inputs
| Name | Description | Range |
|------|-------------|-------|
| time_to_sound_cue | Seconds from current frame to sound cue (negative = before) | continuous |
| day_of_training | Days since first session for this mouse | 0-92 |
| time_since_trial_start | Seconds since trial start | >= 0 |
| reward_availability | Whether reward is available in session | 0 or 1 |

### Outputs
| Name | Description | Values |
|------|-------------|--------|
| visual_stimulus | Texture category integer ID | 0-14 (15 textures) |
| licking | Binary licking indicator | 0 or 1 |
| position | Position bin in corridor | 0-3 (4 bins of 1m each, 0-4m) |
| running_speed | Speed quartile bin | 0-3 (quartile edges: 12.2, 25.0, 40.5 cm/s) |

## Decoder Results

| Output | Val Balanced Acc | Chance |
|--------|-----------------|--------|
| visual_stimulus | 0.587 | 0.067 |
| licking | 0.849 | 0.500 |
| position | 0.321 | 0.250 |
| running_speed | 0.373 | 0.250 |

All outputs decode above chance.

## Key Processing Decisions
- **Neural data**: Suite2p deconvolved fluorescence, concatenated across planes
- **Frame selection**: Only frames where `ft_move > 0` AND `ft_CorrSpc` (corridor/texture area while VR moving)
- **Neuron filtering**: Excludes `iarea == -1` (unassigned) and `iarea == 7` (non-visual)
- **Brain regions**: V1 (iarea=8), mHV (0,1,2,9), lHV (5,6), aHV (3,4)
- **Speed quartiles**: Computed across all sessions (edges: 12.2, 25.0, 40.5 cm/s)

## Reproduction
```bash
# Generate full dataset
python convert_data.py --full

# Generate sample (2 sessions)
python convert_data.py --sample

# Train decoder
python train_decoder.py converted_data.pkl
```

## Files in cache/
Intermediate log files are stored in `cache/`. See `cache/README_CACHE.md`.
