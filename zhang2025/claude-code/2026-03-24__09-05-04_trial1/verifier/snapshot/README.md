# IBL Brain-Wide Map - Neural Decoder Dataset

## Overview
Conversion of the IBL (International Brain Laboratory) Brain-Wide Map Neuropixels electrophysiology dataset into a format suitable for training neural decoders.

## Dataset
- **Source**: IBL Brain-Wide Map (BWM) - 139 mice, 459 sessions, 699 Neuropixels probe insertions
- **Converted**: 392 sessions (67 skipped due to missing data), 129 subjects, 534,911 neurons, 167,487 trials, 275 brain regions

## Files
| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (23 GB, neural as uint8) |
| `converted_data_subset.pkl` | Subset for decoder training (98 sessions, 5.7 GB) |
| `sample_data.pkl` | 2-session sample for testing |
| `convert_data.py` | Conversion script |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder module |
| `CONVERSION_NOTES.md` | Detailed conversion notes and decisions |

## Data Format
```python
{
    'neural': list of sessions, each a list of trials, each (n_neurons, 100) uint8 array
    'input': list of sessions, each a list of trials, each (2, 100) float32 array
    'output': list of sessions, each a list of trials, each (4, 100) int array
    'subjects': list of subject names
    'subject_idx': (n_sessions,) int array
    'brain_regions': list of Beryl atlas region names
    'brain_region_idx': list of (n_neurons,) int arrays per session
    'input_names': ['time_since_stimulus_onset', 'trial_number_in_block']
    'output_names': ['choice', 'prior_probability_left', 'wheel_speed', 'whisker_motion_energy']
    'output_values': [['left','right'], ['0.2','0.5','0.8'], ['low','medium','high'], ['low','medium','high']]
    'metadata': {...}
}
```

## Processing Parameters
- **Alignment**: Stimulus onset (stimOn_times)
- **Time window**: -0.5s to 1.5s (100 bins at 20ms)
- **Spike sorting**: Kilosort 2.5 (pykilosort), all clusters (qc=None)
- **Wheel velocity**: 1kHz interpolation, Butterworth low-pass (20Hz, order 8), absolute value for speed
- **Whisker motion energy**: From leftCamera ROI (fallback to rightCamera)
- **Discretization**: Quantile-based 3 bins for wheel speed and whisker ME

## Decoder Results
| Variable | Validation BA | Chance |
|----------|--------------|--------|
| Choice | 0.559 | 0.500 |
| Prior | 0.597 | 0.333 |
| Wheel speed | 0.593 | 0.333 |
| Whisker ME | 0.574 | 0.333 |

## Usage
```bash
# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder (subset, fits in 64GB memory)
python train_decoder.py converted_data_subset.pkl --cpu --plot-samples

# Reconvert from raw data
python convert_data.py converted_data.pkl --full
```

## References
- "A brain-wide map of neural activity during complex behaviour" (IBL, 2023)
- "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., 2025)
