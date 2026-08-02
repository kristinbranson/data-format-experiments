# Neural Decoder Data Conversion

Conversion of hippocampal CA1 2-photon calcium imaging data from Sosa, Plitt & Giocomo (2025) "A flexible hippocampal population code for experience relative to reward" (Nature Neuroscience) into a standardized format for neural decoder training.

## Data Source
- **DANDI Archive**: Dandiset 001361
- **Format**: Neurodata Without Borders (NWB)
- **Reference code**: https://github.com/GiocomoLab/Sosa_et_al_2024

## Files

| File | Description |
|------|-------------|
| `convert_data.py` | Main conversion script |
| `converted_data.pkl` | Full converted dataset (152 sessions, ~9 GB) |
| `sample_data.pkl` | Sample dataset for quick testing (5 sessions, ~14 MB) |
| `CONVERSION_NOTES.md` | Detailed conversion decisions and validation |
| `train_decoder.py` | Decoder training and validation script |
| `decoder.py` | Decoder model implementation |
| `conversion_full_out.txt` | Full conversion log |
| `conversion_sample_out.txt` | Sample data verification log |
| `verification_full_out.txt` | Full data format verification log |
| `verification_sample_out.txt` | Sample data format verification log |
| `train_decoder_full_out.txt` | Full decoder training output |
| `train_decoder_sample_out.txt` | Sample decoder training output |

## Task Description

Mice navigate a 450cm virtual linear track with a hidden reward zone at one of three possible locations (A: 80-130cm, B: 200-250cm, C: 320-370cm). The reward zone switches locations across sessions. Two distinct virtual environments (ENV1, ENV2) are used.

## Data Structure

```python
data = {
    'neural': [[trial_array, ...], ...],  # (n_neurons, T) per trial
    'input': [[input_array, ...], ...],    # (4, T) per trial
    'output': [[output_array, ...], ...],  # (6, T) per trial
    'subjects': ['m11', 'm12', ...],       # 11 subjects
    'subject_idx': np.array([...]),        # session-to-subject mapping
    'brain_regions': ['CA1'],
    'brain_region_idx': [...],
    'input_names': ['time_from_trial_start', 'environment_type', 'trial_number', 'previous_trial_outcome'],
    'output_names': ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome'],
    'output_values': [[...], ...],         # class labels for each output
    'metadata': {...}
}
```

## Usage

```bash
# Convert NWB data to decoder format
python convert_data.py

# Verify data format
python train_decoder.py converted_data.pkl --verify-only

# Train decoder
python train_decoder.py converted_data.pkl --cpu

# Quick test with sample data
python train_decoder.py sample_data.pkl --cpu --plot-samples
```

## Dataset Statistics

- **11 subjects**, 152 sessions, 12,216 trials
- **912 neurons/session** on average (range: 155-2339)
- **80.4 trials/session** on average
- **15.5 Hz** imaging rate (64.48 ms time bins)
- **15.3%** reward omission rate
