# Neural Decoder Data Conversion

Converts NWB data from Sosa, Plitt, & Giocomo (2025) — "A flexible hippocampal population code for experience relative to reward" — into a format suitable for training a neural decoder.

## Quick Start

```bash
# Convert NWB data to decoder format
python3 convert_data.py

# Verify the converted data
python3 train_decoder.py converted_data.pkl --verify-only --plot-samples --cpu

# Train decoder on sample data
python3 train_decoder.py sample_data.pkl --cpu --plot-samples

# Train decoder on full data
python3 train_decoder.py converted_data.pkl --cpu --plot-samples
```

## Files

### Scripts
- `convert_data.py` — Main conversion script (NWB → decoder format)
- `train_decoder.py` — Decoder training and evaluation script

### Data
- `converted_data.pkl` — Full converted dataset (152 sessions, 12,216 trials, ~19.5 GB)
- `sample_data.pkl` — Sample subset (10 sessions, 800 trials, ~961 MB)

### Logs
- `conversion_full_out.txt` — Output from full data conversion
- `conversion_sample_out.txt` — Output from sample data inspection
- `verification_full_out.txt` — Full dataset format verification
- `verification_sample_out.txt` — Sample dataset format verification
- `train_decoder_full_out.txt` — Full dataset decoder training results
- `train_decoder_sample_out.txt` — Sample dataset decoder training results

### Documentation
- `CONVERSION_NOTES.md` — Detailed conversion decisions and validation
- `README.md` — This file

## Data Structure

The converted pickle file contains:

```python
data = {
    'neural': [[array(n_neurons, T), ...], ...],  # deconvolved events per session/trial
    'input': [[array(4, T), ...], ...],            # decoder inputs
    'output': [[array(6, T), ...], ...],           # decoder outputs (int)
    'input_names': ['time_from_trial_start', 'environment_type', 'trial_number', 'previous_trial_outcome'],
    'output_names': ['distance_to_reward_zone', 'absolute_position', 'speed', 'lick', 'reward_zone_location', 'reward_outcome'],
    'output_values': [...],  # list of possible values per output dim
    'subjects': ['m11', 'm12', ...],
    'subject_idx': array,    # subject index per session
    'brain_region': ['CA1'],
    'brain_region_idx': array,
    'metadata': {'time_bin_size': 64.48, 'brain_region': 'CA1', 'frame_rate': 15.5078125},
}
```

## Source Data

- **Species**: Mouse (11 subjects)
- **Brain region**: Hippocampal CA1
- **Imaging**: 2-photon calcium imaging (GCaMP7f), 15.5 Hz
- **Task**: Virtual reality linear track navigation with hidden reward zones
- **NWB files**: Located in `data/` directory, organized by subject
