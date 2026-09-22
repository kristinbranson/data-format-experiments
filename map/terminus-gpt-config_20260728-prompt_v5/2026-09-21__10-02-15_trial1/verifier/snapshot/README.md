# Converted Neural Decoder Dataset

This repository contains a converted dataset saved at:

- `/app/converted_data.pkl`

## Source dataset
Brain-wide neural activity underlying memory-guided movement NWB sessions from `/app/data`.

## Conversion summary
- Alignment event: Go cue onset
- Window: -2.5 s to +1.5 s around go cue
- Bin size: 50 ms
- Neuron filter: `classification == 'good'`
- Trial validity: trials restricted to valid `is_good_trials` coverage when available; residual all-zero neural trials removed

## Decoder inputs
1. `time_from_tone_onset_sec`
2. `photostimulation_on`

## Decoder outputs
1. `choice` = left / right / no_lick
2. `outcome` = ignore / miss / hit
3. `early_lick` = no / yes
4. `tongue_y_position` = lt_40th / 40th_to_60th / gt_60th / not_visible

## Data format
The pickle stores a Python dictionary with keys:
- `neural`
- `input`
- `output`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

Each session contains a list of trials. Each neural trial is an array of shape `(n_neurons, 80)`, corresponding to 80 time bins of 50 ms across the 4 s window.

## Basic usage
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(len(data['neural']))  # number of sessions
print(data['input_names'])
print(data['output_names'])
```

## Validation
The converted dataset was checked with:
```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
