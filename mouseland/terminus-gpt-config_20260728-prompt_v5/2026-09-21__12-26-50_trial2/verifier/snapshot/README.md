# Converted Neural Decoder Dataset

This repository contains a converted neuroscience dataset saved in `/app/converted_data.pkl` for use with `/app/train_decoder.py`.

## Dataset summary
- Neural data: deconvolved calcium activity from virtual-reality corridor sessions
- Temporal alignment: trial start / corridor entry
- Sessions included: 75 converted sessions with matched behavior and neural data
- Sample file: `/app/sample_data.pkl`

## Decoder inputs
1. `time_to_sound_cue`
2. `day_of_training`
3. `time_since_trial_start`
4. `reward_availability`

## Decoder outputs
1. `visual_stimulus_category`
2. `licking`
3. `position_bin`
4. `running_speed_bin`

## File format
The pickle contains a dictionary with keys:
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

## Usage
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Validation
- Sample verification: `/app/verification_sample_out.txt`
- Full verification: `/app/verification_full_out.txt`
- Sample decoder training: `/app/train_decoder_sample_out.txt`
- Full decoder training: `/app/train_decoder_full_out.txt`
