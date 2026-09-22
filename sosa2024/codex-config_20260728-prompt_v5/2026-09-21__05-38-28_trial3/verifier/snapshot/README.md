# Converted Dataset README

## Dataset
- Source dataset: *A flexible hippocampal population code for experience relative to reward*
- Raw inputs: NWB session files in [`/app/data`](/app/data)
- Converted output: [`/app/converted_data.pkl`](/app/converted_data.pkl)

## What was converted
- Sessions: `152`
- Subjects: `11`
- Brain region: `hippocampus, CA1`
- Curated neurons: `138,678`
- Converted trials: `12,147`
- Trial alignment: start of trial (`trial_start` to `teleport`)
- Neural signal: curated deconvolved calcium activity from NWB `processing/ophys/Deconvolved`
- Trial exclusions: `69` lick-artifact trials removed with the reference-code-compatible QC rule

## Decoder-oriented representation
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: list of sessions/trials, each trial shaped `(4, n_timepoints)`
- `output`: list of sessions/trials, each trial shaped `(6, n_timepoints)`

### Inputs
1. `time_from_trial_start_sec`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

### Outputs
1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Important processing choices
- Trials are defined from NWB `trial_start` to the next `teleport`, matching the reference analysis structure and excluding tunnel/teleport fragments.
- Multi-plane sessions are handled by reading each deconvolved plane series separately and concatenating curated cells across planes.
- Reward outcome is reconstructed from sparse NWB reward timestamps.
- Reward-zone identity is inferred from session scene metadata with the same A/B/C mapping used in the paper and code:
  - A: `80-130 cm`
  - B: `200-250 cm`
  - C: `320-370 cm`

## How to load
```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['input_names'])
print(data['output_names'])
```

## Metadata summary
- Time bin size: `64.48362720403011 ms`
- Temporal alignment event: `trial start (entry onto the 450 cm track)`
- `off_start`: `0.0`
- `off_end`: `None` because trials have variable duration

## Validation files
- Sample conversion log: [`/app/conversion_sample_out.txt`](/app/conversion_sample_out.txt)
- Sample verification log: [`/app/verification_sample_out.txt`](/app/verification_sample_out.txt)
- Sample training log: [`/app/train_decoder_sample_out.txt`](/app/train_decoder_sample_out.txt)
- Full conversion log: [`/app/conversion_full_out.txt`](/app/conversion_full_out.txt)
- Full verification log: [`/app/verification_full_out.txt`](/app/verification_full_out.txt)
- Full training log: [`/app/train_decoder_full_out.txt`](/app/train_decoder_full_out.txt)
- Conversion notes: [`/app/CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md)

## Re-running conversion
```bash
python3 -u /app/convert_data.py /app/converted_data.pkl --full
```

## Re-running validation
```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python3 -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
