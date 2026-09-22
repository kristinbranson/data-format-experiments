# Converted Neural Decoder Dataset

This repository now includes a decoder-ready dataset at `/app/converted_data.pkl` derived from the Hasnain, Birnbaum et al. ALM electrophysiology and video dataset accompanying *Separating cognitive and motor processes in the behaving mouse*.

## Dataset summary

- Sessions: 44
- Subjects: 14 manifest subject IDs
- Trials: 13,935
- Neurons: 2,456
- Brain region: `ALM`
- Alignment event: `goCue`
- Trial window: `[-2.5, 2.5]` s relative to go cue
- Time bin size: `10 ms`
- Neural representation: smoothed firing rate

The converted dataset contains curated ALM electrophysiology sessions from:

- `Ephys_Behavior`: 25 sessions
- `RandomizedDelay_Ephys_Behavior`: 19 sessions

Early trials were excluded. Two sessions contained behavioral trials beyond the neural recording range; those all-zero-neural tail trials were removed during conversion.

## Decoder inputs and outputs

Input:

- `time_from_go_cue`

Outputs:

- `lick_direction`: `left`, `right`, `none`
- `behavioral_context`: `DR`, `WC`
- `outcome`: `incorrect`, `correct`, `ignore`
- `tongue_velocity`: `low`, `high`, `not_visible`
- `paw_velocity`: `low`, `high`, `not_visible`
- `motion_energy`: `low`, `high`, `no_video`

Velocity and motion-energy outputs are time-varying categorical labels built on the same 500-bin time axis as the neural data.

## File format

`/app/converted_data.pkl` stores a Python dictionary with these top-level keys:

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

Per trial:

- `neural[session][trial]`: `(n_neurons, 500)`
- `input[session][trial]`: `(1, 500)`
- `output[session][trial]`: `(6, 500)`

## Loading example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

session_id = data["metadata"]["session_ids"][0]
neural_trial = data["neural"][0][0]
input_trial = data["input"][0][0]
output_trial = data["output"][0][0]

print(session_id)
print(neural_trial.shape)  # (n_neurons, 500)
print(input_trial.shape)   # (1, 500)
print(output_trial.shape)  # (6, 500)
```

## Re-running conversion and validation

Conversion:

```bash
python3 -u /app/convert_data.py /app/converted_data.pkl --full
```

Format verification:

```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Full decoder training:

```bash
python3 -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Validation snapshot

Full decoder validation balanced accuracy:

- `lick_direction`: `0.6253`
- `behavioral_context`: `0.8535`
- `outcome`: `0.5973`
- `tongue_velocity`: `0.5804`
- `paw_velocity`: `0.5726`
- `motion_energy`: `0.7728`

Detailed conversion decisions, sanity checks, and validation results are documented in `/app/CONVERSION_NOTES.md`.
