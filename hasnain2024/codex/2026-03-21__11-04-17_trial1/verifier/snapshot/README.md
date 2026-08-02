# Neural Decoder Conversion

This directory contains a decoder-ready conversion of the dataset from "Separating cognitive and motor processes in the behaving mouse".

The main output is [`converted_data.pkl`](/app/converted_data.pkl), a Python pickle with:

- `neural`: list of sessions, each a list of trials with `(n_neurons, n_timepoints)` neural matrices
- `input`: list of sessions, each a list of trials with `(1, n_timepoints)` time-from-go-cue input vectors
- `output`: list of sessions, each a list of trials with `(6, n_timepoints)` categorical targets
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Dataset summary

- Sessions: 44
- Subjects: 14
- Trials: 11,955
- Units: 2,455
- Brain region: ALM
- Alignment event: `goCue`
- Time window: `[-2.5, 2.5] s`
- Time bin: `0.005 s`

## Outputs

The decoder targets are:

1. `lick_direction`: `left=0`, `right=1`
2. `behavioral_context`: `WC=0`, `DR=1`
3. `outcome`: `incorrect=0`, `correct=1`
4. `tongue_velocity_bin`: session-median split
5. `paw_velocity_bin`: session-median split
6. `motion_energy_bin`: session-median split

Trial-level labels are stored as constant time series so every output has the same `(n_output, n_timepoints)` layout.

## Load example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

session0_trial0_neural = data["neural"][0][0]
session0_trial0_input = data["input"][0][0]
session0_trial0_output = data["output"][0][0]

print(session0_trial0_neural.shape)
print(session0_trial0_input.shape)
print(session0_trial0_output.shape)
print(data["input_names"])
print(data["output_names"])
```

## Rebuild and validate

- Sample conversion:
  - `python -u convert_data.py sample_data.pkl --sample --show-processing`
- Full conversion:
  - `python -u convert_data.py converted_data.pkl --full`
- Format verification:
  - `python -u train_decoder.py converted_data.pkl --verify-only`
- Full decoder training:
  - `python -u train_decoder.py converted_data.pkl --plot-samples`

## Notes

- Full rationale, checks, and debugging history are in [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md).
- Reusable investigation utilities are in [`cache/`](/app/cache).
