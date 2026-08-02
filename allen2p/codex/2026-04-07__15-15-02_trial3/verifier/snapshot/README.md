# Visual Behavior Ophys Conversion

This repository now contains a decoder-ready conversion of the local Allen Brain Observatory Visual Behavior 2P subset in `converted_data.pkl`.

## Dataset Summary
- Source data: local `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb`
- Included sessions: active task sessions only
- Trial filter: keep `go` and `catch`; exclude `aborted` and `auto_rewarded`
- Neural signal: NWB `processing/ophys/event_detection/data`
- Temporal alignment: absolute ophys time, then segmented into trials
- Common binning: 30 Hz (`33.333... ms`)

Full converted dataset statistics:
- Subjects: `38`
- Sessions: `199`
- Trials: `51,075`
- Neurons: `29,168`
- Brain regions: `VISp`, `VISl`

## Outputs
The decoder outputs are stored in this order:
1. `image_identity`
2. `image_change`
3. `running_speed_bin`
4. `pupil_diameter_bin`
5. `trial_outcome`

Category sets:
- `image_identity`: `gray` plus 16 flashed image identities
- `image_change`: `no_change`, `change`
- `running_speed_bin`: `q1` to `q5`
- `pupil_diameter_bin`: `q1` to `q5`
- `trial_outcome`: `hit`, `miss`, `false_alarm`, `correct_reject`

## File Format
`converted_data.pkl` stores a Python dictionary with:
- `neural`: list of sessions, each a list of trials, each trial `(n_neurons, n_timepoints)`
- `input`: empty `(0, T)` arrays for every trial because this task specifies no decoder inputs
- `output`: list of sessions/trials, each trial `(5, n_timepoints)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## How To Use
Load the dataset with Python:

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))          # sessions
print(data["output_names"])         # decoded variables
print(data["neural"][0][0].shape)   # first trial neural matrix
print(data["output"][0][0].shape)   # first trial output matrix
```

Re-run the converter:

```bash
python3 -u convert_data.py converted_data.pkl --full
```

Run validation only:

```bash
python3 -u train_decoder.py converted_data.pkl --verify-only
```

Run full decoder training:

```bash
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Validation Snapshot
- Full verification completed successfully.
- Full training completed successfully on GPU.
- Validation balanced accuracy:
  - `image_identity`: `0.1903`
  - `image_change`: `0.5890`
  - `running_speed_bin`: `0.2440`
  - `pupil_diameter_bin`: `0.2714`
  - `trial_outcome`: `0.2684`

Detailed rationale, sanity checks, and review results are in `CONVERSION_NOTES.md`.
