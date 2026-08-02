# Visual Behavior Conversion

This repository contains a converted subset of the Allen Brain Observatory Visual Behavior 2P dataset formatted for `train_decoder.py`.

## Dataset
- Source: Allen Institute Visual Behavior Ophys NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments`
- Included sessions: active Visual Behavior ophys experiments with usable eye tracking
- Excluded sessions: passive sessions and 3 active sessions missing pupil-tracking data
- Neural signal: precomputed calcium `events`
- Temporal grid: common 30 Hz grid built from raw trial `start_time` / `stop_time`

## Converted Output
- File: `converted_data.pkl`
- Structure: Python dictionary with `neural`, `input`, `output`, subject metadata, brain-region metadata, output labels, and conversion metadata
- Decoder inputs:
  - none (`input_names = []`, each trial stores a `(0, T)` array)
- Decoder outputs:
  - `image_identity`
  - `image_change`
  - `running_speed_bin`
  - `pupil_diameter_bin`
  - `trial_outcome`

## Key Statistics
- Sessions: 199
- Subjects: 38
- Trials: 51,075
- Neurons: 29,168
- Brain regions: `VISp`, `VISl`
- Trial length: 211-377 bins
- Neurons/session: 4-666

## How To Load
```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["output_names"])
print(data["metadata"]["session_ophys_experiment_ids"][:5])
```

## Recreate The Conversion
```bash
python3 -u convert_data.py converted_data.pkl --full
python3 -u train_decoder.py converted_data.pkl --verify-only
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes
- Trial inclusion follows the NWB/SDK trial taxonomy: keep `go` and `catch`, exclude `aborted` and `auto_rewarded`.
- `image_change` is labeled during the changed-image presentation window, using raw stimulus `is_change` intervals.
- Running speed and pupil diameter are discretized with global quintile edges stored in `data["metadata"]`.
- Additional audit scripts used during conversion review are in `cache/`.
