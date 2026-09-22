# Converted hippocampal decoder dataset

This repository contains a converted version of the NWB dataset from:
- **Paper**: *A flexible hippocampal population code for experience relative to reward*
- **Converted file**: `/app/converted_data.pkl`

## Contents
The pickle stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: decoder inputs per trial
- `output`: decoder targets per trial
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Neural signal
- Deconvolved calcium activity from NWB `processing/ophys/Deconvolved/plane0/data`

## Temporal alignment
- Trials are aligned to **trial start**
- Trial windows are defined from trial-start onset to teleport, excluding off-trial samples (`trial number = -1`) and requiring active scanning samples

## Decoder inputs
1. `time_from_trial_start`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

## Decoder outputs
1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

## Usage
Verify format:
```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Train decoder:
```bash
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Load in Python:
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Notes
- Some sessions had off-by-one mismatches between neural and behavioral stream lengths; aligned streams were trimmed to the common minimum length per session.
- Reward outcome was derived from the NWB `BehavioralTimeSeries/Reward` signal, aligned by its own timestamps when necessary.
- Reward-zone A/B/C categories were inferred from per-trial reward-zone-associated positions.
