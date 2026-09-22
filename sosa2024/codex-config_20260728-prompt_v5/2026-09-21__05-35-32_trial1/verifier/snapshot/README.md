# Neural Decoder Conversion

This directory contains a converted version of the dataset from *A flexible hippocampal population code for experience relative to reward* formatted for `/app/train_decoder.py`.

## Converted file

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample export used for early validation

## Alignment and signal choices

- Temporal alignment event: `trial_start`
- Trial window: from `trial_start` through the frame immediately before `teleport`
- Time bin size: `64.483627 ms`
- Neural signal: deconvolved calcium activity from NWB `processing/ophys/Deconvolved/plane*`, pooled across planes by ROI `planeIdx`
- ROI inclusion: `iscell[:, 0] > 0.5`
- Lick-QC rule: drop trials where more than 30% of imaging-frame samples have cumulative lick count `> 2`

## Dataset summary

- Subjects: 11 mice
- Sessions: 152
- Kept trials: 12,135
- Complete raw paired trials before lick QC: 12,216
- Removed bad-lick trials: 81
- Kept neurons across sessions: 138,678
- Brain region: `CA1`

## Decoder inputs

The exported input order is:

1. `time_from_trial_start_s`
2. `environment_type`
3. `trial_number`
4. `previous_trial_outcome`

Each trial stores inputs as a `(4, T)` `float32` array.

## Decoder outputs

The exported output order is:

1. `distance_to_reward_zone`
2. `absolute_position`
3. `speed`
4. `lick`
5. `reward_zone_location`
6. `reward_outcome`

Each trial stores outputs as a `(6, T)` `int16` array. Per-trial targets are repeated across time for a uniform decoder format.

Output category values:

- `distance_to_reward_zone`: `lt_-50`, `-50_to_-10`, `-10_to_lt0`, `0`, `gt0_to_10`, `gt10_to_50`, `gt50`
- `absolute_position`: `lt_90`, `90_to_180`, `180_to_270`, `270_to_360`, `gt_360`
- `speed`: `lt_2`, `2_to_10`, `10_to_20`, `20_to_40`, `gt_40`
- `lick`: `no`, `yes`
- `reward_zone_location`: `A`, `B`, `C`
- `reward_outcome`: `no`, `yes`

## Loading example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))               # sessions
print(data["input_names"])               # input variable order
print(data["output_names"])              # output variable order
print(data["neural"][0][0].shape)        # (n_neurons, T) for session 0, trial 0
print(data["input"][0][0].shape)         # (4, T)
print(data["output"][0][0].shape)        # (6, T)
```

## Reproducing conversion and validation

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Full decoder result summary

Validation balanced accuracy on the full export:

- `distance_to_reward_zone`: `0.3540`
- `absolute_position`: `0.4660`
- `speed`: `0.3812`
- `lick`: `0.6414`
- `reward_zone_location`: `0.8112`
- `reward_outcome`: `0.5237`

`reward_outcome` is the hardest target because reward omission was randomized and the per-trial label is repeated across the whole trial, including pre-reward periods.
