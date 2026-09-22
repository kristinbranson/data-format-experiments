# Sosa et al. CA1 Decoder Dataset

## Overview

This directory contains a decoder-ready conversion of the DANDI 001361 dataset associated with **“A flexible hippocampal population code for experience relative to reward.”** The data comprise dorsal CA1 two-photon calcium imaging and virtual-linear-track behavior from 11 switch-task mice.

The full converted file is:

- `/app/converted_data.pkl` — 152 sessions, 12,216 complete trials, 138,678 accepted session-cells

A two-session test file is also provided:

- `/app/sample_data.pkl` — 2 sessions, 160 trials, 323 accepted session-cells

Detailed scientific and processing decisions are in `/app/CONVERSION_NOTES.md`. Reproducible conversion code is in `/app/convert_data.py`.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))                 # 152 sessions
print(data['neural'][0][0].shape)         # neurons x time
print(data['input'][0][0].shape)          # 4 x time
print(data['output'][0][0].shape)         # 6 x time
```

The full pickle is approximately 9.84 GB, so loading it requires substantial RAM.

## Structure

The top-level dictionary contains:

- `neural[session][trial]`: float32 accepted-cell deconvolved calcium activity, shape `(n_neurons, n_timepoints)`.
- `input[session][trial]`: float32 decoder inputs, shape `(4, n_timepoints)`.
- `output[session][trial]`: int64 categorical targets, shape `(6, n_timepoints)`.
- `subjects`: unique mouse IDs.
- `subject_idx`: subject index for every session.
- `brain_regions = ['CA1']` and per-session `brain_region_idx`.
- `input_names`, `output_names`, and `output_values`.
- `metadata`: alignment, bin duration, source, reward-zone intervals, and session information.

Trials have variable duration. Within every trial, neural, input, and output arrays have identical time length.

## Decoder Inputs

Rows of each input matrix are:

1. Time from trial-start pulse in seconds.
2. Environment type: ENV1=0, ENV2=1.
3. Native zero-based trial number.
4. Previous complete trial outcome: omitted=0, rewarded=1. The first trial is 0 because no preceding within-session trial exists.

## Decoder Outputs

Rows of each output matrix are:

1. **Signed distance to reward-zone interval**: 7 requested categories. Distance is zero anywhere inside the interval, negative before it, and positive after it.
2. **Absolute position**: 5 requested bins over the 450 cm track.
3. **Speed**: 5 requested bins in cm/s.
4. **Lick**: no lick=0, lick=1.
5. **Reward-zone location**: A=0, B=1, C=2.
6. **Reward outcome**: omitted=0, rewarded=1.

Per-trial zone and outcome targets are repeated across time so all targets share one `(6, time)` representation.

Reference reward-zone intervals are A=[80,130], B=[200,250], and C=[320,370] cm. Switch sessions change zone on trial 30, matching the reference `get_reward_zones` logic.

## Neural Processing and Alignment

- Neural activity is the supplied Suite2p/OASIS `Deconvolved` calcium signal for `iscell` ROIs. It is not interpreted as spike rate.
- Accepted cells from multiple imaging planes are concatenated in plane order.
- Native synchronized samples are retained at approximately 15.5078 Hz (64.4836 ms/bin).
- Each trial begins at the `trial_start` pulse and ends immediately before its paired `teleport` sample.
- Sparse reward-event timestamps determine rewarded versus omission outcomes.
- Ten source sessions have one extra terminal neural row; only that terminal excess is trimmed, with no shift or interpolation.

## Key Statistics

| Statistic | Value |
|---|---:|
| Subjects | 11 |
| Sessions | 152 |
| Trials | 12,216 |
| Trials/session | 41-100 (mean 80.37) |
| Accepted session-cells | 138,678 |
| Cells/session | 155-2,341 (mean 912.36) |
| Rewarded trials | 10,342 (84.66%) |
| Omitted trials | 1,874 (15.34%) |
| Zone A/B/C trials | 4,186 / 4,010 / 4,020 |
| ENV1/ENV2 trials | 6,226 / 5,990 |

## Reproducing and Validating

```bash
# Two-session conversion with processing plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full conversion
python -u /app/convert_data.py /app/converted_data.pkl --full

# Format validation
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Full decoder training and plots
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation completed with no format errors or warnings. Full validation balanced accuracies were 0.3473 (distance), 0.5293 (position), 0.3916 (speed), 0.6514 (lick), 0.7991 (zone), and 0.5153 (outcome); all exceeded uniform chance.
