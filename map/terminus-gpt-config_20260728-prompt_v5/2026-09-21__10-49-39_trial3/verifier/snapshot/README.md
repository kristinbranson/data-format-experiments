# Converted MAP decoder dataset

This repository contains a converted neuroscience dataset saved at `/app/converted_data.pkl`.

## Contents
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, 80)`
- `input`: list of sessions/trials with 2 inputs over 80 bins
- `output`: list of sessions/trials with 4 categorical outputs over 80 bins

## Alignment and binning
- Alignment event: go cue onset
- Window: -2.5 s to +1.5 s
- Bin size: 50 ms

## Inputs
1. `time_from_tone_onset_sec`
2. `photostimulation_on`

## Outputs
1. `choice` = left / right / no lick
2. `outcome` = ignore / miss / hit
3. `early_lick` = no / yes
4. `tongue_y_position` = lt_40th / 40th_to_60th / gt_60th / not_visible

## Summary
- Subjects: 28
- Sessions: 174
- Trials: 90,999
- Mean neurons/session: ~406.06

## Usage
```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```
