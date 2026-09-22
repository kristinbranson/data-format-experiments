# Sosa et al. Hippocampal Dataset Conversion

## Dataset

This conversion reformats the released NWB data for **“A flexible hippocampal population code for experience relative to reward”** (DANDI 001361) into neural-decoder trials aligned to the start of each virtual-navigation trial.

The full output is `/app/converted_data.pkl`. It contains 152 dorsal-CA1 calcium-imaging sessions from 11 mice, 138,678 Suite2p-classified cells across sessions, and 12,135 valid trials.

## Reproduce

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

`--sample` converts two sessions. `--show-processing` saves plots for up to two sessions.

## Load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First trial of first session
neural = data['neural'][0][0]   # float32: neurons x time
inputs = data['input'][0][0]    # float32: 4 x time
outputs = data['output'][0][0]  # int64: 6 x time
```

Trials use a 64.4836 ms time bin (about 15.5 Hz), include the `trial_start` sample, and stop before teleport.

## Variables

Inputs:
1. time from trial start (seconds)
2. environment (ENV1=0, ENV2=1)
3. source trial number
4. previous source-trial reward outcome

Categorical outputs:
1. signed distance to the nearest point in the current reward zone (7 bins)
2. absolute corridor position (5 bins)
3. speed (5 bins)
4. lick (binary)
5. reward-zone location (A/B/C)
6. reward outcome (omitted/rewarded)

Per-trial context/output values are repeated over time to provide uniform matrices.

## Processing Summary

- Concatenates synchronized imaging planes and uses released Suite2p deconvolved calcium events.
- Retains ROIs with Suite2p `iscell` flag only; no place-cell or reward-relative selection is imposed.
- Uses explicit start/teleport pulses, not trial-ID boundaries. Teleport samples are excluded.
- Excludes 81 lick-sensor-corrupt trials using the paper criterion (>30% of trial frames with cumulative lick count >2), exactly matching the paper’s reported count.
- Converts valid lick counts to binary.
- Derives A/B/C schedules and fixed zones from the reference code: A=80–130 cm, B=200–250 cm, C=320–370 cm; switches occur at source trial index 30.
- Uses sparse NWB reward timestamps for current and previous outcomes.
- Ten two-plane files contain one extra neural-only trailing frame after behavior; this unusable frame is strictly trimmed.

## Key Statistics

- Subjects: 11
- Sessions: 152
- Trials: 12,135 retained (12,216 complete released trials minus 81 corrupt-lick trials)
- Curated cells: 138,678 total across sessions (155–2,341/session; mean 912.36)
- Reward outcomes: 15.36% omitted, 84.64% rewarded by trial
- Framewise lick: 22.26% positive
- Reward zones are approximately balanced across frame samples (A 33.16%, B 33.59%, C 33.25%).

Full validation and decoder results are documented in `/app/CONVERSION_NOTES.md` and the `*_out.txt` logs. Full validation balanced accuracies were 0.3538 (distance), 0.5037 (position), 0.4044 (speed), 0.5957 (lick), 0.8153 (zone), and 0.5044 (reward outcome).
