# Sosa et al. CA1 Decoder Dataset

This directory contains a decoder-ready conversion of the NWB data accompanying *A flexible hippocampal population code for experience relative to reward*.

## Main files

- `converted_data.pkl`: full dataset (152 sessions, 11 mice, 12,135 retained trials; ~9.0 GiB)
- `sample_data.pkl`: two representative switch sessions for quick testing
- `convert_data.py`: reproducible NWB-to-pickle converter
- `CONVERSION_NOTES.md`: processing rationale, paper/code comparisons, QC, and decoder results
- `verification_full_out.txt` and `train_decoder_full_out.txt`: full validation/training logs

## Loading

The full pickle needs roughly its on-disk size plus Python overhead in available RAM.

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First session, first trial
neural = data["neural"][0][0]  # curated CA1 cells x native time bins
inputs = data["input"][0][0]   # 4 x time bins
outputs = data["output"][0][0] # 6 x time bins
```

All trials preserve the native 64.483627-ms imaging/behavior bin. Trial lengths vary because laps were not resampled. `neural`, `input`, and `output` have identical time lengths within a trial.

Inputs, in order, are time from trial start, environment (ENV1=0/ENV2=1), zero-based trial number, and previous-trial outcome (omitted=0/rewarded=1). Outputs are the requested categorical reward-zone distance, absolute position, speed, lick, reward-zone identity, and reward outcome. See `output_values` in the pickle for class labels.

## Processing summary

- Used author-computed OASIS-deconvolved calcium events.
- Applied each NWB's manual Suite2p `iscell` mask and pooled both planes for m17/m18.
- Sliced zero-based NWB intervals `[trial_start, teleport)`, excluding teleport/ITI periods.
- Removed 81 lick-sensor artifact trials using the paper criterion (>30% of frames with cumulative lick count >2).
- Inferred A/B/C schedules from observed zone-entry positions separately before and after the documented trial-30 switch; no observation contradicted its inferred schedule.
- Defined signed distance as negative before the active 50-cm interval, zero anywhere inside it, and positive after it.

The result contains 138,678 curated cell-session entries from 312,110 raw ROIs and 2,576,026 retained timepoints. Formal validation reported no errors or warnings. Full validation balanced accuracies were 0.3703 (distance), 0.4888 (position), 0.4317 (speed), 0.6098 (lick), 0.8137 (zone), and 0.5192 (outcome), all above chance.

## Reproducing

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The independent raw-data audit and its log are archived under `cache/`.
