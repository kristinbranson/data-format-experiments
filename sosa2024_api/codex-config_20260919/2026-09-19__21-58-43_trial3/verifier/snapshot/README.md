# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the NWB data associated with
*A flexible hippocampal population code for experience relative to reward*. The conversion
uses `pynwb` throughout and recreates the paper's per-trial maximin dF/F and OASIS calcium-event
processing from fluorescence and neuropil signals.

## Main Files

- `converted_data.pkl`: complete converted dataset (152 sessions)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: detailed decisions, reference comparisons, and validation record
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: captured run logs
- `processing_sub-*.png`: raw-to-converted processing/alignment checks
- `sample_trials.png`, `predictions.png`: full-decoder diagnostic plots

## Dataset Summary

| Item | Value |
|------|-------|
| Subjects | 11 mice |
| Sessions | 152 (77 switch sessions) |
| Source / retained trials | 12,216 / 12,135 |
| Manually curated / retained neuron-sessions | 138,678 / 138,276 |
| Brain region | CA1 |
| Retained time bins | 2,576,026 |
| Nominal bin duration | 64.484 ms (15.5078 Hz) |
| Trial interval | trial-start sample inclusive to teleport sample exclusive |
| Rewarded trials | 10,271 / 12,135 (84.64%) |

The 81 excluded trials exactly meet the paper's lick-sensor corruption rule (>30% of frames
with cumulative lick count >2). Manual Suite2p cell labels are followed by the paper's putative
interneuron exclusion (dF/F–speed Pearson correlation >0.5). All provided sessions are retained;
place-cell-only filtering is not applied because the requested outputs also include movement,
licking, zone identity, and reward outcome.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First trial of the first session
neural = data["neural"][0][0]  # neurons x time
inputs = data["input"][0][0]   # 4 x time
outputs = data["output"][0][0] # 6 x time
```

The top-level keys are `neural`, `input`, `output`, `subjects`, `subject_idx`,
`brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and
`metadata`. Sessions and trials have matching list order across the first three fields.

Input rows are:

1. time from trial start (seconds, time-varying)
2. environment (ENV1=0, ENV2=1; per trial)
3. source trial number (continuous, per trial)
4. previous source-trial outcome (omitted=0, rewarded=1; per trial)

Output rows are categorical:

1. signed distance to the active reward zone (7 classes)
2. absolute corridor position (5 classes)
3. speed (5 classes)
4. lick presence (2 classes)
5. reward-zone identity A/B/C (3 classes)
6. trial reward outcome (2 classes)

Exact class labels and boundaries are stored in `output_values` and documented in
`CONVERSION_NOTES.md`. Variable-length trials preserve every native imaging-frame sample.

## Reproducing and Validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The completed full decoder achieved validation balanced accuracies of 0.574 (zone distance),
0.696 (absolute position), 0.603 (speed), 0.751 (lick), 0.849 (zone identity), and 0.579
(reward outcome). All exceed chance. Reward outcome is intrinsically harder because random
omission/delivery cannot be known from neural activity in the pre-reward portion of a trial,
while the required label is repeated across the whole trial.

See `CONVERSION_NOTES.md` for raw-data `np.allclose` checks, edge-case tests, reconciliations
against the paper, and the complete decoder review.
