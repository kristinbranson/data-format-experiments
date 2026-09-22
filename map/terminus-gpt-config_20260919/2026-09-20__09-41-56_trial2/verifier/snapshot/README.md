# MAP Neural Decoder Dataset Conversion

## Overview

This directory contains a decoder-ready conversion of the **Mesoscale Activity Map (MAP) Dataset** (DANDI `000363`, version `0.230822.0128`), originally presented in *Brain-wide neural activity underlying memory-guided movement* and reused in *Brain-wide analysis reveals movement encoding structured across and within brain areas*.

The final dataset is `/app/converted_data.pkl`. Neural activity is aligned to **go cue onset**, spans **-2.5 to +1.5 seconds**, and is represented as firing rates in **80 contiguous 50-ms bins**.

## Files

- `converted_data.pkl`: full converted dataset (173 sessions)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: complete exploration, decisions, validation, and review record
- `conversion_sample_out.txt`, `conversion_full_out.txt`: conversion logs
- `verification_sample_out.txt`, `verification_full_out.txt`: format validation logs
- `train_decoder_sample_out.txt`, `train_decoder_full_out.txt`: decoder training logs
- `processing_*.png`: conversion diagnostic plots
- `sample_trials.png`, `predictions.png`: full decoder sample/prediction plots

## Reproduce the Conversion

```bash
# Two-session test with diagnostic plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full dataset (default mode is also full)
python -u /app/convert_data.py /app/converted_data.pkl --full

# Validate
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Train and create sample plots
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Load the Data

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First trial of first session
neural = data['neural'][0][0]   # (n_neurons, 80), float32, spikes/s
inputs = data['input'][0][0]    # (2, 80), float32
outputs = data['output'][0][0]  # (4, 80), int64
```

The pickle is approximately 11.94 GB, so loading requires sufficient RAM.

## Data Structure

- `neural[session][trial]`: classifier-curated neuron firing rates, shape `(n_neurons, 80)`
- `input[session][trial]`: shape `(2, 80)`
  1. time from auditory tone onset in seconds
  2. photostimulation state (0/1)
- `output[session][trial]`: shape `(4, 80)`
  1. lick direction: left=0, right=1, no lick=2
  2. outcome: ignore=0, miss=1, hit=2
  3. early lick: no=0, yes=1
  4. tongue y-position: below session q40=0, q40–q60=1, above q60=2, not visible=3
- `subjects` and `subject_idx`: subject lookup and per-session indices
- `brain_regions` and `brain_region_idx`: bilateral electrode target-region lookup and per-neuron indices
- `metadata`: bin timing, alignment, source, curation, tongue threshold, and session statistics

Per-trial choice, outcome, and early-lick labels are tiled across 80 time bins so they can coexist in one rectangular output array with time-varying tongue position.

## Processing Summary

### Neural activity

- Units must have NWB `units/classification == 'good'`, the paper's region-specific classifier QC result.
- To maintain a fixed population within each session, units must also be valid on every represented source trial according to `is_good_trials`.
- Nine sessions contain behavior after neural recording ended; only the neural represented-trial prefix is used.
- Population-wide four-second source spike gaps are excluded as invalid acquisition periods (2,576 trials). No final trial has all-zero population activity.
- Spike counts use half-open bins and are divided by 0.05 s to produce spikes/s.

### Alignment and inputs

- The unique go-start event within each trial is time zero.
- Tone onset is the latest sample-start event before go cue. Long tone-to-go intervals on early-lick trials reflect real state-machine delays.
- Photostimulation uses exact NWB start/stop event intervals and is sampled at each bin center.

### Outputs

- Choice is derived from authoritative instruction/outcome labels: hit uses the instructed side, miss uses the opposite side, and ignore is no lick. This agrees with the first post-go lick on 99.742% of responsive non-early trials.
- Tongue tracking uses side-camera DeepLabCut y and likelihood. Likelihood below 0.9 is “not visible.” Visible velocity outliers above five standard deviations are interpolated, following the method paper. The 40th/60th percentiles are computed independently over each session's cleaned visible y samples.

## Final Statistics

| Statistic | Value |
|---|---:|
| Subjects | 28 |
| Sessions | 173 |
| Trials | 90,734 |
| Retained neurons summed across sessions | 68,888 |
| Trials/session | min 159, mean 524.47, max 800 |
| Neurons/session | min 90, mean 398.20, max 923 |
| Time bins/trial | 80 |
| Brain-region labels | 14 bilateral target labels |

Output fractions:

- Choice: left 42.847%, right 42.222%, no lick 14.930%
- Outcome: ignore 14.930%, miss 16.640%, hit 68.430%
- Early lick: no 88.448%, yes 11.552%
- Tongue: low 6.212%, middle 3.175%, high 6.511%, not visible 84.103%

## Validation and Decoder Results

`verification_full_out.txt` reports:

> Data format is valid, no errors or warnings.

Full validation balanced accuracies:

| Output | Balanced accuracy | Uniform chance |
|---|---:|---:|
| Lick direction choice | 0.6645 | 0.3333 |
| Outcome | 0.6547 | 0.3333 |
| Early lick | 0.7519 | 0.5000 |
| Tongue y-position | 0.5990 | 0.2500 |

Training loss decreased from 27.4954 to 0.7073 over 200 epochs. Direct original-NWB `np.allclose` checks for neural rates, both inputs, and all outputs are documented in `CONVERSION_NOTES.md`.
