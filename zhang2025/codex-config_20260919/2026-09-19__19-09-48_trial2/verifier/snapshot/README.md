# IBL Brain-Wide Map Neural Decoder Dataset

This directory contains a stimulus-onset-aligned conversion of the International Brain
Laboratory Brain-Wide Map release. It is designed for `/app/train_decoder.py`: neural spike
counts are decoder features, while choice, prior, wheel speed, and whisker motion energy are
categorical prediction targets.

## Main files

- `converted_data.pkl` — full converted dataset (12.559 GiB)
- `sample_data.pkl` — two-session test subset
- `convert_data.py` — reproducible converter
- `CONVERSION_NOTES.md` — detailed decisions, reference comparisons, audits, and results
- `conversion_full_out.txt`, `verification_full_out.txt`, and
  `train_decoder_full_out.txt` — full-run logs
- `processing_*.png` and `predictions.png` — processing and decoder visual checks

## Dataset summary

| Item | Value |
|---|---:|
| Sessions | 444 |
| Subjects | 136 |
| Trials | 188,925 |
| Curated session-units | 73,044 |
| Beryl region labels represented | 266 |
| Time bins per trial | 100 |
| Bin width | 20 ms |
| Alignment window | -0.5 to +1.5 s from stimulus onset |

The source freeze contains 459 sessions, 699 probes, 139 mice, 621,733 sorted units, and
75,708 well-isolated units. Fourteen sessions lack usable whisker motion energy and one has
no camera samples spanning eligible trial windows, leaving 444 sessions. Trials must pass
the reference event, 0.08–2.00-s reaction-time, <=10-s duration, and nonzero-choice rules,
and must have complete wheel and motion-energy coverage. Units satisfy the data paper's
well-isolated criterion (`clusters.metrics.label >= 1`). Simultaneous probes are combined.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# Example: first trial in first session
neural = data["neural"][0][0]  # (n_neurons, 100), float32 spike counts
inputs = data["input"][0][0]   # (2, 100), float32
outputs = data["output"][0][0] # (4, 100), int64
```

Loading the full pickle requires at least roughly 13 GiB of available memory, with additional
headroom recommended for analysis.

## Variables

Inputs, in order:

1. `time_since_stimulus_onset`: right edges of the 20-ms bins, -0.48 through +1.50 s.
2. `trial_number_in_block`: zero-based native trial number within the current
   `probabilityLeft` block, repeated over time.

Outputs, in order:

1. `choice`: left = 0, right = 1; constant within a trial.
2. `prior_probability_left`: 0.2 = 0, 0.5 = 1, 0.8 = 2; constant within a trial.
3. `wheel_speed`: low/medium/high = 0/1/2, using session-specific tertiles of absolute,
   20-Hz-low-pass-filtered wheel velocity.
4. `whisker_motion_energy`: low/medium/high = 0/1/2, using session-specific tertiles;
   left camera is preferred and right camera is the documented fallback.

Behavior streams use the reference interpolation convention at bin right edges. Neural bins
are half-open intervals spanning `[-0.5, 1.5)` relative to stimulus onset.

## Dictionary layout

`neural`, `input`, and `output` are nested lists indexed `[session][trial]`.
`subject_idx` maps sessions into `subjects`; each `brain_region_idx[session]` maps that
session's neurons into `brain_regions`. `metadata.session_info` records source session IDs,
native trial indices, probe/cluster identities, filter counts, cameras, and tertile thresholds.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The completed full decoder achieved held-out balanced accuracies of 0.6196 (choice), 0.6724
(prior), 0.6164 (wheel), and 0.6012 (whisker), versus chance values of 0.5, 1/3, 1/3, and
1/3. Training loss decreased from 2.0332 to 0.7296.

See `CONVERSION_NOTES.md` for the complete reference-code comparison, raw-file `np.allclose`
checks, warning adjudication, and accuracy review.
