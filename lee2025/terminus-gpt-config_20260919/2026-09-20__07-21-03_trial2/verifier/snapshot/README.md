# Converted CA1 Geometry Dataset

## Overview

This directory contains a decoder-ready conversion of the dataset from **Lee et al. (2025), “Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping.”** Mice freely explored a 75 × 75 cm arena whose accessible partitions varied on a 3 × 3 grid. The decoding task predicts the mouse's 3 × 3 position class from dorsal CA1 calcium-event activity. Arena accessibility is provided as static contextual input.

Primary output: `/app/converted_data.pkl`

## Key statistics

- 7 mice
- 207 recording-day sessions
- 8,187 complete, contiguous one-minute trials
- 39 or 40 trials/session
- 600 timepoints/trial at 100 ms/timepoint
- 113–564 simultaneously present neurons/session (mean 336.93)
- 69,744 session-neuron instances
- One brain region: dorsal CA1
- Nine categorical position classes
- Output class fractions: `[0.09966, 0.09853, 0.13512, 0.07541, 0.05723, 0.07685, 0.11570, 0.14121, 0.20029]`

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First session, first one-minute trial
neural = data['neural'][0][0]   # (n_neurons, 600), float32
context = data['input'][0][0]   # (9,), float32 accessibility mask
position = data['output'][0][0] # (1, 600), int8 classes 0..8
```

## Dictionary fields

- `neural[session][trial]`: float32 matrix `(n_neurons, 600)`. Deposited binary rising-event traces were Gaussian-smoothed at sigma 3 native frames and averaged in non-overlapping three-frame bins, matching the reference decoder's temporal processing.
- `input[session][trial]`: static float32 vector `(9,)`, row-major 3 × 3 arena accessibility (`1=accessible`, `0=blocked`).
- `output[session][trial]`: int8 matrix `(1,600)`, time-varying row-major position class `class = y_bin*3 + x_bin`.
- `subjects`: seven animal IDs.
- `subject_idx`: subject index for each of 207 sessions.
- `brain_regions`: `['CA1']`.
- `brain_region_idx[session]`: zero-valued region index for every neuron.
- `input_names`: names of the nine accessibility entries.
- `output_names`: `['position_bin']`.
- `output_values`: human-readable names for the nine position classes.
- `metadata`: processing description, temporal information, and per-session provenance.

## Processing summary

1. Load the authors' preprocessed compressed joblib files.
2. Treat each recording day as one session.
3. Retain every manually curated cell present on that day; remove cross-day registered cells represented entirely by NaN on that day.
4. Preserve the provided binary calcium rising-phase events rather than recomputing dF/F or deconvolution.
5. Match reference temporal decoder processing: Gaussian smoothing (`sigma=3` native 30 Hz frames), then stride-3 mean pooling of neural and position streams.
6. Discretize pooled x-y coordinates into 25 cm bins on the required 3 × 3 grid.
7. Split each recording into complete, non-overlapping 60-second trials. Trailing sub-minute frames are not padded.
8. Correct 152 rare pooled tracking points in nominally blocked partitions by snapping to the nearest accessible class, analogous to reference valid-map snapping.

See `/app/CONVERSION_NOTES.md` for detailed rationale, source comparisons, edge cases, and validation history. The conversion is reproducible with:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

Use `--sample` for two sessions and `--show-processing` to generate processing plots.

## Validation

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Final format validation reports no errors or warnings. Independent raw-source comparisons passed `np.allclose` for neural, input, and output streams. Full decoder results:

- Training balanced accuracy: **0.6925**
- Validation balanced accuracy: **0.6053**
- Uniform chance: **0.1111**
- Training loss: 2.327492 → 1.159876 over 200 epochs
- Test loss: 1.136346

## Important files

- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: comprehensive conversion record
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: run logs
- `processing_*.png`, `sample_trials.png`, `predictions.png`: visual checks
- `cache/`: investigation scripts and extracted summaries
