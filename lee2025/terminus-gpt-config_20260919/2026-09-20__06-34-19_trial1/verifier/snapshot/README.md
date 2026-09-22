# CA1 Geometry Dataset — Decoder Conversion

## Overview

This conversion reformats the dataset from Lee, Keinath, Cianfarano & Brandon, *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping* (Neuron, 2025) for neural decoding.

The decoder uses processed CA1 calcium-event activity to predict mouse position in one of nine 25 × 25 cm bins of a 75 × 75 cm arena. A static 3 × 3 binary mask indicates which arena partitions are blocked. Continuous animal-day recordings are divided into complete non-overlapping one-minute trials.

## Files

- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: conversion and validation logs
- `processing_*.png`: conversion/alignment diagnostic plots
- `sample_trials.png`, `predictions.png`: full decoder sample plots

## Reproduce the conversion

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

For a two-session test with diagnostic plots:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Validate or train:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

first_neural_trial = data['neural'][0][0]  # neurons × 600 time bins
first_geometry = data['input'][0][0]       # length-9 static blocked mask
first_position = data['output'][0][0]      # 1 × 600 integer classes
```

## Format

- `neural[session][trial]`: float32 `(n_neurons, 600)` arrays. Values are binary rising-phase calcium events after reference Gaussian temporal smoothing and 100-ms pooling.
- `input[session][trial]`: float32 `(9,)` static row-major geometry mask; 1 means blocked.
- `output[session][trial]`: int64 `(1, 600)` position classes 0–8.
- `subjects`: seven mouse IDs.
- `subject_idx`: subject index for each session.
- `brain_regions`: `['CA1']`.
- `brain_region_idx[session]`: all-zero CA1 indices, one per retained neuron.
- `input_names`: blocked NW, N, NE, W, center, E, SW, S, SE.
- `output_names`: `['position_bin']`.
- `output_values[0]`: NW, N, NE, W, center, E, SW, S, SE.
- `metadata`: timing, processing description, source provenance, and detailed per-session information.

The 100-ms time bins match the reference decoder's three-frame processing at the 30-Hz source rate. Each trial spans 60 seconds (`off_start=0`, `off_end=60`).

## Key statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Sessions | 207 animal-days |
| Trials | 8,187 one-minute trials |
| Trials/session | 39 or 40 |
| Time bins/trial | 600 (100 ms each) |
| Source day-cell recordings | 69,744 |
| Retained decoder cell recordings | 68,862 |
| Brain region | CA1 |
| Geometries | 10 |
| Output classes | 9 |
| Output class fraction range | 5.73%–20.03% |
| Full validation balanced accuracy | 60.81% |
| Chance balanced accuracy | 11.11% |

## Processing summary

1. Load the seven author-provided animal joblib files.
2. Select finite cells and match reference decoder curation: Gaussian-smoothed speed >5 cm/s and more than five events during moving frames.
3. Gaussian-smooth each binary event trace with sigma three source frames and average non-overlapping three-frame groups, matching reference decoder processing.
4. Average aligned x/y samples over the same frames and assign fixed 25-cm row-major position classes.
5. Expand blocked partition indices to a static nine-element binary mask.
6. Split from session start into complete contiguous 60-second trials; record and discard only the incomplete tail.

See `CONVERSION_NOTES.md` for rationale, exact source comparisons, raw `np.allclose` sanity checks, and all validation results.
