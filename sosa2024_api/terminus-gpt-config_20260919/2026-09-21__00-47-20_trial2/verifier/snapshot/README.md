# Reward-Relative CA1 Dataset: Decoder Conversion

## Overview

This directory contains a decoder-compatible conversion of the NWB release associated with **“A flexible hippocampal population code for experience relative to reward.”** The dataset contains longitudinal two-photon calcium imaging from hippocampal CA1 while mice navigate a 450 cm virtual corridor in two visual environments with hidden reward zones A, B, and C.

The conversion reads NWB exclusively through `pynwb`, retains Suite2p-classified cells, aligns trials to trial start, and preserves the native imaging-frame interval.

## Main Files

- `converted_data.pkl` — full converted dataset (152 sessions; approximately 13.3 GB)
- `sample_data.pkl` — two-session test conversion
- `convert_data.py` — reproducible converter
- `CONVERSION_NOTES.md` — detailed decisions, source comparisons, checks, and decoder results
- `conversion_full_out.txt`, `verification_full_out.txt` — full conversion/format logs
- `train_decoder_full_out.txt` — completed full decoder training log

## Reproduce Conversion

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Validate or train:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Load the Dataset

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural_trial = data['neural'][0][0]  # neurons x time
input_trial = data['input'][0][0]    # 4 x time
output_trial = data['output'][0][0]  # 6 x time
```

The pickle is large; loading it requires sufficient RAM.

## Data Structure

- `neural[session][trial]`: float32 Suite2p deconvolved activity, shape `(n_cells, n_time)`
- `input[session][trial]`: float32 array, shape `(4, n_time)`
  1. time from trial start (seconds)
  2. environment (0=ENV1, 1=ENV2)
  3. native zero-based trial number
  4. previous trial outcome (0=omitted/no prior, 1=rewarded)
- `output[session][trial]`: int8 categorical array, shape `(6, n_time)`
  1. signed distance to the active reward-zone interval (7 classes)
  2. absolute corridor position (5 classes)
  3. speed (5 classes)
  4. lick (binary)
  5. active reward-zone location (A/B/C)
  6. trial reward outcome (binary)
- `subjects`, `subject_idx`: subject lookup and per-session indices
- `brain_regions=['CA1']`, `brain_region_idx`: region lookup and per-neuron indices
- `metadata`: sampling interval, alignment, zone geometry, signal type, and per-session provenance

Per-trial context and outcomes are repeated over time so all output rows share one shape.

## Processing Summary

- Neural signal: NWB `ophys/Deconvolved`
- Cell curation: Suite2p `iscell[:, 0] == 1`
- Multi-plane sessions: all RoiResponseSeries are mapped through their NWB DynamicTableRegion links and concatenated
- Time bin: 64.483627 ms (~15.508 Hz)
- Alignment: first frame of each complete numbered trial (`off_start=0`)
- Reward zones: A=80–100 cm, B=200–220 cm, C=320–340 cm
- Reward outcome: timestamped NWB `Reward` event within a trial
- Lick output: native lick value >0
- Trial filtering: one malformed three-frame terminal fragment was excluded; rewarded and omitted trials were retained

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Valid trials | 12,216 |
| Curated CA1 cells summed across sessions | 138,678 |
| Cells/session | 155–2,341; mean 912.36 |
| Typical trials/session | 80 |
| Rewarded trials | 10,342 (84.66%) |
| Omitted trials | 1,874 (15.34%) |

## Full Decoder Results

Validation balanced accuracy:

| Output | Accuracy | Uniform chance |
|--------|----------|----------------|
| Distance to reward zone | 0.3772 | 0.1429 |
| Absolute position | 0.5000 | 0.2000 |
| Speed | 0.4084 | 0.2000 |
| Lick | 0.6318 | 0.5000 |
| Reward-zone location | 0.7375 | 0.3333 |
| Reward outcome | 0.5076 | 0.5000 |

All outputs exceeded chance. See `CONVERSION_NOTES.md` for interpretation and independent raw-NWB `np.allclose` audits.
