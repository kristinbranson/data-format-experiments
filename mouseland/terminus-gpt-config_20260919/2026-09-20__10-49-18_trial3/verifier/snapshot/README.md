# Zhong et al. Neural Decoder Dataset

This directory contains a decoder-compatible conversion of the imaging dataset from **“Unsupervised pretraining in biological neural networks”**.

## Main files

- `converted_data.pkl` — complete converted dataset (89 sessions, 19 mice, 38,110 trials; 296.435 GB decimal).
- `sample_data.pkl` — representative two-session dataset for fast structural/testing workflows.
- `convert_data.py` — reproducible conversion script.
- `CONVERSION_NOTES.md` — detailed decisions, reference comparisons, validation, and decoder results.
- `verification_full_out.txt` — full format validation report.
- `train_decoder_full_out.txt` — complete full-dataset decoder training log.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))       # 89 sessions
print(data['input_names'])
print(data['output_names'])
```

The full pickle requires substantial RAM and disk I/O. Use `sample_data.pkl` for lightweight inspection.

## Format

`neural`, `input`, and `output` are nested lists indexed as `[session][trial]`:

- `neural[s][t]`: float32 `(neurons, time)` Suite2p deconvolved activity.
- `input[s][t]`: float32 `(4, time)`:
  1. signed time to sound cue (seconds),
  2. calendar day from the subject's first included recording,
  3. time since corridor entry (seconds),
  4. reward availability (0/1).
- `output[s][t]`: integer `(4, time)`:
  1. visual category (8 classes),
  2. binary licking,
  3. position in four 1 m bins,
  4. running speed in four global quartile bins.

Trials use native imaging frames (~315 ms) from `ceil(StartFr)` through, but excluding, `ceil(GrayFr)`, aligned to visual corridor entry. Per-trial variables are broadcast over time to satisfy the common aligned time axis.

## Key statistics

- Subjects: 19
- Sessions: 89
- Trials: 38,110
- Session-ROI traces: 4,691,034 total; 20,547–89,577 per session
- Retained trial frames: 1,375,142
- Brain regions: V1, mHV, lHV, aHV, unassigned
- Speed thresholds: 0, 8.3812388, 30.1893780 source units
- Full validation: no errors or warnings

## Reproducing conversion

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
```

## Decoder results (full dataset)

| Output | Validation balanced accuracy | Chance |
|---|---:|---:|
| Visual category | 0.4939 | 0.125 |
| Licking | 0.8565 | 0.500 |
| Position bin | 0.3270 | 0.250 |
| Speed quartile | 0.4308 | 0.250 |

See `CONVERSION_NOTES.md` for processing rationale and critical reviews.
