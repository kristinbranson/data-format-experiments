# Zhong et al. Neural Decoder Dataset

This directory contains a decoder-ready conversion of the calcium-imaging data from **“Unsupervised pretraining in biological neural networks”** (Zhong et al., 2025).

## Main files

- `converted_data.pkl` — full dataset: 89 recordings, 19 mice, 38,110 trials, 141.126 GiB.
- `sample_data.pkl` — two-session test dataset: 705 trials, 3.154 GiB.
- `convert_data.py` — reproducible conversion script.
- `CONVERSION_NOTES.md` — full decision log, source comparisons, sanity checks, and decoder results.
- `verification_full_out.txt` and `train_decoder_full_out.txt` — format validation and complete decoder-training logs.

## Loading

The full pickle requires substantial RAM. Load it with:

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)
```

The top-level fields are:

```text
neural[session][trial]       float32 (neurons, time)
input[session][trial]        float32 (4, time)
output[session][trial]       int8    (4, time)
subjects                     list[str]
subject_idx                  integer array (sessions,)
brain_regions                [V1, mHV, lHV, aHV]
brain_region_idx[session]    integer array (neurons,)
input_names/output_names/output_values
metadata                     provenance, thresholds, and session details
```

Inputs, in order:

1. signed seconds to the sound cue (positive before cue),
2. training day/stage,
3. seconds since textured-corridor entry,
4. reward availability for the corridor.

Outputs, in order:

1. visual category: circle, leaf, rock, or wood,
2. binary licking,
3. position: four 1-m bins over the 0–4 m textured corridor,
4. global running-speed quartile.

## Processing summary

- Neural values are the supplied float32 Suite2p non-negative deconvolved traces; no new dF/F or normalization is applied.
- Imaging planes are concatenated in reference order.
- Cells outside mapped visual cortex (`iarea` -1 or 7) are excluded; V1 and three grouped higher-visual regions are retained.
- Frame streams are truncated to their common neural/behavior length, matching the reference code.
- Trials align to textured-corridor entry. Included samples satisfy `ft_CorrSpc & (ft_move > 0)`, matching the paper’s running-timepoint curation.
- Timing inputs use original frame timestamps, so stationary gaps remain represented in elapsed time even though stationary neural samples are omitted.
- Native imaging rate is 3.17 Hz (315.46 ms per frame).
- Speed thresholds in the full data are 12.42242, 25.352615, and 40.854578 native units; the four classes contain 205,394–205,395 samples each.

## Key statistics

| Statistic | Full dataset |
|---|---:|
| Sessions | 89 |
| Subjects | 19 |
| Trials | 38,110 |
| Raw recorded ROIs | 4,691,034 |
| Retained mapped visual-cortex ROIs | 4,105,393 |
| Retained ROIs/session | 17,363–78,815 |
| Curated timepoints | 821,579 |
| Position-bin fractions | 0.250, 0.249, 0.250, 0.252 |
| Speed-quartile fractions | 0.250 each |
| Licking-positive fraction | 0.037 |

Full held-out balanced accuracies were 0.7007 (visual category), 0.8539 (licking), 0.4002 (position), and 0.3573 (speed), versus chance values 0.25, 0.50, 0.25, and 0.25.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

See `CONVERSION_NOTES.md` for alias handling, day-field resolution, raw `np.allclose` comparisons, edge-case review, and the complete rationale for every mapping decision.
