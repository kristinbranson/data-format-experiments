# Brain-wide Memory-Guided Movement Decoder Dataset

## Overview

`converted_data.pkl` reformats the supplied NWB release of **Brain-wide neural activity underlying memory-guided movement** for neural decoding. Mice performed an auditory delayed-response task while brain-wide Neuropixels activity and high-speed orofacial video were recorded.

The conversion:

- loads NWB files exclusively with `pynwb`;
- aligns every trial to auditory **go-cue onset**;
- extracts **−2.5 to +1.5 seconds** in **80 non-overlapping 50-ms bins**;
- converts classifier-curated spike times to firing rates in Hz;
- supplies time from tone onset and photostimulation state as decoder inputs;
- supplies actual lick choice, outcome, early lick, and tongue y-position category as outputs.

## Files

- `converted_data.pkl` — final full converted dataset (about 11.76 GB)
- `sample_data.pkl` — two-session test dataset
- `convert_data.py` — reproducible converter
- `CONVERSION_NOTES.md` — detailed decisions, source comparisons, checks, iterations, and decoder results
- `conversion_full_out.txt`, `verification_full_out.txt` — full conversion and format-validation logs
- `train_decoder_full_out.txt` — completed full decoder training log
- `processing_*.png` — sample processing/alignment diagnostics
- `cache/` — investigation and sanity-check scripts

## Key Statistics

| Statistic | Final value |
|-----------|-------------|
| Subjects | 28 |
| Sessions | 173 |
| Trials | 90,859 |
| Selected neurons | 68,888 |
| Neurons/session | 90–923 |
| Trials/session | 159–800 |
| Time points/trial | 80 |
| Bin width | 50 ms |
| Alignment | go-cue onset |
| Brain-region categories | 14 lateralized recording targets |

One NWB file was excluded because all units lacked classifier labels. Units require `classification == "good"` and all-true `is_good_trials`. A further 565 classifier-good units with invalid trial flags were removed to keep neuron identity fixed within session. Trials with zero total raw spikes across every selected unit over the complete four-second window were treated as acquisition gaps and excluded; all behavioral classes otherwise remain included.

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))                 # 173 sessions
print(data['neural'][0][0].shape)         # neurons x 80
print(data['input'][0][0].shape)          # 2 x 80
print(data['output'][0][0].shape)         # 4 x 80
```

The three per-trial categorical outputs are repeated over 80 points so that they share one uniform output matrix with time-varying tongue position.

## Dictionary Fields

- `neural[session][trial]`: float32 firing-rate matrix, neurons × 80, Hz
- `input[session][trial]`: float32 2 × 80 matrix
  1. continuous seconds from the latest tone/sample onset before go
  2. binary photostimulation state at each bin center
- `output[session][trial]`: int8 4 × 80 matrix
  1. lick choice: left=0, right=1, no lick=2
  2. outcome: ignore=0, miss=1, hit=2
  3. early lick: no=0, yes=1
  4. tongue y: below p40=0, p40–p60=1, above p60=2, not visible=3
- `subjects`, `subject_idx`: subject lookup and session mapping
- `brain_regions`, `brain_region_idx`: 14 lateralized coarse recording targets and per-neuron indices
- `input_names`, `output_names`, `output_values`: dimension labels
- `metadata`: time grid, filters, source files, raw trial indices, session tongue thresholds, unit IDs, and processing details

Tongue percentiles are calculated independently for each session from all finite Camera0 side-view tongue-y samples with DeepLabCut likelihood ≥0.9. A bin is not visible if confidence is lower or no camera sample lies within 10 ms of its center.

## Reproducing the Conversion

```bash
# Two-session sample with processing plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full conversion
python -u /app/convert_data.py /app/converted_data.pkl --full

# Validate format
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Train full decoder and create sample plots
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Final Decoder Performance

Balanced validation accuracy on the complete dataset:

| Output | Accuracy | Uniform chance |
|--------|----------|----------------|
| Lick direction choice | 0.6880 | 0.3333 |
| Outcome | 0.6583 | 0.3333 |
| Early lick | 0.7483 | 0.5000 |
| Tongue y-position | 0.6161 | 0.2500 |

See `CONVERSION_NOTES.md` for paper comparisons, raw NWB `np.allclose()` checks, filtering rationale, edge-case analysis, and all corrected issues.
