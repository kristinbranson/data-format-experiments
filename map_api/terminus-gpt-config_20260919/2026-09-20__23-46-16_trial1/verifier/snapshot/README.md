# Converted MAP Neural Decoder Dataset

## Dataset

This directory contains a decoder-ready conversion of the Mesoscale Activity Map (MAP) electrophysiology dataset from **“Brain-wide neural activity underlying memory-guided movement.”** The supplied NWB archive contains auditory delayed-response behavior, Neuropixels spikes, photostimulation, and high-speed behavioral tracking.

The conversion follows the published classifier-based spike-sorting quality control and uses `pynwb` for all source-data access. Detailed decisions, checks, discrepancies, and decoder results are in [`CONVERSION_NOTES.md`](CONVERSION_NOTES.md).

## Main files

- `converted_data.pkl` — full converted dataset (173 sessions, 89,068 trials)
- `sample_data.pkl` — two-session test conversion
- `convert_data.py` — reproducible converter
- `conversion_full_out.txt`, `verification_full_out.txt` — full conversion and format validation logs
- `train_decoder_full_out.txt` — completed full decoder training log
- `processing_*.png` — sample processing/alignment plots

## Reproduce

```bash
# Two-session test with plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full conversion (also the default mode)
python -u /app/convert_data.py /app/converted_data.pkl --full

# Validate or train
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Load

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

rates = data['neural'][0][0]   # (neurons, 80), float32 Hz
inputs = data['input'][0][0]   # (2, 80), float32
outputs = data['output'][0][0] # (4, 80), integer categories
```

The file is approximately 11 GiB; loading it requires corresponding RAM.

## Processing summary

- **Alignment:** go cue onset
- **Window:** `[-2.5, +1.5)` seconds
- **Bins:** 80 non-overlapping 50-ms bins
- **Neural values:** spike counts divided by 0.05 s, in Hz
- **Units:** 69,453 session-neurons with NWB `classification == "good"`
- **Subjects:** 28 mice
- **Sessions:** 173 usable sessions (one NWB had zero classifier-good units)
- **Trials:** 89,068 after excluding assisted-water and invalid/unrecorded neural trials
- **Regions:** 293 raw Allen-style anatomical labels

Short per-unit validity vectors are mapped to the correct behavioral trial block using NWB `obs_intervals`. Residual all-neuron-zero boundary trials are excluded.

## Fields

- `neural[session][trial]`: `(n_neurons, 80)` float32 firing rates
- `input[session][trial]`: `(2, 80)`
  1. continuous seconds from auditory tone onset
  2. binary photostimulation-on state
- `output[session][trial]`: `(4, 80)` categorical integers
  1. choice: left, right, no lick
  2. outcome: ignore, miss, hit
  3. early lick: no, yes
  4. tongue y: below 40th percentile, 40th–60th, above 60th, not visible
- `subjects`, `subject_idx`: subject vocabulary and per-session indices
- `brain_regions`, `brain_region_idx`: anatomical vocabulary and per-neuron indices
- `metadata`: timing, filters, source-session information, source trial indices, and tongue thresholds

Trial-level choice/outcome/early-lick labels are repeated across 80 bins so they can coexist with time-varying tongue output in one matrix.

Tongue percentiles are computed per session over frames with tracking likelihood at least 0.9. Low-confidence frames are category 3 (“not visible”).

## Full decoder results

| Output | Training balanced accuracy | Validation balanced accuracy | Chance |
|---|---:|---:|---:|
| Lick direction choice | 0.7108 | 0.6803 | 0.3333 |
| Outcome | 0.6953 | 0.6569 | 0.3333 |
| Early lick | 0.7912 | 0.7495 | 0.5000 |
| Tongue y-position | 0.6757 | 0.6194 | 0.2500 |

Training completed 200 epochs successfully; loss decreased from 21.4482 to 0.6725.
