# Track2p Neural Decoder Dataset

This directory contains a decoder-ready conversion of the longitudinal Track2p dataset from Majnik et al. (2025), recorded in layer 2/3 mouse barrel cortex during spontaneous behavior.

## Main files

- `converted_data.pkl`: complete converted dataset (41 sessions, 1,090 trials).
- `sample_data.pkl`: two-session test conversion (50 trials).
- `convert_data.py`: reproducible conversion script.
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, validation, and decoder results.
- `verification_full_out.txt` and `train_decoder_full_out.txt`: full format-validation and decoder-training logs.
- `processing_*.png`: visual audits of neural, behavioral, alignment, binning, and discretization steps.

## Dataset summary

| Property | Value |
|----------|-------|
| Subjects | 6 (`jm031`, `jm032`, `jm038`, `jm039`, `jm040`, `jm046`) |
| Sessions | 41 (6–7 daily sessions per mouse) |
| Unique tracked neurons | 2,998 |
| Neurons per session | 221–746 (mean 498.66) |
| Brain region | Primary somatosensory barrel cortex, layer 2/3 |
| Trials | 1,090 non-overlapping 60-s blocks |
| Timepoints per trial | 180 |
| Time-bin size | 333.333 ms (means of 10 frames acquired at 30 Hz) |
| Decoder input | Absolute elapsed seconds from session start |
| Decoder output | Session-specific motion-energy quintile (classes 0–4) |
| Class distribution | Exactly 20% per class in every complete session |

Neural activity is `F - 0.7*Fneu` followed by the Suite2p maximin baseline subtraction used in the reference code and a 10-frame mean. Sparse missing camera frames (276 of 1,962,000 behavior frames after reconstruction) are localized from timestamp gaps and linearly interpolated before identical temporal averaging. Provided neural rows were already filtered at Suite2p cell probability >0.5 and Track2p-curated as complete longitudinal tracks.

## Loading the data

```python
import pickle

with open("/app/converted_data.pkl", "rb") as handle:
    data = pickle.load(handle)

# First session, first 60-second trial
neural = data["neural"][0][0]  # (n_neurons, 180), float32
decoder_input = data["input"][0][0]  # (1, 180), float32
decoder_output = data["output"][0][0]  # (1, 180), int64 in 0..4
```

The top-level dictionary contains `neural`, `input`, and `output` nested as session → trial, plus `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, variable names/value labels, and detailed processing/session metadata.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The full conversion takes about 44 seconds in the supplied environment. Validation reports no errors or warnings. The supplied decoder achieved 0.3084 held-out balanced accuracy versus 0.2000 chance; full details and negative-control leakage checks are in `CONVERSION_NOTES.md`.

