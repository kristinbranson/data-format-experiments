# Track2p Neural Decoder Dataset

This directory contains a decoder-ready conversion of the longitudinal calcium-imaging dataset from Majnik et al. (2025), *Longitudinal tracking of neuronal activity from the same cells in the developing brain using Track2p*.

## Dataset summary

- 6 mice, 41 daily imaging sessions from layer 2/3 barrel cortex
- 2,998 unique longitudinal cell tracks; 20,445 session-neuron recordings
- 1,090 non-overlapping 60-second trials
- 180 time bins per trial (3 Hz; 333.333 ms bins)
- Decoder input: elapsed time from session start, in seconds
- Decoder output: aligned motion energy classified into five independently selected per-session percentile bins
- Exactly 20% of timepoints occupy each output class

Neural data are Suite2p-style baseline-subtracted fluorescence. The converter calculates `F - 0.7*Fneu`, subtracts a Gaussian/maximin baseline (sigma 10 native frames, 60-second window), and averages non-overlapping groups of 10 native 30-Hz frames. Motion is averaged on the same grid. Documented dropped camera frames are located from timestamp gaps and linearly interpolated before averaging.

## Load the converted data

```python
import pickle

with open("/app/converted_data.pkl", "rb") as handle:
    data = pickle.load(handle)

# First session, first trial
neural = data["neural"][0][0]   # (n_neurons, 180), float32
inputs = data["input"][0][0]    # (1, 180), float32
outputs = data["output"][0][0]  # (1, 180), int64 values 0..4
```

Top-level fields are `neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and `metadata`. `metadata["session_info"]` records each session's source dimensions, gap repairs, class thresholds, and trial count.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The official full validation reports no errors or warnings. Full decoder validation balanced accuracy is 0.3069 versus 0.20 chance; training balanced accuracy is 0.6045. See `CONVERSION_NOTES.md` for all source/reference comparisons, decisions, independent checks, and accuracy analysis.

## Important source discrepancy

The released files contain different tracked-cell counts from Figure 5 for four mice, and four mice contain 30-minute sessions although the paper states 20 minutes. The conversion preserves every valid released row and aligned timepoint rather than inventing absent cells or discarding valid recordings. These differences are fully documented in `CONVERSION_NOTES.md`.

