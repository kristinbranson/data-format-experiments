# Converted Track2p Development Dataset

## Summary
This repository contains a converted decoder-ready version of the Majnik et al. 2025 longitudinal calcium imaging dataset from mouse barrel cortex.

- Source data: `/app/data`
- Conversion script: `/app/convert_data.py`
- Converted dataset: `/app/converted_data.pkl`
- Notes and validation record: `/app/CONVERSION_NOTES.md`

## Conversion choices
- Sessions are continuous recordings split into non-overlapping 60 s trials.
- Neural signal uses an approximate fluorescence-based dF/F reconstruction from Suite2p outputs:
  - neuropil correction: `F - 0.7 * Fneu`
  - low-percentile baseline normalization using Suite2p baseline percentile metadata
  - non-overlapping 10-frame averaging to match the paper's denoising step
- Decoder input: time elapsed from session start in seconds
- Decoder output: motion energy discretized into 5 equal-percentile bins separately for each session
- Missing behavior frames are handled by interpolating motion energy onto neural frame times before binning.

## Output format
The pickle file stores a dictionary with keys:
- `neural`: list of sessions, each a list of trials, each array `(n_neurons, n_timepoints)`
- `input`: list of sessions, each a list of trials, each array `(1, n_timepoints)` for `time_elapsed_sec`
- `output`: list of sessions, each a list of trials, each array `(1, n_timepoints)` for `motion_energy_quantile`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Basic statistics
- Subjects: 6
- Sessions: 41
- Trials: 1090
- Total neurons across sessions: 20445
- Trial length: 180 bins (60 s with 10-frame bins at 30 Hz)

## Usage
Verify format:
```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Train decoder:
```bash
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Re-run conversion:
```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```
