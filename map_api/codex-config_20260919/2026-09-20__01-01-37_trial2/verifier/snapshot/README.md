# Mesoscale Activity Map decoder dataset

This directory contains a decoder-ready conversion of DANDI:000363/0.230822.0128, the electrophysiology and behavior dataset from *Brain-wide neural activity underlying memory-guided movement*. The conversion uses 69,453 released classifier-QC units from 28 mice and 173 sessions.

## Use

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# First trial of first session
neural = data["neural"][0][0]   # neurons × 80 time bins, firing rate in Hz
inputs = data["input"][0][0]    # 2 × 80
outputs = data["output"][0][0]  # 4 × 80 categorical values
```

Reproduce conversion and validation with:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Use `--sample --show-processing` with the converter for a two-session test and diagnostic plots.

## Format and processing

Trials are aligned to go-cue onset and span -2.5 to +1.5 seconds in 80 nonoverlapping 50-ms bins. Neural matrices contain firing rates from half-open spike bins. Inputs are continuous time from the first tone/sample onset and binary photostimulation state. Outputs are lick choice (left/right/no lick), outcome (ignore/miss/hit), early lick (no/yes), and time-varying tongue y-position (below the session 40th percentile, 40th–60th percentile, above the 60th percentile, or not visible). Tongue visibility uses side-camera DeepLabCut likelihood ≥0.9.

Per-trial choice, outcome, and early-lick labels are repeated across the 80 time bins so all outputs share a rectangular time-varying representation. `brain_region_idx` maps curated units to 293 fine CCF annotation names. Full source-session details, thresholds, counts, and curation diagnostics are in `metadata["session_info"]`.

## Key statistics

- Sessions: 173
- Subjects: 28
- Trials with valid neural data: 90,734
- Curated session-units: 69,453
- Neurons/session: median 390, mean 401.46, range 90–923
- Trials/session: median 522, mean 524.47, range 159–800
- Choice fractions: left 0.428, right 0.422, no lick 0.149
- Outcome fractions: ignore 0.149, miss 0.166, hit 0.684
- Early-lick fractions: no 0.884, yes 0.116
- Tongue fractions: low 0.062, middle 0.032, high 0.065, not visible 0.841

Full validation balanced accuracies were 0.6727 (choice), 0.6580 (outcome), 0.7592 (early lick), and 0.6204 (tongue). See [CONVERSION_NOTES.md](/app/CONVERSION_NOTES.md) for the complete rationale, reference comparisons, discovered edge cases, and independent raw-data checks.

