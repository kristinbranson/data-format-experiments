# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the two-photon CA1 data accompanying *A flexible hippocampal population code for experience relative to reward*.

## Main files

- `converted_data.pkl`: complete dataset (152 sessions, 12,135 retained trials)
- `sample_data.pkl`: first two sessions for quick tests
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: full decision log, source reconciliation, sanity checks, and decoder results
- `verification_full_out.txt`: supplied format validator output
- `train_decoder_full_out.txt`: complete 200-epoch decoder run

Regenerate and validate with:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Use `--sample` instead of `--full` for two sessions. Adding `--show-processing` saves diagnostic plots for up to two sessions.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# Session 0, trial 0
neural = data["neural"][0][0]  # neurons x time, float32
inputs = data["input"][0][0]   # 4 x time, float32
outputs = data["output"][0][0] # 6 x time, int8
```

Sessions are sorted by numeric mouse ID and day. Trials retain their original zero-based within-session numbers after filtering.

## Variables

Inputs, in row order:

1. seconds from trial start
2. environment (`0=ENV1`, `1=ENV2`)
3. original zero-based trial number
4. previous raw-trial outcome (`0=omitted`, `1=rewarded`; first trial is 0)

Categorical outputs, in row order:

1. signed distance to the nearest point in the active reward zone: 7 classes
2. absolute position on the 450 cm corridor: 5 classes
3. speed: 5 classes
4. lick (`0=no`, `1=yes`)
5. zone (`0=A`, `1=B`, `2=C`)
6. current trial outcome (`0=omitted`, `1=rewarded`)

Exact class names and boundaries are stored in `output_values` and documented in `CONVERSION_NOTES.md`.

## Processing and curation

- Trials are aligned to `trial_start` and contain native imaging rows through, but not including, `teleport`; bin size is 64.4836 ms (15.5078125 Hz).
- Neural activity is recomputed from raw fluorescence and neuropil using the paper's trial-wise maximin dF/F processing, then Suite2P OASIS deconvolution.
- Manual Suite2P `iscell` curation is applied, followed by the paper's dF/F–speed Pearson `r>0.5` putative-interneuron exclusion.
- Trials are removed when more than 30% of frames have corrupted cumulative lick count greater than 2, reproducing the paper's 81 exclusions.
- m17/m18 plane-1 segmentation is present, but the supplied NWBs contain response arrays only for plane 0. This unavoidable source limitation is recorded in metadata.

## Key statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Raw paired imaging trials | 12,216 |
| Retained trials | 12,135 |
| Manual curated plane-0 cells | 118,493 session-neurons |
| Retained cells | 118,169 session-neurons |
| Neurons/session | mean 777.43; range 154–1,780 |
| Rewarded raw trials | 84.66% |
| Brain region | CA1 |

The supplied validator reports no errors or warnings. Full held-out balanced accuracies are 0.5206 (zone distance), 0.6187 (position), 0.5721 (speed), 0.7591 (lick), 0.8423 (zone), and 0.5799 (outcome); all exceed chance.

