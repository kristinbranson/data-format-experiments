# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the NWB release accompanying *A flexible hippocampal population code for experience relative to reward*. The dataset comprises dorsal CA1 two-photon recordings from 11 mice navigating a 450-cm virtual corridor with hidden reward zones and approximately 15% reward omissions.

## Main files

- `converted_data.pkl`: complete converted dataset (152 sessions).
- `sample_data.pkl`: two representative environment-switch sessions.
- `convert_data.py`: reproducible PyNWB-only conversion.
- `CONVERSION_NOTES.md`: source exploration, mapping decisions, statistics, sanity checks, and decoder results.
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: captured conversion, validation, and training logs.
- `processing_*.png`, `sample_trials.png`, `predictions.png`: processing and decoder diagnostics.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

trial_neural = data["neural"][0][0]  # neurons x time
trial_input = data["input"][0][0]    # 4 x time
trial_output = data["output"][0][0]  # 6 x time
```

The pickle is approximately 8.86 GiB, so loading it requires comparable free RAM.

## Format

The session and trial nesting is identical for `neural`, `input`, and `output`. Neural values are float32 OASIS-deconvolved calcium activity. Inputs are float32 and ordered as:

1. time from trial start (seconds)
2. environment (0=ENV1, 1=ENV2)
3. zero-based raw trial number
4. previous raw trial outcome (0=omitted, 1=rewarded)

Outputs are int8 categorical time series ordered as:

1. signed distance to active reward zone (7 classes)
2. absolute corridor position (5 classes)
3. speed (5 classes)
4. lick (2 classes)
5. reward-zone location (0=A, 1=B, 2=C)
6. trial reward outcome (0=omitted, 1=rewarded)

Class names and exact boundary semantics are stored in `output_values` and documented in `CONVERSION_NOTES.md`. Per-trial variables are repeated over the trial time axis. Trials align to the `trial_start` flag and include `[trial_start, teleport)` at a common nominal bin size of 64.483627 ms.

Additional top-level fields provide subject indices, CA1 brain-region indices, variable names, and detailed session metadata including source paths and curation counts.

## Processing and curation

All NWB access uses `pynwb.NWBHDF5IO`. For each manually curated cell (`iscell[:,0]`), fluorescence is neuropil-corrected and processed trial by trial using the paper's maximin dF/F procedure, two-sample smoothing, and Suite2p OASIS deconvolution. Both imaging planes are pooled. Cells with dF/F-speed Pearson r>0.5 are excluded. Low-speed frames remain because speed is a requested output. Trials are excluded only for the paper-defined lick-sensor failure (>30% of samples with cumulative count >2).

## Key statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions | 152 |
| Raw / retained trials | 12,216 / 12,135 |
| Manual / retained session-neurons | 138,678 / 138,276 |
| Trial time bins | 2,576,026 |
| Retained reward rate | 84.639% |
| Brain region | dorsal CA1 |

The full format validator reported no errors or warnings. Full validation balanced accuracies were 0.4982 distance, 0.6098 position, 0.5416 speed, 0.7249 lick, 0.8462 zone, and 0.5523 outcome; every value is above chance.

## Reproducing

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

See `CONVERSION_NOTES.md` before changing filtering or alignment: it records the resolved differences between NWB metadata, exported Suite2p deconvolution, reference code, and manuscript methods.
