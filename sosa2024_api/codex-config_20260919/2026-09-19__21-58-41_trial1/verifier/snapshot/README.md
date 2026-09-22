# Neural Decoder Dataset

This directory contains a decoder-ready conversion of the NWB data accompanying *A flexible hippocampal population code for experience relative to reward*. The full dataset is `converted_data.pkl`; `sample_data.pkl` contains two representative sessions for quick testing.

## Dataset summary

- 11 mice, 152 CA1 imaging sessions
- 12,216 raw trials; 12,135 retained after excluding the paper's 81 bad-lick-sensor trials
- 138,276 retained cell-session recordings after manual Suite2p `iscell` curation and exclusion of 402 putative interneurons with dF/F-speed Pearson r > 0.5
- 2,576,026 aligned imaging/behavior samples at 15.5078125 Hz (64.483627 ms)
- Neural signal: manuscript-matched, within-trial maximin dF/F followed by OASIS deconvolution
- Trial window: `[trial_start, teleport)`; trials have variable duration

All source NWB files are read through `pynwb`. See `CONVERSION_NOTES.md` for the complete source audit, processing rationale, raw-to-converted checks, and decoder results.

## Loading

The full pickle is about 8.9 GB and should be loaded on a machine with sufficient RAM:

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

session = 0
trial = 0
neural = data["neural"][session][trial]  # (neurons, time)
inputs = data["input"][session][trial]   # (4, time), float32
outputs = data["output"][session][trial] # (6, time), int8
```

Top-level fields are `neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and `metadata`. Session and trial ordering is shared by the first three fields. `metadata["session_info"]` records source files, source/retained trial counts, neuron counts, scene, and QC information for every session.

## Variables

Decoder inputs, in row order:

1. Time from trial start in seconds
2. Environment (`ENV1=0`, `ENV2=1`)
3. Zero-based source trial number
4. Previous source-trial outcome (`omitted=0`, `rewarded=1`)

Categorical decoder outputs, in row order:

1. Signed distance to the active reward-zone interval: 7 requested classes
2. Absolute 450-cm corridor position: 5 requested classes
3. Speed: 5 requested classes
4. Lick: binary
5. Reward-zone location: A/B/C
6. Reward outcome: omitted/rewarded

The exact class labels and boundary conventions are stored in `output_values` and metadata.

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The full validator reports no errors or warnings. Full validation balanced accuracies were 0.5411 distance, 0.6646 position, 0.5656 speed, 0.7492 lick, 0.8440 reward zone, and 0.5607 reward outcome; every value is above its categorical chance level. Logs and processing/prediction plots are retained in this directory.

