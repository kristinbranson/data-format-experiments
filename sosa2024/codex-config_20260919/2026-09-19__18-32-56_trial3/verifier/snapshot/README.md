# Reward-Relative CA1 Decoder Dataset

This directory contains a decoder-ready conversion of the two-photon CA1 data associated with *A flexible hippocampal population code for experience relative to reward*.

## Main files

- `converted_data.pkl`: complete converted dataset (152 sessions, 12,135 trials)
- `sample_data.pkl`: representative two-session test dataset
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed source review, decisions, statistics, and validation record
- `verification_full_out.txt`: full format-validation report
- `train_decoder_full_out.txt`: completed full decoder training log
- `sample_trials.png`, `predictions.png`: full-dataset inspection plots
- `processing_m12_ses-10.png`, `processing_m18_ses-11.png`: conversion-stage plots for representative one- and two-plane sessions

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First trial from the first session
neural = data["neural"][0][0]   # neurons × time
inputs = data["input"][0][0]    # 4 × time
outputs = data["output"][0][0]  # 6 × time
```

Trials have variable duration but use a common 64.4836-ms sample interval. Neural values are author-produced OASIS-deconvolved calcium events (`float32`). Inputs are `float32`; categorical outputs are `int8`.

Input rows are:

1. time from explicit trial start (seconds)
2. environment (ENV1=0, ENV2=1)
3. native zero-based trial number
4. previous raw trial outcome (omitted=0, rewarded=1)

Output rows are:

1. signed distance-to-reward-zone class (7 classes)
2. absolute 450-cm track-position class (5 classes)
3. speed class (5 classes)
4. lick (0/1)
5. reward-zone location (A=0, B=1, C=2)
6. current reward outcome (0/1)

Per-trial variables are broadcast across the time dimension. Exact class labels and boundaries are stored in `output_values` and `metadata["output_discretization"]`.

## Key statistics

| Statistic | Value |
|---|---:|
| Subjects | 11 |
| Sessions | 152 |
| Retained trials | 12,135 |
| Retained neurons summed over sessions | 138,269 |
| Neurons/session | 154–2,323 (mean 909.66) |
| Samples/trial | 96–3,359 (mean 215.53) |
| Sampling rate | 15.5078125 Hz |
| Rewarded trials | 84.64% |
| Brain region | CA1 |

Manual Suite2p cells were retained, then 409 cells with paper-method dF/F–speed Pearson `r > 0.5` were excluded. Trials are sliced from the explicit start frame through the frame before teleport. Eighty-one trials matching the paper's faulty-lick criterion were excluded.

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The full validator reports no errors or warnings. Validation balanced accuracies were 0.4048 distance, 0.4998 position, 0.3850 speed, 0.6162 lick, 0.8067 zone, and 0.5144 outcome; all exceed chance.

