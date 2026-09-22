# Track2p Neural Decoder Dataset

This project converts the Majnik et al. (2025) longitudinal Track2p dataset into 60-second trials for decoding spontaneous mouse motion from layer 2/3 barrel-cortex calcium activity.

## Main output

`converted_data.pkl` contains:

- 6 mice and 41 daily sessions (7/7/7/7/6/7 sessions per mouse)
- 2,998 unique longitudinally matched neurons; 20,445 neuron-session rows
- 820 trials (20 consecutive 60-second trials per session)
- 180 time points per trial at 333.333 ms per bin
- neural matrices shaped `(n_neurons, 180)`
- one time-varying input: elapsed seconds from session start
- one time-varying output: session-specific motion-energy quintile, labels 0–4
- exactly 20% of session samples in each output class

The distributed neuron counts are 221, 370, 685, 746, 541, and 435 for `jm031`, `jm032`, `jm038`, `jm039`, `jm040`, and `jm046`, respectively.

## Load the data

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

neural_trial = data["neural"][0][0]  # neurons × time
time_trial = data["input"][0][0]     # 1 × time
motion_class = data["output"][0][0] # 1 × time, integer 0..4
```

The dictionary follows the requested session/trial structure and includes `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, variable names/value labels, and detailed processing/session metadata.

## Processing summary

For each session, the converter:

1. retains the released Track2p-matched cells (already filtered at Suite2p cell probability >0.5 and tracked across all days);
2. applies stored Suite2p defaults: `F - 0.7*Fneu`, Gaussian/maximin baseline estimation, and baseline subtraction;
3. processes the full available fluorescence trace for baseline context, then retains the first 36,000 frames (20 minutes), matching the paper;
4. reconstructs skipped camera-trigger positions from timestamp interval multiples and linearly interpolates motion onto imaging frame indices;
5. averages neural activity and motion over non-overlapping 10-frame groups, matching the reference decoder preprocessing;
6. computes motion quintiles separately per session and splits the result into consecutive 60-second trials.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full conversion takes about 34 seconds in the supplied environment. Format verification reports no errors or warnings. Full validation balanced accuracy is 0.2950 versus 0.2000 chance; a paper-style continuous ridge check reproduces the expected developmental rise (example P8/P14 R² 0.2078/0.6431 versus approximately 0.25/0.69 in the paper).

See `CONVERSION_NOTES.md` for all decisions, paper/release discrepancies, checks, plots, and decoder results.

