# Allen Visual Behavior Neural Decoder Dataset

This workspace converts the locally cached Allen Brain Observatory **VisualBehavior** 2-photon dataset into trial-structured neural-decoder data. It includes 165 active VISp imaging sessions from 37 mice, 42,470 eligible go/catch trials, and 28,821 valid cell-session observations. Three otherwise eligible sessions are excluded because their AllenSDK eye-tracking tables are empty and the required pupil output cannot be constructed.

## Files

- `converted_data.pkl`: complete converted dataset (7.6 GiB)
- `sample_data.pkl`: deterministic two-session test dataset
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: processing decisions, reference comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: captured run logs
- `processing_<experiment_id>.png`: sample processing/alignment diagnostics
- `sample_trials.png`, `predictions.png`: full-decoder sample plots
- `cache/`: investigation scripts and intermediate review artifacts

## Reproduce

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The converter uses detailed data only through `VisualBehaviorOphysProjectCache`; it never directly opens NWB files.

## Load and use

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# First session, first trial
neural = data["neural"][0][0]   # float32, (neurons, time)
inputs = data["input"][0][0]    # float32, (0, time): no inputs requested
outputs = data["output"][0][0]  # int16, (5, time)
```

All streams use a 30 Hz (33.333 ms) trial-relative grid derived from synchronized ophys timestamps. Trials are half-open SDK intervals `[start_time, stop_time)`. Retained trials are active go/catch trials excluding aborted and auto-rewarded trials.

Output rows are:

1. `image_identity`: gray plus 16 natural-image identities (sets A and B)
2. `image_change`: 1 throughout the 250 ms changed-image flash
3. `running_speed_quintile`: cohort-wide bins 0–4
4. `pupil_diameter_quintile`: cohort-wide bins 0–4
5. `trial_outcome`: hit, miss, false alarm, or correct reject; constant within a trial

Neural values are AllenSDK raw L0 calcium event magnitudes from valid ROIs, linearly interpolated from synchronized ophys frames to 30 Hz. Running uses the SDK’s reference-filtered cm/s stream. Pupil diameter is the cleaned major ellipse axis; blink/outlier gaps are interpolated before cohort quintiling.

## Key results

Format validation passes with no errors. The verifier reports 1,717 all-zero neural trials (4.04%); independent SDK checks confirm these are genuine trials with no detected raw calcium events, not conversion failures.

Full validation balanced accuracies were:

| Output | Accuracy | Chance |
|---|---:|---:|
| Image identity | 0.2063 | 0.0588 |
| Image change | 0.6001 | 0.5000 |
| Running quintile | 0.2537 | 0.2000 |
| Pupil quintile | 0.2819 | 0.2000 |
| Trial outcome | 0.2662 | 0.2500 |

See `CONVERSION_NOTES.md` for full rationale, source statistics, exclusions, and independent `np.allclose` checks.
