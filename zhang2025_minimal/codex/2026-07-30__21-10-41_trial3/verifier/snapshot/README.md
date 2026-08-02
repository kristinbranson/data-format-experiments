# IBL BWM Neural Decoder Conversion

This workspace converts the IBL brain-wide map release into the `train_decoder.py` pickle format used in this task.

## Files

- `convert_data.py`: conversion script.
- `converted_data.pkl`: full converted dataset.
- `sample_data.pkl`: smaller converted subset for quick validation.
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation notes.
- `conversion_sample_out.txt`: stdout/stderr from the sample conversion run.
- `verification_sample_out.txt`: verifier output for `sample_data.pkl`.
- `train_decoder_sample_out.txt`: decoder training output for `sample_data.pkl`.
- `conversion_full_out.txt`: stdout/stderr from the full conversion run.
- `verification_full_out.txt`: verifier output for `converted_data.pkl`.
- `train_decoder_full_out.txt`: decoder training output for `converted_data.pkl`.

## Conversion Summary

The converter:

1. Uses the 2025 BWM release table in [bwm_release.csv](/app/code/code_zhang2025/data/bwm_release.csv).
2. Loads session data through ONE from the local cache at `/app/data`, downloading missing ALF files if needed.
3. Applies the trial filtering described in the papers/code:
   - required non-NaN trial events
   - reaction time between `0.08` and `2.0` s
   - no-choice trials excluded
   - maximum trial duration of `10.0` s
4. Keeps well-isolated units with `clusters.metrics.label >= 1`.
5. Restricts units to grey-matter regions and drops `void`, `root`, and `grey`.
6. Maps retained unit acronyms to the `Beryl` atlas, then keeps only regions with at least `5` neurons in a session and at least `2` sessions overall.
7. Merges all probes within a session.
8. Aligns trials to stimulus onset with a `[-0.5, 1.5]` s window and `20` ms bins.
9. Uses wheel speed from interpolated + filtered wheel velocity and whisker motion energy from the left camera only.
10. Discretizes wheel speed and whisker motion energy into 3 global tertile bins.

## Commands

Sample conversion:

```bash
python /app/convert_data.py \
  --output /app/sample_data.pkl \
  --summary-json /app/sample_summary.json \
  --max-sessions 8
```

Sample verification:

```bash
python /app/train_decoder.py /app/sample_data.pkl --verify-only --plot-samples --cpu
```

Sample training:

```bash
python /app/train_decoder.py /app/sample_data.pkl --plot-samples --cpu
```

Full conversion:

```bash
python /app/convert_data.py \
  --output /app/converted_data.pkl \
  --summary-json /app/full_summary.json
```

Full verification:

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only --plot-samples --cpu
```

Full training:

```bash
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Current Sample Stats

The current sample run contains:

- 8 sessions
- 4 subjects
- 2,830 trials
- 441 neurons
- 10 Beryl brain regions

The current full run contains:

- 433 sessions
- 133 subjects
- 184,664 trials
- 59,751 neurons
- 208 Beryl brain regions

The full verifier passes with no warnings, and the decoder validation balanced accuracies are:

- `choice`: `0.6145`
- `prior_probability_left`: `0.6575`
- `wheel_speed_bin`: `0.6276`
- `whisker_motion_energy_bin`: `0.7306`
