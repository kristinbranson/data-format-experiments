# Allen Visual Behavior Neural-Decoder Dataset

This directory contains a decoder-ready conversion of the local Allen Brain Observatory Visual Behavior 2P release. It uses active `VisualBehavior` experiments with usable eye tracking, released valid ROIs, unfiltered FastLZero L0 calcium-event magnitudes, and native ophys timestamps.

## Files

- `converted_data.pkl`: complete conversion (165 sessions)
- `sample_data.pkl`: two-session test conversion
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: processing decisions, reference comparisons, and validation record
- `conversion_*_out.txt`, `verification_*_out.txt`, and `train_decoder_*_out.txt`: captured run logs
- `processing_<session_id>.png`: sample alignment/discretization diagnostics
- `cache/`: investigation scripts and auxiliary plots

Reproduce the outputs with:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Loading and structure

The pickle contains the requested nested dictionary. Trial neural matrices have shape `(n_neurons, n_timepoints)` and `float32` dtype. There are no non-neural decoder inputs, so each input matrix has shape `(0, n_timepoints)`. Output matrices have shape `(5, n_timepoints)` and integer categorical values.

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

neural_trial = data["neural"][0][0]
output_trial = data["output"][0][0]
print(neural_trial.shape, output_trial.shape)
print(data["output_names"])
```

The five output rows are:

1. `image_identity`: `gray` plus 16 globally consistent natural-image identities.
2. `image_change`: 1 throughout the changed image's on-screen presentation, otherwise 0.
3. `running_speed_quintile`: within-session quintiles after interpolation to ophys frames.
4. `pupil_diameter_quintile`: within-session quintiles of blink-masked pupil major-axis diameter after interpolation.
5. `trial_outcome`: one of `hit`, `miss`, `false_alarm`, or `correct_reject`, repeated across the trial.

Each variable-length trial is the half-open native interval `start_time <= ophys_timestamp < stop_time`. The median frame interval is 32.31 ms. Retained trials satisfy `(go OR catch) AND NOT aborted AND NOT auto_rewarded`.

## Key statistics

| Statistic | Value |
|-----------|------:|
| Sessions | 165 |
| Mice | 37 |
| Eligible trials | 42,470 |
| Session-neuron recordings | 28,821 |
| Unique longitudinal cell IDs | 13,627 |
| Aggregate trial frames | 11,192,974 |
| Neurons/session | mean 174.67, range 6–666 |
| Trial frames | mean 262.02, range 217–389 |
| Go / catch trials | 37,143 / 5,327 |
| Hit / miss / false alarm / correct reject | 13,569 / 23,574 / 814 / 4,513 |

The supplied full decoder completed 200 epochs successfully. Validation balanced accuracies were 0.1605 for image identity (chance 0.0588), 0.5833 for image change (0.5), 0.2182 for running quintile (0.2), 0.2142 for pupil quintile (0.2), and 0.2612 for trial outcome (0.25).

For rationale and caveats—including the three excluded eye-absent experiments, source-version count differences, genuine all-zero sparse-event trials, and direct raw-data `np.allclose` checks—see `CONVERSION_NOTES.md`.
