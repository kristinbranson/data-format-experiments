# IBL Neural Decoder Dataset

`converted_data.pkl` is a stimulus-onset-aligned conversion of the International Brain Laboratory brain-wide map electrophysiology release for neural decoding. It contains 444 usable sessions from 136 mice, 188,922 trials, and 599,865 session-summed Kilosort clusters. Fourteen release sessions lack both usable left- and right-camera whisker motion energy; one additional session has no trials jointly covered by all required streams, so those sessions are recorded as exclusions rather than assigned fabricated targets.

## Load the data

The full pickle is 106.081 GB and needs a high-memory machine to load:

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

# First trial of the first session
neural = data["neural"][0][0]  # (898 neurons, 100 time bins)
inputs = data["input"][0][0]   # (2 inputs, 100 time bins)
outputs = data["output"][0][0] # (4 outputs, 100 time bins)
```

For lightweight testing, `sample_data.pkl` contains two sessions in the identical schema.

## Representation

- Alignment: visual stimulus onset (`stimOn_times`).
- Window: `[-0.5, +1.5)` seconds.
- Binning: 100 half-open 20-ms bins; neural values are raw float32 spike counts.
- Behavior sampling: the right edge of each neural bin (`-0.48, ..., +1.50` seconds).
- Inputs: time since stimulus onset and zero-based trial number within the raw probability block, both time-varying arrays.
- Outputs: choice (`left=0`, `right=1`), prior probability left (`0.2=0`, `0.5=1`, `0.8=2`), and within-session tertile classes (`low=0`, `medium=1`, `high=2`) for wheel speed and whisker motion energy.
- Native choice convention is mapped explicitly: IBL `+1` is left and becomes 0; IBL `-1` is right and becomes 1.
- Brain regions are cluster-channel CCF locations mapped to Beryl acronyms.

Each trial has neural shape `(n_neurons, 100)`, input shape `(2, 100)`, and output shape `(4, 100)`. Static choice/prior values are repeated across time so static and dynamic targets share one consistent trial array.

## Key statistics

| Statistic | Value |
|-----------|-------|
| Sessions | 444 |
| Subjects | 136 |
| Trials | 188,922 |
| Session-summed neurons | 599,865 |
| Neurons/session | mean 1,351.05; range 135–3,140 |
| Beryl region labels | 281 |
| Choice distribution | 50.8% left, 49.2% right |
| Prior distribution | 41.7% / 14.1% / 44.2% for 0.2 / 0.5 / 0.8 |
| Wheel classes | approximately 33.3% each |
| Whisker classes | 33.3% / 33.3% / 33.5% (ties explain rounding) |

The provided validator reports no errors or warnings. Final validation balanced accuracies are 0.5667 choice, 0.5856 prior, 0.5807 wheel, and 0.5669 whisker; all exceed chance.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Detailed decisions, source comparisons, exclusions, sanity checks, and review iterations are in `CONVERSION_NOTES.md`. Exact command outputs are retained in the `conversion_*`, `verification_*`, and `train_decoder_*` text logs. `sample_trials.png`, `predictions.png`, and the two `processing_<eid>.png` files provide visual audits.

