# Allen Visual Behavior Ophys Decoder Dataset

## Description

`converted_data.pkl` reformats the supplied Allen Brain Observatory Visual Behavior Ophys 1.1.0 NWB subset for neural decoding. Each converted session is one imaging experiment/plane, matching the reference paper's decoding unit. Trials are native **Go** and **Catch** trials; aborted and auto-rewarded trials are excluded.

Neural data are released AllenSDK detected calcium-event magnitudes, resampled by linear interpolation to a 30 Hz grid anchored to each experiment's ophys timestamps. Released valid ROIs are retained. Three experiments without pupil tracking are documented in `metadata['excluded_sessions']` and excluded because pupil diameter is a required output.

## Key Statistics

- 281 imaging-plane sessions from 38 mice
- 84,313 trials
- 41,871 neurons (session sum; 4–666 per plane)
- Regions: VISp (41,633 neurons) and VISl (238 neurons)
- Fixed bin duration: 33.333 ms; native trial durations are preserved
- Full pickle size: approximately 13.7 GB

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First session, first trial
neural = data['neural'][0][0]  # neurons x time, float32 event magnitude
inputs = data['input'][0][0]   # shape (0, time): no decoder inputs
outputs = data['output'][0][0] # shape (5, time), categorical integers
print(neural.shape, inputs.shape, outputs.shape)
```

Because the file is large, loading requires sufficient RAM. Session-level metadata in `data['metadata']['session_info']` include source IDs, mouse, region, cell type, experience level, trial/neuron counts, and quintile edges.

## Output Rows

Rows of each `output` trial follow `output_names`:

1. `image_identity`: gray plus 16 image identities. Gray includes the 500 ms inter-image interval and omitted presentations; image labels occur only during the actual non-gray presentation.
2. `image_change`: binary one-bin pulse at the first 30 Hz sample at or after a true image-change onset.
3. `running_speed_quintile`: five within-experiment percentile bins from filtered running speed.
4. `pupil_diameter_quintile`: five within-experiment percentile bins. Diameter is the equivalent-circle diameter derived from filtered pupil ellipse area.
5. `trial_outcome`: hit, miss, false alarm, or correct reject. This static trial label is repeated over time to coexist with time-varying outputs in one matrix.

Category names and ordering are in `output_values`. There are intentionally no decoder inputs, so `input_names` is empty and each input matrix has shape `(0, T)`.

## Reproduction and Validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The full validator reports warnings for 3,915 trials with no detected events. Independent checks against original NWBs confirmed these are genuine all-zero event intervals, not conversion loss. Full validation accuracies and all methodological decisions are documented in `CONVERSION_NOTES.md`.
