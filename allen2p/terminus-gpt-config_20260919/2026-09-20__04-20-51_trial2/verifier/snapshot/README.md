# Allen Visual Behavior Ophys Decoder Dataset

## Overview

`converted_data.pkl` contains a decoder-ready conversion of the supplied Allen Brain Observatory Visual Behavior 2P NWB subset. It includes active go/no-go image-change task experiments and uses released detected calcium events as neural activity.

### Curation

- Active task experiments only; passive viewing experiments are excluded.
- Go and catch trials are included.
- Aborted and auto-rewarded trials are excluded.
- Three active experiments without eye tracking are excluded because pupil diameter is a required output.
- All released post-QC cell ROIs are retained; no additional activity threshold is applied.
- Each ophys experiment/imaging plane is represented as one decoder session.

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Decoder sessions | 199 ophys experiments |
| Subjects | 38 mice |
| Trials | 51,075 |
| Neurons | 29,168 experiment-specific ROIs |
| Neurons/session | 4–666, mean 146.57 |
| Trials/session | 39–409, mean 256.66 |
| Brain regions | VISp (29,006 neurons), VISl (162 neurons) |
| Time bin | 100 ms |
| Trial length | 70–125 bins |

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# First trial in first session
neural = data['neural'][0][0]   # (neurons, time)
inputs = data['input'][0][0]    # (0, time): no decoder inputs
outputs = data['output'][0][0]  # (5, time)
print(neural.shape, outputs.shape)
```

## Output Variables

The rows of every output matrix are:

1. `image_identity`: gray plus 16 natural-image identities.
2. `image_change`: one in the first 100 ms bin at/after an actual image change; zero otherwise.
3. `running_speed_bin`: experiment-specific running-speed quintile (0–4).
4. `pupil_diameter_bin`: experiment-specific blink-filtered equivalent pupil-diameter quintile (0–4).
5. `trial_outcome`: hit, miss, false alarm, or correct reject, broadcast across the trial time axis.

`output_values` maps integer values to category names. The input dimension is zero because this decoder task specifies no decoder inputs.

## Temporal Processing

- Alignment is to the NWB trial start on the ophys session clock.
- Native detected calcium events at approximately 11 or 31 Hz are averaged into common 100 ms bins.
- Running and pupil streams are interpolated from their native timestamps to the same bin centers.
- Natural images are labeled during their 250 ms visible interval; 500 ms inter-image periods and omissions are labeled gray.
- Full native trial boundaries are retained, so trial lengths vary while bin width remains fixed.

## Reproducing the Conversion

```bash
# Two-session test with diagnostic plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full conversion
python -u /app/convert_data.py /app/converted_data.pkl --full

# Format validation
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Full decoder training
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

See `CONVERSION_NOTES.md` for detailed decisions, source comparisons, sanity checks, and decoder results.
