# Allen Visual Behavior Ophys Decoder Dataset

## Overview

`converted_data.pkl` contains Allen Brain Observatory Visual Behavior two-photon calcium-imaging data reformatted for neural decoding. It includes active go/catch change-detection trials from 199 imaging-plane experiments, 38 mice, 29,168 SDK-curated cells, and 48,112 trials. Data were loaded exclusively through AllenSDK `VisualBehaviorOphysProjectCache`.

Neural activity is released detrended dF/F, aligned with stimulus, running, and pupil streams on a common 100 ms grid expressed in each experiment's ophys clock. Aborted, auto-rewarded, passive, and incomplete-stream trials are excluded.

## Files

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed decisions, reference comparisons, and validation reports
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: conversion and validation logs
- `processing_<id>.png`: sample alignment/processing plots

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))       # 199 sessions
print(data['output_names'])
```

Each `data['neural'][session][trial]` is a float32 `(n_neurons, n_timepoints)` array. Each input is an empty `(0, n_timepoints)` array because this task has no decoder inputs. Each output is an integer `(5, n_timepoints)` array ordered as:

1. `image_identity`: `none/gray` or one of 16 natural-image identities
2. `image_change`: no-change/change onset event
3. `running_speed_bin`: global quintile 0–4
4. `pupil_diameter_bin`: global quintile 0–4
5. `trial_outcome`: hit/miss/false alarm/correct reject, constant within trial

Use `output_values` to map integer codes to labels. `subject_idx` maps sessions to `subjects`; `brain_region_idx` maps neurons to `brain_regions` (`VISp`/V1 and `VISl`/LM).

## Key Statistics

| Statistic | Value |
|---|---:|
| Sessions | 199 imaging planes |
| Subjects | 38 mice |
| Trials | 48,112 |
| Neurons | 29,168 |
| Time bin | 100 ms |
| VISp neurons | 29,006 |
| VISl neurons | 162 |
| Trial outcomes | 14,979 hit; 27,107 miss; 878 false alarm; 5,148 correct reject |

Three active experiments were excluded because no eligible trial had usable pupil coverage. All 38 mice remain represented.

## Reproducing and Validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation balanced accuracies were 0.3722 image identity, 0.6457 image change, 0.3713 running quintile, 0.4104 pupil quintile, and 0.2923 trial outcome; every output exceeded chance. See `CONVERSION_NOTES.md` for interpretation and independent source comparisons.
