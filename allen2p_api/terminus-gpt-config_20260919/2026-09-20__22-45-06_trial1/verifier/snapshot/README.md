# Allen Visual Behavior Neural Decoder Dataset

## Overview

`converted_data.pkl` contains Allen Brain Observatory Visual Behavior 2P recordings reformatted for neural decoding. Source data were read exclusively through the AllenSDK `VisualBehaviorOphysProjectCache`; NWB files were never opened directly.

The retained cohort contains active visual change-detection recordings. Each target session is one ophys experiment/imaging plane. Passive sessions are excluded, as are three active planes without a usable SDK-processed pupil stream.

## Key statistics

- 199 imaging-plane sessions from 38 mice
- 51,075 go/catch trials after excluding aborted and auto-rewarded trials
- 29,168 summed valid cell records
- Regions: VISp (V1) and VISl (LM)
- Neural signal: AllenSDK detected calcium-event magnitudes summed into 100 ms bins
- Variable native trial durations: 70–125 bins (7.0–12.5 seconds)
- No decoder inputs
- Five categorical decoder outputs:
  1. image identity: gray plus 16 natural-image identities
  2. image change: binary, positive during the first 400 ms after a real change
  3. running speed: session-wise percentile quintile
  4. pupil diameter: session-wise percentile quintile
  5. trial outcome: hit, miss, false alarm, or correct reject, repeated across time

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(len(data['neural']))          # 199 sessions
print(data['neural'][0][0].shape) # neurons x time
print(data['input'][0][0].shape)  # (0, time), because there are no inputs
print(data['output'][0][0].shape) # (5, time)
```

Top-level fields follow the requested format: `neural`, `input`, `output`, `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `input_names`, `output_names`, `output_values`, and `metadata`.

Per-session provenance is available in `data['metadata']['session_info']`, including Allen experiment/session IDs, mouse, Cre line, region, experience level, cell/trial counts, outcome counts, and percentile thresholds.

## Reproducing the conversion

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
```

Full mode uses eight experiment-level workers and requires the local AllenSDK cache under `/app/data`. Processing plots are written as `processing_<experiment_id>.png` for up to two sessions when `--show-processing` is used.

## Validation and decoder

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation balanced accuracies were:

| Output | Accuracy | Uniform chance |
|---|---:|---:|
| Image identity | 0.2561 | 0.0588 |
| Image change | 0.5888 | 0.5000 |
| Running quintile | 0.2364 | 0.2000 |
| Pupil quintile | 0.2306 | 0.2000 |
| Trial outcome | 0.2750 | 0.2500 |

The verifier reports some all-zero neural trials. These are genuine sparse detected-event trials, especially in low-cell planes, rather than missing or invalid arrays. See `CONVERSION_NOTES.md` for full decisions, checks, warnings, reference comparisons, and issue-resolution history.
