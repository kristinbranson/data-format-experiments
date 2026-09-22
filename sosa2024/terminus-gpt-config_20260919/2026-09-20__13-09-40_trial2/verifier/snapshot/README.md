# Converted CA1 Reward-Relative Navigation Dataset

## Dataset

This conversion contains two-photon CA1 recordings from **11 mice, 152 sessions, 12,216 complete corridor trials, and 138,678 Suite2p-accepted session-cells** from Sosa et al., *A flexible hippocampal population code for experience relative to reward*.

Mice ran along a 450 cm virtual corridor. Reward was available in one of three 50 cm zones beginning near 80, 200, or 320 cm (A/B/C), with reward-location switches and omission trials. Neural activity and behavior are aligned to imaging frames at approximately 15.5078 Hz (64.4836 ms bins).

## Files

- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: two-session test dataset
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: detailed decisions, checks, statistics, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: execution logs
- `processing_*.png`, `sample_trials.png`, `predictions.png`: processing/training visualizations

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Session 0, trial 0
neural = data['neural'][0][0]  # accepted neurons x time
inputs = data['input'][0][0]   # 4 x time
outputs = data['output'][0][0] # 6 x time
```

The complete pickle is about 13.3 GB, so loading requires sufficient RAM.

## Variables

### Neural

Suite2p deconvolved calcium events from ROIs with `iscell[:, 0] == 1`, stored as `float32` neuron-by-time matrices. Multi-plane recordings concatenate accepted cells from all planes.

### Decoder inputs

1. Time from trial start (seconds)
2. Environment type (ENV1=0, ENV2=1)
3. Native zero-based trial number
4. Previous native-trial outcome (omitted=0, rewarded=1; first trial=0)

### Decoder outputs

1. Signed distance to the reward-zone interval: 7 requested categories
2. Absolute corridor position: 5 requested 90 cm categories
3. Speed: 5 requested categories
4. Lick presence per imaging frame: no=0, yes=1
5. Reward-zone location: A=0, B=1, C=2
6. Trial reward outcome: omitted=0, rewarded=1

Trials are aligned to corridor entry and retain all in-trial frames, including low-speed periods. Intertrial samples and one uniquely incomplete 3-frame recording-edge fragment lacking a trial-start marker are excluded.

## Validation Summary

The supplied validator reports no errors or warnings. Independent raw-NWB `np.allclose` checks pass for neural, input, and output streams in one- and two-plane sessions.

Full validation balanced accuracies:

| Output | Accuracy | Chance |
|---|---:|---:|
| Distance to reward zone | 0.3950 | 0.1429 |
| Absolute position | 0.4986 | 0.2000 |
| Speed | 0.4303 | 0.2000 |
| Lick | 0.6025 | 0.5000 |
| Reward-zone location | 0.7537 | 0.3333 |
| Reward outcome | 0.5163 | 0.5000 |

See `CONVERSION_NOTES.md` for processing rationale, edge cases, output distributions, and detailed review.

## Reproducing

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```
