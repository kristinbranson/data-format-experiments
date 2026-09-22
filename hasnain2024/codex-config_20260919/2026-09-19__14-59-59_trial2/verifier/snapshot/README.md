# Go-Cue-Aligned Neural Decoder Dataset

This directory contains a decoder-ready conversion of the two-context ALM electrophysiology cohort from *Separating cognitive and motor processes in the behaving mouse*. The conversion uses the 12 sessions and ALM probes selected by the supplied Figure 8 loaders, aligns every stream to go-cue onset, and preserves the reference spike processing and curation.

## Main files

- `converted_data.pkl`: full 12-session dataset
- `sample_data.pkl`: first two sessions for quick testing
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: scientific decisions, source comparisons, audits, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: captured run logs
- `processing_*.png`, `sample_trials.png`, `predictions.png`: processing and decoder diagnostics
- `cache/`: independent audit scripts and their logs

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as stream:
    data = pickle.load(stream)

neural_trial = data["neural"][0][0]  # neurons x 1,000 time bins
input_trial = data["input"][0][0]    # 1 x 1,000
output_trial = data["output"][0][0]  # 6 x 1,000
```

The 5-ms bin centers span -2.4975 to +2.4975 seconds relative to the go cue. Neural values are causally smoothed firing rates (`float32`). The decoder input is continuous time from the go cue (`float32`). Outputs are categorical `int8` rows in this order:

1. lick direction: left, right, none
2. behavioral context: WC, DR
3. outcome: incorrect, correct, ignore
4. tongue velocity: below median, at/above median, not visible
5. paw velocity: below median, at/above median, not visible
6. motion energy: below median, at/above median, no video

Velocity and motion-energy medians are computed independently within each session over finite retained samples. Static trial labels are repeated across time so the target dimensions remain consistent.

## Dataset summary

| Statistic | Value |
|---|---:|
| Sessions | 12 |
| Native subject IDs | 7 |
| Retained trials | 3,116 |
| Session-summed ALM units | 521 |
| Units/session | 27–67 (mean 43.42) |
| Bins/trial | 1,000 |
| Window | -2.5 to +2.5 s |

The deposited archive and executable reference filters yield 521 units (213 manually well-isolated); the paper reports 522 (214), indicating a one-unit archive/version difference. The archive exposes seven native IDs while the paper describes six mice. Native IDs are retained because the source supplies no defensible cross-ID mapping.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Full validation reports no format errors or warnings. Full held-out balanced accuracies are 0.556 lick direction, 0.711 context, 0.546 outcome, 0.552 tongue velocity, 0.525 paw velocity, and 0.712 motion energy. A paper-comparable binary delay-choice audit reaches ROC-AUC 0.924 ± 0.084.

