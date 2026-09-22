# Two-context ALM neural-decoder dataset

This directory contains a decoder-ready conversion of the electrophysiology and high-speed-video data accompanying *Separating cognitive and motor processes in the behaving mouse*.

## Converted cohort

The conversion uses the 12 sessions explicitly selected by the paper's two-context neural-analysis scripts. These sessions contain alternating delayed-response (DR) and water-cued (WC) blocks, ALM electrophysiology, two-camera DLC tracking, and motion energy.

- 12 sessions from 7 source animal IDs
- 3,116 retained trials (early-lick and stimulation trials excluded)
- 518 ALM session-units after quality rejection and a strict mean firing-rate >1 Hz rule
- 500 samples/trial, spanning bin centers −2.495 to +2.495 s around go cue/water presentation
- 10-ms bins; reference 15-sample causal Gaussian smoothing

The paper reports 522 units and six mice for this cohort. Applying the published algorithm to the supplied data snapshot produces 518 units, and the explicit session loaders resolve to seven distinct IDs. These small source-version discrepancies are documented in `CONVERSION_NOTES.md` rather than hidden by changing thresholds or identities.

## Files

- `converted_data.pkl`: complete 12-session dataset
- `sample_data.pkl`: first two sessions for quick testing
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: decisions, source comparisons, checks, and decoder results
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: complete run logs
- `processing_*.png`: conversion/alignment/threshold diagnostics
- `sample_trials.png`, `predictions.png`: full-decoder diagnostic plots
- `cache/`: independent audit utility and its documentation

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural_trial = data['neural'][0][0]   # (n_neurons, 500), float32 spikes/s
input_trial = data['input'][0][0]     # (1, 500), time from go cue in seconds
output_trial = data['output'][0][0]   # (6, 500), integer categories
```

The six output rows are:

1. lick direction: `left`, `right`, `none`
2. behavioral context: `WC`, `DR`
3. outcome: `incorrect`, `correct`, `ignore`
4. tongue velocity: below median, at/above median, not visible
5. paw velocity: below median, at/above median, not visible
6. motion energy: below median, at/above median, no video

Rows 1–3 are trial-level values broadcast across time. Rows 4–6 vary over time. Velocity/motion thresholds are computed separately per session from finite samples in retained trials; equality belongs to the high class.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
python -u /app/cache/sanity_checks.py
```

Full held-out balanced accuracies were 0.6040 lick direction, 0.7480 context, 0.5769 outcome, 0.6339 tongue velocity, 0.5119 paw velocity, and 0.6028 motion energy. All exceed uniform chance.
