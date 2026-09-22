# MAP Neural Decoder Dataset

This workspace converts the Mesoscale Activity Map NWB release (DANDI:000363) into a decoder-ready pickle. It contains 173 sessions from 28 mice, 69,453 classifier-QC neurons, and 90,378 valid recorded trials.

## Reproduce the conversion

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

For a quick two-session run with diagnostic plots:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

The converter accesses NWB files exclusively through `pynwb`. It keeps units labeled `good` by the paper's region-specific multimetric classifier, resolves insertion-local valid-trial intervals, aligns all streams to go onset, and computes firing rates in 80 non-overlapping 50-ms bins over `[-2.5, +1.5)` seconds.

## Load and use

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

trial_rates = data["neural"][0][0]   # neurons × 80, spikes/s
trial_inputs = data["input"][0][0]   # 2 × 80
trial_outputs = data["output"][0][0] # 4 × 80 categorical integers
```

Inputs are continuous seconds since the completed tone/sample onset and a binary photostimulation-on signal. Outputs are lick choice (left/right/no lick), outcome (ignore/miss/hit), early lick (no/yes), and session-percentile tongue y-position (low/middle/high/not visible). Trial-level outputs are broadcast through time; tongue position varies by bin.

The remaining fields provide subjects/session indices, fine Allen CCF brain-region labels/neuron indices, category names, timing metadata, per-session thresholds, source filenames, and a conversion audit.

## Validation summary

`train_decoder.py --verify-only` reports a valid structure with no errors or warnings. Full validation balanced accuracies were:

| Output | Accuracy | Chance |
|---|---:|---:|
| Lick direction choice | 0.6876 | 0.3333 |
| Outcome | 0.6632 | 0.3333 |
| Early lick | 0.7527 | 0.5000 |
| Tongue y-position | 0.6159 | 0.2500 |

See `CONVERSION_NOTES.md` for full decisions, reference comparisons, raw-NWB sanity checks, edge cases, and validation details.
