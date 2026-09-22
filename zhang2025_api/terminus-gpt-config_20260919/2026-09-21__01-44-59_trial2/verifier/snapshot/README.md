# IBL Neural Decoder Dataset

This directory contains a stimulus-aligned conversion of International Brain Laboratory Brain-Wide Map electrophysiology and behavior data for neural decoding.

## Main files

- `converted_data.pkl` — full converted dataset (22 sessions)
- `sample_data.pkl` — two-session test dataset
- `convert_data.py` — reproducible conversion script
- `CONVERSION_NOTES.md` — detailed decisions, reference comparisons, sanity checks, and decoder results
- `conversion_full_out.txt`, `verification_full_out.txt` — full conversion/validation logs
- `train_decoder_full_out.txt` — completed full decoder training log

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Structure

```text
data['neural'][session][trial]  # float32, (n_neurons, 100), spike counts
data['input'][session][trial]   # float32, (2, 100)
data['output'][session][trial]  # int64,   (4, 100)
```

Inputs:

1. Time since stimulus onset, bin centers from -0.49 to +1.49 seconds
2. Zero-based trial number in the current probability block, repeated over time

Outputs:

1. Choice: left = 0, right = 1
2. Prior probability of left: 0.2 = 0, 0.5 = 1, 0.8 = 2
3. Wheel speed: low/medium/high = 0/1/2
4. Whisker ROI motion energy: low/medium/high = 0/1/2

Choice and prior are per-trial labels repeated over the 100 time bins. Wheel and whisker outputs vary over time.

## Processing summary

- Alignment: visual stimulus onset (`stimOn_times`)
- Window: -0.5 to +1.5 seconds
- Bin width: 20 ms (100 bins)
- Neural signal: spike counts, with probes merged within each session
- Unit QC: merged cluster `label >= 1` and resolved positive atlas ID
- Anatomy: Allen IDs remapped to the Beryl atlas
- Trial QC: required task events finite, supported choice/prior, positive duration no longer than 10 seconds, and complete wheel/camera support
- Wheel speed: absolute timestamp-aware derivative of wheel position
- Whisker signal: left-camera ROI motion energy when valid, otherwise right-camera fallback
- Dynamic discretization: within-session behavioral tertiles; exact thresholds are stored in `metadata['session_info']`
- All source neuroscience arrays are accessed through `one.api.ONE` and brainbox loaders

## Full dataset statistics

| Statistic | Value |
|---|---:|
| Sessions | 22 |
| Subjects | 22 |
| Trials | 12,943 |
| QC-passing session-units | 3,984 |
| Neurons/session | 53–468 (mean 181.1) |
| Trials/session | 359–839 (mean 588.3) |
| Brain-region categories | 66 |
| Pickle size | 992.5 MB |

Output distributions:

- Choice: left 0.4749, right 0.5251
- Prior: 0.2 = 0.4166, 0.5 = 0.1472, 0.8 = 0.4362
- Wheel bins: approximately one third each
- Whisker bins: approximately one third each

## Decoder results

Full validation balanced accuracy:

| Output | Accuracy | Chance |
|---|---:|---:|
| Choice | 0.6138 | 0.5000 |
| Prior | 0.6830 | 0.3333 |
| Wheel speed bin | 0.6038 | 0.3333 |
| Whisker motion energy bin | 0.6108 | 0.3333 |

Training completed all 200 epochs; loss decreased from 1.666 to 0.734. Train/validation gaps were small for every output.

## Reproduction

```bash
# Sample conversion with diagnostic plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full conversion
python -u /app/convert_data.py /app/converted_data.pkl --full

# Format verification
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Full decoder training and sample plots
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The converter builds a local ONE index with ONE's own `make_parquet_db`, because the supplied offline release manifests contain stale revision metadata. Local path-hash session IDs are mapped to methods-paper sessions through ONE session metadata. See `CONVERSION_NOTES.md` for complete rationale and checks.
