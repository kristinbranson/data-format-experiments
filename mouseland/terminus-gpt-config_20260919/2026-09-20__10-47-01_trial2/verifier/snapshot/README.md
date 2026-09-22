# Zhong et al. Neural Decoder Dataset

## Overview

`converted_data.pkl` contains calcium-imaging and synchronized behavioral data from **Unsupervised pretraining in biological neural networks**, converted for neural decoding.

- 19 mice
- 89 unique physical imaging sessions
- 38,110 trials
- 4,691,034 neuron/session instances
- 137.855 GiB pickle
- Native temporal sampling: median 314.8 ms per imaging frame
- Trials aligned to corridor entry and restricted to the 0–4 m visual corridor

The source neural signal is Suite2p non-negative deconvolved fluorescence (`spks`). No dF/F was recomputed. Neural values are stored as `float16` to reduce disk usage; inputs are `float32`, and outputs are categorical integers.

## Files

- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: representative two-session test dataset
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, checks, and decoder results
- `conversion_sample_out.txt`, `conversion_full_out.txt`: conversion logs
- `verification_sample_out.txt`, `verification_full_out.txt`: format-validation logs
- `train_decoder_sample_out.txt`, `train_decoder_full_out.txt`: decoder logs
- `processing_<session>.png`: sample conversion/alignment plots
- `predictions.png`: full-decoder prediction plot produced by `--plot-samples`
- `cache/`: investigation scripts and reports

## Loading

Loading the full pickle requires substantial RAM because it is approximately 138 GiB on disk.

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))       # 89 sessions
print(data['neural'][0][0].shape)  # neurons x timepoints
print(data['input_names'])
print(data['output_names'])
```

For lightweight inspection, use `/app/sample_data.pkl`.

## Structure

```python
data = {
    'neural':          list[session][trial],  # neurons x time, float16
    'input':           list[session][trial],  # 4 x time, float32
    'output':          list[session][trial],  # 4 x time, categorical integers
    'subjects':        list[str],
    'subject_idx':     array[n_sessions],
    'brain_regions':   ['V1', 'mHV', 'lHV', 'aHV', 'unmapped'],
    'brain_region_idx': list[array[n_neurons]],
    'input_names':     list[str],
    'output_names':    list[str],
    'output_values':   list[list[str]],
    'metadata':        dict,
}
```

Every neural, input, and output array for a trial has the same number of temporal samples.

## Decoder Inputs

1. **Time to sound cue (s)** — signed, time-varying; positive before and negative after cue.
2. **Day of training (elapsed days)** — calendar days since that mouse's first supplied imaging session, repeated over the trial.
3. **Time since trial start (s)** — actual imaging timestamps re-zeroed at corridor entry.
4. **Reward availability** — 1 for a rewarded corridor, otherwise 0, repeated over the trial.

## Decoder Outputs

1. **Visual stimulus category** — 15 concrete `WallName` identities: circle, leaf, swap, rock, and wood variants.
2. **Licking** — binary per imaging frame; lick events are assigned to the nearest retained frame within their declared trial.
3. **Position in corridor** — four 1 m bins spanning 0–4 m.
4. **Running speed quartile** — four global classes using source-derived edges 0, 8.3270, and 30.1568 cm/s.

Because 20.38% of valid frames have speed exactly zero, identical values cannot be divided into exact equal-count bins. Full timepoint-weighted speed fractions are 9.8%, 40.2%, 25.0%, and 25.0%.

## Regions

Reference retinotopy labels are mapped as follows:

- V1: atlas label 8
- mHV: labels 0, 1, 2, 9
- lHV: labels 5, 6
- aHV: labels 3, 4
- unmapped: label 7, retained rather than discarded

## Reproducing Conversion

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
```

Full conversion took 11.14 minutes in the provided environment and writes approximately 138 GiB.

## Validation

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Format validation completed with **no errors or warnings**. Full validation balanced accuracies were:

| Output | Validation accuracy | Uniform chance |
|---|---:|---:|
| Visual stimulus | 0.5087 | 0.0667 |
| Licking | 0.8635 | 0.5000 |
| Position | 0.3196 | 0.2500 |
| Running speed | 0.4159 | 0.2500 |

The GPU run exceeded device memory after initialization; the provided trainer automatically retried on CPU and completed all 200 epochs successfully.

See `CONVERSION_NOTES.md` for full rationale, raw-source `np.allclose` checks, edge-case analysis, and accuracy review.
