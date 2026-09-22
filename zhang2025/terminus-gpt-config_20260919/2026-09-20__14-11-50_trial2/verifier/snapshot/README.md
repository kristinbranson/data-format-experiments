# IBL Brain-Wide Neural Decoder Dataset

## Overview

This directory contains a stimulus-aligned conversion of the International Brain Laboratory (IBL) brain-wide electrophysiology dataset for neural decoding. The conversion follows the frozen BWM release and preprocessing in the supplied data/method papers and `code_zhang2025`, except where the requested decoder task explicitly requires categorical outputs.

The final dataset is:

- `/app/converted_data.pkl` — full converted dataset
- `/app/sample_data.pkl` — deterministic two-session sample
- `/app/convert_data.py` — reproducible conversion script
- `/app/CONVERSION_NOTES.md` — detailed decisions, references, checks, and results

## Dataset Summary

| Statistic | Full dataset |
|-----------|--------------|
| Sessions / subjects | 134 / 134 |
| Trials | 55,475 |
| Summed session neuron populations | 21,396 |
| Time bins per trial | 100 |
| Bin width | 20 ms |
| Alignment/window | stimulus onset, −0.5 to +1.5 s |
| Brain-region ontology | Beryl (202 labels represented) |
| Camera streams | 131 left, 3 right fallback |
| Pickle size | approximately 3.3 GB |

Five of 139 reference-selected subject sessions were excluded because neither left nor right whisker motion-energy data were available. Trial and neuron filtering details are in `CONVERSION_NOTES.md`.

## Decoder Variables

### Inputs (`2 × 100`, float32)

1. `time_since_stimulus_onset_s`: behavior sampling times/right bin edges, −0.48 through +1.50 s.
2. `trial_number_in_session`: original zero-based raw trial index, repeated over time.

### Outputs (`4 × 100`, categorical int8)

1. `choice`: left = 0, right = 1; repeated over time.
2. `prior_probability_left`: 0.2 = 0, 0.5 = 1, 0.8 = 2; repeated over time.
3. `wheel_speed_bin`: low/medium/high, time-varying pooled tertiles.
4. `whisker_motion_energy_bin`: low/medium/high, time-varying pooled tertiles.

Neural arrays are `n_neurons × 100` float32 spike-count matrices. Neurons pass all IBL RIGOR unit criteria (`label >= 1`) and have valid Beryl anatomy.

## Loading the Data

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
# neural, input, output, subjects, subject_idx, brain_regions,
# brain_region_idx, input_names, output_names, output_values, metadata

session = 0
trial = 0
print(data['neural'][session][trial].shape)  # (n_neurons, 100)
print(data['input'][session][trial].shape)   # (2, 100)
print(data['output'][session][trial].shape)  # (4, 100)
```

`metadata['session_info']` contains each EID, source paths, original retained trial indices, camera side, neuron UUIDs, exclusion counts, and processing timing. Discretization thresholds and category mappings are also stored in metadata.

## Reproducing the Conversion

The environment provides a staged read-only ONE cache at `/mnt/dataset/one_cache`.

```bash
# Two-session test with processing plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Full reference cohort (also the default mode)
python -u /app/convert_data.py /app/converted_data.pkl --full

# Validate format
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only

# Train full decoder and make sample plots
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Validation Results

Full validation reported no errors or warnings. Validation balanced accuracies were:

| Output | Accuracy | Chance |
|--------|----------|--------|
| Choice | 0.6072 | 0.5000 |
| Prior probability left | 0.6954 | 0.3333 |
| Wheel speed bin | 0.6144 | 0.3333 |
| Whisker motion-energy bin | 0.7038 | 0.3333 |

Independent raw-file sanity checks reconstruct neural spike histograms, inputs, choice/prior mappings, wheel speed, and whisker categories with `np.allclose`; see `/app/cache/raw_sanity_checks.py`.

## Logs

- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`

See `CONVERSION_NOTES.md` for complete provenance, paper comparisons, edge-case handling, and critical reviews.
