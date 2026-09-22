# MAP Neural Decoder Dataset Conversion

## Overview

This directory contains a decoder-ready conversion of the electrophysiology and behavior data from **Brain-wide neural activity underlying memory-guided movement**, also analyzed in **Brain-wide analysis reveals movement encoding structured across and within brain areas**.

The conversion reads the source NWB files with `pynwb`, aligns every retained trial to go-cue onset, and stores 50-ms firing rates and task/behavior variables in `/app/converted_data.pkl`.

## Main files

- `converted_data.pkl` — complete converted dataset (approximately 12 GB)
- `sample_data.pkl` — two-session test dataset
- `convert_data.py` — reproducible converter
- `CONVERSION_NOTES.md` — detailed decisions, paper/code comparisons, iterations, and validation
- `conversion_*_out.txt` — conversion logs
- `verification_*_out.txt` — format-validation logs
- `train_decoder_*_out.txt` — sample and full decoder-training logs
- `processing_*.png` — sample processing/alignment visualizations
- `sample_trials.png`, `predictions.png` — full decoder diagnostic plots

## Loading the converted data

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))                 # 173 sessions
print(data['neural'][0][0].shape)         # (n_neurons, 80)
print(data['input'][0][0].shape)          # (2, 80)
print(data['output'][0][0].shape)         # (4, 80)
```

Loading the complete pickle requires substantial memory because it contains approximately 12 GB of arrays.

## Data organization

The top-level dictionary contains:

- `neural[session][trial]`: `float32`, shape `(n_neurons, 80)`, firing rate in Hz
- `input[session][trial]`: `float32`, shape `(2, 80)`
- `output[session][trial]`: integer, shape `(4, 80)`
- `subjects`: sorted unique mouse IDs
- `subject_idx`: session-to-subject indices
- `brain_regions`: exact NWB Allen CCF anatomical labels
- `brain_region_idx[session]`: neuron-to-region indices
- `input_names`, `output_names`, `output_values`
- `metadata`: timing, curation, discretization, and per-session audit information

### Time axis

- Alignment: go-cue onset at 0 s
- Window edges: -2.5 to +1.5 s
- Bin width: 50 ms
- Number of bins: 80
- Bin centers: -2.475 to +1.475 s

### Neural data

Spikes are counted in non-overlapping 50-ms bins and divided by 0.05 s to produce Hz. Units must have NWB classifier label `good`. Unit-local `obs_intervals` and `is_good_trials` are mapped to behavioral trials, and each session retains the common valid trial intersection. Trials with no spikes from any selected unit anywhere in the requested four-second window are treated as recording gaps and excluded.

### Decoder inputs

1. `time from tone onset (s)`: continuous elapsed time at each bin center. Each go cue is paired to the latest preceding auditory sample onset.
2. `photostimulation on`: binary state at each bin center, derived from NWB photostimulation start/stop events.

### Decoder outputs

The first three per-trial values are repeated across the 80 time bins to satisfy the uniform decoder format.

1. `lick direction choice`: left, right, no lick
   - hit: instructed direction
   - miss: direction opposite instruction
   - ignore: no lick
2. `outcome`: ignore, miss, hit
3. `early lick`: no, yes
4. `tongue y-position`: time-varying classes
   - 0: below session visible-frame 40th percentile
   - 1: 40th through 60th percentile
   - 2: above 60th percentile
   - 3: not visible, outside video coverage, or likelihood below 0.9

## Final statistics

| Statistic | Value |
|---|---:|
| Subjects | 28 |
| Sessions | 173 |
| Retained trials | 90,378 |
| Summed neurons across sessions | 69,453 |
| Neurons/session | 90–923 (mean 401.46) |
| Brain-region labels | 293 |
| Time bins/trial | 80 |

The papers report 69,943 good units. The supplied NWBs contain 69,453 units with classifier label `good`; the released NWB labels are used as authoritative. One of 174 NWB files has zero classifier-good units, yielding the paper-matching 173 usable sessions.

## Validation and decoder results

Final format validation reported **no errors or warnings**. Independent direct-NWB checks made 108 `np.allclose` comparisons of neural, input, and output arrays; all passed.

Full decoder validation balanced accuracies:

| Output | Train | Validation | Chance |
|---|---:|---:|---:|
| Lick direction choice | 0.7118 | 0.6818 | 0.3333 |
| Outcome | 0.6979 | 0.6626 | 0.3333 |
| Early lick | 0.7889 | 0.7531 | 0.5000 |
| Tongue y-position | 0.6739 | 0.6175 | 0.2500 |

Training completed for 200 epochs; loss decreased from 18.6044 to 0.6657 and test loss was 0.6576.

## Reproducing the conversion

```bash
# Two-session sample with processing plots
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing

# Complete dataset
python -u /app/convert_data.py /app/converted_data.pkl --full

# Validation
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```

See `CONVERSION_NOTES.md` for detailed rationale and all discovered/fixed edge cases.
