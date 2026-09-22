# Track2p Barrel-Cortex Neural Decoder Dataset

## Overview

`converted_data.pkl` contains longitudinal two-photon calcium imaging and synchronized motion-energy labels from mouse layer 2/3 barrel cortex. The source experiment followed six mice over 6–7 daily sessions during the second postnatal week. Continuous recordings were converted into consecutive, non-overlapping 60-second trials for decoding animal motion state from neural activity.

The conversion follows the reference methods where applicable:

- Track2p-curated neurons passing the Suite2p cell probability threshold of 0.5
- neuropil correction using the saved Suite2p coefficient
- full-recording Suite2p maximin baseline correction
- non-overlapping 10-frame means for neural and behavioral streams
- frame-ordinal synchronization at the documented 30 Hz acquisition rate

Motion energy is discretized separately within each session into five equal-percentile classes, as required by the decoder task.

## Files

- `converted_data.pkl`: complete converted dataset
- `sample_data.pkl`: first two sessions for quick testing
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: detailed decisions, reference comparisons, checks, and decoder results
- `conversion_sample_out.txt`, `conversion_full_out.txt`: conversion logs
- `verification_sample_out.txt`, `verification_full_out.txt`: format-validation logs
- `train_decoder_sample_out.txt`, `train_decoder_full_out.txt`: decoder-training logs
- `processing_*.png`: conversion-stage diagnostics for two sample sessions
- decoder-generated sample plots (`neural_sample.png`, `input_sample.png`, `output_sample.png`)
- `cache/`: investigation scripts and extracted reference-paper text

## Key Statistics

| Statistic | Value |
|---|---:|
| Subjects | 6 |
| Sessions | 41 |
| Sessions per subject | 7, 7, 7, 7, 6, 7 |
| Unique longitudinally tracked neuron identities | 2,998 |
| Repeated neuron-session entries | 20,445 |
| Neurons per session | 221–746; mean 498.66 |
| Retained 60-second trials | 1,081 |
| Trials per session | 19–30 |
| Samples per trial | 180 |
| Converted sampling rate | 3 Hz |
| Time-bin size | 333.333 ms |
| Motion classes | 5; approximately 20% each |
| Brain region | Layer 2/3 barrel cortex |

Nine sessions had a behavior stream ending 1–148 native frames before the nominal neural endpoint. Their incomplete final 60-second windows were excluded rather than interpolated, accounting for the difference between 1,090 nominal and 1,081 retained trials.

## Loading the Data

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(len(data['neural']))                   # 41 sessions
print(data['neural'][0][0].shape)           # (221 neurons, 180 time bins)
print(data['input'][0][0].shape)            # (1, 180)
print(data['output'][0][0].shape)           # (1, 180)
```

The primary fields are nested as `[session][trial]`:

- `neural`: float32 baseline-corrected fluorescence, shape `(n_neurons, 180)`
- `input`: float32 elapsed time from session start in seconds, shape `(1, 180)`; time continues across trial boundaries
- `output`: int64 motion-energy quintile, shape `(1, 180)`, with values 0–4
- `subjects`: sorted subject IDs
- `subject_idx`: subject index for every session
- `brain_regions`: `['barrel cortex L2/3']`
- `brain_region_idx`: region index for every neuron/session
- `metadata`: processing description, timing, and per-session provenance/statistics

Output labels are:

0. lowest motion energy
1. low motion energy
2. middle motion energy
3. high motion energy
4. highest motion energy

## Reproducing the Conversion

Full dataset:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full
```

Two-session sample with diagnostic plots:

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

Validation and decoder training:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Validation Results

The provided validator reports no errors or warnings. Full decoder balanced accuracy for `motion_energy_quintile` was:

- training: 0.6168
- validation: 0.2931
- chance: 0.2000

See `CONVERSION_NOTES.md` for independent raw-file `np.allclose()` checks, endpoint checks, reference-code comparisons, and accuracy review.
