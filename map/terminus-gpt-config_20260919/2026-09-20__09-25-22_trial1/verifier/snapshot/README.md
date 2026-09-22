# Brain-wide Neural Decoder Dataset

## Overview

`converted_data.pkl` is a decoder-ready conversion of the NWB release associated with **“Brain-wide neural activity underlying memory-guided movement”**, using processing conventions from the supplied papers and code where compatible with the requested decoder task.

The data contain go-cue-aligned Neuropixels population firing rates, task inputs, and behavioral outputs from an auditory delayed-response task.

## Key Statistics

- **Subjects:** 28 mice
- **Sessions:** 173
- **Trials:** 90,378
- **Curated session-units:** 69,453
- **Brain-region labels:** 293 detailed Allen atlas annotations
- **Time window:** −2.5 to +1.5 s relative to auditory go cue
- **Bin width:** 50 ms
- **Timepoints per trial:** 80
- **File size:** approximately 11.2 GiB

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data['neural']))              # 173 sessions
print(data['neural'][0][0].shape)      # (n_neurons, 80)
print(data['input'][0][0].shape)       # (2, 80)
print(data['output'][0][0].shape)      # (4, 80)
```

Because the pickle is large, loading requires substantial RAM.

## Data Structure

- `neural[session][trial]`: float32 firing rate in Hz, shape `(n_neurons, 80)`.
- `input[session][trial]`: float32 array, shape `(2, 80)`:
  1. continuous seconds from the final tone/sample onset before go cue;
  2. binary photostimulation state.
- `output[session][trial]`: integer categorical array, shape `(4, 80)`:
  1. lick choice: left, right, no lick;
  2. outcome: ignore, miss, hit;
  3. early lick: no, yes;
  4. tongue y-position: below session 40th percentile, 40th–60th percentile, above 60th percentile, not visible.
- `subjects` and `subject_idx`: subject vocabulary and per-session indices.
- `brain_regions` and `brain_region_idx`: global detailed atlas vocabulary and per-neuron indices.
- `metadata`: task, binning, alignment, filters, tongue threshold, and source-session information.

Trial-level outputs are repeated across all 80 time bins; tongue position is time-varying.

## Processing Summary

1. Retain NWB units with final region-classifier label `classification == "good"`.
2. Exclude the one source session with no retained units.
3. Map each unit’s NWB observation rows to behavioral trials and intersect manual `is_good_trials` across retained units.
4. Remove population windows containing no spikes in any retained neuron, which indicate missing neural segments.
5. Bin raw spikes into non-overlapping 50-ms bins and convert counts to Hz.
6. Align all streams to go cue onset.
7. Select the final sample/tone onset within each trial, correctly handling early-lick epoch replays.
8. Interpolate tongue y and likelihood at neural bin centers. Frames with likelihood below 0.9 are “not visible”; visible y is discretized using session-wide 40th and 60th percentiles.

See `CONVERSION_NOTES.md` for detailed rationale, paper/code comparisons, corrections, and independent raw-data checks.

## Validation

`verification_full_out.txt` reports no format errors or warnings. Full decoder validation balanced accuracies were:

| Output | Validation balanced accuracy | Chance |
|--------|------------------------------|--------|
| Lick choice | 0.6738 | 0.3333 |
| Outcome | 0.6553 | 0.3333 |
| Early lick | 0.7533 | 0.5000 |
| Tongue y-position | 0.6114 | 0.2500 |

Training details are in `train_decoder_full_out.txt`; plots are in `sample_trials.png` and `predictions.png`.

## Reproducing the Conversion

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
```
