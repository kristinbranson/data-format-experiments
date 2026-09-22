# Converted Neural Decoder Dataset

This directory contains a decoder-ready conversion of the MAP delayed-response dataset provided in `/app/data`.

## Summary
- Alignment event: `Go cue onset`
- Neural window: `[-2.5 s, +1.5 s)` around go cue
- Neural bin size: `50 ms`
- Subjects: `28`
- Sessions: `173`
- Classifier-good units: `69,453`
- Converted trials: `88,779`
- Main file: `/app/converted_data.pkl`

## Conversion choices
- Units: kept only `units.classification == "good"`.
- Sessions: excluded the single NWB session with zero classifier-good units.
- Trials:
  - kept only trials covered by `units.obs_intervals`
  - excluded `auto_water` and `free_water`
  - kept early-lick, ignore, and photostim trials because they are required decoder labels / inputs
  - dropped residual trials with all-zero neural activity
- Tongue output:
  - source: `Camera0_side_TongueTracking`
  - visibility: likelihood threshold + 5-sigma velocity-outlier rejection
  - classes:
    - `0`: below session 40th percentile
    - `1`: session 40th-60th percentile
    - `2`: above session 60th percentile
    - `3`: not visible

## Data format
Load with:

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)
```

Top-level fields:
- `neural`: `list[session][trial]`, each trial is `(n_neurons, 80)` firing rates
- `input`: `list[session][trial]`, each trial is `(2, 80)`
- `output`: `list[session][trial]`, each trial is `(4, 80)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

Input order:
1. `time_from_tone_onset_s`
2. `photostimulation_on`

Output order:
1. `choice`
2. `outcome`
3. `early_lick`
4. `tongue_y`

Categorical values:
- `choice`: `["left", "right", "no lick"]`
- `outcome`: `["ignore", "miss", "hit"]`
- `early_lick`: `["no", "yes"]`
- `tongue_y`: `["<40th percentile", "40th-60th percentile", ">60th percentile", "not visible"]`

## Validation
- Format verification: `/app/verification_full_out.txt`
- Sample conversion log: `/app/conversion_sample_out.txt`
- Full conversion log: `/app/conversion_full_out.txt`
- Sample decoder training log: `/app/train_decoder_sample_out.txt`
- Full decoder training log: `/app/train_decoder_full_out.txt`
- Detailed rationale and audit trail: `/app/CONVERSION_NOTES.md`

Full decoder results:
- choice validation balanced accuracy: `0.6856`
- outcome validation balanced accuracy: `0.6582`
- early_lick validation balanced accuracy: `0.7506`
- tongue_y validation balanced accuracy: `0.6133`
