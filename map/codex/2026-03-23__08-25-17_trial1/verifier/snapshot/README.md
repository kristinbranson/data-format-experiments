# Neural Decoder Conversion

This repository now contains a decoder-ready conversion of the mesoscopic Neuropixels delayed-response dataset described in `datapaper.pdf` and reused in `methodpaper.pdf`.

## Converted Dataset

- File: `converted_data.pkl`
- Sessions: 173
- Subjects: 28
- Good units: 69,453
- Trials kept in converted dataset: 51,346
- Alignment: go cue onset
- Window: `[-2.5 s, +1.5 s]`
- Bin size: `50 ms`

The converted pickle follows the required dictionary structure:

- `neural`: list of sessions, each a list of `(n_neurons, 80)` firing-rate matrices
- `input`: list of sessions, each a list of `(2, 80)` arrays
- `output`: list of sessions, each a list of `(4, 80)` arrays
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Inputs

Order of `input_names`:

1. `time_from_tone_onset_s`
2. `photostim_on`

## Outputs

Order of `output_names`:

1. `choice`
2. `outcome`
3. `early_lick`
4. `tongue_y_bin`

Categorical values:

- `choice`: `left=0`, `right=1`
- `outcome`: `ignore=0`, `miss=1`, `hit=2`
- `early_lick`: `no=0`, `yes=1`
- `tongue_y_bin`: `<40th=0`, `40th-60th=1`, `>60th=2`

## Important Conversion Rules

- Units are kept only if `units/classification == good`.
- Sessions with zero good units are excluded.
- Trials are aligned to raw go-cue timestamps.
- A trial is kept only if its full `[-2.5 s, +1.5 s]` neural window is supported by the raw session observation interval.
- When the NWB `units/is_good_trials` matrix matches the selected trial set, it is used directly. Otherwise trial support falls back to raw `obs_intervals`.
- Tongue y is aligned from `Camera0_side_TongueTracking` by taking the last frame in each 50 ms bin, then discretized per session at the 40th and 60th percentiles.

## Load Example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["input_names"])
print(data["output_names"])
print(len(data["neural"]), "sessions")
print(data["neural"][0][0].shape)   # (n_neurons, 80)
print(data["input"][0][0].shape)    # (2, 80)
print(data["output"][0][0].shape)   # (4, 80)
```

## Validation Files

The main validation artifacts are kept at the repository top level because they are required outputs:

- `conversion_sample_out.txt`
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `conversion_full_out.txt`
- `verification_full_out.txt`
- `train_decoder_full_out.txt`
- `sample_data.pkl`
- `converted_data.pkl`
- `CONVERSION_NOTES.md`

## Notes

- `CONVERSION_NOTES.md` contains the full step-by-step conversion record, sanity checks, and decoder results.
- `train_decoder.py --plot-samples` produced `sample_trials.png` and `predictions.png`.
