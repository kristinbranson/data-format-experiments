# Neural Decoder Conversion

This directory contains a decoder-ready conversion of the Mesoscale Activity Map delayed-response dataset from the supplied NWB files in `/app/data`.

The converted dataset is stored at `/app/converted_data.pkl`. It contains go-cue-aligned neural firing rates and decoder targets for:
- `choice`: `left`, `right`, `no lick`
- `outcome`: `ignore`, `miss`, `hit`
- `early_lick`: `no`, `yes`
- `tongue_y_position_discrete`: `lt_p40`, `p40_to_p60`, `gt_p60`, `not_visible`

## Conversion Summary

- Source dataset: Mesoscale Activity Map Dataset (`DANDI:000363`)
- Sessions retained: `173`
- Subjects retained: `28`
- Good units retained: `69,453`
- Trials retained: `88,943`
- Alignment event: `Go cue onset`
- Decoder window: `-2.5 s` to `+1.5 s`
- Neural bin width: `50 ms`

Conversion decisions follow the supplied papers and reference code where applicable:
- Keep sessions with at least one `units/classification == "good"` unit.
- Keep only units with `units/classification == "good"`.
- Exclude `auto_water` and `free_water` trials.
- Retain `early_lick`, `ignore`, and photostimulation trials because they are required by the decoder task.
- Compute firing rates from spike counts in non-overlapping 50 ms bins.
- Align all streams to go cue.
- Derive `choice` from the first lick in the 1.5 s post-go response window.
- Discretize tongue y-position per session using visible-frame percentiles.

## File Format

The pickle stores a Python dictionary with these top-level keys:

- `neural`
- `input`
- `output`
- `subjects`
- `subject_idx`
- `brain_regions`
- `brain_region_idx`
- `input_names`
- `output_names`
- `output_values`
- `metadata`

Per-trial array shapes:
- `neural[session][trial]`: `(n_neurons, 80)`
- `input[session][trial]`: `(2, 80)`
- `output[session][trial]`: `(4, 80)`

Input names:
- `time_from_tone_onset_s`
- `photostim_on`

Output names:
- `choice`
- `outcome`
- `early_lick`
- `tongue_y_position_discrete`

## Loading Example

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["input_names"])
print(data["output_names"])
print(len(data["neural"]))  # sessions
print(data["neural"][0][0].shape)  # first trial neural matrix
```

## Validation

Format-only validation:

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Full decoder training:

```bash
python /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The final full-dataset run completed successfully and produced validation balanced accuracies of:
- `choice`: `0.6841`
- `outcome`: `0.6604`
- `early_lick`: `0.7489`
- `tongue_y_position_discrete`: `0.6121`

## Notes

- Detailed processing decisions, sanity checks, and validation results are documented in `/app/CONVERSION_NOTES.md`.
- Investigation utilities that are not part of the main deliverables are kept in `/app/cache`.
