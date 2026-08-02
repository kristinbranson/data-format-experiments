# MAP Dataset Conversion

This directory contains a decoder-ready conversion of the MAP NWB dataset from the movement-alignment papers:

- `datapaper.pdf`: "Brain-wide neural activity underlying memory-guided movement"
- `methodpaper.pdf`: "Brain-wide analysis reveals movement encoding structured across and within brain areas"

## Converted Output

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample dataset used for debugging and validation

The converted dataset is a Python dictionary with these top-level keys:

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

## Conversion Summary

- Temporal alignment: go cue onset
- Window: `-2.5 s` to `+1.5 s`
- Bin size: `50 ms`
- Sessions: `173`
- Subjects: `28`
- Good units: `69,453`
- Trials retained: `73,910`

Decoder inputs:

- `time_from_tone_onset_s`
- `photostim_on`

Decoder outputs:

- `choice`: `left=0`, `right=1`
- `outcome`: `ignore=0`, `miss=1`, `hit=2`
- `early_lick`: `no=0`, `yes=1`
- `tongue_y_position`: `lt_40pct=0`, `40_to_60pct=1`, `gt_60pct=2`

## Loading Example

```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["input_names"])
print(data["output_names"])
print(len(data["neural"]), "sessions")
print(len(data["neural"][0]), "trials in session 0")
print(data["neural"][0][0].shape, "neural shape for session 0 trial 0")
```

## Regenerating

Full conversion:

```bash
python -u convert_data.py converted_data.pkl --full
```

Sample conversion:

```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
```

Validation:

```bash
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes

- Sessions with zero good units are excluded.
- Good units are defined from NWB `classification == "good"`.
- Trials are retained only when the requested neural window is available and contains nonzero spikes across the good-unit population.
- The papers' reported good-unit total (`69,943`) differs from the local NWB release (`69,453`); the conversion follows the local NWB release because that is the source data available here.
