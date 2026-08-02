# Neural Decoder Conversion

This repository now contains a decoder-ready conversion of the brain-wide memory-guided movement dataset into `converted_data.pkl`.

## Dataset Summary
- Source dataset: brain-wide Neuropixels recordings from the delayed auditory licking task described in `datapaper.pdf`, `methodpaper.pdf`, and `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- Included sessions: 173
- Included subjects: 28
- Included good units: 69,453 (`units.classification == "good"`)
- Included trials: 73,910 trials with a complete `[-2.5, +1.5] s` go-aligned neural window
- Neural representation: 50 ms firing-rate bins, aligned to go cue onset

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: 2-session sample conversion
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: full audit trail of decisions, checks, and validation results
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`: sample-run logs
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`: full-run logs

## Data Format
The pickle contains a Python dictionary with these top-level keys:
- `neural`: list of sessions, each session a list of trial arrays shaped `(n_neurons, 80)`
- `input`: list of sessions, each trial shaped `(2, 80)` with:
  - `time_from_tone_onset_s`
  - `photostim_on`
- `output`: list of sessions, each trial shaped `(4, 80)` with:
  - `choice`
  - `outcome`
  - `early_lick`
  - `tongue_y_bin`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Output Definitions
- `choice`: `left=0`, `right=1`
- `outcome`: `ignore=0`, `miss=1`, `hit=2`
- `early_lick`: `no=0`, `yes=1`
- `tongue_y_bin`:
  - `0`: below session 40th percentile
  - `1`: 40th to 60th percentile
  - `2`: above session 60th percentile

## Loading Example
```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data["metadata"]["n_sessions"])
print(data["input_names"])
print(data["output_names"])

trial_neural = data["neural"][0][0]   # (n_neurons, 80)
trial_input = data["input"][0][0]     # (2, 80)
trial_output = data["output"][0][0]   # (4, 80)
```

## Reproducing The Conversion
Run the sample conversion:
```bash
python -u convert_data.py sample_data.pkl --sample --show-processing
python -u train_decoder.py sample_data.pkl --verify-only
python -u train_decoder.py sample_data.pkl
```

Run the full conversion:
```bash
python -u convert_data.py converted_data.pkl --full
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes
- The conversion follows the reference go-cue alignment and classifier-based unit filtering, but uses 50 ms bins and a `[-2.5, +1.5] s` window because the decoder task requires that format.
- Trials are excluded unless the full go-aligned window lies inside a good-unit observation interval. This avoids invalid all-zero neural windows at the beginning or end of recordings.
- Ignore trials do not carry a native ground-truth choice label in the raw data. For those trials, choice is assigned from the earliest lick in the trial when present, otherwise the instructed side. This is documented in `CONVERSION_NOTES.md`.
