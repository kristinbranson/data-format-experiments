# MAP Decoder Conversion

Converted dataset for the MAP delayed-response task from the paper *Brain-wide neural activity underlying memory-guided movement*, prepared for `/app/train_decoder.py`.

## Dataset Summary
- Source archive: local NWB copy of `DANDI:000363/0.230822.0128`
- Retained sessions: `173`
- Subjects: `28`
- Retained good units: `69,453`
- Retained trials: `90,844`
- Neural representation: firing rate in `50 ms` bins
- Alignment event: go cue onset
- Time window: `[-2.5 s, +1.5 s)`

## Conversion Rules
- Units are kept when `units/classification == "good"`.
- Trials are kept when the full decoder window lies inside the shared `units/obs_intervals` support of retained units, required go/sample events are present, and the binned neural tensor is not all zero.
- Tone timing is derived from the last `sample_start` event before the go cue within each trial.
- Choice is derived from the first lick in `[go_start, min(go_start + 1.5 s, trial_stop)]`.
- Tongue y is discretized per session:
  - `0`: below 40th percentile of visible tongue y
  - `1`: 40th to 60th percentile
  - `2`: above 60th percentile
  - `3`: not visible

## Decoder Variables
- Inputs:
  - `time_from_tone_onset_s`
  - `photostim_on`
- Outputs:
  - `choice`: `left`, `right`, `no lick`
  - `outcome`: `ignore`, `miss`, `hit`
  - `early_lick`: `no`, `yes`
  - `tongue_y_bin`: `lt40`, `40to60`, `gt60`, `not visible`

## File Format
The converted file is `/app/converted_data.pkl`. It is a Python dictionary with:
- `neural`: list of sessions, each a list of trial matrices shaped `(n_neurons, 80)`
- `input`: list of sessions, each a list of trial matrices shaped `(2, 80)`
- `output`: list of sessions, each a list of trial matrices shaped `(4, 80)`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Loading Example
```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data["neural"]))                 # sessions
print(data["neural"][0][0].shape)         # (n_neurons, 80)
print(data["input_names"])
print(data["output_names"])
```

## Validation Outputs
Required conversion and decoder logs are kept in the top-level `/app` directory:
- `conversion_sample_out.txt`
- `verification_sample_out.txt`
- `train_decoder_sample_out.txt`
- `conversion_full_out.txt`
- `verification_full_out.txt`
- `train_decoder_full_out.txt`

Additional review artifacts and plots are stored in `/app/cache`.
