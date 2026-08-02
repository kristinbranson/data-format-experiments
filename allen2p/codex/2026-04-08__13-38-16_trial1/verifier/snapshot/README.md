# Visual Behavior 2P Conversion

This directory contains a decoder-ready conversion of the Allen Brain Observatory Visual Behavior 2P dataset local subset in [`converted_data.pkl`](/app/converted_data.pkl).

## Dataset Summary
- Source data: local NWB files under [`data/visual-behavior-ophys-1.1.0`](/app/data/visual-behavior-ophys-1.1.0)
- Included sessions: `281` ophys experiment files
- Excluded sessions: `3` experiments with missing eye tracking
- Subjects: `38` mice
- Trials: `84,313` valid go/catch trials after excluding aborted and auto-rewarded trials
- Neurons: `41,871` valid ROIs
- Brain regions: `VISp`, `VISl`

## Conversion Choices
- Neural signal: ophys event-detection magnitudes from the NWB `event_detection` table
- Trial definition: go and catch trials only; aborted and auto-rewarded trials excluded
- Temporal alignment: all outputs are aligned to ophys timestamps on a common `100 ms` trial-relative grid
- Alignment event: trial start
- Outputs:
  - `image_identity`
  - `image_change`
  - `running_speed_bin`
  - `pupil_diameter_bin`
  - `trial_outcome`
- Inputs: none for this decoder task, so each `input` trial has shape `(0, T)`

## Files
- [`convert_data.py`](/app/convert_data.py): conversion script
- [`converted_data.pkl`](/app/converted_data.pkl): full converted dataset
- [`sample_data.pkl`](/app/sample_data.pkl): 2-session sample dataset
- [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md): full audit trail, checks, and results
- [`conversion_full_out.txt`](/app/conversion_full_out.txt): full conversion log
- [`verification_full_out.txt`](/app/verification_full_out.txt): full format-verification log
- [`train_decoder_full_out.txt`](/app/train_decoder_full_out.txt): full decoder training log

## Load the Converted Data
```python
import pickle

with open("converted_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(len(data["neural"]), "sessions")
print(data["output_names"])
```

Each trial is stored as:
- `data["neural"][session_idx][trial_idx]`: `(n_neurons, n_timepoints)`
- `data["input"][session_idx][trial_idx]`: `(0, n_timepoints)`
- `data["output"][session_idx][trial_idx]`: `(5, n_timepoints)`

## Reproduce
```bash
python3 -u convert_data.py sample_data.pkl --sample --show-processing
python3 -u train_decoder.py sample_data.pkl --verify-only
python3 -u convert_data.py converted_data.pkl --full
python3 -u train_decoder.py converted_data.pkl --verify-only
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Notes
- Full verification reports warnings for some all-zero neural trials. These were checked directly against the raw NWB event-detection matrices and are genuine sparse-event trials, not conversion mismatches.
- The detailed reasoning, sanity checks, and decoder results are documented in [`CONVERSION_NOTES.md`](/app/CONVERSION_NOTES.md).
