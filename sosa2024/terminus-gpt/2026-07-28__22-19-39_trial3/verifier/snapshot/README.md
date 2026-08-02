# Converted Reward-Relative Hippocampal Dataset

This repository contains a converted version of the reward-relative hippocampal 2P imaging dataset from:
- **Paper**: *A flexible hippocampal population code for experience relative to reward*

## Files
- `converted_data.pkl`: full converted dataset for decoder training
- `sample_data.pkl`: small sample conversion used for validation
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `train_decoder_full_out.txt`: full decoder training output
- `verification_full_out.txt`: full dataset format verification output

## Converted data format
The pickle file stores a Python dictionary with keys:
- `neural`: list of sessions, each a list of trials, each trial shaped `(n_neurons, n_timepoints)`
- `input`: decoder inputs per trial
- `output`: decoder outputs per trial
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Decoder inputs
1. time from trial start (s)
2. environment (ENV1/ENV2)
3. trial number
4. previous trial outcome

## Decoder outputs
1. distance to reward zone (7 bins)
2. absolute corridor position (5 bins)
3. speed (5 bins)
4. lick (binary)
5. reward zone location (A/B/C)
6. reward outcome (binary)

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples --cpu
```

Run conversion:
```bash
python3 -u convert_data.py converted_data.pkl --full
```

## Notes
- Trials are aligned to `trial_start`.
- Reward outcomes are derived from NWB `BehavioralTimeSeries/Reward` event timestamps.
- Reward zone location is inferred from per-trial reward-zone position dynamics.
