# Converted Track2p Motion Decoding Dataset

## Summary
This repository now includes `converted_data.pkl`, a decoder-ready dataset derived from longitudinal barrel-cortex calcium imaging and synchronized videography motion-energy measurements.

## Contents
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample conversion on 2 sessions
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`

## Conversion choices
- Neural signal: Suite2p `spks.npy` deconvolved activity
- Cell inclusion: `iscell[:,0] > 0.5`
- Behavior output: `move_deve/motion_energy_glob.npy`
- Temporal binning: averages over 10 consecutive frames
- Pseudo-trials: consecutive 2-minute windows from continuous recordings
- Decoder input: elapsed time from session start (seconds)
- Decoder output: motion energy discretized into 5 global percentile bins

## Data format
The pickle contains a dictionary with keys:
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

## Basic statistics
- Subjects: 6
- Sessions: 41
- Total pseudo-trials: 536
- Total neurons: 20445
- Brain region: barrel cortex

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl
```

Re-run conversion:
```bash
python3 -u convert_data.py converted_data.pkl --full
```
