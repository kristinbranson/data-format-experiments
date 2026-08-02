# Converted Neural Decoder Dataset

This repository contains a converted dataset for decoding spontaneous mouse motion energy from longitudinal two-photon calcium imaging in developing barrel cortex.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Conversion summary
- Source data: Suite2p outputs (`F.npy`, `Fneu.npy`, `iscell.npy`, `ops.npy`) plus motion-energy arrays in `move_deve/`
- Neuron filtering: keep ROIs with `iscell[:,1] > 0.5`
- Neural signal: fluorescence-based trace from neuropil-corrected fluorescence with low-percentile baseline normalization
- Temporal binning: non-overlapping 10-frame bins at 30 Hz (~333 ms)
- Trialization: consecutive 2-minute blocks per session
- Decoder input: elapsed time from start of session (seconds), time-varying
- Decoder output: motion energy discretized into 5 global equal-percentile bins, time-varying

## Data format
The pickle contains a dictionary with keys:
- `neural`, `input`, `output`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Usage
Verify format:
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Train decoder:
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

Run conversion:
```bash
python3 -u convert_data.py converted_data.pkl --full
```
