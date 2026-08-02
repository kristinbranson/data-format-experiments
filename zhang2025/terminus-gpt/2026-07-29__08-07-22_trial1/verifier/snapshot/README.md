# Converted IBL Brain-Wide Map Dataset

This repository contains a converted version of the IBL brain-wide map ephys dataset formatted for `train_decoder.py`.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small sample dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes

## Decoder task
Neural activity is aligned to stimulus onset and used to decode:
- choice
- prior probability of left
- wheel speed bin
- whisker motion energy bin

## Data format
The pickle contains a dictionary with keys:
- `neural`
- `input`
- `output`
- `subjects`, `subject_idx`
- `brain_regions`, `brain_region_idx`
- `input_names`, `output_names`, `output_values`
- `metadata`

## Notes
- Time bin size: 20 ms
- Alignment event: stimulus onset
- Per-trial variables are broadcast across time bins
- Brain regions are currently stored as `unknown` in this conversion
