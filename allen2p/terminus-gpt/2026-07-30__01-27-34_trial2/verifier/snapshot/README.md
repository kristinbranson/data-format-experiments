# Converted Allen Visual Behavior Ophys Dataset

This repository contains a converted decoder-ready dataset saved as `converted_data.pkl`.

## Files
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion notes and validation results
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`

## Dataset summary
- Sessions: 284
- Subjects: 38
- Trials: 963758
- Neurons: 42147
- Brain regions: VISl, VISp

## Outputs decoded
- image_identity
- image_change
- running_speed_bin
- pupil_diameter_bin
- trial_outcome

## Usage
Run verification:

```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

Run decoder training:

```bash
python3 train_decoder.py converted_data.pkl --cpu
```
