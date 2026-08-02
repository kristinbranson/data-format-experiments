# Converted Neural Decoder Dataset

This repository contains a converted neuroscience dataset saved in `converted_data.pkl` for decoder training with `train_decoder.py`.

## Contents
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample converted dataset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: detailed conversion log and validation notes
- `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`

## Dataset summary
- Sessions kept: 33
- Subjects kept: 12
- Total trials: 8598
- Total neurons: 4176
- Brain region: ALM
- Alignment event: go cue onset
- Time bin size: 75 ms

## Decoder variables
### Input
- `time_from_go_cue`

### Outputs
- `lick_direction`: left/right
- `behavioral_context`: WC/DR
- `outcome`: incorrect/correct
- `tongue_velocity`: low/high (per-session median split)
- `paw_velocity`: low/high (per-session median split)
- `motion_energy`: low/high (per-session median split)

## Load the dataset
```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

## Run verification
```bash
python3 train_decoder.py converted_data.pkl --verify-only
```

## Run decoder training
```bash
python3 train_decoder.py converted_data.pkl --plot-samples
```

## Conversion notes
See `CONVERSION_NOTES.md` for details on reference code alignment, filtering, sanity checks, and debugging history.
