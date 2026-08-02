# Neural Decoder Conversion

This directory contains a reproducible conversion of the paper dataset into the pickle format expected by `train_decoder.py`.

Files:
- `convert_data.py`: conversion script.
- `converted_data.pkl`: full converted dataset.
- `sample_data.pkl`: smaller 4-session subset for quick checks.
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results.
- `conversion_full_out.txt`: stdout from running `python convert_data.py`.
- `conversion_sample_out.txt`: summary of the saved sample subset.
- `verification_full_out.txt`: stdout from `python train_decoder.py converted_data.pkl --verify-only --cpu`.
- `verification_sample_out.txt`: stdout from `python train_decoder.py sample_data.pkl --verify-only --cpu`.
- `train_decoder_full_out.txt`: stdout from `python train_decoder.py converted_data.pkl --cpu`.
- `train_decoder_sample_out.txt`: stdout from `python train_decoder.py sample_data.pkl --cpu`.

Regeneration:

```bash
python convert_data.py
python train_decoder.py converted_data.pkl --verify-only --cpu
python train_decoder.py sample_data.pkl --verify-only --cpu
python train_decoder.py converted_data.pkl --cpu
python train_decoder.py sample_data.pkl --cpu
```

Dataset summary:
- Sessions: 12
- Subjects: 7 packaged subject IDs
- Usable trials: 2,415
- Retained ALM units: 520
- Time window: `[-3.0, 2.5]` s from go cue
- Time bin: `10 ms`

Decoder summary:
- Full validation balanced accuracy: lick direction `0.6285`, context `0.7673`, outcome `0.6272`, tongue velocity `0.8314`, paw velocity `0.6182`, motion energy `0.7922`.
- Sample validation balanced accuracy: lick direction `0.6322`, context `0.7536`, outcome `0.6227`, tongue velocity `0.8201`, paw velocity `0.5926`, motion energy `0.8088`.
