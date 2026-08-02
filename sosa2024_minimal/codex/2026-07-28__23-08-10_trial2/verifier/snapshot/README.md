# Neural Decoder Conversion

This workspace contains a conversion of the supplied hippocampal NWB sessions into the decoder-ready pickle format expected by `train_decoder.py`.

Main outputs:

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: smaller representative subset
- `convert_data.py`: conversion script
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results

## Files

- `conversion_full_out.txt`: stdout from full conversion
- `conversion_sample_out.txt`: stdout from sample-dataset creation
- `verification_full_out.txt`: format verification for full dataset
- `verification_sample_out.txt`: format verification for sample dataset
- `train_decoder_full_out.txt`: decoder training run on full dataset
- `train_decoder_sample_out.txt`: decoder training run on sample dataset

## Regenerate

Run the conversion:

```bash
python /app/convert_data.py --full-only > /app/conversion_full_out.txt 2>&1
python /app/convert_data.py --sample-only --sample-from-full /app/converted_data.pkl > /app/conversion_sample_out.txt 2>&1
```

Run verification:

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only --cpu > /app/verification_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --verify-only --cpu > /app/verification_sample_out.txt 2>&1
```

Run decoder training:

```bash
python /app/train_decoder.py /app/converted_data.pkl --cpu > /app/train_decoder_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --cpu > /app/train_decoder_sample_out.txt 2>&1
```

## Summary

- Full dataset: `152` sessions, `12135` kept trials, `11` subjects
- Sample dataset: `11` sessions, `110` trials
- Full verification passed with no warnings
- Decoder training succeeded on both full and sample datasets

For the exact preprocessing and validation details, see `CONVERSION_NOTES.md`.
