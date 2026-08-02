# Neural Decoder Conversion

This workspace contains a Python conversion of the two-context ALM electrophysiology dataset from Hasnain, Birnbaum et al. into the decoder format expected by `train_decoder.py`.

## Files

- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: smaller stratified subset for quick checks
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results
- `conversion_full_out.txt`: stdout from the conversion run
- `conversion_sample_out.txt`: duplicate conversion stdout for the sample artifact requirement
- `verification_full_out.txt`: `train_decoder.py --verify-only --plot-samples` on the full dataset
- `verification_sample_out.txt`: `train_decoder.py --verify-only --plot-samples` on the sample dataset
- `train_decoder_full_out.txt`: full decoder training log
- `train_decoder_sample_out.txt`: sample decoder training log

## Reproduce

```bash
python convert_data.py --stats-json conversion_stats.json
python train_decoder.py sample_data.pkl --verify-only --plot-samples --cpu
python train_decoder.py sample_data.pkl --cpu
python train_decoder.py converted_data.pkl --verify-only --plot-samples --cpu
python train_decoder.py converted_data.pkl --cpu
```

## Dataset Summary

- Sessions: 12
- Subjects: 7
- Trials: 2415
- Units after filtering: 520
- Neural bins: 10 ms
- Alignment: `goCue` field from the released session objects

Full details are in `CONVERSION_NOTES.md`.
