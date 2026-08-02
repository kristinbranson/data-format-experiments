# Neural Decoder Data Conversion

This workspace converts the NWB release in `/app/data` into the pickle structure expected by `/app/train_decoder.py`.

Files produced:

- `/app/convert_data.py`: conversion script
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: small subset for quick checks
- `/app/conversion_summary.json`: per-session and global conversion summary
- `/app/conversion_full_out.txt`: stdout from full conversion
- `/app/conversion_sample_out.txt`: stdout from sample conversion summary checks
- `/app/verification_full_out.txt`: `train_decoder.py --verify-only` on the full dataset
- `/app/verification_sample_out.txt`: `train_decoder.py --verify-only` on the sample dataset
- `/app/train_decoder_full_out.txt`: full decoder training output
- `/app/train_decoder_sample_out.txt`: sample decoder training output
- `/app/CONVERSION_NOTES.md`: processing decisions and validation notes

## Rerun Conversion

```bash
python /app/convert_data.py > /app/conversion_full_out.txt 2>&1
```

## Verify Data Format

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only > /app/verification_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --verify-only > /app/verification_sample_out.txt 2>&1
```

## Train Decoder

```bash
python /app/train_decoder.py /app/sample_data.pkl > /app/train_decoder_sample_out.txt 2>&1
python /app/train_decoder.py /app/converted_data.pkl > /app/train_decoder_full_out.txt 2>&1
```

Add `--cpu` if GPU memory is constrained.

## Dataset Summary

The current converted dataset contains 173 usable sessions, 89,068 kept trials, 28 subjects, and 69,453 classifier-good units. All trials are aligned to go cue onset with 50 ms bins over `[-2.5 s, +1.5 s]`.
