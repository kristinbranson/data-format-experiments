# Neural Decoder Conversion

This workspace converts the IBL brain-wide-map cache in `/app/data/one_cache` into the decoder format required by `train_decoder.py`.

Files produced:

- `convert_data.py`: offline conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: first 8 converted sessions, remapped to only the used subjects/regions
- `conversion_full_out.txt`: stdout from the full conversion run
- `conversion_sample_out.txt`: summary of the derived sample dataset
- `verification_full_out.txt`: `train_decoder.py --verify-only` on the full dataset
- `verification_sample_out.txt`: `train_decoder.py --verify-only` on the sample dataset
- `train_decoder_sample_out.txt`: decoder training log for the sample dataset
- `train_decoder_full_out.txt`: decoder training log for the full dataset
- `CONVERSION_NOTES.md`: detailed processing notes and validation summary

Reproduction commands:

```bash
python -u convert_data.py \
  --output /app/converted_data.pkl \
  --sample-output /app/sample_data.pkl \
  --sample-sessions 8 \
  --stats-json /app/conversion_stats_full.json \
  > /app/conversion_full_out.txt 2>&1

python train_decoder.py /app/sample_data.pkl --verify-only --cpu > /app/verification_sample_out.txt 2>&1
python train_decoder.py /app/sample_data.pkl --cpu > /app/train_decoder_sample_out.txt 2>&1
python train_decoder.py /app/converted_data.pkl --verify-only --cpu > /app/verification_full_out.txt 2>&1
python train_decoder.py /app/converted_data.pkl --cpu > /app/train_decoder_full_out.txt 2>&1
```
