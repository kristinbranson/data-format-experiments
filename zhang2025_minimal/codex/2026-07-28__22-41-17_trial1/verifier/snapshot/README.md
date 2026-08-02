# Neural Decoder Conversion

This workspace contains a stimulus-aligned conversion of the IBL brain-wide-map release into the format expected by `train_decoder.py`.

## Outputs

- `converted_data.pkl`: full converted dataset.
- `sample_data.pkl`: small reproducible subset built from the first 5 included sessions.
- `convert_data.py`: conversion script.
- `CONVERSION_NOTES.md`: detailed processing and validation notes.
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

## Reproduce

```bash
python /app/convert_data.py --output /app/converted_data.pkl --summary-json /app/full_summary.json
python /app/convert_data.py --max-sessions 5 --output /app/sample_data.pkl --summary-json /app/sample_summary.json

python /app/train_decoder.py /app/sample_data.pkl --verify-only --stats-json /app/sample_verify_stats.json
python /app/train_decoder.py /app/sample_data.pkl --stats-json /app/sample_train_stats.json

python /app/train_decoder.py /app/converted_data.pkl --verify-only --stats-json /app/full_verify_stats.json
python /app/train_decoder.py /app/converted_data.pkl
```

## High-Level Summary

The converter follows the provided paper/code conventions where they are compatible with the requested task:

- trials are aligned to `stimOn_times`
- spike counts are binned into 20 ms bins over `[-0.5, 1.5]` s
- trial exclusion matches the reference mask (`choice != 0`, required events present, reaction time `0.08-2.0` s, max trial length `10` s)
- neurons are restricted to QC-passed clusters with `clusters.metrics.label >= 1`
- wheel speed uses the IBL wheel interpolation and filtered-velocity pipeline
- whisker motion energy uses the left camera when available and falls back to the right camera, matching the provided code

The full dataset contains 438 sessions, 186,245 trials, 135 subjects, and 72,757 QC-passed neurons from included sessions.
