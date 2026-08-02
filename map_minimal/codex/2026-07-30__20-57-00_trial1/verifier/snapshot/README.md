# MAP Decoder Conversion

This workspace converts the NWB sessions in `data/` into the decoder format expected by `train_decoder.py`.

Files:
- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small subset used for quick verification/training
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results

Re-run:

```bash
python /app/convert_data.py --full-out /app/converted_data.pkl --sample-out /app/sample_data.pkl --sample-sessions 5
python /app/train_decoder.py /app/sample_data.pkl --verify-only --plot-samples
python /app/train_decoder.py /app/sample_data.pkl --plot-samples
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl
```

Core processing choices:
- Align all trials to go cue onset.
- Use 50 ms non-overlapping bins from `-2.5 s` to `+1.5 s`.
- Keep units with `classification == "good"`.
- Exclude NWB sessions with zero good units.
- Restrict each session to the trials actually covered by ephys using `units/obs_intervals`.
- Keep all ephys-covered trials, including photostim, early-lick, and ignore trials, because the requested decoder outputs explicitly require those conditions.
