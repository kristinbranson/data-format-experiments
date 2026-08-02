# Neural Decoder Conversion

This workspace contains a conversion of the `georepca1` CA1 dataset into the decoder format required by `train_decoder.py`.

Created files:

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `CONVERSION_NOTES.md`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

Reproducible commands:

```bash
python /app/convert_data.py --sanity-json /app/conversion_full_stats.json
python /app/train_decoder.py /app/converted_data.pkl --verify-only --cpu --stats-json /app/verification_full_stats.json
python /app/train_decoder.py /app/sample_data.pkl --verify-only --cpu --stats-json /app/verification_sample_stats.json
python /app/train_decoder.py /app/converted_data.pkl --cpu --stats-json /app/train_decoder_full_stats.json
python /app/train_decoder.py /app/sample_data.pkl --cpu --stats-json /app/train_decoder_sample_stats.json
```

Key results:

- Full dataset matches the paper headline counts exactly: `207` sessions, `5413` unique neurons, `69744` session-by-neuron maps.
- Full converted dataset: `8187` one-minute trials.
- Full decoder validation balanced accuracy: `0.5729` for `position_bin` vs `0.1111` uniform chance.
- Sample decoder validation balanced accuracy: `0.4748` for `position_bin` vs `0.1111` uniform chance.

Details of the conversion choices and sanity checks are in `CONVERSION_NOTES.md`.
