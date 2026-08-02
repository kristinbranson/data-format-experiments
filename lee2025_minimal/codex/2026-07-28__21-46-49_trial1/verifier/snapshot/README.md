# CA1 Geometry Decoder Conversion

This workspace converts the `georepca1` CA1 dataset into the decoder format expected by `train_decoder.py`.

Files produced:

- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: smaller representative subset
- `conversion_full_out.txt`: stdout from the full conversion run
- `conversion_sample_out.txt`: summary of the sample subset
- `verification_full_out.txt`: `train_decoder.py --verify-only` on the full dataset
- `verification_sample_out.txt`: `train_decoder.py --verify-only` on the sample dataset
- `train_decoder_sample_out.txt`: sample decoder training run
- `train_decoder_full_out.txt`: full decoder training run log from the launched full-dataset run; this file may still be growing if the long training job is in progress
- `CONVERSION_NOTES.md`: processing decisions and sanity checks

Key dataset choices:

- One decoder session = one recording day from one mouse.
- Each 40 min recording is split into 40 nominal 1 min trials.
- Neural data are the paper's rise-extracted CA1 event traces.
- Decoder input is a 9D static blocked-partition mask for the 3x3 arena.
- Decoder output is a time-varying 9-class 3x3 spatial bin.

Run the conversion again:

```bash
python /app/convert_data.py --stats-json /app/conversion_stats.json
```

Run verification:

```bash
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/sample_data.pkl --verify-only
```

Run decoder training:

```bash
python /app/train_decoder.py /app/sample_data.pkl --cpu
python /app/train_decoder.py /app/converted_data.pkl
```
