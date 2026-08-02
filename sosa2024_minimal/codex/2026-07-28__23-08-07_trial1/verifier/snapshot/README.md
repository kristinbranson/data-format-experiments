# Neural Decoder Conversion

This directory contains a conversion of the Sosa, Plitt, Giocomo hippocampal NWB dataset into the decoder format expected by `train_decoder.py`.

Files:
- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: smaller subset for quick validation
- `CONVERSION_NOTES.md`: processing decisions and sanity checks
- `conversion_full_out.txt`: summary of the full converted dataset
- `conversion_sample_out.txt`: summary of the sample subset
- `verification_full_out.txt`: `train_decoder.py --verify-only` output for the full dataset
- `verification_sample_out.txt`: `train_decoder.py --verify-only` output for the sample dataset
- `train_decoder_full_out.txt`: full decoder training output
- `train_decoder_sample_out.txt`: sample decoder training output

Reproduction:

```bash
python /app/convert_data.py --mode both
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/sample_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --cpu
python /app/train_decoder.py /app/sample_data.pkl --cpu
```

Dataset summary:
- Full dataset: 152 sessions, 11 mice, 12,216 trials.
- Sample dataset: 8 sessions from `m3`, 670 trials.
- Brain region: `CA1`.
- Neural signal: NWB-exported deconvolved calcium events.
- Temporal alignment: trial start / entry onto the 0 cm corridor position.
