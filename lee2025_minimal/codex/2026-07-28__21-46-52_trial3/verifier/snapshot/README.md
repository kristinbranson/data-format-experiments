# Neural Decoder Conversion

This workspace converts the CA1 geometry-remapping dataset from Lee, Keinath, Cianfarano and Brandon (2025) into the decoder format required by `train_decoder.py`.

Files:
- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: sample subset used for fast checks
- `CONVERSION_NOTES.md`: processing decisions and validation summary
- `conversion_full_out.txt`: stdout from full conversion run
- `conversion_sample_out.txt`: stdout from sample conversion run
- `verification_full_out.txt`: stdout from `train_decoder.py --verify-only` on full data
- `verification_sample_out.txt`: stdout from `train_decoder.py --verify-only` on sample data
- `train_decoder_full_out.txt`: stdout from full decoder training
- `train_decoder_sample_out.txt`: stdout from sample decoder training

Core representation:
- `neural`: CA1 rise-event traces, restricted to movement frames, Gaussian-smoothed and pooled to 100 ms bins
- `input`: static 3x3 environment-open mask for each trial
- `output`: time-varying mouse position as a single 9-class `position_bin`

How to rerun:
```bash
python convert_data.py
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl
```

Sample subset:
- `sample_data.pkl` contains the first 11 sessions from `QLAK-CA1-08`, covering one full square-to-square geometry sequence.
