# Visual Behavior Conversion

This workspace contains a converted Allen Visual Behavior ophys dataset for the decoder interface in `train_decoder.py`.

## Files

- `convert_data.py`: conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small stratified sample dataset
- `CONVERSION_NOTES.md`: cohort definition, processing decisions, sanity checks, and validation results
- `conversion_full_out.txt`: stdout from the full conversion run
- `conversion_sample_out.txt`: stdout from the sample conversion run
- `verification_full_out.txt`: `train_decoder.py --verify-only` output for the full dataset
- `verification_sample_out.txt`: `train_decoder.py --verify-only` output for the sample dataset
- `train_decoder_full_out.txt`: full decoder training output on `converted_data.pkl`
- `train_decoder_sample_out.txt`: full decoder training output on `sample_data.pkl`

## Core choices

- Neural signal: AllenSDK `events` traces
- Trial filter: keep `go` and `catch`; drop `aborted` and `auto_rewarded`
- Time axis: successive active `change_detection` stimulus intervals, median duration `0.75061 s`
- Image labels: image identity per interval, with `omitted` retained as its own category
- Continuous outputs: running speed and blink-filtered pupil width averaged per interval, then discretized into global quintiles

## Re-run

```bash
python convert_data.py
python train_decoder.py converted_data.pkl --verify-only --cpu
python train_decoder.py converted_data.pkl
```

For the sample dataset:

```bash
python convert_data.py --mode sample --sample-sessions 9 --sample-trials 18 --output sample_data.pkl
python train_decoder.py sample_data.pkl --verify-only --cpu
python train_decoder.py sample_data.pkl --cpu
```
