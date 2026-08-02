# Visual Behavior Decoder Conversion

This directory contains a conversion of the locally available Allen Visual Behavior ophys NWB files into the decoder format expected by `train_decoder.py`.

## Main files

- `/app/convert_data.py`: conversion script
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: small representative subset for fast validation
- `/app/CONVERSION_NOTES.md`: processing decisions, sanity checks, and results
- `/app/train_decoder.py`: validator / training entry point

## Reproduce the full dataset

Convert:

```bash
python /app/convert_data.py --output /app/converted_data.pkl > /app/conversion_full_out.txt 2>&1
```

Verify format only:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only > /app/verification_full_out.txt 2>&1
```

Train the decoder:

```bash
python -u /app/train_decoder.py /app/converted_data.pkl > /app/train_decoder_full_out.txt 2>&1
```

Use `--cpu` if GPU memory is limited.

## Reproduce the sample dataset

The sample uses the same conversion code on a fixed experiment subset:

```bash
python /app/convert_data.py \
  --output /app/sample_data.pkl \
  --experiment-ids 775614751,792813858,794381992,795073741,795076128,788490510,796105304,951980475,958527485,957759566,960410038 \
  > /app/conversion_sample_out.txt 2>&1
```

Verify:

```bash
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only > /app/verification_sample_out.txt 2>&1
```

Train on CPU for a fast smoke test:

```bash
python -u /app/train_decoder.py /app/sample_data.pkl --cpu > /app/train_decoder_sample_out.txt 2>&1
```

## Notes

- Only NWB files already present under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments` are used.
- Each decoder session corresponds to one Allen `BehaviorOphysExperiment` file.
- Trials keep `go` and `catch`, and exclude `aborted` and `auto_rewarded`.
- Neural activity uses AllenSDK discrete calcium events.
- Time series are aligned on Allen trial start times and resampled to 100 ms bins.
