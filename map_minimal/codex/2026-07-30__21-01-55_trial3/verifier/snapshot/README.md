# Neural Decoder Data Conversion

This directory contains a conversion of the MAP auditory delayed-response NWB sessions into the decoder format required by `train_decoder.py`.

Created files:

- `convert_data.py`: deterministic conversion script
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: small subset built from the first 5 included sessions
- `conversion_full_out.txt`: stdout from full conversion
- `verification_full_out.txt`: `train_decoder.py --verify-only` output for the full dataset
- `train_decoder_full_out.txt`: full decoder training output for the full dataset
- `conversion_sample_out.txt`: stdout from sample conversion
- `verification_sample_out.txt`: `train_decoder.py --verify-only` output for the sample dataset
- `train_decoder_sample_out.txt`: full decoder training output for the sample dataset
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation notes

## Reproduce

Full dataset:

```bash
python /app/convert_data.py --output /app/converted_data.pkl
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/converted_data.pkl --cpu
```

Sample dataset:

```bash
python /app/convert_data.py --output /app/sample_data.pkl --session-limit 5
python /app/train_decoder.py /app/sample_data.pkl --verify-only
python /app/train_decoder.py /app/sample_data.pkl --cpu
```

## Conversion Summary

The converter:

- aligns all trials to go cue onset
- extracts `[-2.5, 1.5)` seconds around the go cue
- bins spikes into exact 50 ms bins and stores firing rates in spikes/s
- uses decoder inputs `time_from_tone_onset_s` and `photostim_on`
- uses decoder outputs `choice`, `outcome`, `early_lick`, and discretized `tongue_y`
- keeps early-lick, ignore, and photostimulation trials because they are required decoder targets or inputs
- excludes `auto_water` and `free_water` trials, sessions with zero good units, trials without full video coverage for the aligned window, and trials that still produce all-zero neural activity after alignment

## Full Dataset Summary

- Sessions included: 173
- Sessions skipped: 1
- Total trials: 88,654
- Total good units: 69,453
- Brain regions: 293
- Time bins per trial: 80
- Full pickle size: about 5.5 GB

`train_decoder.py --verify-only` reports that the full dataset is valid with no errors or warnings.

## Sample Dataset Summary

- Sessions included: 5
- Total trials: 1,756
- Total good units: 2,117
- Brain regions: 36
- Time bins per trial: 80

`train_decoder.py --verify-only` reports that the sample dataset is valid with no errors or warnings.

For the full processing rationale and validation details, see `CONVERSION_NOTES.md`.
