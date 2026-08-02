# Neural Decoder Conversion

This directory contains a reproducible conversion of the dataset from "A flexible hippocampal population code for experience relative to reward" into the pickle format expected by `train_decoder.py`.

The main conversion script is `/app/convert_data.py`. It reads the NWB files under `/app/data`, reconstructs framewise trials from `trial_start` to `teleport`, selects curated CA1 neurons from the published deconvolved `plane0` response series, and writes:

- `/app/converted_data.pkl`
- `/app/sample_data.pkl`

The converted full dataset contains 11 mice, 152 sessions, and 12,147 kept trials. All sessions use the same frame-aligned bin size of 64.483627 ms.

## Files

- `/app/convert_data.py`: conversion code
- `/app/converted_data.pkl`: full converted dataset
- `/app/sample_data.pkl`: smaller representative subset for fast checks
- `/app/CONVERSION_NOTES.md`: detailed processing decisions, sanity checks, and validation results
- `/app/conversion_full_out.txt`: stdout from the full conversion run
- `/app/conversion_sample_out.txt`: sample-dataset portion of the conversion output
- `/app/verification_full_out.txt`: `train_decoder.py --verify-only` output for the full dataset
- `/app/verification_sample_out.txt`: `train_decoder.py --verify-only` output for the sample dataset
- `/app/train_decoder_full_out.txt`: full decoder training output on the full dataset
- `/app/train_decoder_sample_out.txt`: full decoder training output on the sample dataset

## Regeneration

Run these commands from `/app`:

```bash
python /app/convert_data.py > /app/conversion_full_out.txt 2>&1
awk 'BEGIN{print "Sample dataset summary extracted from convert_data.py output"; print ""} /^Sample dataset summary:/{flag=1} flag{print}' /app/conversion_full_out.txt > /app/conversion_sample_out.txt
python /app/train_decoder.py /app/sample_data.pkl --verify-only --cpu > /app/verification_sample_out.txt 2>&1
python /app/train_decoder.py /app/converted_data.pkl --verify-only --cpu > /app/verification_full_out.txt 2>&1
python /app/train_decoder.py /app/sample_data.pkl --cpu > /app/train_decoder_sample_out.txt 2>&1
python /app/train_decoder.py /app/converted_data.pkl > /app/train_decoder_full_out.txt 2>&1
```

`train_decoder.py` will also write `sample_trials.png` during verification.

## Summary

The conversion matches the paper and reference code on the main structural points:

- Trials are aligned to trial start.
- Neural activity uses deconvolved calcium events sampled at the imaging frame rate.
- Reward-zone and environment switches are recovered with the expected 30-trial split.
- Lick-sensor error handling follows the reference code logic by dropping affected trials.
- Multi-plane mice `m17` and `m18` use only the published `plane0` response series that is actually linked in the NWB files.

See `/app/CONVERSION_NOTES.md` for the full rationale and validation details.
