# Track2p Decoder Conversion

This workspace contains a conversion of the Majnik et al. 2025 Track2p developmental barrel-cortex dataset into the decoder format expected by `train_decoder.py`.

## Files

- `convert_data.py`: conversion script.
- `converted_data.pkl`: full converted dataset with all 41 sessions.
- `sample_data.pkl`: sample converted dataset containing the first session from each of the 6 subjects.
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results.
- `conversion_sample_out.txt`: stdout from sample conversion.
- `verification_sample_out.txt`: stdout from sample `--verify-only` validation.
- `train_decoder_sample_out.txt`: stdout from sample decoder training.
- `conversion_full_out.txt`: stdout from full conversion.
- `verification_full_out.txt`: stdout from full `--verify-only` validation.
- `train_decoder_full_out.txt`: stdout from full decoder training.

## Reproduce

Sample export:

```bash
python convert_data.py --mode sample --output sample_data.pkl --summary-json sample_summary.json --cpu
python train_decoder.py sample_data.pkl --verify-only --cpu
python train_decoder.py sample_data.pkl --cpu
```

Full export:

```bash
python convert_data.py --mode full --output converted_data.pkl --summary-json full_summary.json --cpu
python train_decoder.py converted_data.pkl --verify-only --cpu
python train_decoder.py converted_data.pkl --cpu
```

## Conversion Summary

- Neural traces use the Track2p-provided matched-cell Suite2p outputs and are processed with Suite2p-style neuropil subtraction plus baseline correction using the parameters stored in each session’s `ops.npy`.
- Motion energy is aligned to imaging frames, repaired only for sessions where the motion trace is shorter than the imaging trace, then linearly interpolated over inserted missing values.
- Both neural and motion traces are averaged in non-overlapping bins of 10 imaging frames to match the paper’s decoding preprocessing.
- Sessions are split into consecutive 2-minute trials, matching the paper’s block-wise decoding unit.
- Decoder input is absolute time from session start.
- Decoder output is per-session min-max normalized motion energy, discretized into five global quintile bins within each export.
