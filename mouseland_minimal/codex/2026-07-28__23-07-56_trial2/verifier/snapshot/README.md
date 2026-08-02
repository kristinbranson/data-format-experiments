# Neural Decoder Conversion

This directory contains a trialized decoder dataset derived from the Zhong et al. imaging release in `data/`, using the task and preprocessing conventions described in `methods.txt`, `paper.pdf`, and the supplied analysis code in `code/`.

## Files

- `convert_data.py`: conversion script.
- `converted_data.pkl`: full converted dataset.
- `sample_data.pkl`: smaller subset for quick checks.
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results.
- `conversion_full_out.txt`: stdout from the full conversion run.
- `conversion_sample_out.txt`: stdout summary for the sample dataset.
- `verification_full_out.txt`: `train_decoder.py --verify-only` output for the full dataset.
- `verification_sample_out.txt`: `train_decoder.py --verify-only` output for the sample dataset.
- `train_decoder_full_out.txt`: full decoder training output.
- `train_decoder_sample_out.txt`: sample decoder training output.

## Re-run

```bash
python convert_data.py
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --cpu
```

## Conversion Summary

- Sessions are reconstructed at the level of the 89 physical recordings in `data/spk`, not the 99 analysis-specific behavior views.
- Only moving frames inside the corridor are used, matching the paper’s running-only corridor analyses.
- Decoder trials are summarized over the 4 required 1 m corridor bins.
- Visual-cortex neurons are retained from retinotopy labels and then capped deterministically per session to keep the exported trial structure tractable for downstream decoding.

See `CONVERSION_NOTES.md` for details and final validation statistics.
