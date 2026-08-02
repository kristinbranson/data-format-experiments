# Neural Decoder Conversion

This workspace converts the Track2p developmental barrel-cortex release into the decoder format expected by `train_decoder.py`.

Files created by the conversion workflow:

- `convert_data.py`: conversion script for full and sample datasets
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: representative sample dataset
- `CONVERSION_NOTES.md`: processing decisions, sanity checks, and validation results
- `conversion_full_out.txt`, `conversion_sample_out.txt`: conversion logs
- `verification_full_out.txt`, `verification_sample_out.txt`: format-verification logs
- `train_decoder_full_out.txt`, `train_decoder_sample_out.txt`: decoder-training logs

Usage:

```bash
python convert_data.py --mode full
python convert_data.py --mode sample
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl --cpu
```

The conversion follows the paper and release structure by:

- treating each recording day as a session
- splitting each continuous recording into consecutive 2-minute blocks, matching the paper's decoding splits
- averaging neural and behavior traces in non-overlapping 10-frame bins
- using Suite2p-style neuropil subtraction and baseline estimation to reconstruct `dF/F`
- repairing missing camera frames from `interframe_int.npy` before motion-energy binning

See `CONVERSION_NOTES.md` for exact implementation details and validation outcomes.
