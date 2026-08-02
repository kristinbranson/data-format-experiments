# Neural Decoder Conversion

Files produced by the conversion pipeline:

- `convert_data.py`: converts the Zhong et al. imaging data into the decoder-ready format
- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: smaller subset for quick verification and training
- `CONVERSION_NOTES.md`: loading decisions, reductions, and validation summary
- `conversion_full_out.txt`: stdout from the full conversion run
- `conversion_sample_out.txt`: stdout from the sample conversion run
- `verification_full_out.txt`: `train_decoder.py --verify-only` on `converted_data.pkl`
- `verification_sample_out.txt`: `train_decoder.py --verify-only` on `sample_data.pkl`
- `train_decoder_full_out.txt`: decoder training run on `converted_data.pkl`
- `train_decoder_sample_out.txt`: decoder training run on `sample_data.pkl`

Usage:

```bash
python /app/convert_data.py
python /app/train_decoder.py /app/converted_data.pkl --verify-only
python /app/train_decoder.py /app/sample_data.pkl --plot-samples
```

The converter uses the imaging-session metadata in `data/beh/Imaging_Exp_info.npy`, canonicalizes duplicate behavior entries across the paper’s figure-specific `.npy` files, keeps only running frames inside the textured corridor, and then downsamples for decoder tractability.
