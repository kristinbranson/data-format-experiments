# Neural Decoder Conversion

This workspace converts the Zhong et al. 2025 imaging data into the decoder format expected by `train_decoder.py`.

Files produced:

- `convert_data.py`
- `converted_data.pkl`
- `sample_data.pkl`
- `CONVERSION_NOTES.md`
- `conversion_full_out.txt`
- `conversion_sample_out.txt`
- `verification_full_out.txt`
- `verification_sample_out.txt`
- `train_decoder_full_out.txt`
- `train_decoder_sample_out.txt`

Main rerun commands:

```bash
python convert_data.py
python train_decoder.py sample_data.pkl --verify-only
python train_decoder.py sample_data.pkl
python train_decoder.py converted_data.pkl --verify-only
python train_decoder.py converted_data.pkl
```

The exported dataset uses the rewarded task-mouse imaging sessions only, because the decoder input specification requires per-trial reward availability. Neural activity is built from running-only, position-interpolated deconvolved traces, then restricted to retinotopically defined visual-cortex neurons with strong familiar rewarded-vs-unrewarded selectivity so the provided decoder can train tractably.
