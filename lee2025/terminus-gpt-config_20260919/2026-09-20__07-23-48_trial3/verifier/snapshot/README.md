# CA1 Geometry Dataset Conversion

## Dataset

This conversion reformats the released data from Lee, Keinath, Cianfarano, and Brandon (2025), *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping*, for neural decoding.

Seven mice explored ten geometries formed by blocking parts of a 75 × 75 cm arena. Dorsal CA1 calcium activity and head position were recorded simultaneously at 30 Hz in approximately 40-minute sessions. The released neural stream is the authors' quality-controlled binary rising-phase event representation.

## Files

- `converted_data.pkl`: full converted dataset
- `sample_data.pkl`: first two animals for rapid testing
- `convert_data.py`: reproducible conversion
- `CONVERSION_NOTES.md`: detailed decisions, reference comparisons, validation, and review
- `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt`: required run logs
- `sample_trials.png`, `predictions.png`: full-decoder visualizations
- `processing_*.png`: sample processing/alignment visualizations
- `cache/`: investigation scripts and outputs

## Key Statistics

| Statistic | Value |
|---|---:|
| Subjects | 7 |
| Sessions | 207 |
| One-minute trials | 8,187 |
| Trials/session | 39–40 |
| Time bins/trial | 600 |
| Time-bin size | 100 ms |
| Session-neurons | 69,744 |
| Neurons/session | 113–564 (mean 336.93) |
| Decoder inputs | 9 blocked-partition indicators |
| Decoder output | one 9-class time-varying position variable |
| Full validation balanced accuracy | 0.6084 (chance 0.1111) |

## Loading

```python
import pickle

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
print(data['neural'][0][0].shape)  # (n_neurons, 600)
print(data['input'][0][0].shape)   # (9,), static geometry
print(data['output'][0][0].shape)  # (1, 600), values 0..8
```

The dictionary fields follow the requested schema:

- `neural[session][trial]`: float32 smoothed event activity, neurons × time
- `input[session][trial]`: float32 9-vector, where 1 means the row-major arena partition is blocked
- `output[session][trial]`: uint8 mouse-position class, shape 1 × time; class `row*3 + col`
- `subjects`, `subject_idx`: animal identity and session mapping
- `brain_regions`, `brain_region_idx`: all neurons are dorsal CA1
- `input_names`, `output_names`, `output_values`: variable labels
- `metadata`: processing, alignment, timing, and per-session provenance

## Processing Summary

1. Load each animal's compressed joblib file using the reference representation.
2. Treat each animal-day as one session.
3. Keep every neuron with a finite trace on that day; remove all-NaN unregistered rows only.
4. Match the reference decoder by Gaussian-smoothing binary traces with sigma 3 source frames and mean-pooling non-overlapping groups of 3 frames.
5. Mean-pool position over the identical frame groups and discretize x-y into natural 25 cm arena partitions. Output labels are row-major (`y_bin*3 + x_bin`).
6. Encode raw `blocked` indices as a static 9-element input.
7. Split each recording into complete consecutive 60-second trials; discard only the incomplete final remainder.
8. Snap 152 of 4,912,200 pooled positions (0.0031%) that interpolation placed in blocked cells to the nearest accessible partition, following the reference decoder's nearest-valid-bin cleanup principle.

## Reproducing and Validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Final format verification reports no errors or warnings. See `CONVERSION_NOTES.md` for independent raw-file `np.allclose` checks and complete rationale.
