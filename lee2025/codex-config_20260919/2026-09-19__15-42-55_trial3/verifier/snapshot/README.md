# CA1 Geometry Decoder Dataset

This directory contains a decoder-ready conversion of the dataset from *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping*. It predicts a mouse's 3x3 arena position from dorsal CA1 calcium-event activity. Arena geometry is supplied as a static nine-partition decoder input.

## Main files

- `converted_data.pkl`: complete converted dataset (6.173 GiB)
- `sample_data.pkl`: two-session test subset
- `convert_data.py`: reproducible conversion script
- `CONVERSION_NOTES.md`: source audit, decisions, checks, statistics, and decoder results
- `verification_full_out.txt`: full format-validation report
- `train_decoder_full_out.txt`: complete full-dataset training log
- `processing_QLAK-CA1-08_day00.png` and `processing_QLAK-CA1-08_day01.png`: conversion diagnostics
- `sample_trials.png` and `predictions.png`: full decoder diagnostics

## Dataset summary

| Quantity | Value |
|---|---:|
| Subjects | 7 mice |
| Sessions | 207 animal-days |
| Trials | 8,187 non-overlapping complete minutes |
| Trials/session | 39 for 93 sessions; 40 for 114 sessions |
| Unique within-animal CellReg identities | 5,413 |
| Session-neuron instances | 69,744 |
| Neurons/session | mean 336.93; range 113--564 |
| Source sampling | 30 Hz |
| Converted sampling | 10 Hz (100 ms/bin) |
| Timepoints/trial | 600 |
| Brain region | CA1 |

The source `trace` is the paper's already-preprocessed binary rising-phase calcium-event series. For consistency with the reference position decoder, activity is Gaussian-smoothed with sigma 3 source frames and mean-pooled in non-overlapping three-frame windows. Smoothing is done separately inside each one-minute trial to prevent leakage across trial splits. Position is mean-pooled over identical windows and discretized into 25-cm bins.

## Load and inspect

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

session = 0
trial = 0
neural = data["neural"][session][trial]  # (n_neurons, 600), float32
geometry = data["input"][session][trial] # (9,), float32; 1 means blocked
position = data["output"][session][trial] # (1, 600), uint8; values 0..8
```

Position class `3*y_bin + x_bin` uses row-major labels:

```text
0 1 2
3 4 5
6 7 8
```

The corresponding names are `y0_x0` through `y2_x2`. `input_names` use `blocked_partition_0` through `blocked_partition_8` in the same order. `metadata.session_info` preserves animal/day identifiers, arena name, blocked indices, raw/used/tail frame counts, trial starts, and retained source-neuron indices.

## Reproduce and validate

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/train_decoder.py /app/sample_data.pkl

python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The supplied full validation reports no errors or warnings. Full decoder loss fell from 2.318670 to 1.156360. Balanced position accuracy was 0.6982 on training data and 0.6078 on validation data, versus 0.1111 uniform chance.

See `CONVERSION_NOTES.md` for the rationale for complete-minute segmentation, cell/frame inclusion, coordinate orientation, boundary handling, independent raw-data `np.allclose` checks, and comparison with the paper's 15x15 Euclidean-error decoder.
