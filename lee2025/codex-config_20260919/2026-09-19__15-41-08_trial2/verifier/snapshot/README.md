# CA1 Geometry Neural Decoder Dataset

This workspace converts the Lee, Keinath, Cianfarano, and Brandon dataset from *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping* into the supplied neural-decoder format. The task predicts a mouse's location in a 3x3 arena grid from dorsal CA1 activity, with the blocked arena geometry supplied as context.

## Main files

- `converted_data.pkl`: full converted dataset (6.205 GiB)
- `sample_data.pkl`: two-session test subset
- `convert_data.py`: reproducible converter
- `CONVERSION_NOTES.md`: processing rationale, source comparisons, sanity checks, and validation results
- `verification_full_out.txt`: full format-verification log
- `train_decoder_full_out.txt`: completed full decoder-training log

## Dataset summary

| Item | Value |
|---|---:|
| Mice | 7 |
| Recording sessions | 207 |
| Complete one-minute trials | 8,187 |
| Time bins per trial | 600 |
| Time-bin width | 100 ms |
| Unique within-animal registered cells | 5,413 |
| Session-specific neuron instances | 69,744 |
| Neurons/session | 113-564 (mean 336.93) |
| Decoder inputs | 9 static blocked-location bits |
| Decoder output | one time-varying 9-class position variable |

All nine position classes occur. Their full-data fractions are `[0.099663, 0.075412, 0.115696, 0.098495, 0.057266, 0.141213, 0.135121, 0.076848, 0.200287]`.

## Processing

- Each recording day is one session. Consecutive complete 60-second windows are trials; only an incomplete final window is discarded.
- Supplied binary calcium rising-phase events are used directly—dF/F is not recomputed. Following the paper's position decoder, activity is Gaussian-smoothed with sigma 3 source frames and mean-pooled over three 30 Hz frames. Position is mean-pooled over the identical frames.
- All manually curated cells registered on a given day are retained. All-NaN absent registrations are removed; no place-cell-only selection is applied.
- Position is clipped to the 75 cm arena, divided into 25 cm bins, and encoded as `class = x_bin * 3 + y_bin`.
- Native blocked-partition matrices are stored as `(y,x)` and are transposed to the decoder's `(x,y)` class order. Input value 1 means blocked.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

# Session 0, trial 0
neural = data["neural"][0][0]   # float32, (n_neurons, 600)
context = data["input"][0][0]   # float32, (9,), 1 = blocked
position = data["output"][0][0] # int64, (1, 600), values 0..8
```

Session provenance—including subject, day, environment, original/used frames, discarded tail, neuron count, and blocked locations—is in `data["metadata"]["session_info"]`.

## Reproducing and validating

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

Final full decoder balanced accuracy was 0.6979 on training trials and 0.6083 on held-out trials, versus 0.1111 chance. Training loss decreased from 2.324551 to 1.157662 over 200 epochs.

