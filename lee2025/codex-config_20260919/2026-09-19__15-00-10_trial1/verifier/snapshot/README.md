# CA1 Neural Decoder Dataset

This directory contains a decoder-ready conversion of the Lee et al. (2025) dataset, *Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping*. Mice freely explored ten geometries made by blocking partitions of a 75 x 75 cm arena while dorsal CA1 calcium activity and head position were recorded at 30 Hz.

## Main files

- `converted_data.pkl`: complete converted dataset (6.173 GiB).
- `sample_data.pkl`: first two sessions for quick tests.
- `convert_data.py`: reproducible converter.
- `CONVERSION_NOTES.md`: detailed decisions, source comparisons, sanity checks, and decoder results.
- `verification_full_out.txt`: full format/statistics validation.
- `train_decoder_full_out.txt`: completed full decoder training log.

## Loading

```python
import pickle

with open("/app/converted_data.pkl", "rb") as f:
    data = pickle.load(f)

neural_trial = data["neural"][0][0]  # (n_neurons, 600)
geometry = data["input"][0][0]       # (9,), 1=blocked
position = data["output"][0][0]      # (1, 600), integer 0..8
```

Loading the full pickle requires more than its 6.2 GB on-disk size in available memory. Use `sample_data.pkl` for lightweight inspection.

## Representation

- Each source recording day is one session; each session is split into consecutive complete 60-second trials.
- Neural rows are all cells registered in that session. Released binary calcium-transient rise events are Gaussian-smoothed with sigma 3 source frames and averaged in non-overlapping 3-frame bins, matching the paper's reference decoder preprocessing.
- The resulting time bin is 100 ms (10 Hz), giving exactly 600 timepoints per trial.
- Decoder inputs are nine static binary features named `blocked_x0_y0` through `blocked_x2_y2`. Source blocked-grid axes are transposed to match position `(x,y)` indexing.
- The output is one time-varying categorical variable. Both coordinates are divided into 25 cm bins and encoded as `position_class = 3*x_bin + y_bin`. `output_values` names classes `x0_y0` through `x2_y2`.
- Only terminal segments shorter than 60 seconds are omitted. Unregistered all-NaN cell rows are excluded; no place-cell, activity, movement, trial, session, or subject filtering is added.

## Key statistics

| Statistic | Value |
|---|---:|
| Subjects | 7 |
| Sessions | 207 |
| Longitudinal cell identities in source | 5,413 |
| Registered session-neuron instances | 69,744 |
| Neurons/session | mean 336.93, range 113-564 |
| Complete one-minute trials | 8,187 |
| Trials/session | 39 or 40 |
| Timepoints/trial | 600 |
| Position-class fractions | 0.0997, 0.0754, 0.1157, 0.0985, 0.0573, 0.1412, 0.1351, 0.0768, 0.2003 |

The full validator reported no errors or warnings. Full decoder validation balanced accuracy was **0.6088**, compared with uniform chance **0.1111**; training balanced accuracy was 0.6965.

## Reproduction and validation

```bash
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/convert_data.py /app/converted_data.pkl --full
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

The converter processes animals sequentially and validates shapes, dtypes, finiteness, class ranges, and trial alignment before writing. Independent raw-file comparisons are archived under `cache/`.
