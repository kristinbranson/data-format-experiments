# CA1 Geometric Deformation Dataset - Decoder Format

## Dataset Description

Converted calcium imaging data from hippocampal subregion CA1 during a geometric deformation task. Original data from:

> Lee, Keinath, Cianfarano & Brandon (2025). "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping." Neuron 113(2): 307-320.

7 mice navigated a 75x75 cm square arena with systematic 3x3 partition manipulations creating 10 distinct geometries. Neural populations in CA1 were recorded with miniscope calcium imaging (GCaMP6f) at 30 Hz across 207 sessions.

## Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 7 mice |
| Sessions | 207 (31 per mouse, 21 for one) |
| Trials | 8,187 (1-minute segments) |
| Neurons per session | 113-564 (mean 337) |
| Total neuron-sessions | 69,744 |
| Geometries | 10 (square, o, t, u, rectangle, +, i, l, bit donut, glenn) |
| Time bin | 33.33 ms (30 Hz) |
| Timepoints per trial | 1,800 |

## Decoder Task

- **Input**: Environment geometry as flattened 3x3 binary matrix (9 values). Static per trial.
- **Output**: Mouse position discretized into 3x3 spatial bins (9 classes). Time-varying.
- **Neural**: Binary calcium events (0/1) from CA1, at native 30 Hz frame rate.

## Loading the Data

```python
import pickle

with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access neural data: list of sessions, each a list of trials
neural_session0_trial0 = data['neural'][0][0]  # shape: (n_neurons, 1800)

# Access inputs (environment geometry)
input_session0_trial0 = data['input'][0][0]  # shape: (9,)

# Access outputs (position bins)
output_session0_trial0 = data['output'][0][0]  # shape: (1, 1800)

# Metadata
print(data['subjects'])        # ['QLAK-CA1-08', ...]
print(data['brain_regions'])   # ['CA1']
print(data['input_names'])     # ['partition_0', ..., 'partition_8']
print(data['output_names'])    # ['position']
print(data['output_values'])   # [['x[0-25]_y[0-25]', ...]]
```

## Output Format Specification

See `CONVERSION_NOTES.md` for full details on the data dictionary structure.

## Files

| File | Description |
|------|-------------|
| `converted_data.pkl` | Full converted dataset (207 sessions) |
| `sample_data.pkl` | Sample dataset (2 sessions) |
| `convert_data.py` | Conversion script |
| `CONVERSION_NOTES.md` | Detailed conversion notes and validation |
| `train_decoder.py` | Decoder training script |
| `decoder.py` | Decoder module |

## Decoder Results

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| position | 0.6229 | 0.5513 | 0.1111 |

## Running the Conversion

```bash
# Sample (2 sessions, fast)
python3 -u convert_data.py sample_data.pkl --sample

# Full (207 sessions, ~4 min)
python3 -u convert_data.py converted_data.pkl --full

# With processing plots
python3 -u convert_data.py sample_data.pkl --sample --show-processing
```

## Running the Decoder

```bash
# Verify data format
python3 -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python3 -u train_decoder.py converted_data.pkl --cpu
```
