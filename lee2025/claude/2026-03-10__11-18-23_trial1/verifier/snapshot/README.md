# CA1 Neural Decoder Dataset

Converted from the georepca1 dataset (Lee, Keinath, Cianfarano & Brandon, 2025. Neuron 113(2): 307-320).

## Dataset Description

CA1 hippocampal calcium imaging recordings from 7 mice freely exploring 10 geometrically distinct environments. Each environment is a 75x75cm open field with different partitions blocked to create unique shapes (square, +, o, t, u, i, l, rectangle, bit donut, glenn).

### Task
Decode mouse position (discretized into 3x3 = 9 spatial bins) from binary calcium event traces recorded in CA1, with environment geometry provided as decoder input.

## Data Format

The converted data is saved as `converted_data.pkl` (Python pickle). Load with:

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)
```

### Structure

| Key | Type | Description |
|-----|------|-------------|
| `neural` | list of lists of arrays | Binary calcium events, shape (n_neurons, n_timepoints) per trial |
| `input` | list of lists of arrays | Environment geometry (9 values: 3x3 grid, 1=open, 0=blocked), shape (9,) per trial |
| `output` | list of lists of arrays | Position bin (0-8, row-major 3x3), shape (1, n_timepoints) per trial |
| `subjects` | list of str | 7 animal IDs |
| `subject_idx` | array | Animal index per session |
| `brain_regions` | list of str | ['CA1'] |
| `brain_region_idx` | list of arrays | Region index per neuron per session |
| `input_names` | list of str | 9 env partition names |
| `output_names` | list of str | ['position_bin'] |
| `output_values` | list of lists | [['x0y0', 'x0y1', ..., 'x2y2']] |
| `metadata` | dict | Task description, time bin size, etc. |

### Key Statistics

| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Sessions | 207 |
| Trials per session | 40 (1-minute segments) |
| Total trials | 8,280 |
| Neuron-sessions | 69,744 |
| Neurons per session | 113-564 (mean 337) |
| Time bin | 33.33 ms (30 Hz) |
| Output classes | 9 (3x3 position bins) |

### Decoder Performance

| Metric | Value |
|--------|-------|
| Validation balanced accuracy | 55.16% |
| Chance level | 11.11% (1/9) |
| Accuracy / chance | 4.96x |

## How to Run

```bash
# Convert data (full)
python3 -u convert_data.py converted_data.pkl --full

# Convert data (sample - 2 animals)
python3 -u convert_data.py sample_data.pkl --sample

# Verify data format
python3 -u train_decoder.py converted_data.pkl --verify-only

# Train decoder
python3 -u train_decoder.py converted_data.pkl --plot-samples
```

## Reference

Lee, J.Q., Keinath, A.T., Cianfarano, E., & Brandon, M.P. (2025). Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping. Neuron, 113(2), 307-320.
